/* Whisper-Stufe-3 M3: Ein Encoder-Layer als TPU-Package-Kette mit CPU-Glue.
 * Modelle in tpud-Reihenfolge: 0=proj(16x8) 1=core(16x8, geteilt) 2=woc(int8) 3=ffnresc(int8).
 * Glue: Requant proj->core, q-Slice-Tausch je Block, Division num/den + Head-Transpose,
 * Residual x1=x+att, Requant je Kante. Verify gegen out_ref.f32 (f32), Stufen-Timing.
 *   wsp_layer <x0.in.bin(int16)> <out_ref.f32> <params.txt> [iters]
 * params.txt: key=value je Zeile (proj_in_isc, proj_out_osc, core_in_isc, core_out_osc,
 *             woc_isc, woc_izp, woc_osc, woc_ozp, ffn_isc, ffn_izp, ffn_osc, ffn_ozp)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "barra.h"

#define S 1500
#define D 512
#define H 8
#define HD 64
#define BQ 375
#define NB (S/BQ)

static double nowms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static double gp(const char* txt, const char* key){
  const char* p=strstr(txt,key); if(!p){ fprintf(stderr,"param %s fehlt\n",key); exit(1); }
  return atof(p+strlen(key)+1);
}
static void* rdfile(const char* fn, long* n){
  FILE* f=fopen(fn,"rb"); if(!f){ perror(fn); exit(1); }
  fseek(f,0,SEEK_END); *n=ftell(f); fseek(f,0,SEEK_SET);
  void* b=malloc(*n+1); fread(b,1,*n,f); fclose(f); ((char*)b)[*n]=0; return b;
}

int main(int argc,char**argv){
  if(argc<4){ fprintf(stderr,"usage: wsp_layer <x0.int16> <out_ref.f32> <params.txt> [iters]\n"); return 1; }
  int iters=argc>4?atoi(argv[4]):1;
  long n; short* x0q=rdfile(argv[1],&n);           if(n!=S*D*2){ fprintf(stderr,"x0 size %ld\n",n); return 1; }
  float* oref=rdfile(argv[2],&n);                  if(n!=S*D*4){ fprintf(stderr,"ref size %ld\n",n); return 1; }
  char* par=rdfile(argv[3],&n);
  double pin_isc=gp(par,"proj_in_isc"), pout=gp(par,"proj_out_osc"), cin=gp(par,"core_in_isc"),
         cout=gp(par,"core_out_osc"), wisc=gp(par,"woc_isc"), wosc=gp(par,"woc_osc"),
         fisc=gp(par,"ffn_isc"), fosc=gp(par,"ffn_osc");
  int wizp=(int)gp(par,"woc_izp"), wozp=(int)gp(par,"woc_ozp"),
      fizp=(int)gp(par,"ffn_izp"), fozp=(int)gp(par,"ffn_ozp");

  /* ZWEI tpud-Instanzen: Kern allein (Softmax-Package liefert Nullen, sobald andere Graphen
   * im selben tpud registriert sind — Befund 21.8.), proj/woc/ffnres zusammen. */
  const char* dm=getenv("WSP_SOCK_MAIN"); const char* dc=getenv("WSP_SOCK_CORE");
  if(!dm||!dc){ fprintf(stderr,"[wsp] WSP_SOCK_MAIN/WSP_SOCK_CORE setzen\n"); return 1; }
  barra_tpu T, TC;
  setenv("BARRA_SOCK_DIR",dm,1); if(barra_tpu_open(&T)){ return 1; }
  setenv("BARRA_SOCK_DIR",dc,1); if(barra_tpu_open(&TC)){ return 1; }
  uint32_t is,os,nm,nmc; barra_tpu_info(&T,0,&is,&os,&nm); barra_tpu_info(&TC,0,0,0,&nmc);
  fprintf(stderr,"[wsp] tpud main: %u Modelle (proj in/out %u/%u), core-tpud: %u\n",nm,is,os,nmc);
  if(nm<3||nmc<1){ fprintf(stderr,"[wsp] Modelle fehlen\n"); return 1; }

  barra_zbuf bX,bY,bC,bO,bW,bA,bF,bG;
  if(barra_zc_alloc(&bX,S*D*2)||barra_zc_alloc(&bY,(long)H*3*S*HD*2)||barra_zc_alloc(&bC,(long)H*(BQ+2*S)*HD*2)||
     barra_zc_alloc(&bO,(long)H*BQ*(HD+1)*2)||barra_zc_alloc(&bW,S*D)||barra_zc_alloc(&bA,S*D)||
     barra_zc_alloc(&bF,S*D)||barra_zc_alloc(&bG,S*D)){ fprintf(stderr,"[wsp] zc_alloc FAIL\n"); return 1; }
  float* ctx=malloc((size_t)S*D*4);
  double t_proj=0,t_core=0,t_wo=0,t_ffn=0,t_tot=0;

  for(int it=0; it<iters; it++){
    memcpy(bX.map,x0q,S*D*2);
    double t0=nowms();
    uint32_t us=0;
    if(barra_tpu_infer(&T,0,&bX,&bY,&us)){ return 1; }                 /* proj */
    double t1=nowms(); t_proj+=t1-t0;

    /* k|v einmal je Layer in bC (Zeilen 375..3375 je Kopf), Requant pout->cin */
    short* Y=(short*)bY.map; short* Cm=(short*)bC.map;
    float rq=(float)(pout/cin);
    for(int h=0;h<H;h++){
      short* src=Y+((long)h*3*S)*HD + (long)S*HD;                       /* k+v = 2S Zeilen ab S */
      short* dst=Cm+((long)h*(BQ+2*S))*HD + (long)BQ*HD;
      for(long i=0;i<(long)2*S*HD;i++){ float v=src[i]*rq; dst[i]=(short)lrintf(v); }
    }
    double t2=nowms();

    for(int b=0;b<NB;b++){
      for(int h=0;h<H;h++){                                             /* q-Block-Slice tauschen */
        short* src=Y+((long)h*3*S)*HD + (long)b*BQ*HD;
        short* dst=Cm+((long)h*(BQ+2*S))*HD;
        for(long i=0;i<(long)BQ*HD;i++){ float v=src[i]*rq; dst[i]=(short)lrintf(v); }
      }
      double ta=nowms();
      if(barra_tpu_infer(&TC,0,&bC,&bO,&us)){ return 1; }              /* core Block b */
      double tb=nowms(); t_core+=tb-ta;
      if(it==0 && b==0 && getenv("WSP_DUMP")){
        FILE* f=fopen("/data/local/tmp/wsptpu/dump_C0.bin","wb"); fwrite(bC.map,1,bC.size,f); fclose(f);
        f=fopen("/data/local/tmp/wsptpu/dump_O0.bin","wb"); fwrite(bO.map,1,bO.size,f); fclose(f);
      }
      short* O=(short*)bO.map;                                          /* Division + Transpose */
      for(int h=0;h<H;h++) for(int r=0;r<BQ;r++){
        short* row=O+((long)h*BQ+r)*(HD+1);
        float den=row[HD]*(float)cout; if(den<1e-6f) den=1e-6f;
        float inv=1.0f/den;
        float* c=ctx+((long)(b*BQ+r))*D + h*HD;
        for(int d=0;d<HD;d++) c[d]=row[d]*(float)cout*inv;
      }
    }
    double t3=nowms(); (void)t2; (void)t3;

    signed char* Wq8=(signed char*)bW.map;                              /* ctx -> woc int8 */
    for(long i=0;i<(long)S*D;i++){ int v=(int)lrintf(ctx[i]/(float)wisc)+wizp; if(v>127)v=127; if(v<-128)v=-128; Wq8[i]=(signed char)v; }
    double t4=nowms();
    if(barra_tpu_infer(&T,1,&bW,&bA,&us)){ return 1; }                 /* woc */
    double t5=nowms(); t_wo+=t5-t4;

    signed char* A=(signed char*)bA.map; signed char* F=(signed char*)bF.map;
    for(long i=0;i<(long)S*D;i++){                                      /* x1 = x + att -> int8 */
      float x=x0q[i]*(float)pin_isc;
      float att=(A[i]-wozp)*(float)wosc;
      int v=(int)lrintf((x+att)/(float)fisc)+fizp; if(v>127)v=127; if(v<-128)v=-128; F[i]=(signed char)v;
    }
    double t6=nowms();
    if(barra_tpu_infer(&T,2,&bF,&bG,&us)){ return 1; }                 /* ffnres */
    double t7=nowms(); t_ffn+=t7-t6; t_tot+=t7-t0;
    if(it==0 && getenv("WSP_DUMP")){
      FILE* f;
      f=fopen("/data/local/tmp/wsptpu/dump_Y.bin","wb");   fwrite(bY.map,1,bY.size,f); fclose(f);
      f=fopen("/data/local/tmp/wsptpu/dump_ctx.f32","wb"); fwrite(ctx,4,(size_t)S*D,f); fclose(f);
      f=fopen("/data/local/tmp/wsptpu/dump_att.bin","wb"); fwrite(bA.map,1,bA.size,f); fclose(f);
      f=fopen("/data/local/tmp/wsptpu/dump_x1.bin","wb");  fwrite(bF.map,1,bF.size,f); fclose(f);
      fprintf(stderr,"[wsp] Dumps geschrieben\n");
    }
    if(it==0){
      signed char* G=(signed char*)bG.map;                              /* Verify */
      double dab=0,daa=0,dbb=0; float mx=0;
      for(long i=0;i<(long)S*D;i++){
        float v=(G[i]-fozp)*(float)fosc, r=oref[i];
        dab+=(double)v*r; daa+=(double)v*v; dbb+=(double)r*r;
        float e=fabsf(v-r); if(e>mx)mx=e;
      }
      double cos=dab/(sqrt(daa)*sqrt(dbb)+1e-12);
      double rel=sqrt(daa-2*dab+dbb)/(sqrt(dbb)+1e-12);
      fprintf(stderr,"[wsp] LAYER-E2E: cos=%.5f rel_l2=%.2f%% max|e|=%.4f\n",cos,100.0*rel,mx);
    }
  }
  fprintf(stderr,"[wsp] Timing/Iter (n=%d): proj %.2f  core(4x) %.2f  wo %.2f  ffn %.2f  glue %.2f ms  GESAMT %.2f ms\n",
    iters,t_proj/iters,t_core/iters,t_wo/iters,t_ffn/iters,
    (t_tot-t_proj-t_core-t_wo-t_ffn)/iters, t_tot/iters);
  fprintf(stderr,"WSP_LAYER_DONE\n");
  return 0;
}
