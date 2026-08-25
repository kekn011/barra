/* Whisper-Stufe-3 M3b: Kompletter base-Encoder (NL Layer) als TPU-Package-Kette mit CPU-Glue.
 * Main-tpud: je Layer proj(16x8), woc(int8), ffnresc(int8) -> mids 3L/3L+1/3L+2.
 * Core-tpud: das eine geteilte Kern-Package (mid 0), 4 Aufrufe je Layer.
 *   WSP_SOCK_MAIN=<dir> WSP_SOCK_CORE=<dir> wsp_enc <x0.int16> <out_ref.f32> <enc_params.txt> [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "barra.h"

/* saettigende int16-Requantisierung (kein Wrap bei rq>1 nahe der Grenze) */
static inline short qs16(float v){ long r=lrintf(v); return (short)(r>32767?32767:(r<-32768?-32768:r)); }

#define S 1500
#define D 512
#define H 8
#define HD 64
#define BQ 375
#define NB (S/BQ)
#define MAXL 8

static double nowms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static double gp(const char* txt, const char* key){
  const char* p=strstr(txt,key); if(!p){ fprintf(stderr,"param %s fehlt\n",key); exit(1); }
  return atof(p+strlen(key)+1);
}
static double gpl(const char* txt, int L, const char* suf){
  char k[64]; snprintf(k,sizeof k,"L%d_%s",L,suf); return gp(txt,k);
}
static void* rdfile(const char* fn, long* n){
  FILE* f=fopen(fn,"rb"); if(!f){ perror(fn); exit(1); }
  fseek(f,0,SEEK_END); *n=ftell(f); fseek(f,0,SEEK_SET);
  void* b=malloc(*n+1); fread(b,1,*n,f); fclose(f); ((char*)b)[*n]=0; return b;
}

static void ln_row(const float* x, const float* g, const float* b, float* o){
  double m=0; for(int d=0;d<D;d++) m+=x[d]; m/=D;
  double v=0; for(int d=0;d<D;d++){ double e=x[d]-m; v+=e*e; } v/=D;
  float inv=(float)(1.0/sqrt(v+1e-5));
  for(int d=0;d<D;d++) o[d]=(float)((x[d]-m)*inv)*g[d]+b[d];
}

int main(int argc,char**argv){
  if(argc<5){ fprintf(stderr,"usage: wsp_enc <x0.f32> <out_ref.f32> <enc_params.txt> <enc_ln.f32> [iters]\n"); return 1; }
  int iters=argc>5?atoi(argv[5]):1;
  long n; float* x0f=rdfile(argv[1],&n);  if(n!=S*D*4){ fprintf(stderr,"x0 size %ld\n",n); return 1; }
  float* oref=rdfile(argv[2],&n);         if(n!=S*D*4){ fprintf(stderr,"ref size %ld\n",n); return 1; }
  char* par=rdfile(argv[3],&n);
  float* LNW=rdfile(argv[4],&n);         /* [NL][4][D]: g1 b1 g2 b2 */
  double cin=gp(par,"core_in"), cout=gp(par,"core_out");
  int NL=(int)gp(par,"nlayers"); if(NL<1||NL>MAXL) return 1;
  double pin[MAXL],pout[MAXL],wisc[MAXL],wosc[MAXL],fisc[MAXL],fosc[MAXL];
  int wizp[MAXL],wozp[MAXL],fizp[MAXL],fozp[MAXL];
  for(int L=0;L<NL;L++){
    pin[L]=gpl(par,L,"pin"); pout[L]=gpl(par,L,"pout");
    wisc[L]=gpl(par,L,"wisc"); wizp[L]=(int)gpl(par,L,"wizp");
    wosc[L]=gpl(par,L,"wosc"); wozp[L]=(int)gpl(par,L,"wozp");
    fisc[L]=gpl(par,L,"fisc"); fizp[L]=(int)gpl(par,L,"fizp");
    fosc[L]=gpl(par,L,"fosc"); fozp[L]=(int)gpl(par,L,"fozp");
  }
  const char* dm=getenv("WSP_SOCK_MAIN"); const char* dc=getenv("WSP_SOCK_CORE");
  if(!dm||!dc){ fprintf(stderr,"[wsp] WSP_SOCK_MAIN/WSP_SOCK_CORE setzen\n"); return 1; }
  barra_tpu T, TC;
  setenv("BARRA_SOCK_DIR",dm,1); if(barra_tpu_open(&T)) return 1;
  setenv("BARRA_SOCK_DIR",dc,1); if(barra_tpu_open(&TC)) return 1;
  uint32_t nm=0,nmc=0; barra_tpu_info(&T,0,0,0,&nm); barra_tpu_info(&TC,0,0,0,&nmc);
  fprintf(stderr,"[wsp] main %u Modelle (erwartet %d), core %u\n",nm,3*NL,nmc);
  if((int)nm<3*NL||nmc<1) return 1;

  barra_zbuf bX,bY,bC,bO,bW,bA,bF,bG;
  if(barra_zc_alloc(&bX,S*D*2)||barra_zc_alloc(&bY,(long)H*3*S*HD*2)||barra_zc_alloc(&bC,(long)H*(BQ+2*S)*HD*2)||
     barra_zc_alloc(&bO,(long)H*BQ*(HD+1)*2)||barra_zc_alloc(&bW,S*D)||barra_zc_alloc(&bA,S*D)||
     barra_zc_alloc(&bF,S*D*2)||barra_zc_alloc(&bG,S*D*2)){ fprintf(stderr,"[wsp] zc_alloc FAIL\n"); return 1; }
  float* xf=malloc((size_t)S*D*4);
  float* hb=malloc((size_t)S*D*4);
  float* ctx=malloc((size_t)S*D*4);
  double t_proj=0,t_core=0,t_wo=0,t_ffn=0,t_tot=0;

  for(int it=0; it<iters; it++){
    memcpy(xf,x0f,(size_t)S*D*4);
    double t0=nowms();
    for(int L=0;L<NL;L++){
      uint32_t us=0;
      const float* G1=LNW+((long)L*4+0)*D; const float* B1=LNW+((long)L*4+1)*D;
      const float* G2=LNW+((long)L*4+2)*D; const float* B2=LNW+((long)L*4+3)*D;
      for(int r=0;r<S;r++) ln_row(xf+(long)r*D,G1,B1,hb+(long)r*D);        /* LN1 in float */
      short* X=(short*)bX.map;
      for(long i=0;i<(long)S*D;i++){ long v=lrintf(hb[i]/(float)pin[L]); if(v>32767)v=32767; if(v<-32768)v=-32768; X[i]=(short)v; }
      double ta=nowms();
      if(barra_tpu_infer(&T,3*L,&bX,&bY,&us)) return 1;                 /* proj */
      double tb=nowms(); t_proj+=tb-ta;

      short* Y=(short*)bY.map; short* Cm=(short*)bC.map;
      float rq=(float)(pout[L]/cin);
      for(int h=0;h<H;h++){
        short* src=Y+((long)h*3*S)*HD + (long)S*HD;
        short* dst=Cm+((long)h*(BQ+2*S))*HD + (long)BQ*HD;
        for(long i=0;i<(long)2*S*HD;i++) dst[i]=qs16(src[i]*rq);
      }
      for(int b=0;b<NB;b++){
        for(int h=0;h<H;h++){
          short* src=Y+((long)h*3*S)*HD + (long)b*BQ*HD;
          short* dst=Cm+((long)h*(BQ+2*S))*HD;
          for(long i=0;i<(long)BQ*HD;i++) dst[i]=qs16(src[i]*rq);
        }
        double tc=nowms();
        if(barra_tpu_infer(&TC,0,&bC,&bO,&us)) return 1;                /* core */
        t_core+=nowms()-tc;
        short* O=(short*)bO.map;
        for(int h=0;h<H;h++) for(int r=0;r<BQ;r++){
          short* row=O+((long)h*BQ+r)*(HD+1);
          float den=row[HD]*(float)cout; if(den<1e-6f) den=1e-6f;
          float inv=1.0f/den;
          float* c=ctx+((long)(b*BQ+r))*D + h*HD;
          for(int d=0;d<HD;d++) c[d]=row[d]*(float)cout*inv;
        }
      }
      signed char* Wq8=(signed char*)bW.map;
      for(long i=0;i<(long)S*D;i++){ int v=(int)lrintf(ctx[i]/(float)wisc[L])+wizp[L]; if(v>127)v=127; if(v<-128)v=-128; Wq8[i]=(signed char)v; }
      double td=nowms();
      if(barra_tpu_infer(&T,3*L+1,&bW,&bA,&us)) return 1;               /* woc */
      double te=nowms(); t_wo+=te-td;
      signed char* A=(signed char*)bA.map;
      for(long i=0;i<(long)S*D;i++) xf[i]=xf[i]+(A[i]-wozp[L])*(float)wosc[L];   /* x1 = x+att (float) */
      for(int r=0;r<S;r++) ln_row(xf+(long)r*D,G2,B2,hb+(long)r*D);              /* LN2 in float */
      short* F=(short*)bF.map;
      for(long i=0;i<(long)S*D;i++){
        long v=lrintf(hb[i]/(float)fisc[L])+fizp[L]; if(v>32767)v=32767; if(v<-32768)v=-32768; F[i]=(short)v;
      }
      double tf2=nowms();
      if(barra_tpu_infer(&T,3*L+2,&bF,&bG,&us)) return 1;               /* ffnres */
      double tg=nowms(); t_ffn+=tg-tf2;
      short* G=(short*)bG.map;
      for(long i=0;i<(long)S*D;i++) xf[i]+= (G[i]-fozp[L])*(float)fosc[L];       /* out = x1 + f */
    }
    double t9=nowms(); t_tot+=t9-t0;
    if(it==0){
      double dab=0,daa=0,dbb=0; float mx=0;
      for(long i=0;i<(long)S*D;i++){
        float v=xf[i], r=oref[i];
        dab+=(double)v*r; daa+=(double)v*v; dbb+=(double)r*r;
        float e=fabsf(v-r); if(e>mx)mx=e;
      }
      fprintf(stderr,"[wsp] ENCODER-E2E (%d Layer): cos=%.5f rel_l2=%.2f%% max|e|=%.4f\n",
        NL,dab/(sqrt(daa)*sqrt(dbb)+1e-12),100.0*sqrt(daa-2*dab+dbb)/(sqrt(dbb)+1e-12),mx);
    }
  }
  fprintf(stderr,"[wsp] Timing/Iter (n=%d): proj %.1f  core(%dx) %.1f  wo %.1f  ffn %.1f  glue %.1f ms  GESAMT %.1f ms\n",
    iters,t_proj/iters,4*NL,t_core/iters,t_wo/iters,t_ffn/iters,
    (t_tot-t_proj-t_core-t_wo-t_ffn)/iters, t_tot/iters);
  fprintf(stderr,"WSP_ENC_DONE\n");
  return 0;
}
