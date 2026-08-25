// Whisper-Stufe-3 M4/M5: barra-TPU-Encoder als Bibliothek fuer whisper.cpp — GENERISCH
// (base UND turbo). Dims aus enc_params.txt: D, HTOT (Koepfe), HG (Koepfe je Kern-Package),
// PROJW (1 = flacher qkv-Output [S,3D], 0 = Head-Major [HTOT][3S][HD]), nlayers.
// Kette je Layer: LN1(float) -> proj -> je Kopfgruppe: k/v-Assembly + 4x(q-Swap+Kern) ->
// Division/ctx -> woc -> Residual -> LN2 -> ffn -> Residual. Kanten int16/int8 wie gehabt.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

extern "C" {
#include "barra.h"
}

#define WSP_S 1500
#define WSP_HD 64
#define WSP_BQ 375
#define WSP_NB (WSP_S/WSP_BQ)
#define WSP_MAXL 40

static double wsp_gp(const char* txt, const char* key, double defv, int req){
  const char* p=strstr(txt,key);
  if(!p){ if(req){ fprintf(stderr,"[wsp-barra] param %s fehlt\n",key); exit(1);} return defv; }
  return atof(p+strlen(key)+1);
}
static double wsp_gpl(const char* txt, int L, const char* suf){
  char k[64]; snprintf(k,sizeof k,"L%d_%s",L,suf); return wsp_gp(txt,k,0,1);
}
static void* wsp_rdfile(const char* fn, long* n){
  FILE* f=fopen(fn,"rb"); if(!f){ perror(fn); return NULL; }
  fseek(f,0,SEEK_END); *n=ftell(f); fseek(f,0,SEEK_SET);
  void* b=malloc(*n+1); if(fread(b,1,*n,f)!=(size_t)*n){ fclose(f); free(b); return NULL; }
  fclose(f); ((char*)b)[*n]=0; return b;
}

static struct {
  int ok, NL, D, HTOT, HG, PROJW, NG;
  double cin, cout;
  double pin[WSP_MAXL],pout[WSP_MAXL],wisc[WSP_MAXL],wosc[WSP_MAXL],fisc[WSP_MAXL],fosc[WSP_MAXL];
  int wizp[WSP_MAXL],wozp[WSP_MAXL],fizp[WSP_MAXL],fozp[WSP_MAXL];
  int NX;                                  /* Cross-K/V-Packages (0 = aus) */
  double xisc[WSP_MAXL],xosc[WSP_MAXL]; int xizp[WSP_MAXL],xozp[WSP_MAXL];
  int CONV, NMEL;                          /* Conv-Frontend: 2 = Split conv1|conv2, 0 = aus */
  double c1isc,c1osc,c2isc,c2osc; int c1izp,c1ozp,c2izp,c2ozp;
  float* LNW;
  int pair, ovl;                           /* K=2-Kern-Paare via INFER2 (WSP_PAIR=0 aus); ovl = submit2/wait2-Overlap (WSP_OVL=0 aus) */
  barra_tpu T, TC;
  barra_zbuf bX,bY,bC,bO,bW,bA,bF,bG,bJ,bC2,bO2,bC3,bO3,bC4,bO4;
  float *hb, *ctx;
} g;

static double wsp_now(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec*1e3+t.tv_nsec/1e6;
}

/* Mini-Threadpool fuer die Glue (battn_par-Muster): wsp_par(n, f) teilt [0,n) auf
   Worker+Main auf; f(i0,i1) muss zeilen-/elementdisjunkt schreiben. WSP_THREADS
   (inkl. Main, Default 6) steuert die Breite; 1 = seriell wie bisher. */
#define WSP_MAXTH 8
static struct {
  int nth, quit, seq, done; long n;
  void (*fn)(long,long,void*); void* u;
  pthread_mutex_t mu; pthread_cond_t cv, cvd;
  pthread_t th[WSP_MAXTH];
} pl;

static void* wsp_worker(void* arg){
  long id=(long)arg; int myseq=0;
  /* Worker auf Big-Cores (A715 4-7, X3 8): Scheduler legt sie sonst auf die
     A510-Littles und die Barrier wartet auf den langsamsten Kern. WSP_AFF=0 aus. */
  const char* ae=getenv("WSP_AFF");
  if(!ae||atoi(ae)!=0){
    cpu_set_t cs; CPU_ZERO(&cs);
    int c=3+(int)id; if(c>8)c=8; CPU_SET(c,&cs);
    sched_setaffinity(0,sizeof cs,&cs);
  }
  for(;;){
    pthread_mutex_lock(&pl.mu);
    while(pl.seq==myseq && !pl.quit) pthread_cond_wait(&pl.cv,&pl.mu);
    if(pl.quit){ pthread_mutex_unlock(&pl.mu); return NULL; }
    myseq=pl.seq;
    void (*fn)(long,long,void*)=pl.fn; void* u=pl.u; long n=pl.n;
    pthread_mutex_unlock(&pl.mu);
    long P=pl.nth+1, c=(n+P-1)/P, a=id*c, b=a+c;
    if(a>n)a=n; if(b>n)b=n;
    if(a<b) fn(a,b,u);
    pthread_mutex_lock(&pl.mu);
    if(++pl.done==pl.nth) pthread_cond_signal(&pl.cvd);
    pthread_mutex_unlock(&pl.mu);
  }
}

static void wsp_par_raw(void (*fn)(long,long,void*), void* u, long n){
  if(pl.nth==0 || n<256){ fn(0,n,u); return; }
  pthread_mutex_lock(&pl.mu);
  pl.fn=fn; pl.u=u; pl.n=n; pl.done=0; pl.seq++;
  pthread_cond_broadcast(&pl.cv);
  pthread_mutex_unlock(&pl.mu);
  long P=pl.nth+1, c=(n+P-1)/P;
  fn(0,c<n?c:n,u);
  pthread_mutex_lock(&pl.mu);
  while(pl.done<pl.nth) pthread_cond_wait(&pl.cvd,&pl.mu);
  pthread_mutex_unlock(&pl.mu);
}

template<class F> static void wsp_tramp(long a, long b, void* u){ (*(F*)u)(a,b); }
template<class F> static void wsp_par(long n, F f){ wsp_par_raw(wsp_tramp<F>, &f, n); }

static void wsp_pool_init(void){
  static int done=0; if(done) return; done=1;
  int t=6; const char* e=getenv("WSP_THREADS"); if(e) t=atoi(e);
  if(t<1)t=1; if(t>WSP_MAXTH+1)t=WSP_MAXTH+1;
  pl.nth=t-1; pl.quit=0; pl.seq=0;
  pthread_mutex_init(&pl.mu,NULL); pthread_cond_init(&pl.cv,NULL); pthread_cond_init(&pl.cvd,NULL);
  for(long i=1;i<=pl.nth;i++) pthread_create(&pl.th[i-1],NULL,wsp_worker,(void*)i);
}

/* Encode-Rest-Helfer fuer den whisper.cpp-Patch (parallel ueber Zeilen t). */
extern "C" void wsp_barra_addpe(const float* conv, const float* pe, float* x0, int T, int NS){
  wsp_pool_init();
  wsp_par(T,[&](long t0,long t1){
    for(long t=t0;t<t1;t++) for(long c=0;c<NS;c++)
      x0[t*NS+c]=conv[c*T+t]+pe[t*NS+c];
  });
}
static int wsp_init(void);

/* f16-Ausgabe: auf dem Geraet nativ (__fp16), im Host-Build (x86, nur Kompilierbarkeit
   fuer whisper-quantize etc.) Bit-Konvertierung. */
#if defined(__aarch64__)
typedef __fp16 wsp_f16;
static inline wsp_f16 wsp_tof16(float v){ return (__fp16)v; }
#else
typedef unsigned short wsp_f16;
static inline wsp_f16 wsp_tof16(float v){
  union { float f; unsigned u; } x; x.f=v;
  unsigned s=(x.u>>16)&0x8000u; int e=(int)((x.u>>23)&0xff)-127+15; unsigned m=x.u&0x7fffffu;
  if(e<=0) return (wsp_f16)s;
  if(e>=31) return (wsp_f16)(s|0x7c00u);
  return (wsp_f16)(s|((unsigned)e<<10)|(m>>13));
}
#endif

/* Cross-K/V auf der TPU (ersetzt build_graph_cross): Package l = Modell 3*NL+l im
   Main-Daemon, Output-Zeile = [k(D)|v(D)] int16 -> direkt f16 (Ziel-Layout kv_cross). */
extern "C" int wsp_barra_cross_layers(void){
  if(wsp_init()) return 0;
  return g.NX;
}
extern "C" int wsp_barra_cross(int l, const float* enc, void* kdst, void* vdst){
  if(!g.ok||l<0||l>=g.NX) return -1;
  const int D=g.D; const long SD=(long)WSP_S*D;
  short* X=(short*)g.bX.map;
  const float isc=(float)g.xisc[l]; const int izp=g.xizp[l];
  wsp_par(SD,[&](long i0,long i1){
    for(long i=i0;i<i1;i++){ long v=lrintf(enc[i]/isc)+izp; if(v>32767)v=32767; if(v<-32768)v=-32768; X[i]=(short)v; }
  });
  uint32_t us=0;
  if(barra_tpu_infer(&g.T,3*g.NL+l,&g.bX,&g.bY,&us)) return -1;
  short* Y=(short*)g.bY.map;
  const float osc=(float)g.xosc[l]; const int ozp=g.xozp[l];
  wsp_f16* kh=(wsp_f16*)kdst; wsp_f16* vh=(wsp_f16*)vdst;
  wsp_par(WSP_S,[&](long s0,long s1){
    for(long s=s0;s<s1;s++){
      short* row=Y+s*2L*D;
      for(long d=0;d<D;d++){
        kh[s*D+d]=wsp_tof16((row[d]-ozp)*osc);
        vh[s*D+d]=wsp_tof16((row[D+d]-ozp)*osc);
      }
    }
  });
  return 0;
}

/* Conv-Frontend auf der TPU: mel [80,3000] (kanal-major wie wstate.inp_mel) ->
   Package (conv1 GELU conv2 GELU) -> x0 [1500,1280] float inkl. e_pe-Addition. */
extern "C" int wsp_barra_conv_ready(void){
  if(wsp_init()) return 0;
  return g.CONV;
}
extern "C" int wsp_barra_conv(const float* mel, const float* pe, float* x0){
  if(!g.ok||!g.CONV) return -1;
  const int D=g.D; const long SD=(long)WSP_S*D;
  const long TM=2*WSP_S, NM=g.NMEL;
  short* X=(short*)g.bX.map;
  const float i1=(float)g.c1isc; const int z1=g.c1izp;
  wsp_par(TM,[&](long t0,long t1){
    for(long t=t0;t<t1;t++) for(long c=0;c<NM;c++){
      long v=lrintf(mel[c*TM+t]/i1)+z1; if(v>32767)v=32767; if(v<-32768)v=-32768;
      X[t*NM+c]=(short)v;
    }
  });
  uint32_t us=0;
  if(barra_tpu_infer(&g.T,3*g.NL+g.NX,&g.bX,&g.bY,&us)) return -1;      /* conv1 */
  short* Y=(short*)g.bY.map;
  short* J=(short*)g.bJ.map;
  const float rq=(float)(g.c1osc/g.c2isc); const int zo1=g.c1ozp, z2=g.c2izp;
  wsp_par(WSP_S,[&](long t0,long t1){                                    /* im2col + Naht-Requant */
    for(long t=t0;t<t1;t++){
      short* dst=J+t*3*D;
      const long src[3]={2*t-1,2*t,2*t+1};
      for(int k=0;k<3;k++){
        if(src[k]<0){ for(long d=0;d<D;d++) dst[k*D+d]=(short)z2; continue; }
        const short* s=Y+src[k]*D;
        for(long d=0;d<D;d++){
          long v=lrintf((s[d]-zo1)*rq)+z2; if(v>32767)v=32767; if(v<-32768)v=-32768;
          dst[k*D+d]=(short)v;
        }
      }
    }
  });
  if(barra_tpu_infer(&g.T,3*g.NL+g.NX+1,&g.bJ,&g.bG,&us)) return -1;    /* conv2w */
  short* G=(short*)g.bG.map;
  const float o2=(float)g.c2osc; const int zo2=g.c2ozp;
  wsp_par(SD,[&](long i0,long i1_){
    for(long i=i0;i<i1_;i++) x0[i]=(G[i]-zo2)*o2+pe[i];
  });
  return 0;
}

extern "C" void wsp_barra_lnpost(float* enc, const float* w, const float* b, int T, int NS){
  wsp_pool_init();
  wsp_par(T,[&](long t0,long t1){
    for(long t=t0;t<t1;t++){
      float* r=enc+(long)t*NS;
      double m=0; for(long c=0;c<NS;c++) m+=r[c]; m/=NS;
      double v=0; for(long c=0;c<NS;c++){ double e=r[c]-m; v+=e*e; } v/=NS;
      float inv=(float)(1.0/sqrt(v+1e-5));
      for(long c=0;c<NS;c++) r[c]=(float)((r[c]-m)*inv)*w[c]+b[c];
    }
  });
}

static void wsp_ln_row(const float* x, const float* gm, const float* b, float* o){
  const int D=g.D;
  double m=0; for(int d=0;d<D;d++) m+=x[d]; m/=D;
  double v=0; for(int d=0;d<D;d++){ double e=x[d]-m; v+=e*e; } v/=D;
  float inv=(float)(1.0/sqrt(v+1e-5));
  for(int d=0;d<D;d++) o[d]=(float)((x[d]-m)*inv)*gm[d]+b[d];
}

/* proj-Out-Element (part 0=q 1=k 2=v, globaler Kopf h, Zeile s, Dim d) */
static inline short wsp_yget(const short* Y, int part, int h, int s, int d){
  if(g.PROJW) return Y[(long)s*3*g.D + (long)part*g.D + (long)h*WSP_HD + d];
  return Y[(((long)h*3 + part)*WSP_S + s)*WSP_HD + d];
}

static int wsp_init(void){
  if(g.ok) return 0;
  const char* dir=getenv("WSP_PKG_DIR"); if(!dir){ fprintf(stderr,"[wsp-barra] WSP_PKG_DIR fehlt\n"); return -1; }
  char fn[512]; long n;
  snprintf(fn,sizeof fn,"%s/enc_params.txt",dir);
  char* par=(char*)wsp_rdfile(fn,&n); if(!par) return -1;
  snprintf(fn,sizeof fn,"%s/enc_ln.f32",dir);
  g.LNW=(float*)wsp_rdfile(fn,&n); if(!g.LNW) return -1;
  g.cin=wsp_gp(par,"core_in",0,1); g.cout=wsp_gp(par,"core_out",0,1);
  g.NL=(int)wsp_gp(par,"nlayers",0,1); if(g.NL<1||g.NL>WSP_MAXL) return -1;
  g.D   =(int)wsp_gp(par,"dim",512,0);
  g.HTOT=(int)wsp_gp(par,"heads",g.D/WSP_HD>8?20:8,0);
  g.HG  =(int)wsp_gp(par,"headgrp",g.HTOT,0);
  g.PROJW=(int)wsp_gp(par,"projw",0,0);
  if(g.HTOT%g.HG){ fprintf(stderr,"[wsp-barra] HTOT%%HG\n"); return -1; }
  g.NG=g.HTOT/g.HG;
  if((long)n!=(long)g.NL*4*g.D*4) fprintf(stderr,"[wsp-barra] WARN enc_ln.f32 %ld B (erwartet %ld)\n",n,(long)g.NL*4*g.D*4);
  for(int L=0;L<g.NL;L++){
    g.pin[L]=wsp_gpl(par,L,"pin"); g.pout[L]=wsp_gpl(par,L,"pout");
    g.wisc[L]=wsp_gpl(par,L,"wisc"); g.wizp[L]=(int)wsp_gpl(par,L,"wizp");
    g.wosc[L]=wsp_gpl(par,L,"wosc"); g.wozp[L]=(int)wsp_gpl(par,L,"wozp");
    g.fisc[L]=wsp_gpl(par,L,"fisc"); g.fizp[L]=(int)wsp_gpl(par,L,"fizp");
    g.fosc[L]=wsp_gpl(par,L,"fosc"); g.fozp[L]=(int)wsp_gpl(par,L,"fozp");
  }
  free(par);
  const char* dm=getenv("WSP_SOCK_MAIN"); const char* dc=getenv("WSP_SOCK_CORE");
  if(!dm||!dc){ fprintf(stderr,"[wsp-barra] WSP_SOCK_MAIN/CORE fehlen\n"); return -1; }
  setenv("BARRA_SOCK_DIR",dm,1); if(barra_tpu_open(&g.T)) return -1;
  setenv("BARRA_SOCK_DIR",dc,1); if(barra_tpu_open(&g.TC)) return -1;
  /* Core-Daemon laeuft IMMER mit TPU_ZC_BOUNCE (LUT-Pflicht) -> CPU-only auf dem Mapping,
     Cache-Sync-Klammern unnoetig (sparen ~2,5GB Cache-Maintenance/Encoder). WSP_SYNC=1 erzwingt sie. */
  { const char* sy=getenv("WSP_SYNC"); if(!(sy&&atoi(sy)==1)) g.TC.nosync=1; }
  uint32_t nm=0,nmc=0; barra_tpu_info(&g.T,0,0,0,&nm); barra_tpu_info(&g.TC,0,0,0,&nmc);
  if((int)nm<3*g.NL||nmc<1){ fprintf(stderr,"[wsp-barra] Modelle fehlen (%u/%u)\n",nm,nmc); return -1; }
  /* Cross-K/V optional: cross_params.txt + Packages als Modelle 3*NL.. im Main-Daemon */
  g.NX=0;
  snprintf(fn,sizeof fn,"%s/cross_params.txt",dir);
  char* xp=(char*)wsp_rdfile(fn,&n);
  if(xp){
    int nx=(int)wsp_gp(xp,"ntext",0,1);
    if(nx>0&&nx<=WSP_MAXL&&(int)nm>=3*g.NL+nx){
      for(int L=0;L<nx;L++){
        char k[32];
        snprintf(k,sizeof k,"X%d_isc",L); g.xisc[L]=wsp_gp(xp,k,0,1);
        snprintf(k,sizeof k,"X%d_izp",L); g.xizp[L]=(int)wsp_gp(xp,k,0,1);
        snprintf(k,sizeof k,"X%d_osc",L); g.xosc[L]=wsp_gp(xp,k,0,1);
        snprintf(k,sizeof k,"X%d_ozp",L); g.xozp[L]=(int)wsp_gp(xp,k,0,1);
      }
      g.NX=nx;
    } else fprintf(stderr,"[wsp-barra] cross: ntext=%d aber nur %u Modelle geladen — aus\n",nx,nm);
    free(xp);
  }
  /* Conv-Frontend optional: conv_params.txt + Package als Modell 3*NL+NX */
  g.CONV=0;
  snprintf(fn,sizeof fn,"%s/conv_params.txt",dir);
  char* cvp=(char*)wsp_rdfile(fn,&n);
  if(cvp){
    if((int)wsp_gp(cvp,"conv",0,1)==2&&(int)nm>=3*g.NL+g.NX+2){
      g.c1isc=wsp_gp(cvp,"C1_isc",0,1); g.c1izp=(int)wsp_gp(cvp,"C1_izp",0,1);
      g.c1osc=wsp_gp(cvp,"C1_osc",0,1); g.c1ozp=(int)wsp_gp(cvp,"C1_ozp",0,1);
      g.c2isc=wsp_gp(cvp,"C2_isc",0,1); g.c2izp=(int)wsp_gp(cvp,"C2_izp",0,1);
      g.c2osc=wsp_gp(cvp,"C2_osc",0,1); g.c2ozp=(int)wsp_gp(cvp,"C2_ozp",0,1);
      g.NMEL=(int)wsp_gp(cvp,"C_nmel",80,0);
      g.CONV=2;
    } else fprintf(stderr,"[wsp-barra] conv: Packages fehlen (%u Modelle) — aus\n",nm);
    free(cvp);
  }
  long SD=(long)WSP_S*g.D;
  if(barra_zc_alloc(&g.bX,SD*2)||barra_zc_alloc(&g.bY,SD*3*2)||
     barra_zc_alloc(&g.bC,(long)g.HG*(WSP_BQ+2*WSP_S)*WSP_HD*2)||
     barra_zc_alloc(&g.bO,(long)g.HG*WSP_BQ*(WSP_HD+1)*2)||
     barra_zc_alloc(&g.bW,SD)||barra_zc_alloc(&g.bA,SD)||
     barra_zc_alloc(&g.bF,SD*2)||barra_zc_alloc(&g.bG,SD*2)) return -1;
  if(g.CONV&&barra_zc_alloc(&g.bJ,(long)WSP_S*3*g.D*2)) return -1;      /* conv2w-Fenster */
  { const char* pe_=getenv("WSP_PAIR"); g.pair=!(pe_&&atoi(pe_)==0); }
  { const char* ov=getenv("WSP_OVL");  g.ovl=g.pair&&!(ov&&atoi(ov)==0)&&WSP_NB==4; }
  if(g.pair&&(barra_zc_alloc(&g.bC2,(long)g.HG*(WSP_BQ+2*WSP_S)*WSP_HD*2)||
              barra_zc_alloc(&g.bO2,(long)g.HG*WSP_BQ*(WSP_HD+1)*2))) return -1;
  if(g.ovl&&(barra_zc_alloc(&g.bC3,(long)g.HG*(WSP_BQ+2*WSP_S)*WSP_HD*2)||
             barra_zc_alloc(&g.bO3,(long)g.HG*WSP_BQ*(WSP_HD+1)*2)||
             barra_zc_alloc(&g.bC4,(long)g.HG*(WSP_BQ+2*WSP_S)*WSP_HD*2)||
             barra_zc_alloc(&g.bO4,(long)g.HG*WSP_BQ*(WSP_HD+1)*2))) return -1;
  if(g.ovl){
    /* Vorab-Import: ein lazy Import in submit2 wuerde seine Antwort mit der noch
       ausstehenden Paar-Reply verschraenken (Protokoll-Versatz). */
    barra_zbuf* v[8]={&g.bC,&g.bO,&g.bC2,&g.bO2,&g.bC3,&g.bO3,&g.bC4,&g.bO4};
    if(barra_tpu_import(&g.TC,v,8)){ fprintf(stderr,"[wsp-barra] ovl: Vorab-Import fehlgeschlagen\n"); return -1; }
  }
  g.hb=(float*)malloc(SD*4);
  g.ctx=(float*)malloc(SD*4);
  wsp_pool_init();
  g.ok=1;
  fprintf(stderr,"[wsp-barra] init ok: %d Layer, D=%d, %d Koepfe (%d Gruppen a %d), projw=%d, %d Glue-Threads\n",
          g.NL,g.D,g.HTOT,g.NG,g.HG,g.PROJW,pl.nth+1);
  return 0;
}

extern "C" int wsp_barra_encode(const float* x0, float* out, int S_, int D_){
  if(wsp_init()) return -1;
  if(S_!=WSP_S||D_!=g.D){ fprintf(stderr,"[wsp-barra] Shape %d/%d passt nicht (D=%d)\n",S_,D_,g.D); return -1; }
  const int D=g.D;
  const long SD=(long)WSP_S*D;
  float* xf=out;
  memcpy(xf,x0,SD*4);
  /* Profiling: Wall-Zeit je Glue-Stufe + je TPU-Package (Wall vs. Daemon-us) */
  struct { double ln,qq,kv,qs,divi,res;
           double proj_w,core_w,woc_w,ffn_w; double proj_d,core_d,woc_d,ffn_d; } p;
  memset(&p,0,sizeof p);
  double t0, tall=wsp_now();
  for(int L=0;L<g.NL;L++){
    uint32_t us=0;
    const float* G1=g.LNW+((long)L*4+0)*D; const float* B1=g.LNW+((long)L*4+1)*D;
    const float* G2=g.LNW+((long)L*4+2)*D; const float* B2=g.LNW+((long)L*4+3)*D;
    t0=wsp_now();
    wsp_par(WSP_S,[&](long r0,long r1){
      for(long r=r0;r<r1;r++) wsp_ln_row(xf+r*D,G1,B1,g.hb+r*D);
    });
    p.ln+=wsp_now()-t0; t0=wsp_now();
    short* X=(short*)g.bX.map;
    wsp_par(SD,[&](long i0,long i1){
      for(long i=i0;i<i1;i++){ long v=lrintf(g.hb[i]/(float)g.pin[L]); if(v>32767)v=32767; if(v<-32768)v=-32768; X[i]=(short)v; }
    });
    p.qq+=wsp_now()-t0; t0=wsp_now();
    if(barra_tpu_infer(&g.T,3*L,&g.bX,&g.bY,&us)) return -1;                /* proj */
    p.proj_w+=wsp_now()-t0; p.proj_d+=us/1e3;
    short* Y=(short*)g.bY.map; short* Cm=(short*)g.bC.map;
    float rq=(float)(g.pout[L]/g.cin);
    for(int grp=0;grp<g.NG;grp++){
      t0=wsp_now();
      wsp_par((long)g.HG*WSP_S,[&](long i0,long i1){                       /* k/v je Gruppe einmal */
        for(long i=i0;i<i1;i++){
          int hl=(int)(i/WSP_S), s=(int)(i%WSP_S);
          int h=grp*g.HG+hl;
          short* dk=Cm+((long)hl*(WSP_BQ+2*WSP_S) + WSP_BQ + s)*WSP_HD;
          short* dv=Cm+((long)hl*(WSP_BQ+2*WSP_S) + WSP_BQ + WSP_S + s)*WSP_HD;
          for(int d=0;d<WSP_HD;d++){
            dk[d]=(short)lrintf(wsp_yget(Y,1,h,s,d)*rq);
            dv[d]=(short)lrintf(wsp_yget(Y,2,h,s,d)*rq);
          }
        }
      });
      p.kv+=wsp_now()-t0;
      short* Cm2=(short*)(g.pair?g.bC2.map:0);
      short* Cm3=(short*)(g.ovl?g.bC3.map:0);
      short* Cm4=(short*)(g.ovl?g.bC4.map:0);
      if(g.pair){                                                          /* k/v-Streifen in die Partner-Puffer */
        t0=wsp_now();
        for(int hl=0;hl<g.HG;hl++){
          long off=((long)hl*(WSP_BQ+2*WSP_S)+WSP_BQ)*WSP_HD;
          long len=(long)2*WSP_S*WSP_HD*2;
          memcpy(Cm2+off,Cm+off,len);
          if(g.ovl){ memcpy(Cm3+off,Cm+off,len); memcpy(Cm4+off,Cm+off,len); }
        }
        p.kv+=wsp_now()-t0;
      }
      auto qslice=[&](int bb, short* CmT){
        wsp_par((long)g.HG*WSP_BQ,[&](long i0,long i1){
          for(long i=i0;i<i1;i++){
            int hl=(int)(i/WSP_BQ), r=(int)(i%WSP_BQ);
            int h=grp*g.HG+hl;
            short* dq=CmT+((long)hl*(WSP_BQ+2*WSP_S) + r)*WSP_HD;
            for(int d=0;d<WSP_HD;d++) dq[d]=(short)lrintf(wsp_yget(Y,0,h,bb*WSP_BQ+r,d)*rq);
          }
        });
      };
      auto divb=[&](int bb, short* O){
        wsp_par((long)g.HG*WSP_BQ,[&](long i0,long i1){
          for(long i=i0;i<i1;i++){
            int hl=(int)(i/WSP_BQ), r=(int)(i%WSP_BQ);
            short* row=O+((long)hl*WSP_BQ+r)*(WSP_HD+1);
            float den=row[WSP_HD]*(float)g.cout; if(den<1e-6f) den=1e-6f;
            float inv=1.0f/den;
            float* c=g.ctx+((long)(bb*WSP_BQ+r))*D + (grp*g.HG+hl)*WSP_HD;
            for(int d=0;d<WSP_HD;d++) c[d]=row[d]*(float)g.cout*inv;
          }
        });
      };
      if(g.ovl){
        /* Overlap: Paar B wird gebaut+abgesetzt, waehrend die TPU Paar A rechnet
           (Daemon liest Request B erst nach Reply A — Socket puffert ihn). */
        t0=wsp_now(); qslice(0,Cm); qslice(1,Cm2); p.qs+=wsp_now()-t0;
        t0=wsp_now();
        if(barra_tpu_submit2(&g.TC,0,&g.bC,&g.bO,&g.bC2,&g.bO2)) return -1;
        p.core_w+=wsp_now()-t0;
        t0=wsp_now(); qslice(2,Cm3); qslice(3,Cm4); p.qs+=wsp_now()-t0;
        t0=wsp_now();
        if(barra_tpu_submit2(&g.TC,0,&g.bC3,&g.bO3,&g.bC4,&g.bO4)) return -1;
        if(barra_tpu_wait2(&g.TC,&g.bC,&g.bO,&g.bC2,&g.bO2,&us)) return -1;
        p.core_w+=wsp_now()-t0; p.core_d+=us/1e3;
        t0=wsp_now(); divb(0,(short*)g.bO.map); divb(1,(short*)g.bO2.map); p.divi+=wsp_now()-t0;
        t0=wsp_now();
        if(barra_tpu_wait2(&g.TC,&g.bC3,&g.bO3,&g.bC4,&g.bO4,&us)) return -1;
        p.core_w+=wsp_now()-t0; p.core_d+=us/1e3;
        t0=wsp_now(); divb(2,(short*)g.bO3.map); divb(3,(short*)g.bO4.map); p.divi+=wsp_now()-t0;
      } else for(int b=0;b<WSP_NB;b+=(g.pair?2:1)){
        t0=wsp_now();
        qslice(b,Cm);
        if(g.pair) qslice(b+1,Cm2);
        p.qs+=wsp_now()-t0; t0=wsp_now();
        if(g.pair){
          if(barra_tpu_infer2(&g.TC,0,&g.bC,&g.bO,&g.bC2,&g.bO2,&us)) return -1;  /* Kern-Paar K=2 */
        } else {
          if(barra_tpu_infer(&g.TC,0,&g.bC,&g.bO,&us)) return -1;                 /* Kern */
        }
        p.core_w+=wsp_now()-t0; p.core_d+=us/1e3; t0=wsp_now();
        divb(b,(short*)g.bO.map);
        if(g.pair) divb(b+1,(short*)g.bO2.map);
        p.divi+=wsp_now()-t0;
      }
    }
    t0=wsp_now();
    signed char* Wq8=(signed char*)g.bW.map;
    wsp_par(SD,[&](long i0,long i1){
      for(long i=i0;i<i1;i++){ int v=(int)lrintf(g.ctx[i]/(float)g.wisc[L])+g.wizp[L]; if(v>127)v=127; if(v<-128)v=-128; Wq8[i]=(signed char)v; }
    });
    p.qq+=wsp_now()-t0; t0=wsp_now();
    if(barra_tpu_infer(&g.T,3*L+1,&g.bW,&g.bA,&us)) return -1;             /* woc */
    p.woc_w+=wsp_now()-t0; p.woc_d+=us/1e3; t0=wsp_now();
    signed char* A=(signed char*)g.bA.map;
    wsp_par(SD,[&](long i0,long i1){
      for(long i=i0;i<i1;i++) xf[i]=xf[i]+(A[i]-g.wozp[L])*(float)g.wosc[L];
    });
    p.res+=wsp_now()-t0; t0=wsp_now();
    wsp_par(WSP_S,[&](long r0,long r1){
      for(long r=r0;r<r1;r++) wsp_ln_row(xf+r*D,G2,B2,g.hb+r*D);
    });
    p.ln+=wsp_now()-t0; t0=wsp_now();
    short* F=(short*)g.bF.map;
    wsp_par(SD,[&](long i0,long i1){
      for(long i=i0;i<i1;i++){ long v=lrintf(g.hb[i]/(float)g.fisc[L])+g.fizp[L]; if(v>32767)v=32767; if(v<-32768)v=-32768; F[i]=(short)v; }
    });
    p.qq+=wsp_now()-t0; t0=wsp_now();
    if(barra_tpu_infer(&g.T,3*L+2,&g.bF,&g.bG,&us)) return -1;             /* ffn */
    p.ffn_w+=wsp_now()-t0; p.ffn_d+=us/1e3; t0=wsp_now();
    short* G=(short*)g.bG.map;
    wsp_par(SD,[&](long i0,long i1){
      for(long i=i0;i<i1;i++) xf[i]+=(G[i]-g.fozp[L])*(float)g.fosc[L];
    });
    p.res+=wsp_now()-t0;
  }
  double tot=wsp_now()-tall;
  fprintf(stderr,"[wsp-barra] TPU-Encoder (%d Layer): %.1f ms\n",g.NL,tot);
  double glue=p.ln+p.qq+p.kv+p.qs+p.divi+p.res;
  double tpuw=p.proj_w+p.core_w+p.woc_w+p.ffn_w;
  fprintf(stderr,"[wsp-barra] prof glue %.0f (ln %.0f, quant %.0f, kv-asm %.0f, q-slice %.0f, div %.0f, resid %.0f)\n",
          glue,p.ln,p.qq,p.kv,p.qs,p.divi,p.res);
  fprintf(stderr,"[wsp-barra] prof tpu %.0f wall / %.0f daemon (proj %.0f/%.0f, core %.0f/%.0f, woc %.0f/%.0f, ffn %.0f/%.0f) rest %.0f\n",
          tpuw,p.proj_d+p.core_d+p.woc_d+p.ffn_d,
          p.proj_w,p.proj_d,p.core_w,p.core_d,p.woc_w,p.woc_d,p.ffn_w,p.ffn_d,tot-glue-tpuw);
  return 0;
}
