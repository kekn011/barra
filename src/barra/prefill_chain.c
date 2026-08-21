#define _GNU_SOURCE
/* M3: volle hybride Prefill-Kette. Attention (CPU, C) + FFN (TPU, Zero-Copy) ueber NL Layer.
   pro Layer: xattn = x + attn(rms(x)) [CPU] -> quant int16 in ZI [dmabuf] -> TPU FFN -> dequant -> x.
   Ausreisserzeilen (Qwen-Massive-Activations, Pos 0: Norm ~1600 vs ~10 normal) passen nicht in den
   int16-Eingangsbereich der TPU-Packages (die ohne Ausreisser kalibriert sind, prefill_ref.py) ->
   Kriterium max|xa|/isc > 32767 -> Zeile auf CPU-FFN (float aus f16-Gewichten), TPU-Zeile genullt. */
#include "barra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>
#ifndef __aarch64__
typedef _Float16 __fp16; /* Host-Build (x86) fuer Attention-Checks mit barra_stub.c */
#endif
#define D 896
#define FF 4864
#define H 14
#define NKV 2
#define HD 64
#define GRP (H/NKV)

/* ---- parallel-for ueber [0,n) mit PERSISTENTEM Thread-Pool: NT Threads (env PF_THREADS, Default 1),
   Worker einmal erzeugt + optional gepinnt (env PF_CPUS="8,4,5,6,7": Thread i -> Liste[i], Thread 0 = Hauptthread),
   DYNAMISCHE Chunks (Atomic-Zaehler; heterogene Kerne holen sich passend viel), Job-Start per Generation-Zaehler:
   Worker spinnen kurz (Kerne bleiben wach zwischen dicht folgenden Aufrufen), dann Cond-Wait (TPU-Phase). ---- */
#include <sched.h>
#include <errno.h>
#include <stdatomic.h>
static int NT=0,PFC[64],NPFC=0;
static struct{void(*fn)(int,int,void*);void*ctx;int n,chunk;atomic_int next,done;atomic_int gen;pthread_mutex_t mu;pthread_cond_t cv;}PF={0};
static void pf_pin(int cpu){if(cpu>=0){cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(cpu,&cs);sched_setaffinity(0,sizeof cs,&cs);}}
static void pf_work(void){for(;;){int lo=atomic_fetch_add(&PF.next,PF.chunk);if(lo>=PF.n)break;int hi=lo+PF.chunk<PF.n?lo+PF.chunk:PF.n;PF.fn(lo,hi,PF.ctx);}}
static void*pf_worker(void*a){long i=(long)a;pf_pin(NPFC?PFC[i%NPFC]:-1);int seen=0;
  for(;;){int spins=0;while(atomic_load_explicit(&PF.gen,memory_order_acquire)==seen){if(++spins<20000)continue;
      pthread_mutex_lock(&PF.mu);while(atomic_load(&PF.gen)==seen)pthread_cond_wait(&PF.cv,&PF.mu);pthread_mutex_unlock(&PF.mu);}
    seen=atomic_load(&PF.gen);pf_work();atomic_fetch_add(&PF.done,1);}
  return NULL;}
static void pf_init(void){const char*e=getenv("PF_THREADS");NT=e?atoi(e):1;if(NT<1)NT=1;
  const char*c=getenv("PF_CPUS");if(c){while(*c&&NPFC<64){PFC[NPFC++]=atoi(c);while(*c&&*c!=',')c++;if(*c)c++;}}
  pthread_mutex_init(&PF.mu,NULL);pthread_cond_init(&PF.cv,NULL);pf_pin(NPFC?PFC[0]:-1);
  for(long i=1;i<NT;i++){pthread_t t;pthread_create(&t,NULL,pf_worker,(void*)i);pthread_detach(t);}}
static void pfor(int n,void(*fn)(int,int,void*),void*ctx){
  if(!NT)pf_init();
  if(NT<=1||n<2){fn(0,n,ctx);return;}
  PF.fn=fn;PF.ctx=ctx;PF.n=n;PF.chunk=n/(NT*4)>0?n/(NT*4):1;atomic_store(&PF.next,0);atomic_store(&PF.done,0);
  pthread_mutex_lock(&PF.mu);atomic_fetch_add_explicit(&PF.gen,1,memory_order_release);pthread_cond_broadcast(&PF.cv);pthread_mutex_unlock(&PF.mu);
  pf_work();
  while(atomic_load_explicit(&PF.done,memory_order_acquire)<NT-1)sched_yield();
}
/* Skalarprodukt f32 x f32, 4 Akkumulatoren + Reassoziation -> NEON */
static inline float dotf(const float*a,const float*b,int n){
#pragma clang fp reassociate(on)
  float s0=0,s1=0,s2=0,s3=0;int i=0;
  for(;i+4<=n;i+=4){s0+=a[i]*b[i];s1+=a[i+1]*b[i+1];s2+=a[i+2]*b[i+2];s3+=a[i+3]*b[i+3];}
  for(;i<n;i++)s0+=a[i]*b[i];
  return (s0+s1)+(s2+s3);
}
static double now_us(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
static float* rd(const char*p,long n){FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"no %s\n",p);exit(1);}float*a=malloc(n*4);if(fread(a,4,n,f)!=(size_t)n){fprintf(stderr,"short %s\n",p);exit(1);}fclose(f);return a;}
/* per-layer attn weights layout (floats): norm[D],Wq[D*D],bq[D],Wk[NKV*HD*D],bk[NKV*HD],Wv[NKV*HD*D],bv[NKV*HD],Wo[D*D] */
#define WSZ (D + D*D + D + NKV*HD*D + NKV*HD + NKV*HD*D + NKV*HD + D*D)
/* per-layer ffn weights (f16): norm[D], Wg[FF*D], Wu[FF*D], Wd[D*FF] */
#define FSZ (D + FF*D + FF*D + D*FF)
static void rmsn(const float*x,const float*g,float*o,int B){for(int b=0;b<B;b++){const float*xr=x+b*D;float s=0;for(int d=0;d<D;d++)s+=xr[d]*xr[d];float r=1.f/sqrtf(s/D+1e-6f);for(int d=0;d<D;d++)o[b*D+d]=xr[d]*r*g[d];}}
/* y[B,M] = h[B,D] @ W[M,D]^T + b[M] */
typedef struct{const float*h,*W,*b;float*y;int B,M;}mm_ctx;
static void mm_blk(int lo,int hi,void*c_){mm_ctx*c=c_;for(int o=lo;o<hi;o++){const float*wr=c->W+(long)o*D;float bo=c->b?c->b[o]:0;for(int i=0;i<c->B;i++)c->y[i*c->M+o]=bo+dotf(c->h+i*D,wr,D);}}
static void mm(const float*h,const float*W,const float*b,float*y,int B,int M){mm_ctx c={h,W,b,y,B,M};pfor(M,mm_blk,&c);}
static void rope(float*v,const float*cs,const float*sn,int n){for(int t=0;t<n;t++){float*p=v+t*HD;float tmp[HD];for(int d=0;d<HD;d++){float rot=(d<HD/2)?-p[d+HD/2]:p[d-HD/2];tmp[d]=p[d]*cs[d]+rot*sn[d];}memcpy(p,tmp,HD*4);}}
/* CPU-FFN fuer EINE Zeile: y = x + Wd(silu(Wg h) * (Wu h)), h = rmsnorm(x)*g. Gewichte f16. */
/* Skalarprodukt f32 x f16 mit 4 unabhaengigen Akkumulatoren (Reassoziation erlaubt -> NEON-vektorisierbar) */
static inline float dot16(const float*a,const __fp16*w,int n){
#pragma clang fp reassociate(on)
  float s0=0,s1=0,s2=0,s3=0; int i=0;
  for(;i+4<=n;i+=4){s0+=a[i]*(float)w[i];s1+=a[i+1]*(float)w[i+1];s2+=a[i+2]*(float)w[i+2];s3+=a[i+3]*(float)w[i+3];}
  for(;i<n;i++)s0+=a[i]*(float)w[i];
  return (s0+s1)+(s2+s3);
}
typedef struct{const float*h;const __fp16*Wg,*Wu,*Wd;float*a;const float*x;float*y;}ffn_ctx;
static void ffn_gu(int lo,int hi,void*c_){ffn_ctx*c=c_;for(int j=lo;j<hi;j++){float sg=dot16(c->h,c->Wg+(long)j*D,D),su=dot16(c->h,c->Wu+(long)j*D,D);c->a[j]=sg/(1.f+expf(-sg))*su;}}
static void ffn_dn(int lo,int hi,void*c_){ffn_ctx*c=c_;for(int d=lo;d<hi;d++)c->y[d]=c->x[d]+dot16(c->a,c->Wd+(long)d*FF,FF);}
static void ffn_row(const float*x,const __fp16*fw,float*y){
  const __fp16*g=fw,*Wg=g+D,*Wu=Wg+(long)FF*D,*Wd=Wu+(long)FF*D;
  float h[D]; float s=0; for(int d=0;d<D;d++)s+=x[d]*x[d]; float r=1.f/sqrtf(s/D+1e-6f); for(int d=0;d<D;d++)h[d]=x[d]*r*(float)g[d];
  static float a[FF];
  ffn_ctx c={h,Wg,Wu,Wd,a,x,y}; pfor(FF,ffn_gu,&c); pfor(D,ffn_dn,&c);
}
typedef struct{const float*q,*k,*v;float*ctx;int B;}att_ctx;
static void att_rows(int lo,int hi,void*c_){att_ctx*c=c_;
  for(int b=lo;b<hi;b++)for(int hh=0;hh<H;hh++){int gg=hh/GRP;const float*qh=c->q+((long)b*H+hh)*HD;float sc[512],mx=-1e30f;
    for(int t2=0;t2<=b;t2++){const float*kh=c->k+((long)t2*NKV+gg)*HD;float s=dotf(qh,kh,HD)/sqrtf(HD);sc[t2]=s;if(s>mx)mx=s;}
    float se=0;for(int t2=0;t2<=b;t2++){sc[t2]=expf(sc[t2]-mx);se+=sc[t2];}
    float*co=c->ctx+((long)b*H+hh)*HD;for(int d=0;d<HD;d++)co[d]=0;
    for(int t2=0;t2<=b;t2++){float wgt=sc[t2]/se;const float*vh=c->v+((long)t2*NKV+gg)*HD;for(int d=0;d<HD;d++)co[d]+=wgt*vh[d];}}}
int main(int argc,char**argv){
  int B=argc>1?atoi(argv[1]):8, NL=argc>2?atoi(argv[2]):24;
  const char*dir=argc>3?argv[3]:".";
  char p[512];
  snprintf(p,sizeof p,"%s/embed_B%d.bin",dir,B); float*x=rd(p,(long)B*D);
  snprintf(p,sizeof p,"%s/attn_weights.bin",dir); float*AW=rd(p,(long)NL*WSZ);
  snprintf(p,sizeof p,"%s/scales_B%d.bin",dir,B); float*SC=rd(p,2L*NL); float*isc=SC,*osc=SC+NL;
  snprintf(p,sizeof p,"%s/ffn_weights.f16",dir); const __fp16*FW=NULL;
  {int fd=open(p,O_RDONLY); if(fd<0){fprintf(stderr,"no %s (CPU-Fallback deaktiviert)\n",p);} else {FW=mmap(NULL,(size_t)NL*FSZ*2,PROT_READ,MAP_PRIVATE,fd,0); if(FW==MAP_FAILED){fprintf(stderr,"mmap ffn fail\n");return 1;} close(fd);
    /* Gewichte resident machen (Prefault) - im Zielsystem sind CPU-Gewichte ohnehin im RAM; sonst misst man Page-Faults */
    size_t fsz=(size_t)NL*FSZ*2; madvise((void*)FW,fsz,MADV_WILLNEED); volatile long acc=0; for(size_t i=0;i<fsz;i+=4096)acc+=((const char*)FW)[i]; }}
  /* rope tables */
  float*cs=malloc(B*HD*4),*sn=malloc(B*HD*4);
  for(int t=0;t<B;t++)for(int d=0;d<HD/2;d++){double inv=pow(1000000.0,-(2.0*d)/HD);float c=cosf(t*inv),s=sinf(t*inv);cs[t*HD+d]=cs[t*HD+d+HD/2]=c;sn[t*HD+d]=sn[t*HD+d+HD/2]=s;}
  barra_tpu t; if(barra_tpu_open(&t)){fprintf(stderr,"tpu_open fail\n");return 1;}
  barra_zbuf ZI,ZO; uint32_t sz=B*D*2; if(barra_zc_alloc(&ZI,sz)||barra_zc_alloc(&ZO,sz)){fprintf(stderr,"alloc fail\n");return 1;}
  float*h=malloc(B*D*4),*q=malloc((long)B*H*HD*4),*k=malloc((long)B*NKV*HD*4),*v=malloc((long)B*NKV*HD*4),*ctx=malloc(B*D*4),*o=malloc(B*D*4),*xa=malloc(B*D*4),*ycpu=malloc(B*D*4);
  int*oncpu=calloc(B,sizeof(int)); long nfall=0;
  double tA=0,tF=0,tC=0; double t0=now_us();
  for(int L=0;L<NL;L++){
    float*w=AW+(long)L*WSZ; float*g=w; float*Wq=g+D; float*bq=Wq+D*D; float*Wk=bq+D; float*bk=Wk+NKV*HD*D; float*Wv=bk+NKV*HD; float*bv=Wv+NKV*HD*D; float*Wo=bv+NKV*HD;
    double a0=now_us();
    rmsn(x,g,h,B);
    mm(h,Wq,bq,q,B,H*HD); mm(h,Wk,bk,k,B,NKV*HD); mm(h,Wv,bv,v,B,NKV*HD);
    for(int b=0;b<B;b++){rope(q+(long)b*H*HD,cs+b*HD,sn+b*HD,H);rope(k+(long)b*NKV*HD,cs+b*HD,sn+b*HD,NKV);}
    {att_ctx ac={q,k,v,ctx,B}; pfor(B,att_rows,&ac);}
    mm(ctx,Wo,NULL,o,B,D);
    for(int i=0;i<B*D;i++)xa[i]=x[i]+o[i];
    tA+=now_us()-a0;
    if(L==0 && getenv("DBG")){char dp[512];snprintf(dp,sizeof dp,"%s/xattn0_tpu.bin",dir);FILE*df=fopen(dp,"wb");fwrite(xa,4,B*D,df);fclose(df);return 0;}
    /* quant xa -> int16 ZI (scale isc[L]); Zeilen ausserhalb des int16-Bereichs -> CPU-Fallback */
    short*zi=(short*)ZI.map; float qi=1.f/isc[L];
    for(int b=0;b<B;b++){float mx=0;for(int d=0;d<D;d++){float a=fabsf(xa[b*D+d]);if(a>mx)mx=a;}
      oncpu[b]=(FW && mx*qi>32767.f);
      if(oncpu[b]){memset(zi+b*D,0,D*2);nfall++;}
      else for(int d=0;d<D;d++){int val=(int)lrintf(xa[b*D+d]*qi); if(val>32767)val=32767; if(val<-32768)val=-32768; zi[b*D+d]=(short)val;}}
    double f0=now_us(); uint32_t us=0; barra_tpu_infer(&t,L,&ZI,&ZO,&us); tF+=now_us()-f0;
    short*zo=(short*)ZO.map; float qo=osc[L];
    /* Ausgangszeile clippt (int16-Rand) -> Zeile passt nicht in den Ausgangsbereich (z.B. L2: FFN ERZEUGT die
       Massive Activation bei unauffaelligem Eingang) -> ebenfalls CPU-Fallback */
    if(FW)for(int b=0;b<B;b++)if(!oncpu[b]){for(int d=0;d<D;d++){short z=zo[b*D+d];if(z>=32767||z<=-32768){oncpu[b]=1;nfall++;break;}}}
    double c0=now_us(); int nf=0; for(int b=0;b<B;b++)if(oncpu[b]){ffn_row(xa+b*D,FW+(long)L*FSZ,ycpu+b*D);nf++;} tC+=now_us()-c0;
    if(nf && getenv("DUMPL")){fprintf(stderr,"  L%d: %d Zeile(n) CPU-Fallback:",L,nf);for(int b=0;b<B;b++)if(oncpu[b])fprintf(stderr," %d",b);fprintf(stderr,"\n");}
    for(int b=0;b<B;b++){if(oncpu[b])memcpy(x+b*D,ycpu+b*D,D*4);else for(int d=0;d<D;d++)x[b*D+d]=zo[b*D+d]*qo;}
    if(getenv("DUMPL")){char dp[512];FILE*df; /* per-layer Diagnose: xattn (vor TPU) + x (nach TPU) */
      snprintf(dp,sizeof dp,"%s/dump/xa_L%d.bin",dir,L);df=fopen(dp,"wb");if(df){fwrite(xa,4,B*D,df);fclose(df);}
      snprintf(dp,sizeof dp,"%s/dump/x_L%d.bin",dir,L);df=fopen(dp,"wb");if(df){fwrite(x,4,B*D,df);fclose(df);}}
  }
  double wall=now_us()-t0;
  snprintf(p,sizeof p,"%s/hidden_tpu_B%d.bin",dir,B); FILE*f=fopen(p,"wb");fwrite(x,4,B*D,f);fclose(f);
  fprintf(stderr,"[prefill] NL=%d B=%d: wall %.1f ms (attn-CPU %.1f, ffn-TPU %.1f, ffn-CPU-fallback %.1f fuer %ld Zeilen) -> %.2f ms/Tok\n",NL,B,wall/1000,tA/1000,tF/1000,tC/1000,nfall,wall/1000/B);
  return 0;
}
