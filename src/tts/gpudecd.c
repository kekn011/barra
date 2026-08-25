// gpudecd — persistenter GPU-f16-HiFi-GAN-Vokoder-Daemon. GPU + Gewichte + Shader EINMAL laden
// (warm: erster Request ~3s Shader-Compile, danach ~0,6s), dann pro Unix-Socket-Verbindung:
//   <- int32 T, dann 192*T float32 (z, channel-major [192,T])
//   -> int32 nsamples, dann nsamples float32 (wav 22050Hz)
// gcc gpudecd.c -o gpudecd -lbarra -lm
#include <barra.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include <errno.h>
#include "convfused16_spv.h"
#include "convfused16s_spv.h"
#include "shuffle16_spv.h"
#include "mrf16_spv.h"
typedef unsigned short u16;
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void* slurp(const char* p,long* n){ FILE* f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
  fseek(f,0,SEEK_END);*n=ftell(f);fseek(f,0,SEEK_SET);void* b=malloc(*n);if(fread(b,1,*n,f)!=(size_t)*n)exit(1);fclose(f);return b; }
static u16 f2h(float f){ unsigned x; memcpy(&x,&f,4); unsigned s=(x>>16)&0x8000; int e=((x>>23)&0xff)-127+15; unsigned m=(x>>13)&0x3ff;
  if(e<=0){ if(e<-10) return (u16)s; unsigned mm=(x&0x7fffff)|0x800000; int sh=14-e; return (u16)(s|(mm>>sh)); } if(e>=31) return (u16)(s|0x7c00); return (u16)(s|(e<<10)|m); }
static float h2f(u16 h){ unsigned s=((unsigned)(h&0x8000))<<16; int e=(h>>10)&0x1f; unsigned m=h&0x3ff,f;
  if(e==0){ if(m==0)f=s; else { e=1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; f=s|((unsigned)(e-15+127)<<23)|(m<<13);} }
  else if(e==31) f=s|0x7f800000|(m<<13); else f=s|((unsigned)(e-15+127)<<23)|(m<<13); float r; memcpy(&r,&f,4); return r; }
static const char* fk(const char* s,const char* e,const char* k){ char pat[48];snprintf(pat,sizeof pat,"\"%s\"",k);size_t kl=strlen(pat);
  for(const char* p=s;p+kl<e;p++) if(!strncmp(p,pat,kl)){p+=kl;while(p<e&&(*p==' '||*p==':'))p++;return p;} return 0; }
static long ji(const char* s,const char* e,const char* k){ const char* v=fk(s,e,k); return v?atol(v):0; }
static void js(const char* s,const char* e,const char* k,char* o,int c){ const char* v=fk(s,e,k);o[0]=0;if(!v||*v!='"')return;v++;int i=0;while(v<e&&*v!='"'&&i<c-1)o[i++]=*v++;o[i]=0; }
static int nextobj(const char** pp,const char* e,const char** os,const char** oe){ const char* p=*pp;while(p<e&&*p!='{')p++;if(p>=e)return 0;int d=0;const char* s=p;
  for(;p<e;p++){if(*p=='{')d++;else if(*p=='}'){if(--d==0){*os=s;*oe=p+1;*pp=p+1;return 1;}}} return 0; }
static int up128(int x){ return (x+127)/128*128; }

#define MAXSLOT 48
static char snm[MAXSLOT][20]; static barra_zbuf sbuf[MAXSLOT]; static int sC[MAXSLOT],sT[MAXSLOT]; static long ssz[MAXSLOT]; static int nsl=0;
static int slot(const char* n){ for(int i=0;i<nsl;i++) if(!strcmp(snm[i],n))return i; if(nsl>=MAXSLOT){ fprintf(stderr,"[gpudecd] zu viele Puffer-Slots (>%d)\n",MAXSLOT); exit(1); } strncpy(snm[nsl],n,19); snm[nsl][19]=0; ssz[nsl]=0; sC[nsl]=sT[nsl]=0; return nsl++; }

#define POOL 30
#define MAXPEND 30
static barra_zbuf ppool[POOL]; static int pcur=0;
static barra_gpu_stage pend[MAXPEND]; static barra_zbuf* pendbuf[MAXPEND][6]; static int npend=0;
static barra_gpu* G; static int g_fl=0;
static barra_zbuf ZW,ZB; static char* J; static long jl; static const char* JE; static const char* oarr;

static int flush_batch(){
  if(npend==0) return 0;
  int rc=barra_gpu_batch(G,pend,npend);
  if(rc) fprintf(stderr,"flush#%d FAIL nstage=%d\n",g_fl,npend);
  g_fl++; npend=0; pcur=0; return rc;
}
static void ensure_room(int n){ if(npend+n>MAXPEND || pcur+n>POOL){ if(flush_batch()){ fprintf(stderr,"flush fail\n"); } } }
static barra_zbuf* addstage(const uint8_t* spv,uint32_t slen,uint32_t gx,uint32_t gy,uint32_t gz,barra_zbuf** bufs,int nbuf){
  int s=npend++; barra_zbuf* PP=&ppool[pcur++];
  pendbuf[s][0]=PP; for(int i=1;i<nbuf;i++) pendbuf[s][i]=bufs[i];
  pend[s].spirv=spv; pend[s].slen=slen; pend[s].gx=gx; pend[s].gy=gy; pend[s].gz=gz; pend[s].bufs=pendbuf[s]; pend[s].nbuf=nbuf;
  return PP;
}

// verarbeitet EIN z[192,T] -> wav; alloc/free der Slots je Request (GPU/Shader bleiben warm)
static float* process(const float* zf,int T,int* out_n){
  nsl=0; npend=0; pcur=0;
  int sz=slot("z"); sT[sz]=T; sC[sz]=192; ssz[sz]=(long)192*T;
  { const char* p=oarr; const char* os,*oe;
    while(nextobj(&p,JE,&os,&oe)){ char op[16]; js(os,oe,"op",op,sizeof op);
      if(!strcmp(op,"conv")){ char sn[20],dn[20]; js(os,oe,"src",sn,20); js(os,oe,"dst",dn,20);
        int Cout=ji(os,oe,"Cout"),outpad=ji(os,oe,"outpad");
        int si=slot(sn); int Tout=sT[si]+outpad;
        int di=slot(dn); if((long)Cout*Tout>ssz[di])ssz[di]=(long)Cout*Tout; sC[di]=Cout; sT[di]=Tout;
      } else if(!strcmp(op,"shuffle")){ char sn[20],dn[20]; js(os,oe,"src",sn,20); js(os,oe,"dst",dn,20);
        int r=ji(os,oe,"r"),Cout=ji(os,oe,"Cout"),pad=ji(os,oe,"pad"),K=ji(os,oe,"K"),Jj=ji(os,oe,"J");
        int si=slot(sn); int Tin=sT[si]-(Jj-1); int outT=(Tin-1)*r+K-2*pad;
        int di=slot(dn); if((long)Cout*outT>ssz[di])ssz[di]=(long)Cout*outT; sC[di]=Cout; sT[di]=outT;
      } else if(!strcmp(op,"mrf")){ char an[20],dn[20]; js(os,oe,"a",an,20); js(os,oe,"dst",dn,20);
        int ai=slot(an); int di=slot(dn); if((long)sC[ai]*sT[ai]>ssz[di])ssz[di]=(long)sC[ai]*sT[ai]; sC[di]=sC[ai]; sT[di]=sT[ai]; }
    }
  }
  for(int i=0;i<nsl;i++) if(barra_zc_alloc(&sbuf[i],(size_t)ssz[i]*2)){ fprintf(stderr,"[gpudecd] zc_alloc Slot %d (%ld B) fehlgeschlagen\n",i,(long)ssz[i]*2); *out_n=0; return 0; }
  barra_zc_cpu_begin(&sbuf[sz]); u16* zt=(u16*)sbuf[sz].map; for(int c=0;c<192;c++) for(int t=0;t<T;t++) zt[c*T+t]=f2h(zf[c*T+t]); barra_zc_cpu_end(&sbuf[sz]);
  sC[sz]=192; sT[sz]=T;

  const char* p=oarr; const char* os,*oe;
  while(nextobj(&p,JE,&os,&oe)){
    char op[16]; js(os,oe,"op",op,sizeof op);
    if(!strcmp(op,"conv")){
      char sn[20],dn[20],rn[20]; js(os,oe,"src",sn,20); js(os,oe,"dst",dn,20); js(os,oe,"res",rn,20);
      int Cin=ji(os,oe,"Cin"),Kk=ji(os,oe,"K"),dil=ji(os,oe,"dil"),pad=ji(os,oe,"pad"),leaky=ji(os,oe,"leaky"),Cout=ji(os,oe,"Cout");
      int Mp=ji(os,oe,"Mp"),Kg=ji(os,oe,"Kg"),woff=ji(os,oe,"woff"),boff=ji(os,oe,"boff"),tanhf_=ji(os,oe,"tanh"),outpad=ji(os,oe,"outpad"),tm=ji(os,oe,"tm");
      if(tm<=0){ fprintf(stderr,"[gpudecd] conv: 'tm' fehlt/0 (Division) - Programm ungueltig\n"); *out_n=0; return 0; }
      int si=slot(sn); int Tin=sT[si]; int Tout=Tin+outpad; int Np=up128(Tout);
      int di=slot(dn); sC[di]=Cout; sT[di]=Tout;
      int hasres = rn[0]!=0; int ri = hasres?slot(rn):sz;
      ensure_room(1);
      const uint8_t* cgspv = (tm==32)?convfused16s_spv:convfused16_spv; uint32_t cgslen=(tm==32)?convfused16s_spv_len:convfused16_spv_len;
      barra_zbuf* cg[6]={0,&ZW,&sbuf[si],&ZB,&sbuf[ri],&sbuf[di]};
      barra_zbuf* PP=addstage(cgspv,cgslen,(uint32_t)Np/128,(uint32_t)Mp/tm,1,cg,6);
      barra_zc_cpu_begin(PP); uint32_t* q=PP->map;
      q[0]=Mp;q[1]=Np;q[2]=Kg;q[3]=Cin;q[4]=Kk;q[5]=dil;q[6]=pad;q[7]=Tout;q[8]=leaky;q[9]=(uint32_t)woff;q[10]=Tin;q[11]=Cout;q[12]=boff;q[13]=(hasres?1u:0u)|(tanhf_?2u:0u);
      barra_zc_cpu_end(PP);
    } else if(!strcmp(op,"shuffle")){
      char sn[20],dn[20]; js(os,oe,"src",sn,20); js(os,oe,"dst",dn,20);
      int r=ji(os,oe,"r"),Cout=ji(os,oe,"Cout"),pad=ji(os,oe,"pad"),K=ji(os,oe,"K"),Jj=ji(os,oe,"J");
      int si=slot(sn); int Tconv=sT[si]; int Tin=Tconv-(Jj-1); int outT=(Tin-1)*r+K-2*pad;
      int di=slot(dn); sC[di]=Cout; sT[di]=outT;
      ensure_room(1);
      barra_zbuf* sb[3]={0,&sbuf[si],&sbuf[di]};
      barra_zbuf* PS=addstage(shuffle16_spv,shuffle16_spv_len,((uint32_t)Cout*outT+255)/256,1,1,sb,3);
      barra_zc_cpu_begin(PS); uint32_t* q=PS->map; q[0]=Cout;q[1]=outT;q[2]=r;q[3]=pad;q[4]=Tconv; barra_zc_cpu_end(PS);
    } else if(!strcmp(op,"mrf")){
      char an[20],bn[20],cn[20],dn[20]; js(os,oe,"a",an,20); js(os,oe,"b",bn,20); js(os,oe,"c",cn,20); js(os,oe,"dst",dn,20);
      int ai=slot(an),bi=slot(bn),ci=slot(cn),C=sC[ai],Tt=sT[ai]; int di=slot(dn); sC[di]=C; sT[di]=Tt;
      ensure_room(1);
      barra_zbuf* mb[5]={0,&sbuf[ai],&sbuf[bi],&sbuf[ci],&sbuf[di]};
      barra_zbuf* PM=addstage(mrf16_spv,mrf16_spv_len,((uint32_t)C*Tt+255)/256,1,1,mb,5);
      barra_zc_cpu_begin(PM); uint32_t* q=PM->map; q[0]=(uint32_t)C*Tt; barra_zc_cpu_end(PM);
    }
  }
  flush_batch();
  int sw=slot("wav"); int Tt=sT[sw];
  barra_zc_cpu_begin(&sbuf[sw]); u16* wv=sbuf[sw].map; float* out=malloc((size_t)Tt*4); if(!out){ *out_n=0; return 0; } for(int t=0;t<Tt;t++) out[t]=h2f(wv[t]);
  /* Per-Request-Handles aus der gpud-zc-Session freigeben, sonst laeuft die Handle-Tabelle
   * (MAXH=64) nach ~3 Requests voll -> "import fail (slot -1)" -> Vokoder bricht ab (Fund F25,
   * am Node am 25.8. verifiziert). ZW/ZB/ppool bleiben importiert (warm); nur die je Request
   * frischen sbuf werden freigegeben. Blockweise, da barra_gpu_release max. ZC_MAXBUF(16) nimmt. */
  for(int i=0;i<nsl;){ barra_zbuf* rel[16]; int k=0; while(k<16 && i<nsl) rel[k++]=&sbuf[i++]; barra_gpu_release(G,rel,k); }
  for(int i=0;i<nsl;i++) barra_zc_free(&sbuf[i]);
  *out_n=Tt; return out;
}

static int readn(int fd,void* b,size_t n){ size_t o=0; while(o<n){ ssize_t r=read(fd,(char*)b+o,n-o); if(r<=0) return -1; o+=r; } return 0; }
static int writen(int fd,const void* b,size_t n){ size_t o=0; while(o<n){ ssize_t r=write(fd,(char*)b+o,n-o); if(r<=0) return -1; o+=r; } return 0; }

int main(int argc,char**argv){
  if(argc<4){ fprintf(stderr,"gpudecd <sock> <program2.json> <gpukit-dir>\n"); return 1; }
  const char* sockpath=argv[1];
  J=slurp(argv[2],&jl); JE=J+jl; oarr=fk(J,JE,"ops");
  char wp[512],bp[512]; snprintf(wp,sizeof wp,"%s/weights16.bin",argv[3]); snprintf(bp,sizeof bp,"%s/bias16.bin",argv[3]);
  long wl,bl; u16* wblob=slurp(wp,&wl); u16* bblob=slurp(bp,&bl);
  signal(SIGPIPE,SIG_IGN);

  barra_gpu g; if(barra_gpu_open(&g)){ fprintf(stderr,"gpu open FAIL\n"); return 1; } G=&g;
  barra_zc_alloc(&ZW,wl); barra_zc_cpu_begin(&ZW); memcpy(ZW.map,wblob,wl); barra_zc_cpu_end(&ZW);
  barra_zc_alloc(&ZB,bl); barra_zc_cpu_begin(&ZB); memcpy(ZB.map,bblob,bl); barra_zc_cpu_end(&ZB);
  for(int i=0;i<POOL;i++) barra_zc_alloc(&ppool[i],64);

  int srv=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sockpath,sizeof(a.sun_path)-1);
  unlink(sockpath);
  if(bind(srv,(struct sockaddr*)&a,sizeof a)<0){ perror("bind"); return 1; }
  chmod(sockpath,0660); listen(srv,4);   /* nur Owner/Gruppe (nicht welt-beschreibbar) */
  fprintf(stderr,"[gpudecd] bereit, lauscht auf %s (Vokoder warm nach 1. Request)\n",sockpath);

  for(;;){
    int c=accept(srv,0,0); if(c<0) continue;
    struct timeval tv={5,0};   /* Lese-Timeout: ein Client, der zu wenig sendet, blockiert den Daemon nicht mehr */
    setsockopt(c,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);
    int T=0; if(readn(c,&T,4)||T<=0||T>100000){ close(c); continue; }
    float* zf=malloc((size_t)192*T*4);
    if(!zf){ close(c); continue; }
    if(readn(c,zf,(size_t)192*T*4)){ free(zf); close(c); continue; }
    double t0=now(); int n=0; float* wav=process(zf,T,&n); double dt=now()-t0;
    fprintf(stderr,"[gpudecd] T=%d -> %d samples (%.2fs Audio) in %.3fs (RTF %.3f)\n",T,n,n/22050.0,dt,dt/(n/22050.0));
    writen(c,&n,4); writen(c,wav,(size_t)n*4);
    free(zf); free(wav); close(c);
  }
  return 0;
}
