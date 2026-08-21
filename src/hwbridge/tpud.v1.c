/* tpud - TPU-Inferenz-Daemon (Bionic), MULTI-MODELL + asymmetrische I/O.
 * Laedt N Modelle (Args), fragt je Modell die Input/Output-Tensor-Groessen via DarwinnApi2 SELBST ab.
 * Protokoll (persistente Verbindung, viele Requests):
 *   Request:  u32 magic 'TPD2', u32 model_id, u32 in_size, [in_size Bytes]
 *   Response: u32 status(0=ok), u32 out_size, [out_size Bytes]
 * Korrektheit: FRISCHE Buffer/Request (Cross-Request-Kontamination), Warmup je Modell (Kaltstart),
 * adaptiver Stability-Poll (async Submit; Sentinel + Fingerprint bis stabil).  15.8. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <errno.h>
#include <signal.h>
#include <time.h>

#define MAGIC 0x54504432u   /* 'TPD2' */
#define MAXMODELS 16

static void *H;
#define S(n) dlsym(H,n)
static void* (*allocBuf)(void*,void*,void*,long,void*);
static void* (*mapHost)(void*,void*);
static void* (*mapDev)(void*);
static void* (*invCache)(void*);
static void* (*bufFree)(void*);
static void* (*createReq)(void*,void*,void*);
static void* (*addIn)(void*,int,void*);
static void* (*addOut)(void*,int,void*);
static void* (*submit)(void*,void*);
static void* (*reqFree)(void*);
static int   (*getInTensor)(void*,int,void**);
static int   (*getOutTensor)(void*,int,void**);
static long  (*tensorSize)(void*);
static void* (*regGraph)(void*,void*,long,int,int,void*);
static long  (*numIn)(void*);
static long  (*numOut)(void*);
static void *g_opt,*g_factory,*g_vdev;

typedef struct { void* graph; long in_size; long out_size; char name[64]; } Model;
static Model g_models[MAXMODELS];
static int g_nmodels=0;

static unsigned char* g_prev=0; static long g_prev_cap=0;
static unsigned char g_sctr=0;
static long g_wait_us=0, g_max_wait_us=500000, g_poll_us=500;

static void* mkbuf(long sz, void** host){
  long descr[4]={1,0,0,0}; void* buf=0; allocBuf(g_factory,g_opt,descr,sz,&buf);
  if(!buf) return 0; void* hp=0; mapHost(buf,&hp); if(host)*host=hp; return buf;
}

/* eine Inferenz auf FRISCHEN Buffern fuer Modell mid. oh=host des Output-Buffers (out_size). */
static int tpu_infer(int mid, void* inBuf, void* outBuf, unsigned char* oh){
  void* g=g_models[mid].graph; long osz=g_models[mid].out_size;
  void* req=0;
  if(g_wait_us>0){
    mapDev(inBuf); mapDev(outBuf); createReq(g_vdev,g,&req); if(!req) return -1;
    addIn(req,0,inBuf); addOut(req,0,outBuf); submit(g_vdev,req); usleep(g_wait_us); if(invCache) invCache(outBuf);
    if(reqFree) reqFree(req); return 0;
  }
  unsigned char sv=++g_sctr; if(!sv) sv=++g_sctr;
  memset(oh, sv, osz);
  mapDev(inBuf); mapDev(outBuf);
  createReq(g_vdev,g,&req); if(!req) return -1;
  addIn(req,0,inBuf); addOut(req,0,outBuf);
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  submit(g_vdev,req);
  #define NSAMP 64
  long stride=osz/NSAMP; if(stride<1) stride=1;
  unsigned long fp,prev_fp=0; long waited=0; int stable=0, started=0;
  #define MKFP(F) do{ F=1469598103934665603UL; for(int s=0;s<NSAMP;s++){ long p=(long)s*stride; if(p<osz){ F^=(unsigned char)oh[p]; F*=1099511628211UL; } } }while(0)
  if(invCache) invCache(outBuf); MKFP(prev_fp);
  while(waited<g_max_wait_us){
    usleep(g_poll_us); waited+=g_poll_us;
    if(invCache) invCache(outBuf);
    if(!started){ for(int s=0;s<NSAMP;s++){ long p=(long)s*stride; if(p<osz && (unsigned char)oh[p]!=sv){ started=1; break; } } }
    MKFP(fp);
    if(started && fp==prev_fp){
      if(++stable>=2){ memcpy(g_prev,oh,osz); usleep(g_poll_us); if(invCache) invCache(outBuf);
        if(memcmp(g_prev,oh,osz)==0) break; stable=0; MKFP(fp); }
    } else stable=0;
    prev_fp=fp;
  }
  clock_gettime(CLOCK_MONOTONIC,&t1);
  if(reqFree) reqFree(req);
  double us=(t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_nsec-t0.tv_nsec)/1e3;
  static long g_cnt=0; static double g_sum=0; g_cnt++; g_sum+=us;
  if(g_cnt<=3 || g_cnt%500==0) fprintf(stderr,"[tpud] reine Inferenz #%ld (M%d): %.2f ms (Schnitt %.2f ms)\n",g_cnt,mid,us/1000.0,g_sum/g_cnt/1000.0);
  return 0;
}

/* Warmup: erster Submit nach Registrierung ist kalt (JIT). Serielle Laeufe bis Output stabil. */
static void warmup(int mid){
  long isz=g_models[mid].in_size, osz=g_models[mid].out_size;
  void *ih=0,*oh=0; void* ib=mkbuf(isz,&ih); void* ob=mkbuf(osz,&oh);
  if(!ib||!ob){ fprintf(stderr,"[tpud] Warmup %s: kein Buffer\n",g_models[mid].name); return; }
  unsigned char* prev=malloc(osz); memset(ih,0x5A,isz);
  int stable=0, it=0;
  for(; it<12 && stable<2; it++){
    void* req=0; mapDev(ib); mapDev(ob); createReq(g_vdev,g_models[mid].graph,&req);
    addIn(req,0,ib); addOut(req,0,ob); submit(g_vdev,req); usleep(200000); if(invCache) invCache(ob);
    if(reqFree) reqFree(req);
    if(it>0 && prev && memcmp(prev,oh,osz)==0) stable++; else stable=0;
    if(prev) memcpy(prev,oh,osz);
  }
  free(prev); if(bufFree){ bufFree(ib); bufFree(ob); }
  fprintf(stderr,"[tpud] Warmup %s: %d Iter -> %s\n",g_models[mid].name,it,stable>=2?"stabil":"WARNUNG");
}

static int load_model(const char* path){
  if(g_nmodels>=MAXMODELS) return -1;
  FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"[tpud] kein Modell %s\n",path); return -1; }
  fseek(f,0,SEEK_END); long psz=ftell(f); fseek(f,0,SEEK_SET);
  void* pdata=malloc(psz); fread(pdata,1,psz,f); fclose(f);
  long pasz=(psz+4095)&~4095L; void* ph=0; void* pbuf=mkbuf(pasz,&ph);
  memcpy(ph,pdata,psz); mapDev(pbuf); free(pdata);
  void* graph=0; regGraph(g_vdev,pbuf,0,1,0,&graph);
  if(!graph){ fprintf(stderr,"[tpud] RegisterGraph FAIL %s\n",path); return -1; }
  int mid=g_nmodels++;
  g_models[mid].graph=graph;
  void* it=0; getInTensor(graph,0,&it);  g_models[mid].in_size = tensorSize(it);
  void* ot=0; getOutTensor(graph,0,&ot); g_models[mid].out_size= tensorSize(ot);
  const char* b=strrchr(path,'/'); snprintf(g_models[mid].name,sizeof g_models[mid].name,"%s",b?b+1:path);
  if(g_models[mid].out_size>g_prev_cap){ g_prev_cap=g_models[mid].out_size; g_prev=realloc(g_prev,g_prev_cap); }
  fprintf(stderr,"[tpud] Modell %d '%s': NumIn=%ld NumOut=%ld, in=%ld out=%ld Bytes\n",
    mid,g_models[mid].name,numIn(graph),numOut(graph),g_models[mid].in_size,g_models[mid].out_size);
  warmup(mid);
  return mid;
}

static int tpu_init(void){
  H=dlopen("/vendor/lib64/libedgetpu_util.so", RTLD_NOW|RTLD_GLOBAL);
  if(!H){fprintf(stderr,"[tpud] dlopen FAIL %s\n",dlerror());return -1;}
  void* (*getSpec)(int*,unsigned char*,int*,int*,void**,unsigned char*)=(void*(*)(int*,unsigned char*,int*,int*,void**,unsigned char*))S("DarwinnDelegate_GetDefaultDeviceSpec");
  void* (*createVD)(long,long,long,long,void*,long,void*)=(void*(*)(long,long,long,long,void*,long,void*))S("DarwinnDelegate_CreateVirtualDevice");
  void* (*getVD)(void*)=(void*(*)(void*))S("DarwinnDelegate_EdgeTpuDevice_GetVirtualDevice");
  void* (*getBF)(void*)=(void*(*)(void*))S("DarwinnApi2_VirtualDevice_GetBufferFactory");
  void* (*getSupported)(void*,void*,int*)=(void*(*)(void*,void*,int*))S("DarwinnApi2_BufferFactory_GetSupportedOptions");
  regGraph=(void*(*)(void*,void*,long,int,int,void*))S("DarwinnApi2_VirtualDevice_RegisterGraphByBuffer");
  numIn=(long(*)(void*))S("DarwinnApi2_Graph_NumInputTensors");
  numOut=(long(*)(void*))S("DarwinnApi2_Graph_NumOutputTensors");
  getInTensor=(int(*)(void*,int,void**))S("DarwinnApi2_Graph_GetInputTensor");
  getOutTensor=(int(*)(void*,int,void**))S("DarwinnApi2_Graph_GetOutputTensor");
  tensorSize=(long(*)(void*))S("DarwinnApi2_TensorInfo_SizeBytes");
  allocBuf=(void*(*)(void*,void*,void*,long,void*))S("DarwinnApi2_BufferFactory_AllocateBuffer");
  mapHost=(void*(*)(void*,void*))S("DarwinnApi2_Buffer_MapToHost");
  mapDev=(void*(*)(void*))S("DarwinnApi2_Buffer_MapToDevice");
  invCache=(void*(*)(void*))S("DarwinnApi2_Buffer_InvalidateCache");
  bufFree=(void*(*)(void*))S("DarwinnApi2_Buffer_Free");
  createReq=(void*(*)(void*,void*,void*))S("DarwinnApi2_VirtualDevice_CreateInferenceRequest");
  addIn=(void*(*)(void*,int,void*))S("DarwinnApi2_InferenceRequest_AddInput");
  addOut=(void*(*)(void*,int,void*))S("DarwinnApi2_InferenceRequest_AddOutput");
  submit=(void*(*)(void*,void*))S("DarwinnApi2_VirtualDevice_Submit");
  reqFree=(void*(*)(void*))S("DarwinnApi2_InferenceRequest_Free");

  int s0,s2,s3; unsigned char s1,s5; void* s4=0; getSpec(&s0,&s1,&s2,&s3,&s4,&s5);
  void* edgeDev=0; createVD((long)s0,(long)s1,(long)s2,(long)s3,s4,(long)s5,&edgeDev);
  g_vdev=getVD(edgeDev); g_factory=getBF(g_vdev);
  void* arr=0; int cnt=0; getSupported(g_factory,&arr,&cnt); g_opt=((void**)arr)[0];
  return 0;
}

static int read_full(int fd,void*b,long n){char*p=b;while(n>0){ssize_t r=read(fd,p,n);if(r<=0)return -1;p+=r;n-=r;}return 0;}
static int write_full(int fd,const void*b,long n){const char*p=b;while(n>0){ssize_t w=write(fd,p,n);if(w<=0)return -1;p+=w;n-=w;}return 0;}
static int ru32(int fd,uint32_t*v){return read_full(fd,v,4);}

static long g_nreq=0;
static void handle_conn(int c){
  for(;;){
    uint32_t magic=0,mid=0,insz=0;
    if(ru32(c,&magic)) break;
    if(magic!=MAGIC){ break; }
    if(ru32(c,&mid)||ru32(c,&insz)) break;
    if(mid>=(uint32_t)g_nmodels || (long)insz!=g_models[mid].in_size){
      uint32_t st=1, z=0; write_full(c,&st,4); write_full(c,&z,4); break;
    }
    long isz=g_models[mid].in_size, osz=g_models[mid].out_size;
    void *ih=0,*oh=0;
    void* inBuf=mkbuf(isz,&ih); if(!inBuf) break;
    if(read_full(c,ih,isz)!=0){ if(bufFree) bufFree(inBuf); break; }
    void* outBuf=mkbuf(osz,&oh); if(!outBuf){ if(bufFree) bufFree(inBuf); break; }
    int ok = tpu_infer((int)mid,inBuf,outBuf,(unsigned char*)oh)==0;
    uint32_t st=ok?0:2, osz32=(uint32_t)osz;
    int wr = (write_full(c,&st,4)||write_full(c,&osz32,4)|| (ok?write_full(c,oh,osz):0));
    if(bufFree){ bufFree(inBuf); bufFree(outBuf); }
    g_nreq++; if((g_nreq%1000)==1) fprintf(stderr,"[tpud] req #%ld (Modell %u)\n",g_nreq,mid);
    if(wr!=0) break;
  }
}

int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: tpud <socket> <model0.package> [model1.package ...]\n"); return 1; }
  const char* sock=argv[1];
  const char* w=getenv("TPU_WAIT_US"); if(w) g_wait_us=atol(w);
  const char* pw=getenv("TPU_POLL_US"); if(pw) g_poll_us=atol(pw);
  signal(SIGPIPE,SIG_IGN);
  if(tpu_init()!=0) return 1;
  for(int i=2;i<argc;i++) if(load_model(argv[i])<0) return 1;
  if(g_nmodels==0){ fprintf(stderr,"[tpud] kein Modell geladen\n"); return 1; }

  int srv=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
  unlink(sock);
  if(bind(srv,(struct sockaddr*)&a,sizeof a)<0){ fprintf(stderr,"[tpud] bind %s: %s\n",sock,strerror(errno)); return 1; }
  chmod(sock,0666); listen(srv,8);
  fprintf(stderr,"[tpud] bereit, lauscht auf %s (%d Modell(e), %s)\n",sock,g_nmodels,g_wait_us>0?"fixe Wartezeit":"adaptiver Poll");
  for(;;){ int c=accept(srv,0,0); if(c<0) continue; handle_conn(c); close(c); }
  return 0;
}
