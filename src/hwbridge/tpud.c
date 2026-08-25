/* tpud - TPU-Inferenz-Daemon (Bionic), MULTI-MODELL + asymmetrische I/O + ZERO-COPY.
 * Laedt N Modelle (Args), fragt je Modell die Input/Output-Tensor-Groessen via DarwinnApi2 SELBST ab.
 *
 * Protokoll (A) TPD2 0x54504432 — inline (persistente Verbindung, viele Requests):
 *   Request:  u32 magic 'TPD2', u32 model_id, u32 in_size, [in_size Bytes]
 *   Response: u32 status(0=ok), u32 out_size, [out_size Bytes]
 *
 * Protokoll (B) TPZ2 0x54505A32 — ZERO-COPY-SESSION (16.8.): der Client alloziert dmabufs
 *   (/dev/dma_heap/system), schickt die fds via SCM_RIGHTS, tpud importiert sie EINMAL als
 *   Tensor-Puffer (DarwinnApi2_BufferFactory_ImportBufferByFd, fd_type 0 = dmabuf) und die TPU
 *   rechnet direkt hinein/heraus; KEINE Tensordaten ueber den Socket. Jeder Befehl = 6*u32
 *   [magic, cmd, a, b, c, d]:
 *   cmd=1 IMPORT : [.,1,nbuf,0,0,0] + SCM_RIGHTS(nbuf fds); dann nbuf*u32 size -> u32 status, nbuf*u32 handle
 *   cmd=2 INFER  : [.,2,model_id,in_h,out_h,0]                                -> u32 status, u32 exec_us
 *   cmd=3 RELEASE: [.,3,nbuf,0,0,0]; dann nbuf*u32 handle                      -> u32 status
 *   cmd=4 INFO   : [.,4,model_id,0,0,0]                                       -> u32 status, u32 in_size, u32 out_size, u32 nmodels
 *   Verbindungsende gibt alle Handles frei. Der Sentinel/Stabilitaets-Poll liest den Output
 *   aus tpuds EIGENER mmap desselben dmabufs (Cache-Invalidate via Buffer_InvalidateCache).
 *
 * Korrektheit: FRISCHE Request-Objekte, Warmup je Modell (Kaltstart), adaptiver Stability-Poll
 * (async Submit; Sentinel + Fingerprint bis stabil).  15.8. / ZC 16.8. */
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <poll.h>
#include <math.h>

#define MAGIC    0x54504432u   /* 'TPD2' inline */
#define MAGIC_ZC 0x54505A32u   /* 'TPZ2' zero-copy session */
#define MAXMODELS 224  /* 36 Layer x 5 Attention-Packages (barra-attn 16x8, q gechunkt) + Reserve */
#define ZC_MAXH   32
#define ZC_MAXBUF 16

static void *H;
#define S(n) dlsym(H,n)
static void* (*allocBuf)(void*,void*,void*,long,void*);
static void* (*impFd)(void*,void*,int,int,long,long,void*);   /* ImportBufferByFd(factory,opt,fd_type=0(dmabuf),fd,size,offset,&out) */
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
/* Fence-Pipeline (async): SubmitFenced(vdev,req,&fence); Fence_IsCompleted(fence)->bool; Fence_Free(fence) */
static void* (*submitFenced)(void*,void*,void**);
static int   (*fenceDone)(void*);
static void* (*fenceDupFd)(void*,int*);
static void* (*fenceFree)(void*);
static void* (*flushCache)(void*);   /* Host-Schreib -> Device sichtbar (Input-Kohaerenz) */
static int   (*isCoherent)(void*);
static int   (*isMapHost)(void*);
static int   (*isMapDev)(void*);
static void* (*copyFrom)(void*,void*,long,long);  /* Buffer -> host dst (kohaerent) */
static void* (*copyTo)(void*,const void*,long,long);
static void *g_opt,*g_factory,*g_vdev;
/* Ein Thread je Verbindung (eine Zero-Copy-Session darf Inline-Clients nicht blockieren);
 * ALLE TPU-Aufrufe (Buffer/Import/Infer/Free, globale Poll-Zustaende) laufen unter g_tpu. */
static pthread_mutex_t g_tpu=PTHREAD_MUTEX_INITIALIZER;
#define TPU_LOCK()   pthread_mutex_lock(&g_tpu)
#define TPU_UNLOCK() pthread_mutex_unlock(&g_tpu)

#define MAXIO 8
typedef struct { void* graph; long in_size; long out_size; char name[64];
                 int n_in, n_out; long in_sizes[MAXIO], out_sizes[MAXIO]; } Model;   /* in_size/out_size = Tensor 0 (Kompat) */
static Model g_models[MAXMODELS];
static int g_nmodels=0;

static unsigned char* g_prev=0; static long g_prev_cap=0;
static unsigned char g_sctr=0;
static long g_wait_us=0, g_max_wait_us=500000, g_poll_us=500;
static int g_use_fence=0, g_fence_yield=0;   /* Fence-Completion statt Sentinel-Poll (siehe tpu_infer_multi) */
#include <sched.h>
/* ---- TPU-Wakelock (18.8.): der edgetpu-Treiber schaltet die TPU nach JEDEM Job ab (Client-Wakelock nur pro
   Request) -> jede Inferenz zahlt ~5 ms Resume (rms-Graph 6.8 ms statt 2.0). Loesung: Debug-Wakelock des Treibers
   (/sys/kernel/debug/edgetpu/rio/wakelock, root) beim ersten Request halten, nach TPU_WAKELOCK_IDLE_MS (Default 2000)
   ohne Requests freigeben, bei SIGTERM/SIGINT freigeben. TPU_WAKELOCK=0 schaltet ab. ---- */
static const char* g_wl_path="/sys/kernel/debug/edgetpu/rio/wakelock";
static int g_wl_enable=1, g_wl_held=0; static long g_wl_idle_ms=2000; static double g_wl_last=0;
static double wl_now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static void wl_write(const char* v){ int fd=open(g_wl_path,O_WRONLY); if(fd<0){ static int warned=0; if(!warned){warned=1; fprintf(stderr,"[tpud] Wakelock %s nicht schreibbar (%s) - TPU schlaeft zwischen Requests (+~5ms/Inferenz)\n",g_wl_path,strerror(errno));} g_wl_enable=0; return; } if(write(fd,v,strlen(v))<0){} close(fd); }
static void wl_touch(void){ g_wl_last=wl_now_ms(); if(g_wl_enable && !g_wl_held){ wl_write("1"); g_wl_held=1; } }   /* unter g_tpu aufrufen */
static void wl_release(void){ if(g_wl_held){ wl_write("0"); g_wl_held=0; } }
static void* wl_idle_thread(void* a){ (void)a; for(;;){ usleep(200000); if(!g_wl_held) continue; TPU_LOCK(); if(g_wl_held && wl_now_ms()-g_wl_last>g_wl_idle_ms) wl_release(); TPU_UNLOCK(); } return 0; }
static void wl_on_signal(int sig){ (void)sig; wl_release(); _exit(0); }

static void* mkbuf(long sz, void** host){
  long descr[4]={1,0,0,0}; void* buf=0; allocBuf(g_factory,g_opt,descr,sz,&buf);
  if(!buf) return 0; void* hp=0; mapHost(buf,&hp); if(host)*host=hp; return buf;
}

/* eine Inferenz fuer Modell mid. oh=host-Sicht des Output-Buffers (out_size). Liefert exec_us in *us. */
/* Multi-I/O: inBufs[n_in], outBufs[n_out]; Sentinel/Poll auf Ausgabe 0 (oh = Host-Sicht von outBufs[0]). */
static int tpu_infer_multi(int mid, void** inBufs, int n_in, void** outBufs, int n_out, unsigned char* oh, double* us_out){
  void* g=g_models[mid].graph; long osz=g_models[mid].out_sizes[0];
  void* req=0; wl_touch();
  if(g_wait_us>0){
    unsigned char sventi=++g_sctr; if(!sventi) sventi=++g_sctr;
    memset(oh, sventi, osz);   /* Sentinel in Output-Host-Sicht */
    for(int k=0;k<n_in;k++){ mapDev(inBufs[k]); if(flushCache) flushCache(inBufs[k]); } for(int k=0;k<n_out;k++) mapDev(outBufs[k]);
    createReq(g_vdev,g,&req); if(!req) return -1;
    for(int k=0;k<n_in;k++) addIn(req,k,inBufs[k]); for(int k=0;k<n_out;k++) addOut(req,k,outBufs[k]);
    struct timespec ws,we; clock_gettime(CLOCK_MONOTONIC,&ws);
    submit(g_vdev,req); usleep(g_wait_us); if(invCache) for(int k=0;k<n_out;k++) invCache(outBufs[k]);
    clock_gettime(CLOCK_MONOTONIC,&we);
    if(getenv("TPU_BREAKDOWN")){   /* wie viel der Ausgabe hat die TPU nach g_wait_us schon geschrieben? */
      long same=0; for(long p=0;p<osz;p++) if(oh[p]==sventi) same++;
      double subw=(we.tv_sec-ws.tv_sec)*1e6+(we.tv_nsec-ws.tv_nsec)/1e3;
      static long wc=0; static double asame=0,asub=0; wc++; asame+=(double)same*100.0/osz; asub+=subw;
      if(wc%20==0) fprintf(stderr,"[WAIT M%d wait=%ldus n=%ld] Rest-Sentinel=%.1f%%  submit+wait=%.0fus\n",mid,g_wait_us,wc,asame/wc,asub/wc);
    }
    if(reqFree) reqFree(req); if(us_out)*us_out=g_wait_us; return 0;
  }
  /* FENCE-Pfad (Default, TPU_FENCE=0 schaltet auf den alten Sentinel-Poll zurueck): SubmitFenced + Fence_IsCompleted.
     Der Sentinel/Stabilitaets-Poll kostet 6-8 ms Fixzeit pro Inferenz (usleep-Schritte + Stabilitaets-Nachlauf) und ist
     der Grund, warum selbst ein leerer rmsnorm-Graph 8 ms "brauchte" (Befund 18.8.). Fence: exakte Completion. */
  if(g_use_fence){
    struct timespec fs0; clock_gettime(CLOCK_MONOTONIC,&fs0);
    for(int k=0;k<n_in;k++){ mapDev(inBufs[k]); if(flushCache) flushCache(inBufs[k]); } for(int k=0;k<n_out;k++) mapDev(outBufs[k]);
    createReq(g_vdev,g,&req); if(!req) return -1;
    for(int k=0;k<n_in;k++) addIn(req,k,inBufs[k]); for(int k=0;k<n_out;k++) addOut(req,k,outBufs[k]);
    struct timespec f0,f1; clock_gettime(CLOCK_MONOTONIC,&f0);
    if(getenv("TPU_BREAKDOWN")){ static long bc=0; static double asu=0; double d=(f0.tv_sec-fs0.tv_sec)*1e6+(f0.tv_nsec-fs0.tv_nsec)/1e3; bc++; asu+=d;
      if(bc%20==0) fprintf(stderr,"[BRK-F M%d n=%ld] setup(mapDev/flush/createReq/addIO)=%.0fus\n",mid,bc,asu/bc); }
    void* fence=0; submitFenced(g_vdev,req,&fence);
    int done=0; long spins=0;
    if(fence){ for(;;){ if(fenceDone(fence)){done=1;break;} if(++spins>200000){ struct timespec tn; clock_gettime(CLOCK_MONOTONIC,&tn);
          if((tn.tv_sec-f0.tv_sec)*1e6+(tn.tv_nsec-f0.tv_nsec)/1e3>g_max_wait_us) break; if(g_fence_yield) sched_yield(); } } }
    else done=1;
    clock_gettime(CLOCK_MONOTONIC,&f1);
    if(invCache) for(int k=0;k<n_out;k++) invCache(outBufs[k]);
    if(fence&&fenceFree) fenceFree(fence); if(reqFree) reqFree(req);
    double us=(f1.tv_sec-f0.tv_sec)*1e6+(f1.tv_nsec-f0.tv_nsec)/1e3; if(us_out)*us_out=us;
    static long fc=0; static double fs=0; fc++; fs+=us;
    if(getenv("TPU_BREAKDOWN")){ static long wc=0; static double aw=0,mx=0,mn=1e9; wc++; aw+=us; if(us>mx)mx=us; if(us<mn)mn=us;
      if(wc%20==0) fprintf(stderr,"[BRK-F M%d n=%ld] submit->fence avg=%.0fus min=%.0f max=%.0f spins=%ld\n",mid,wc,aw/wc,mn,mx,spins); }
    if(fc<=3 || fc%500==0) fprintf(stderr,"[tpud] Fence-Inferenz #%ld (M%d): %.2f ms (Schnitt %.2f ms)%s\n",fc,mid,us/1000.0,fs/fc/1000.0,done?"":" TIMEOUT");
    return done?0:-2;
  }
  void* outBuf=outBufs[0];
  unsigned char sv=++g_sctr; if(!sv) sv=++g_sctr;
  memset(oh, sv, osz);
  struct timespec tsu0,tsu1,tsub; clock_gettime(CLOCK_MONOTONIC,&tsu0);
  for(int k=0;k<n_in;k++) mapDev(inBufs[k]); for(int k=0;k<n_out;k++) mapDev(outBufs[k]);
  createReq(g_vdev,g,&req); if(!req) return -1;
  for(int k=0;k<n_in;k++) addIn(req,k,inBufs[k]); for(int k=0;k<n_out;k++) addOut(req,k,outBufs[k]);
  clock_gettime(CLOCK_MONOTONIC,&tsu1);
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  submit(g_vdev,req);
  clock_gettime(CLOCK_MONOTONIC,&tsub);
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
  if(us_out)*us_out=us;
  static long g_cnt=0; static double g_sum=0; g_cnt++; g_sum+=us;
  if(g_cnt<=3 || g_cnt%500==0) fprintf(stderr,"[tpud] reine Inferenz #%ld (M%d): %.2f ms (Schnitt %.2f ms)\n",g_cnt,mid,us/1000.0,g_sum/g_cnt/1000.0);
  if(getenv("TPU_BREAKDOWN")){
    double d_setup=(tsu1.tv_sec-tsu0.tv_sec)*1e6+(tsu1.tv_nsec-tsu0.tv_nsec)/1e3;
    double d_submit=(tsub.tv_sec-t0.tv_sec)*1e6+(tsub.tv_nsec-t0.tv_nsec)/1e3;
    double d_wait=(t1.tv_sec-tsub.tv_sec)*1e6+(t1.tv_nsec-tsub.tv_nsec)/1e3;
    static long bc=0; static double as=0,asu=0,aw=0; bc++; as+=d_setup; asu+=d_submit; aw+=d_wait;
    if(bc%20==0) fprintf(stderr,"[BRK M%d n=%ld] setup=%.0fus  submit=%.0fus  wait=%.0fus  (total %.0fus)\n",mid,bc,as/bc,asu/bc,aw/bc,(as+asu+aw)/bc);
  }
  return 0;
}
static int tpu_infer(int mid, void* inBuf, void* outBuf, unsigned char* oh, double* us_out){
  void* i1[1]={inBuf}; void* o1[1]={outBuf}; return tpu_infer_multi(mid,i1,1,o1,1,oh,us_out);
}

/* K=2-Fence-Paar: beide Requests sofort submitFenced (2 in-flight wie pipe_bench),
   dann beide Fences einsammeln. Nur im Fence-Modus rufen (g_use_fence). */
static int tpu_infer_pair(int mid, void* iA, void* oA, void* iB, void* oB, double* us_out){
  void* g=g_models[mid].graph; wl_touch();
  void* req[2]={0,0}; void* fen[2]={0,0};
  void* ins[2]={iA,iB}; void* outs[2]={oA,oB};
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  for(int j=0;j<2;j++){
    mapDev(ins[j]); if(flushCache) flushCache(ins[j]); mapDev(outs[j]);
    createReq(g_vdev,g,&req[j]);
    if(!req[j]){ for(int k=0;k<j;k++){ if(fen[k]&&fenceFree) fenceFree(fen[k]); if(reqFree) reqFree(req[k]); } return -1; }
    addIn(req[j],0,ins[j]); addOut(req[j],0,outs[j]);
    submitFenced(g_vdev,req[j],&fen[j]);
  }
  int done=1;
  for(int j=0;j<2;j++){
    int dj=fen[j]?0:1; long spins=0;
    if(fen[j]) for(;;){ if(fenceDone(fen[j])){dj=1;break;} if(++spins>200000){ struct timespec tn; clock_gettime(CLOCK_MONOTONIC,&tn);
          if((tn.tv_sec-t0.tv_sec)*1e6+(tn.tv_nsec-t0.tv_nsec)/1e3>g_max_wait_us) break; if(g_fence_yield) sched_yield(); } }
    if(invCache) invCache(outs[j]);
    if(fen[j]&&fenceFree) fenceFree(fen[j]); if(reqFree) reqFree(req[j]);
    if(!dj) done=0;
  }
  clock_gettime(CLOCK_MONOTONIC,&t1);
  if(us_out)*us_out=(t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_nsec-t0.tv_nsec)/1e3;
  return done?0:-2;
}

/* Warmup: erster Submit nach Registrierung ist kalt (JIT). Serielle Laeufe bis Output stabil. */
static void warmup(int mid){
  Model* M=&g_models[mid]; long osz=M->out_sizes[0];
  void* ib[MAXIO]={0}; void* ob[MAXIO]={0}; void* ih[MAXIO]={0}; void* ohv[MAXIO]={0};
  for(int k=0;k<M->n_in;k++){ ib[k]=mkbuf(M->in_sizes[k],&ih[k]); if(ib[k]) memset(ih[k],0x5A,M->in_sizes[k]); }
  for(int k=0;k<M->n_out;k++) ob[k]=mkbuf(M->out_sizes[k],&ohv[k]);
  for(int k=0;k<M->n_in;k++) if(!ib[k]){ fprintf(stderr,"[tpud] Warmup %s: kein Buffer\n",M->name); return; }
  for(int k=0;k<M->n_out;k++) if(!ob[k]){ fprintf(stderr,"[tpud] Warmup %s: kein Buffer\n",M->name); return; }
  unsigned char* oh=ohv[0]; unsigned char* prev=malloc(osz);
  int stable=0, it=0;
  for(; it<12 && stable<2; it++){
    void* req=0; for(int k=0;k<M->n_in;k++){ mapDev(ib[k]); if(flushCache) flushCache(ib[k]); } for(int k=0;k<M->n_out;k++) mapDev(ob[k]);
    createReq(g_vdev,M->graph,&req);
    for(int k=0;k<M->n_in;k++) addIn(req,k,ib[k]); for(int k=0;k<M->n_out;k++) addOut(req,k,ob[k]);
    submit(g_vdev,req); usleep(200000); if(invCache) for(int k=0;k<M->n_out;k++) invCache(ob[k]);
    if(reqFree) reqFree(req);
    if(it>0 && prev && memcmp(prev,oh,osz)==0) stable++; else stable=0;
    if(prev) memcpy(prev,oh,osz);
  }
  free(prev); if(bufFree){ for(int k=0;k<M->n_in;k++) bufFree(ib[k]); for(int k=0;k<M->n_out;k++) bufFree(ob[k]); }
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
  g_models[mid].n_in=(int)numIn(graph); g_models[mid].n_out=(int)numOut(graph);
  if(g_models[mid].n_in>MAXIO) g_models[mid].n_in=MAXIO; if(g_models[mid].n_out>MAXIO) g_models[mid].n_out=MAXIO;
  for(int k=0;k<g_models[mid].n_in;k++){ void* it=0; getInTensor(graph,k,&it); g_models[mid].in_sizes[k]=tensorSize(it); }
  for(int k=0;k<g_models[mid].n_out;k++){ void* ot=0; getOutTensor(graph,k,&ot); g_models[mid].out_sizes[k]=tensorSize(ot); }
  g_models[mid].in_size=g_models[mid].in_sizes[0]; g_models[mid].out_size=g_models[mid].out_sizes[0];
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
  impFd=(void*(*)(void*,void*,int,int,long,long,void*))S("DarwinnApi2_BufferFactory_ImportBufferByFd");
  mapHost=(void*(*)(void*,void*))S("DarwinnApi2_Buffer_MapToHost");
  mapDev=(void*(*)(void*))S("DarwinnApi2_Buffer_MapToDevice");
  invCache=(void*(*)(void*))S("DarwinnApi2_Buffer_InvalidateCache");
  bufFree=(void*(*)(void*))S("DarwinnApi2_Buffer_Free");
  createReq=(void*(*)(void*,void*,void*))S("DarwinnApi2_VirtualDevice_CreateInferenceRequest");
  addIn=(void*(*)(void*,int,void*))S("DarwinnApi2_InferenceRequest_AddInput");
  addOut=(void*(*)(void*,int,void*))S("DarwinnApi2_InferenceRequest_AddOutput");
  submit=(void*(*)(void*,void*))S("DarwinnApi2_VirtualDevice_Submit");
  reqFree=(void*(*)(void*))S("DarwinnApi2_InferenceRequest_Free");
  submitFenced=(void*(*)(void*,void*,void**))S("DarwinnApi2_VirtualDevice_SubmitFenced");
  fenceDone=(int(*)(void*))S("DarwinnApi2_Fence_IsCompleted");
  fenceDupFd=(void*(*)(void*,int*))S("DarwinnApi2_Fence_GetDupFd");
  fenceFree=(void*(*)(void*))S("DarwinnApi2_Fence_Free");
  flushCache=(void*(*)(void*))S("DarwinnApi2_Buffer_FlushCache");
  isCoherent=(int(*)(void*))S("DarwinnApi2_Buffer_IsCoherent");
  isMapHost=(int(*)(void*))S("DarwinnApi2_Buffer_IsMappedToHost");
  isMapDev=(int(*)(void*))S("DarwinnApi2_Buffer_IsMappedToDevice");
  copyFrom=(void*(*)(void*,void*,long,long))S("DarwinnApi2_Buffer_CopyFrom");
  copyTo=(void*(*)(void*,const void*,long,long))S("DarwinnApi2_Buffer_CopyTo");
  if(!impFd) fprintf(stderr,"[tpud] WARNUNG: ImportBufferByFd fehlt - kein Zero-Copy\n");

  int s0,s2,s3; unsigned char s1,s5; void* s4=0; getSpec(&s0,&s1,&s2,&s3,&s4,&s5);
  void* edgeDev=0; createVD((long)s0,(long)s1,(long)s2,(long)s3,s4,(long)s5,&edgeDev);
  g_vdev=getVD(edgeDev); g_factory=getBF(g_vdev);
  void* arr=0; int cnt=0; getSupported(g_factory,&arr,&cnt);
  int oi=getenv("TPU_OPT_IDX")?atoi(getenv("TPU_OPT_IDX")):0; if(oi<0||oi>=cnt)oi=0;
  g_opt=((void**)arr)[oi];
  fprintf(stderr,"[tpud] %d Alloc-Option(en), nutze #%d\n",cnt,oi);
  return 0;
}

static int read_full(int fd,void*b,long n){char*p=b;while(n>0){ssize_t r=read(fd,p,n);if(r<=0)return -1;p+=r;n-=r;}return 0;}
static int write_full(int fd,const void*b,long n){const char*p=b;while(n>0){ssize_t w=write(fd,p,n);if(w<=0)return -1;p+=w;n-=w;}return 0;}
static int ru32(int fd,uint32_t*v){return read_full(fd,v,4);}

/* n Bytes + evtl. SCM_RIGHTS-fds lesen (fds haengen am ersten Byte der Nachricht) */
static int recv_with_fds(int c, void* buf, long n, int* fds, int* nfd){
  char cbuf[CMSG_SPACE(sizeof(int)*ZC_MAXBUF)];
  struct iovec iov={.iov_base=buf,.iov_len=n};
  struct msghdr msg={.msg_iov=&iov,.msg_iovlen=1,.msg_control=cbuf,.msg_controllen=sizeof cbuf};
  *nfd=0;
  ssize_t r=recvmsg(c,&msg,MSG_WAITALL);
  struct cmsghdr* cm=CMSG_FIRSTHDR(&msg);
  if(cm&&cm->cmsg_type==SCM_RIGHTS){ int k=(int)((cm->cmsg_len-CMSG_LEN(0))/sizeof(int));
    /* fds sind bereits installiert — bei Kurzlesung ODER Überzahl trotzdem schließen (kein Leak) */
    if(r!=n||k>ZC_MAXBUF){ for(int i=0;i<k;i++) close(((int*)CMSG_DATA(cm))[i]); return -1; }
    memcpy(fds,CMSG_DATA(cm),k*sizeof(int)); *nfd=k; }
  else if(r!=n) return -1;
  return 0;
}

static long g_nreq=0;

/* ---- (A) inline TPD2 (magic schon gelesen) ---- */
static void serve_inline(int c){
  for(;;){
    uint32_t mid=0,insz=0;
    if(ru32(c,&mid)||ru32(c,&insz)) break;
    if(mid>=(uint32_t)g_nmodels || (long)insz!=g_models[mid].in_size){
      uint32_t st=1, z=0; write_full(c,&st,4); write_full(c,&z,4); break;
    }
    long isz=g_models[mid].in_size, osz=g_models[mid].out_size;
    /* Eingabe erst in einen Zwischenpuffer lesen (ohne Lock), dann TPU-Arbeit unter dem Mutex */
    unsigned char* tmp=malloc(isz); if(!tmp) break;
    if(read_full(c,tmp,isz)!=0){ free(tmp); break; }
    TPU_LOCK();
    void *ih=0,*oh=0;
    void* inBuf=mkbuf(isz,&ih); if(!inBuf){ TPU_UNLOCK(); free(tmp); break; }
    memcpy(ih,tmp,isz); free(tmp);
    void* outBuf=mkbuf(osz,&oh); if(!outBuf){ if(bufFree) bufFree(inBuf); TPU_UNLOCK(); break; }
    int ok = tpu_infer((int)mid,inBuf,outBuf,(unsigned char*)oh,0)==0;
    if(getenv("TPU_SERVE_DBG")){ long nz=0; for(long q=0;q<osz;q++) if(((unsigned char*)oh)[q]) nz++;
      long inz=0; for(long q=0;q<isz;q++) if(((unsigned char*)ih)[q]) inz++;
      fprintf(stderr,"[tpud] serve M%u: ok=%d in_nonzero=%ld/%ld out_nonzero=%ld/%ld\n",mid,ok,inz,isz,nz,osz); }
    unsigned char* res=0; if(ok){ res=malloc(osz); if(res) memcpy(res,oh,osz); else ok=0; }
    if(bufFree){ bufFree(inBuf); bufFree(outBuf); }
    TPU_UNLOCK();
    uint32_t st=ok?0:2, osz32=(uint32_t)osz;
    int wr = (write_full(c,&st,4)||write_full(c,&osz32,4)|| (ok?write_full(c,res,osz):0));
    free(res);
    g_nreq++; if((g_nreq%1000)==1) fprintf(stderr,"[tpud] req #%ld (Modell %u)\n",g_nreq,mid);
    if(wr!=0) break;
    /* naechster Request: Magic pruefen (Session bleibt inline) */
    uint32_t magic=0; if(ru32(c,&magic)||magic!=MAGIC) break;
  }
}

/* ---- (B) Zero-Copy-Session TPZ2 ---- */
typedef struct { int used; void* buf; void* map; long size; int fd; } ZBuf;
static void zc_release(ZBuf* z){
  if(!z->used) return;
  if(bufFree&&z->buf) bufFree(z->buf);
  if(z->map) munmap(z->map,z->size);
  if(z->fd>=0) close(z->fd);
  memset(z,0,sizeof *z); z->fd=-1;
}
static void serve_zc(int c, uint32_t* hdr, int* fds, int nfd){
  ZBuf h[ZC_MAXH]; memset(h,0,sizeof h); for(int i=0;i<ZC_MAXH;i++) h[i].fd=-1;
  long ninf=0;
  for(;;){
    uint32_t cmd=hdr[1], status=1;
    if(cmd==1){                                     /* IMPORT */
      uint32_t nbuf=hdr[2]; uint32_t sz[ZC_MAXBUF], hd[ZC_MAXBUF];
      if(!impFd||nbuf==0||nbuf>ZC_MAXBUF||(int)nbuf!=nfd||read_full(c,sz,nbuf*4)){ write_full(c,&status,4); goto next; }
      int ok=1;
      TPU_LOCK();
      for(uint32_t i=0;i<nbuf;i++){
        int slot=-1; for(int k=0;k<ZC_MAXH;k++) if(!h[k].used){slot=k;break;}
        if(slot<0){ ok=0; break; }
        if(sz[i]==0||sz[i]>(512u*1024*1024)){ ok=0; fprintf(stderr,"[tpud] Import-Groesse %u ausserhalb 1..512MiB\n",sz[i]); break; }
        void* b=0; int dfd=dup(fds[i]);                     /* die Lib bekommt ein dup; unser fd bleibt fuer mmap/close */
        impFd(g_factory,g_opt,0,dfd,(long)sz[i],0,&b);
        if(!b){ close(dfd); ok=0; fprintf(stderr,"[tpud] ImportBufferByFd FAIL (size %u)\n",sz[i]); break; }
        void* m=mmap(0,sz[i],PROT_READ|PROT_WRITE,MAP_SHARED,fds[i],0);
        if(m==MAP_FAILED){ if(bufFree) bufFree(b); ok=0; fprintf(stderr,"[tpud] mmap(dmabuf) FAIL\n"); break; }
        h[slot].used=1; h[slot].buf=b; h[slot].map=m; h[slot].size=sz[i]; h[slot].fd=fds[i]; fds[i]=-1;   /* fd uebernommen */
        hd[i]=(uint32_t)slot;
      }
      TPU_UNLOCK();
      if(ok){ status=0; write_full(c,&status,4); write_full(c,hd,nbuf*4); } else write_full(c,&status,4);
    } else if(cmd==2){                              /* INFER model in_h out_h */
      uint32_t mid=hdr[2], ih=hdr[3], oh=hdr[4]; uint32_t us32=0;
      if(mid<(uint32_t)g_nmodels && ih<ZC_MAXH && oh<ZC_MAXH && h[ih].used && h[oh].used && ih!=oh
         && h[ih].size>=g_models[mid].in_size && h[oh].size>=g_models[mid].out_size){
        double us=0;
        TPU_LOCK();
        const char* bev=getenv("TPU_ZC_BOUNCE");
        if(bev){
          /* Befund 21.8.: LUT/Softmax-Packages schreiben in IMPORTIERTE dmabufs Nullen,
           * in interne Lib-Puffer korrekt -> intern rechnen und umkopieren. Puffer je Modell gecacht.
           * Modus 2 = Bounce NUR Output (Input direkt aus dem Import; Nullen-Befund war output-seitig). */
          int bm=atoi(bev);
          static void* c_ib[256]; static void* c_ob[256]; static void* c_ih[256]; static void* c_oh[256];
          long is_=g_models[mid].in_size, os_=g_models[mid].out_size;
          if(bm!=2&&!c_ib[mid]) c_ib[mid]=mkbuf(is_,&c_ih[mid]);
          if(!c_ob[mid]) c_ob[mid]=mkbuf(os_,&c_oh[mid]);
          void* ib=(bm==2)?h[ih].buf:c_ib[mid];
          if(ib&&c_ob[mid]){
            if(bm!=2) memcpy(c_ih[mid],h[ih].map,is_);
            if(tpu_infer((int)mid,ib,c_ob[mid],(unsigned char*)c_oh[mid],&us)==0){ memcpy(h[oh].map,c_oh[mid],os_); status=0; us32=(uint32_t)us; }
            else status=2;
          } else status=2;
        } else if(tpu_infer((int)mid,h[ih].buf,h[oh].buf,(unsigned char*)h[oh].map,&us)==0){ status=0; us32=(uint32_t)us; }
        else status=2;
        TPU_UNLOCK();
      }
      ninf++; g_nreq++;
      if(ninf<=3||ninf%1000==0) fprintf(stderr,"[tpud] zc infer #%ld (M%u, in_h=%u out_h=%u) status=%u %.2f ms\n",ninf,mid,ih,oh,status,us32/1000.0);
      write_full(c,&status,4); write_full(c,&us32,4);
    } else if(cmd==7){                              /* INFER2 model; dann 4 Handles a_ih a_oh b_ih b_oh (K=2-Paar) */
      uint32_t mid=hdr[2]; uint32_t hd4[4]; uint32_t us32=0;
      if(read_full(c,hd4,16)){ write_full(c,&status,4); write_full(c,&us32,4); goto next; }
      uint32_t ai=hd4[0],ao=hd4[1],bi=hd4[2],bo=hd4[3];
      int ok = mid<(uint32_t)g_nmodels && ai<ZC_MAXH&&ao<ZC_MAXH&&bi<ZC_MAXH&&bo<ZC_MAXH
        && h[ai].used&&h[ao].used&&h[bi].used&&h[bo].used && ai!=ao&&bi!=bo
        && h[ai].size>=g_models[mid].in_size&&h[bi].size>=g_models[mid].in_size
        && h[ao].size>=g_models[mid].out_size&&h[bo].size>=g_models[mid].out_size;
      if(ok){
        double us=0; int rcp=-1;
        TPU_LOCK();
        const char* bev=getenv("TPU_ZC_BOUNCE");
        if(bev){
          int bm=atoi(bev);                          /* 2 = Bounce nur Output */
          static void* p_ib[256][2]; static void* p_ob[256][2]; static void* p_ih[256][2]; static void* p_oh[256][2];
          long is_=g_models[mid].in_size, os_=g_models[mid].out_size;
          for(int j=0;j<2;j++){
            if(bm!=2&&!p_ib[mid][j]) p_ib[mid][j]=mkbuf(is_,&p_ih[mid][j]);
            if(!p_ob[mid][j]) p_ob[mid][j]=mkbuf(os_,&p_oh[mid][j]);
          }
          void* iA=(bm==2)?h[ai].buf:p_ib[mid][0];
          void* iB=(bm==2)?h[bi].buf:p_ib[mid][1];
          if(iA&&iB&&p_ob[mid][0]&&p_ob[mid][1]){
            if(bm!=2){ memcpy(p_ih[mid][0],h[ai].map,is_); memcpy(p_ih[mid][1],h[bi].map,is_); }
            if(g_use_fence) rcp=tpu_infer_pair(mid,iA,p_ob[mid][0],iB,p_ob[mid][1],&us);
            else { double u1=0,u2=0; int r1=tpu_infer(mid,iA,p_ob[mid][0],(unsigned char*)p_oh[mid][0],&u1);
                   int r2=tpu_infer(mid,iB,p_ob[mid][1],(unsigned char*)p_oh[mid][1],&u2); us=u1+u2; rcp=(r1||r2)?-1:0; }
            if(rcp==0){ memcpy(h[ao].map,p_oh[mid][0],os_); memcpy(h[bo].map,p_oh[mid][1],os_); }
          }
        } else {
          if(g_use_fence) rcp=tpu_infer_pair(mid,h[ai].buf,h[ao].buf,h[bi].buf,h[bo].buf,&us);
          else { double u1=0,u2=0; int r1=tpu_infer(mid,h[ai].buf,h[ao].buf,(unsigned char*)h[ao].map,&u1);
                 int r2=tpu_infer(mid,h[bi].buf,h[bo].buf,(unsigned char*)h[bo].map,&u2); us=u1+u2; rcp=(r1||r2)?-1:0; }
        }
        TPU_UNLOCK();
        if(rcp==0){ status=0; us32=(uint32_t)us; } else status=2;
      }
      ninf+=2; g_nreq+=2;
      if(ninf<=6||ninf%1000<2) fprintf(stderr,"[tpud] zc infer2 #%ld (M%u) status=%u %.2f ms/Paar\n",ninf,mid,status,us32/1000.0);
      write_full(c,&status,4); write_full(c,&us32,4);
    } else if(cmd==3){                              /* RELEASE */
      uint32_t nbuf=hdr[2]; uint32_t hd[ZC_MAXBUF];
      if(nbuf==0||nbuf>ZC_MAXBUF||read_full(c,hd,nbuf*4)){ write_full(c,&status,4); goto next; }
      TPU_LOCK(); for(uint32_t i=0;i<nbuf;i++) if(hd[i]<ZC_MAXH) zc_release(&h[hd[i]]); TPU_UNLOCK();
      status=0; write_full(c,&status,4);
    } else if(cmd==4){                              /* INFO */
      uint32_t mid=hdr[2]; uint32_t r[4]={1,0,0,(uint32_t)g_nmodels};
      if(mid<(uint32_t)g_nmodels){ r[0]=0; r[1]=(uint32_t)g_models[mid].in_size; r[2]=(uint32_t)g_models[mid].out_size; }
      write_full(c,r,16);
    } else if(cmd==5){                              /* INFER_MULTI model n_in n_out; dann n_in+n_out Handles */
      uint32_t mid=hdr[2], nin=hdr[3], nout=hdr[4]; uint32_t hd[2*MAXIO]; uint32_t us32=0;
      if(nin==0||nout==0||nin>MAXIO||nout>MAXIO||read_full(c,hd,(nin+nout)*4)){ write_full(c,&status,4); write_full(c,&us32,4); goto next; }
      int ok=(mid<(uint32_t)g_nmodels && (int)nin==g_models[mid].n_in && (int)nout==g_models[mid].n_out);
      void* ib[MAXIO]; void* ob[MAXIO];
      for(uint32_t k=0;ok&&k<nin;k++){ uint32_t x=hd[k]; if(x>=ZC_MAXH||!h[x].used||h[x].size<g_models[mid].in_sizes[k]) ok=0; else ib[k]=h[x].buf; }
      for(uint32_t k=0;ok&&k<nout;k++){ uint32_t x=hd[nin+k]; if(x>=ZC_MAXH||!h[x].used||h[x].size<g_models[mid].out_sizes[k]) ok=0; else ob[k]=h[x].buf; }
      if(ok){ double us=0; TPU_LOCK();
        if(tpu_infer_multi((int)mid,ib,(int)nin,ob,(int)nout,(unsigned char*)h[hd[nin]].map,&us)==0){ status=0; us32=(uint32_t)us; } else status=2;
        TPU_UNLOCK(); }
      ninf++; g_nreq++;
      if(ninf<=3||ninf%1000==0) fprintf(stderr,"[tpud] zc infer-multi #%ld (M%u, %u in/%u out) status=%u %.2f ms\n",ninf,mid,nin,nout,status,us32/1000.0);
      write_full(c,&status,4); write_full(c,&us32,4);
    } else if(cmd==6){                              /* INFO2 model -> status,n_in,n_out,nmodels, dann in_sizes[], out_sizes[] */
      uint32_t mid=hdr[2]; uint32_t r[4]={1,0,0,(uint32_t)g_nmodels};
      if(mid<(uint32_t)g_nmodels){ r[0]=0; r[1]=(uint32_t)g_models[mid].n_in; r[2]=(uint32_t)g_models[mid].n_out; }
      write_full(c,r,16);
      if(r[0]==0){ uint32_t sz[2*MAXIO]; for(uint32_t k=0;k<r[1];k++) sz[k]=(uint32_t)g_models[mid].in_sizes[k]; for(uint32_t k=0;k<r[2];k++) sz[r[1]+k]=(uint32_t)g_models[mid].out_sizes[k]; write_full(c,sz,(r[1]+r[2])*4); }
    } else { write_full(c,&status,4); }
next:
    for(int i=0;i<nfd;i++) if(fds[i]>=0) close(fds[i]);
    nfd=0;
    if(recv_with_fds(c,hdr,24,fds,&nfd)) break;
    if(hdr[0]!=MAGIC_ZC) break;
  }
  for(int i=0;i<nfd;i++) if(fds[i]>=0) close(fds[i]);
  int live=0; TPU_LOCK(); for(int k=0;k<ZC_MAXH;k++) if(h[k].used){ live++; zc_release(&h[k]); } TPU_UNLOCK();
  fprintf(stderr,"[tpud] zc session ende: %ld Inferenzen, %d Handles freigegeben\n",ninf,live);
}

static void handle_conn(int c){
  uint32_t magic=0; int fds[ZC_MAXBUF]; int nfd=0;
  /* erste 4 Bytes MIT Ancillary lesen: bei TPZ2 haengen die IMPORT-fds am ersten Byte */
  if(recv_with_fds(c,&magic,4,fds,&nfd)) return;
  if(magic==MAGIC){ for(int i=0;i<nfd;i++) close(fds[i]); serve_inline(c); return; }
  if(magic==MAGIC_ZC){
    uint32_t hdr[6]; hdr[0]=magic;
    if(read_full(c,&hdr[1],20)){ for(int i=0;i<nfd;i++) close(fds[i]); return; }
    serve_zc(c,hdr,fds,nfd); return;
  }
  for(int i=0;i<nfd;i++) close(fds[i]);
}

static void* conn_thread(void* a){ int c=*(int*)a; free(a); handle_conn(c); close(c); return 0; }

/* ---- Fence-Pipeline-Benchmark: K Inferenzen gleichzeitig in-flight, async SubmitFenced ----
 * TPU_PIPE=K (in-flight), TPU_PIPE_N=N (Gesamt-Inferenzen), TPU_PIPE_FD=1 (poll dup-fd statt spin).
 * Korrektheit: gleicher Input -> jede Ausgabe MUSS der seriellen Referenz gleichen. */
static double nowms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static int cmp_float(const void*a,const void*b){ float x=*(const float*)a,y=*(const float*)b; return x<y?-1:x>y?1:0; }
static void pipe_bench(int mid){
  Model* M=&g_models[mid];
  if(M->n_in!=1||M->n_out!=1){ fprintf(stderr,"[pipe] nur 1-in/1-out Modelle\n"); return; }
  if(!submitFenced||!fenceDone||!fenceFree){ fprintf(stderr,"[pipe] Fence-Symbole fehlen (SF=%p done=%p free=%p)\n",(void*)submitFenced,(void*)fenceDone,(void*)fenceFree); return; }
  int K=atoi(getenv("TPU_PIPE")); if(K<1)K=1; if(K>32)K=32;
  int N=getenv("TPU_PIPE_N")?atoi(getenv("TPU_PIPE_N")):200; if(N<K)N=K;
  int use_fd=getenv("TPU_PIPE_FD")?1:0;
  long isz=M->in_sizes[0], osz=M->out_sizes[0];

  void* inh=0; void* inb=mkbuf(isz,&inh); if(!inb){fprintf(stderr,"[pipe] in-buf fail\n");return;} mapDev(inb);
  /* Input: aus Datei (echter Tensor) ODER Fuellmuster */
  const char* infile=getenv("TPU_PIPE_IN");
  if(infile){ FILE* f=fopen(infile,"rb"); if(f){ long r=fread(inh,1,isz,f); fclose(f); fprintf(stderr,"[pipe] Input aus %s: %ld/%ld B\n",infile,r,isz);} else { fprintf(stderr,"[pipe] Input-Datei fehlt\n"); memset(inh,0x5A,isz);} }
  else memset(inh,0x5A,isz);
  /* numpy-Referenz (f32) aus Datei -> Genauigkeit statt nur Selbstkonsistenz */
  const char* reffile=getenv("TPU_PIPE_REF");
  float* fref=0; if(reffile){ FILE* f=fopen(reffile,"rb"); if(f){ fref=malloc((size_t)osz*4+16); long r=fread(fref,1,(size_t)osz*4,f); fclose(f); fprintf(stderr,"[pipe] numpy-Ref aus %s: %ld B (f32)\n",reffile,r);} }
  /* Referenz ueber den getrusteten seriellen Poll-Pfad */
  void* refh=0; void* refb=mkbuf(osz,&refh);
  { double us=0; tpu_infer(mid,inb,refb,(unsigned char*)refh,&us); fprintf(stderr,"[pipe] Referenz seriell: %.2f ms\n",us/1000.0); }
  unsigned char* ref=malloc(osz); memcpy(ref,refh,osz);
  if(getenv("TPU_DIAG")){
    fprintf(stderr,"[diag] IN:  coherent=%d mapHost=%d mapDev=%d  isz=%ld\n",
      isCoherent?isCoherent(inb):-9, isMapHost?isMapHost(inb):-9, isMapDev?isMapDev(inb):-9, isz);
    fprintf(stderr,"[diag] OUT: coherent=%d mapHost=%d mapDev=%d  osz=%ld  graph_out0=%ld n_out=%d\n",
      isCoherent?isCoherent(refb):-9, isMapHost?isMapHost(refb):-9, isMapDev?isMapDev(refb):-9, osz, g_models[mid].out_sizes[0], g_models[mid].n_out);
    if(copyFrom){ unsigned char* cb=malloc(osz); memset(cb,0xEE,osz); copyFrom(refb,cb,osz,0);
      int d=memcmp(cb,refh,osz);
      fprintf(stderr,"[diag] OUT CopyFrom vs mapHost: %s\n", d?"UNTERSCHIEDLICH -> mapHost stale!":"gleich");
      memcpy(ref,cb,osz); free(cb); }   /* Genauigkeit unten auf kohaerenter Ruecklesung */
  }
  { /* TPU_PIPE_SAVE=<datei>: seriellen TPU-Output roh wegschreiben (PC-seitige Auswertung) */
    const char* sf=getenv("TPU_PIPE_SAVE");
    if(sf){ FILE* f=fopen(sf,"wb"); if(f){ fwrite(ref,1,osz,f); fclose(f);
      fprintf(stderr,"[pipe] Output gespeichert: %s (%ld B)\n",sf,osz); } else fprintf(stderr,"[pipe] SAVE fopen FAIL %s\n",sf); }
  }
  if(fref){ /* seriellen TPU-Output gegen numpy vergleichen; int8/int16-Dequant via env */
    int odt=getenv("TPU_PIPE_ODT")?atoi(getenv("TPU_PIPE_ODT")):0;  /* 0=f32 1=int8 2=int16 */
    double osc=getenv("TPU_PIPE_OSCALE")?atof(getenv("TPU_PIPE_OSCALE")):1.0;
    int ozp=getenv("TPU_PIPE_OZP")?atoi(getenv("TPU_PIPE_OZP")):0;
    long nf = odt==1?osz : odt==2?osz/2 : osz/4;
    double dab=0,daa=0,dbb=0,sref=0,mae=0; float mx=0; float pv[4];
    for(long i=0;i<nf;i++){
      float v = odt==1 ? (float)(((signed char*)ref)[i]-ozp)*osc
              : odt==2 ? (float)(((short*)ref)[i]-ozp)*osc
              : ((float*)ref)[i];
      float r=fref[i]; double e=fabs((double)v-r);
      mae+=e; sref+=fabs(r); if(e>mx)mx=e; dab+=(double)v*r; daa+=(double)v*v; dbb+=(double)r*r;
      if(i<4)pv[i]=v;
    }
    double rel_l2=sqrt((daa-2*dab+dbb))/(sqrt(dbb)+1e-12);  /* ||a-b||/||b|| */
    double cosv=dab/(sqrt(daa)*sqrt(dbb)+1e-12);
    fprintf(stderr,"[pipe] GENAUIGKEIT TPU-seriell vs numpy (odt=%d osc=%.4g ozp=%d): rel_l2=%.3f%%  cos=%.4f  max|e|=%.5f  rel(L1)=%.2f%%\n",
      odt,osc,ozp,100.0*rel_l2,cosv,mx,100.0*mae/(sref+1e-9));
    if(getenv("TPU_DIAG")){ /* degeneriert vs permutiert vs falsch? */
      float* ta=malloc(nf*4); float* fa=malloc(nf*4);
      for(long i=0;i<nf;i++){ ta[i]= odt==1?(float)(((signed char*)ref)[i]-ozp)*osc : odt==2?(float)(((short*)ref)[i]-ozp)*osc : ((float*)ref)[i]; fa[i]=fref[i]; }
      double tmin=1e30,tmax=-1e30,tsum=0,tsq=0, rmin=1e30,rmax=-1e30,rsum=0,rsq=0;
      for(long i=0;i<nf;i++){ float t=ta[i],r=fa[i]; if(t<tmin)tmin=t; if(t>tmax)tmax=t; tsum+=t; tsq+=(double)t*t; if(r<rmin)rmin=r; if(r>rmax)rmax=r; rsum+=r; rsq+=(double)r*r; }
      double tstd=sqrt(tsq/nf-(tsum/nf)*(tsum/nf)), rstd=sqrt(rsq/nf-(rsum/nf)*(rsum/nf));
      /* sortieren fuer Permutationstest */
      qsort(ta,nf,4,cmp_float); qsort(fa,nf,4,cmp_float);
      double sdab=0,sdaa=0,sdbb=0; for(long i=0;i<nf;i++){ sdab+=(double)ta[i]*fa[i]; sdaa+=(double)ta[i]*ta[i]; sdbb+=(double)fa[i]*fa[i]; }
      double sorted_rel=sqrt(sdaa-2*sdab+sdbb)/(sqrt(sdbb)+1e-12);
      fprintf(stderr,"[diag] TPU-Verteilung: min=%.4f max=%.4f mean=%.4f std=%.4f\n",tmin,tmax,tsum/nf,tstd);
      fprintf(stderr,"[diag] numpy-Verteilung: min=%.4f max=%.4f mean=%.4f std=%.4f\n",rmin,rmax,rsum/nf,rstd);
      fprintf(stderr,"[diag] SORTIERT rel_l2=%.2f%%  -> %s\n",100.0*sorted_rel, sorted_rel<0.15?"PERMUTATION (Layout!)":"KEINE Permutation");
      free(ta); free(fa);
    }
    fprintf(stderr,"[pipe] TPU_OUT[0..3]=%.4f %.4f %.4f %.4f  numpyREF[0..3]=%.4f %.4f %.4f %.4f\n",
      pv[0],pv[1],pv[2],pv[3], fref[0],fref[1],fref[2],fref[3]);
  }

  void* ob[32]; unsigned char* oh[32]; void* fence[32];
  for(int s=0;s<K;s++){ ob[s]=mkbuf(osz,(void**)&oh[s]); if(!ob[s]){fprintf(stderr,"[pipe] out-buf %d fail\n",s);return;} mapDev(ob[s]); fence[s]=0; }

  #define PSUBMIT(s) do{ void* rq=0; mapDev(inb); if(flushCache) flushCache(inb); mapDev(ob[s]); createReq(g_vdev,M->graph,&rq); \
      addIn(rq,0,inb); addOut(rq,0,ob[s]); fence[s]=0; submitFenced(g_vdev,rq,&fence[s]); }while(0)
  #define PWAIT(s) do{ double w0=nowms(); \
      if(use_fd && fenceDupFd && fence[s]){ int fd=-1; fenceDupFd(fence[s],&fd); if(fd>=0){ struct pollfd pf={fd,POLLIN,0}; poll(&pf,1,500); close(fd);} } \
      else { for(;;){ int c=fence[s]?fenceDone(fence[s]):1; if(c) break; if(nowms()-w0>800){to++; break;} } } }while(0)

  int mism=0, to=0; double t0=nowms();
  for(int s=0;s<K;s++) PSUBMIT(s);
  for(int i=0;i<N;i++){
    int s=i%K;
    PWAIT(s);
    if(invCache) invCache(ob[s]);
    if(memcmp(oh[s],ref,osz)!=0) mism++;
    if(fence[s]&&fenceFree) fenceFree(fence[s]);
    PSUBMIT(s);
  }
  for(int s=0;s<K;s++){ PWAIT(s); if(invCache) invCache(ob[s]); if(memcmp(oh[s],ref,osz)!=0) mism++; if(fence[s]&&fenceFree) fenceFree(fence[s]); }
  if(getenv("TPU_FENCE_DUMP")){
    /* Dump des KORREKTEN Fence-Pipeline-Outputs (oh[0]), NICHT des seriellen Poll-Pfads */
    int odt=getenv("TPU_PIPE_ODT")?atoi(getenv("TPU_PIPE_ODT")):1;
    double osc=getenv("TPU_PIPE_OSCALE")?atof(getenv("TPU_PIPE_OSCALE")):1.0;
    int ozp=getenv("TPU_PIPE_OZP")?atoi(getenv("TPU_PIPE_OZP")):0;
    unsigned char* o=oh[0];
    int nf=osz/(odt==0?4:(odt==2?2:1));
    double mn=1e30,mx=-1e30,sum=0,sq=0;
    fprintf(stderr,"[fence] OUT(int8-raw):");
    for(int i=0;i<osz && i<32;i++) fprintf(stderr," %d",(int)(signed char)o[i]);
    fprintf(stderr,"\n[fence] OUT(dequant):");
    for(int i=0;i<nf;i++){
      double v; signed char* s8=(signed char*)o; short* s16=(short*)o; float* f=(float*)o;
      if(odt==0) v=f[i]; else if(odt==2) v=(s16[i]-ozp)*osc; else v=(s8[i]-ozp)*osc;
      if(i<32) fprintf(stderr," %.4f",v);
      if(v<mn)mn=v; if(v>mx)mx=v; sum+=v; sq+=v*v;
    }
    fprintf(stderr,"\n[fence] min=%.4f max=%.4f mean=%.4f std=%.4f\n",mn,mx,sum/nf,sqrt(sq/nf-(sum/nf)*(sum/nf)));
  }
  double t1=nowms(); int total=N+K;
  double per=(t1-t0)/total;
  fprintf(stderr,"[pipe] mid=%d K=%d N=%d (%s): %.3f ms/Inf  %.0f Inf/s  Mismatch=%d/%d  Timeouts=%d\n",
          mid,K,N,use_fd?"dupfd-poll":"spin-IsCompleted",per,1000.0/per,mism,total,to);
}

int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: tpud <socket> <model0.package> [model1.package ...]\n"); return 1; }
  /* CPU-Cluster-Pinning (Jitter-Hygiene): TPU_CPU=<n> pinnt den Submit/Wait-Thread auf einen Core.
     Tensor G3: cpu8=X3(prime), cpu4-7=A715(mid), cpu0-3=A510(little). Default X3 fuer min Jitter. */
  { const char* pc=getenv("TPU_CPU"); int cpu=pc?atoi(pc):-1;
    if(pc && cpu>=0){ unsigned long mask=1UL<<cpu;
      if(syscall(__NR_sched_setaffinity,0,sizeof(mask),&mask)==0) fprintf(stderr,"[tpud] pinned to cpu%d\n",cpu);
      else fprintf(stderr,"[tpud] pin cpu%d failed\n",cpu); } }
  const char* sock=argv[1];
  const char* w=getenv("TPU_WAIT_US"); if(w) g_wait_us=atol(w);
  const char* pw=getenv("TPU_POLL_US"); if(pw) g_poll_us=atol(pw);
  signal(SIGPIPE,SIG_IGN);
  if(tpu_init()!=0) return 1;
  { const char* fe=getenv("TPU_FENCE"); g_use_fence=(submitFenced&&fenceDone&&fenceFree)&&!(fe&&atoi(fe)==0); g_fence_yield=getenv("TPU_FENCE_YIELD")!=NULL;
    fprintf(stderr,"[tpud] Completion: %s\n",g_use_fence?"FENCE (SubmitFenced/IsCompleted, spin)":"Sentinel-Poll (langsam, 6-8ms Fixkosten)"); }
  { const char* w=getenv("TPU_WAKELOCK"); if(w&&atoi(w)==0) g_wl_enable=0; const char* wi=getenv("TPU_WAKELOCK_IDLE_MS"); if(wi) g_wl_idle_ms=atol(wi);
    if(g_wl_enable){ pthread_t wt; pthread_create(&wt,0,wl_idle_thread,0); pthread_detach(wt); signal(SIGTERM,wl_on_signal); signal(SIGINT,wl_on_signal); }
    fprintf(stderr,"[tpud] Wakelock: %s (idle %ld ms)\n",g_wl_enable?"an":"aus",g_wl_idle_ms); }
  for(int i=2;i<argc;i++) if(load_model(argv[i])<0) return 1;
  if(g_nmodels==0){ fprintf(stderr,"[tpud] kein Modell geladen\n"); return 1; }

  if(getenv("TPU_PIPE")){ int pm=getenv("TPU_PIPE_MID")?atoi(getenv("TPU_PIPE_MID")):0; if(pm>=g_nmodels)pm=0; pipe_bench(pm); return 0; }

  if(getenv("TPU_WARMUP")){
    /* Warmup-Inferenz je Modell auf dem MAIN-Thread vor dem Servieren (Befund 21.8.: das
     * Whisper-Softmax-Kern-Package liefert ueber Conn-Threads Nullen, bis einmal auf dem
     * Main-Thread inferiert wurde — Lazy-Init in der Darwinn-Lib). */
    int wreps=atoi(getenv("TPU_WARMUP")); if(wreps<1) wreps=1;
    for(int m=0;m<g_nmodels;m++){
      if(g_models[m].n_in!=1||g_models[m].n_out!=1) continue;
      for(int w=0;w<wreps;w++){
        void *ih=0,*oh=0;
        void* ib=mkbuf(g_models[m].in_size,&ih); void* ob=mkbuf(g_models[m].out_size,&oh);
        if(ib&&ob){ memset(ih,1+w,g_models[m].in_size);
          const char* wf=getenv("TPU_WARMUP_IN");
          if(wf){ FILE* f=fopen(wf,"rb"); if(f){ fread(ih,1,g_models[m].in_size,f); fclose(f); } }
          double us=0;
          int rc=tpu_infer(m,ib,ob,(unsigned char*)oh,&us);
          long nz=0; for(long q=0;q<g_models[m].out_size;q++) if(((unsigned char*)oh)[q]) nz++;
          fprintf(stderr,"[tpud] warmup M%d #%d: rc=%d %.2f ms nonzero=%ld/%ld\n",m,w,rc,us/1000.0,nz,g_models[m].out_size); }
        if(bufFree){ if(ib)bufFree(ib); if(ob)bufFree(ob); }
      }
    }
  }

  int srv=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
  unlink(sock);
  if(bind(srv,(struct sockaddr*)&a,sizeof a)<0){ fprintf(stderr,"[tpud] bind %s: %s\n",sock,strerror(errno)); return 1; }
  chmod(sock,0666); listen(srv,8);
  fprintf(stderr,"[tpud] bereit, lauscht auf %s (%d Modell(e), %s, zero-copy %s)\n",sock,g_nmodels,g_wait_us>0?"fixe Wartezeit":"adaptiver Poll",impFd?"ja":"nein");
  int single=getenv("TPU_SINGLE")?1:0;   /* seriell auf dem MAIN-Thread servieren (Befund 21.8.:
                                            LUT/Softmax-Packages liefern von Conn-Threads Nullen) */
  for(;;){ int c=accept(srv,0,0); if(c<0) continue;
    if(single){ handle_conn(c); close(c); continue; }
    pthread_t th; int* pc=malloc(sizeof(int)); *pc=c;
    if(pthread_create(&th,0,conn_thread,pc)==0) pthread_detach(th); else { handle_conn(c); close(c); free(pc); } }
  return 0;
}
