/* gxp-async — DSP-Durchsatz mit ASYNCHRONER Submission (RunAsync + Request_Wait).
 * Statt pro Job synchron auf den Firmware-Roundtrip (~1 ms) zu warten, halten wir
 * K Requests gleichzeitig "in flight" (Ring): submit K, dann je fertigen Job direkt
 * einen neuen nachschieben. So ueberlappt die Submit-/Completion-Latenz mit der
 * Ausfuehrung. Sweep ueber K; K=1 ~ RunSync-Baseline.
 * Bionic/NDK, root, Metrics-Stub zuerst im LD_LIBRARY_PATH.  gxp-async [N]  */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/types.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)
struct dma_heap_allocation_data { __u64 len; __u32 fd; __u32 fd_flags; __u64 heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0, struct dma_heap_allocation_data)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static void* L;
#define G(t,n) t n=(t)dlsym(L,#n); if(!n){P("MISSING %s\n",#n);return 1;}
typedef int(*f1)(void**); typedef int(*fCD)(void*,void*,void**); typedef int(*fOLB)(void*,void*,uint32_t,void**);
typedef int(*fLL)(void*,const char*,void**); typedef void(*fSF)(void*,void*); typedef void(*fAB)(void*,void*);
typedef int(*fRun)(void*,void*,void*); typedef int(*fAW)(void*,void*); typedef void(*fSCI)(void*,int);
typedef int(*fImpFd)(void*,void*,int,uint32_t,void**); typedef int(*fMap)(void*); typedef int(*fW)(void*); typedef int(*fRel)(void*);

#define MAXK 32
static void *g_dev,*g_es,*g_fn;
static fRun RunSync,RunAsync; static fW ReqWait; static f1 CreateRequest; static fSF SetFunction; static fAB AppendBuffer; static fRel relReq;
static void* g_buf[MAXK];

static void* mkreq(int slot){ void* r=0; CreateRequest(&r); SetFunction(r,g_fn); AppendBuffer(r,g_buf[slot]); return r; }

/* Async-Durchsatz mit K in-flight; gibt Jobs/s zurueck */
static double bench_async(int K,int N){
  void* ring[MAXK];
  for(int k=0;k<K;k++){ ring[k]=mkreq(k); RunAsync(g_dev,g_es,ring[k]); }
  double t0=now();
  for(int i=0;i<N;i++){ int h=i%K; ReqWait(ring[h]); if(relReq)relReq(ring[h]); ring[h]=mkreq(h); RunAsync(g_dev,g_es,ring[h]); }
  double dt=now()-t0;
  for(int k=0;k<K;k++){ ReqWait(ring[k]); if(relReq)relReq(ring[k]); }
  return N/dt;
}
/* Wie bench_async, aber Requests WIEDERVERWENDEN (kein CreateRequest/Release pro Job) */
static double bench_async_reuse(int K,int N){
  void* ring[MAXK];
  for(int k=0;k<K;k++){ ring[k]=mkreq(k); RunAsync(g_dev,g_es,ring[k]); }
  double t0=now();
  for(int i=0;i<N;i++){ int h=i%K; ReqWait(ring[h]); RunAsync(g_dev,g_es,ring[h]); }
  double dt=now()-t0;
  for(int k=0;k<K;k++){ ReqWait(ring[k]); if(relReq)relReq(ring[k]); }
  return N/dt;
}
/* Sync-Baseline (RunSync, 1 in-flight, Request wiederverwendet) */
static double bench_sync(int N){
  void* r=mkreq(0); double t0=now(); for(int i=0;i<N;i++) RunSync(g_dev,g_es,r); double dt=now()-t0; if(relReq)relReq(r); return N/dt;
}

int main(int argc,char**argv){
  int N=argc>1?atoi(argv[1]):3000;
  setvbuf(stderr,0,_IONBF,0);
  L=dlopen("libgxp.so",RTLD_NOW|RTLD_GLOBAL); if(!L){P("dlopen FAIL\n");return 1;}
  G(f1,GxpCapi_Internal_Initialize)G(f1,GxpCapi_CreateRuntime)G(f1,GxpCapi_CreateDeviceSpec)
  G(fCD,GxpCapi_CreateDevice)G(fOLB,GxpCapi_OpenLibraryFromBuffer)G(fLL,GxpCapi_GetFunction)
  G(f1,GxpCapi_CreateRequest)G(fSF,GxpCapi_Request_SetFunction)G(fAB,GxpCapi_Request_AppendBuffer)
  G(f1,GxpCapi_CreateExecutionSpec)G(fRun,GxpCapi_RunSync)G(fRun,GxpCapi_RunAsync)G(fW,GxpCapi_Request_Wait)
  G(f1,GxpCapi_CreateWakelockDescriptor)G(fAW,GxpCapi_AcquireWakelock)
  G(fSCI,GxpCapi_ExecutionSpec_SetCoreId)G(fSCI,GxpCapi_ExecutionSpec_SetCoreCount)
  G(f1,GxpCapi_CreateBufferOptions)G(fImpFd,GxpCapi_ImportBufferFromFd)G(fMap,GxpCapi_MapBufferAllCores)
  CreateRequest=GxpCapi_CreateRequest; SetFunction=GxpCapi_Request_SetFunction; AppendBuffer=GxpCapi_Request_AppendBuffer;
  RunSync=GxpCapi_RunSync; RunAsync=GxpCapi_RunAsync; ReqWait=GxpCapi_Request_Wait;
  relReq=(fRel)dlsym(L,"GxpCapi_ReleaseRequest");
  typedef void(*fSAT)(void*,int); fSAT sat=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAccessType");
  fSAT sds=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAllowDmaBufferSyncOps");
  void(*setPoll)(void*,int)=(void(*)(void*,int))dlsym(L,"GxpCapi_ExecutionSpec_SetWorkloadPollingFrequency");

  int r; void *rt=0,*spec=0,*wl=0,*lib=0,*iopts=0;
  GxpCapi_Internal_Initialize(0); GxpCapi_CreateRuntime(&rt); GxpCapi_CreateDeviceSpec(&spec);
  r=GxpCapi_CreateDevice(rt,spec,&g_dev); if(r){P("CreateDevice=%d\n",r);return 2;}
  GxpCapi_CreateWakelockDescriptor(&wl); GxpCapi_AcquireWakelock(g_dev,wl);
  FILE* f=fopen("/data/adb/hwbridge/vscale.elf","rb"); if(!f){P("kein vscale.elf\n");return 2;}
  fseek(f,0,SEEK_END); long esz=ftell(f); fseek(f,0,SEEK_SET); void* elf=malloc(esz); if(fread(elf,1,esz,f)!=(size_t)esz)return 2; fclose(f);
  GxpCapi_OpenLibraryFromBuffer(g_dev,elf,(uint32_t)esz,&lib); GxpCapi_GetFunction(lib,"tpu_request_sync_submitter",&g_fn);
  GxpCapi_CreateBufferOptions(&iopts); if(sat)sat(iopts,2); if(sds)sds(iopts,1);
  GxpCapi_CreateExecutionSpec(&g_es); GxpCapi_ExecutionSpec_SetCoreId(g_es,0); GxpCapi_ExecutionSpec_SetCoreCount(g_es,1);
  const char* pf=getenv("GXP_POLL"); if(pf&&setPoll){ setPoll(g_es,atoi(pf)); P("SetWorkloadPollingFrequency(%s)\n",pf); }

  int heap=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC);
  for(int k=0;k<MAXK;k++){ struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=4096; d.fd_flags=O_RDWR|O_CLOEXEC;
    ioctl(heap,DMA_HEAP_IOCTL_ALLOC,&d); int dfd=(int)d.fd; int32_t* p=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0);
    p[0]=3;p[1]=5;p[2]=10;p[3]=100; void* b=0; if(GxpCapi_ImportBufferFromFd(g_dev,iopts,dfd,4096,&b)||!b){P("import k=%d fail\n",k);return 3;}
    GxpCapi_MapBufferAllCores(b); g_buf[k]=b; }

  /* Warmup */ { void* r0=mkreq(0); for(int i=0;i<20;i++) RunSync(g_dev,g_es,r0); if(relReq)relReq(r0); }

  double base=bench_sync(N);
  P("Baseline RunSync (K=1):  %6.0f Jobs/s  (%.3f ms/Job)\n", base, 1e3/base);
  int Ks[]={1,2,3,4,6,8,12,16,24,32};
  P("--- RunAsync-Pipeline (K in-flight) ---\n");
  double best=base; int bestK=1;
  for(unsigned i=0;i<sizeof Ks/sizeof*Ks;i++){ int K=Ks[i]; if(K>MAXK)continue;
    double jps=bench_async(K,N);
    P("K=%2d:  %6.0f Jobs/s  (%.3f ms/Job)  x%.2f\n", K, jps, 1e3/jps, jps/base);
    if(jps>best){ best=jps; bestK=K; } }
  P("=> best (Request je Job neu): K=%d, %.0f Jobs/s (x%.2f gegenueber sync)\n", bestK, best, best/base);
  P("--- RunAsync-Pipeline + Request-WIEDERVERWENDUNG ---\n");
  int Kr[]={4,8,16}; double bestr=base; int bestrK=1;
  for(unsigned i=0;i<sizeof Kr/sizeof*Kr;i++){ int K=Kr[i]; double jps=bench_async_reuse(K,N);
    P("K=%2d (reuse):  %6.0f Jobs/s  (%.3f ms/Job)  x%.2f\n", K, jps, 1e3/jps, jps/base);
    if(jps>bestr){bestr=jps;bestrK=K;} }
  P("=> best (reuse): K=%d, %.0f Jobs/s (x%.2f gegenueber sync)\n", bestrK, bestr, bestr/base);
  return 0;
}
