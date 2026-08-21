/* gxp-multicore — DSP-Durchsatz ueber MEHRERE Callisto-Kerne.
 * Aufbauend auf gxp-async: statt alle Requests auf Kern 0 zu fahren, verteilen wir
 * sie per eigener ExecutionSpec je Kern (SetCoreId 0/1/2) round-robin. Jeder Ring-
 * Slot ist fest einem Kern zugeordnet. Zusaetzlich pro Kern eine Korrektheitspruefung
 * (vscale muss [3,15,30,300] liefern), damit wir sicher sind, dass der Kern WIRKLICH
 * rechnet und nicht nur ein leeres Ack schickt.
 * Bionic/NDK, root, Metrics-Stub zuerst.  gxp-multicore [N] [maxCores]  */
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

#define MAXK 48
#define MAXC 3
static void *g_dev,*g_es[MAXC],*g_fn;
static fRun RunSync,RunAsync; static fW ReqWait; static f1 CreateRequest; static fSF SetFunction; static fAB AppendBuffer; static fRel relReq;
static void *g_buf[MAXK]; static int32_t* g_map[MAXK];

static void* mkreq(int slot){ void* r=0; CreateRequest(&r); SetFunction(r,g_fn); AppendBuffer(r,g_buf[slot]); return r; }

/* async-Durchsatz, K in-flight, Slot h fest auf Kern h%nc */
static double bench_mc(int nc,int K,int N){
  void* ring[MAXK];
  for(int k=0;k<K;k++){ ring[k]=mkreq(k); RunAsync(g_dev,g_es[k%nc],ring[k]); }
  double t0=now();
  for(int i=0;i<N;i++){ int h=i%K; ReqWait(ring[h]); if(relReq)relReq(ring[h]); ring[h]=mkreq(h); RunAsync(g_dev,g_es[h%nc],ring[h]); }
  double dt=now()-t0;
  for(int k=0;k<K;k++){ ReqWait(ring[k]); if(relReq)relReq(ring[k]); }
  return N/dt;
}
/* Korrektheit: auf Kern c einmal vscale rechnen und pruefen */
static int check_core(int c){
  g_map[0][0]=3; g_map[0][1]=5; g_map[0][2]=10; g_map[0][3]=100;
  void* r=mkreq(0); int rc=RunSync(g_dev,g_es[c],r); if(relReq)relReq(r);
  int ok = (g_map[0][1]==15 && g_map[0][2]==30 && g_map[0][3]==300);
  P("  Kern %d: RunSync=%d -> [%d %d %d %d] %s\n",c,rc,g_map[0][0],g_map[0][1],g_map[0][2],g_map[0][3], ok?"OK":"RECHNET NICHT");
  return ok;
}

int main(int argc,char**argv){
  int N=argc>1?atoi(argv[1]):3000; int maxc=argc>2?atoi(argv[2]):MAXC; if(maxc>MAXC)maxc=MAXC;
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
  RunSync=GxpCapi_RunSync; RunAsync=GxpCapi_RunAsync; ReqWait=GxpCapi_Request_Wait; relReq=(fRel)dlsym(L,"GxpCapi_ReleaseRequest");
  typedef void(*fSAT)(void*,int); fSAT sat=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAccessType");
  fSAT sds=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAllowDmaBufferSyncOps");

  int r; void *rt=0,*spec=0,*wl=0,*lib=0,*iopts=0;
  GxpCapi_Internal_Initialize(0); GxpCapi_CreateRuntime(&rt); GxpCapi_CreateDeviceSpec(&spec);
  r=GxpCapi_CreateDevice(rt,spec,&g_dev); if(r){P("CreateDevice=%d\n",r);return 2;}
  GxpCapi_CreateWakelockDescriptor(&wl); GxpCapi_AcquireWakelock(g_dev,wl);
  FILE* f=fopen("/data/adb/hwbridge/vscale.elf","rb"); if(!f){P("kein vscale.elf\n");return 2;}
  fseek(f,0,SEEK_END); long esz=ftell(f); fseek(f,0,SEEK_SET); void* elf=malloc(esz); if(fread(elf,1,esz,f)!=(size_t)esz)return 2; fclose(f);
  GxpCapi_OpenLibraryFromBuffer(g_dev,elf,(uint32_t)esz,&lib); GxpCapi_GetFunction(lib,"tpu_request_sync_submitter",&g_fn);
  GxpCapi_CreateBufferOptions(&iopts); if(sat)sat(iopts,2); if(sds)sds(iopts,1);
  /* je Kern eine ExecutionSpec (CoreId c, Count 1) */
  for(int c=0;c<MAXC;c++){ GxpCapi_CreateExecutionSpec(&g_es[c]); GxpCapi_ExecutionSpec_SetCoreId(g_es[c],c); GxpCapi_ExecutionSpec_SetCoreCount(g_es[c],1); }

  int heap=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC);
  for(int k=0;k<MAXK;k++){ struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=4096; d.fd_flags=O_RDWR|O_CLOEXEC;
    ioctl(heap,DMA_HEAP_IOCTL_ALLOC,&d); int dfd=(int)d.fd; int32_t* p=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0);
    p[0]=3;p[1]=5;p[2]=10;p[3]=100; void* b=0; if(GxpCapi_ImportBufferFromFd(g_dev,iopts,dfd,4096,&b)||!b){P("import k=%d fail\n",k);return 3;}
    GxpCapi_MapBufferAllCores(b); g_buf[k]=b; g_map[k]=p; }

  P("=== Korrektheit je Kern (vscale) ===\n");
  int ncok=0; for(int c=0;c<maxc;c++){ if(check_core(c)) ncok=c+1; else break; }
  if(ncok<1){ P("Kern 0 rechnet nicht - Abbruch\n"); return 4; }
  P("nutzbare Kerne (in Folge ab 0): %d\n",ncok);

  /* Warmup */ { void* r0=mkreq(0); for(int i=0;i<20;i++) RunSync(g_dev,g_es[0],r0); if(relReq)relReq(r0); }

  P("=== Durchsatz je Kernzahl (K in-flight) ===\n");
  int Ks[]={4,6,8,12,16,24}; double best=0; int bK=0,bC=0;
  for(int nc=1;nc<=ncok;nc++){
    for(unsigned i=0;i<sizeof Ks/sizeof*Ks;i++){ int K=Ks[i]; if(K>MAXK)continue;
      double jps=bench_mc(nc,K,N);
      P("Kerne=%d K=%2d:  %6.0f Jobs/s  (%.3f ms/Job)\n", nc, K, jps, 1e3/jps);
      if(jps>best){ best=jps; bK=K; bC=nc; } }
    P("  --\n");
  }
  P("=> best: %d Kern(e), K=%d -> %.0f Jobs/s\n", bC, bK, best);
  return 0;
}
