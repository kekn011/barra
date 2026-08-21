/* gxp-lat — zerlegt die Pro-Job-Latenz des DSP-Pfads (warum nur ~550-660 Jobs/s
 * trotz ~1 GHz Takt?). Standalone (ohne Socket/Daemon), importiert EINEN dmabuf,
 * mappt ihn, und misst:
 *   A) RunSync-only  : dieselbe Request-Struktur wiederverwenden -> reiner
 *                      Firmware-Roundtrip (MCU-RPC + Ausfuehrung + Completion).
 *   B) Voll-Zyklus   : CreateRequest+SetFunction+AppendBuffer+RunSync+ReleaseRequest
 *                      pro Iteration -> das, was gxpd3 pro Job macht (ohne Socket).
 * Bionic/NDK, root, Metrics-Stub zuerst im LD_LIBRARY_PATH. */
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
typedef int(*fRun)(void*,void*,void*); typedef int(*fGRV)(void*,int64_t*); typedef int(*fAW)(void*,void*);
typedef void(*fSCI)(void*,int); typedef int(*fImpFd)(void*,void*,int,uint32_t,void**); typedef int(*fMap)(void*);
typedef int(*fRel)(void*);

int main(int argc,char**argv){
  int N = argc>1?atoi(argv[1]):2000;
  setvbuf(stderr,0,_IONBF,0);
  L=dlopen("libgxp.so",RTLD_NOW|RTLD_GLOBAL); if(!L){P("dlopen FAIL\n");return 1;}
  G(f1,GxpCapi_Internal_Initialize)G(f1,GxpCapi_CreateRuntime)G(f1,GxpCapi_CreateDeviceSpec)
  G(fCD,GxpCapi_CreateDevice)G(fOLB,GxpCapi_OpenLibraryFromBuffer)G(fLL,GxpCapi_GetFunction)
  G(f1,GxpCapi_CreateRequest)G(fSF,GxpCapi_Request_SetFunction)G(fAB,GxpCapi_Request_AppendBuffer)
  G(f1,GxpCapi_CreateExecutionSpec)G(fRun,GxpCapi_RunSync)G(fGRV,GxpCapi_Request_GetReturnValue)
  G(f1,GxpCapi_CreateWakelockDescriptor)G(fAW,GxpCapi_AcquireWakelock)
  G(fSCI,GxpCapi_ExecutionSpec_SetCoreId)G(fSCI,GxpCapi_ExecutionSpec_SetCoreCount)
  G(f1,GxpCapi_CreateBufferOptions)G(fImpFd,GxpCapi_ImportBufferFromFd)G(fMap,GxpCapi_MapBufferAllCores)
  fRel relReq=(fRel)dlsym(L,"GxpCapi_ReleaseRequest");
  typedef void(*fSAT)(void*,int); fSAT sat=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAccessType");
  fSAT sds=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAllowDmaBufferSyncOps");

  int r; void *rt=0,*spec=0,*dev=0,*wl=0,*lib=0,*fn=0,*es=0,*iopts=0,*buf=0;
  GxpCapi_Internal_Initialize(0); GxpCapi_CreateRuntime(&rt); GxpCapi_CreateDeviceSpec(&spec);
  r=GxpCapi_CreateDevice(rt,spec,&dev); if(r){P("CreateDevice=%d\n",r);return 2;}
  GxpCapi_CreateWakelockDescriptor(&wl); GxpCapi_AcquireWakelock(dev,wl);
  FILE* f=fopen("/data/adb/hwbridge/vscale.elf","rb"); if(!f){P("kein vscale.elf\n");return 2;}
  fseek(f,0,SEEK_END); long esz=ftell(f); fseek(f,0,SEEK_SET); void* elf=malloc(esz); if(fread(elf,1,esz,f)!=(size_t)esz)return 2; fclose(f);
  GxpCapi_OpenLibraryFromBuffer(dev,elf,(uint32_t)esz,&lib); GxpCapi_GetFunction(lib,"tpu_request_sync_submitter",&fn);
  GxpCapi_CreateBufferOptions(&iopts); if(sat)sat(iopts,2); if(sds)sds(iopts,1);
  GxpCapi_CreateExecutionSpec(&es); GxpCapi_ExecutionSpec_SetCoreId(es,0); GxpCapi_ExecutionSpec_SetCoreCount(es,1);

  int heap=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC);
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=4096; d.fd_flags=O_RDWR|O_CLOEXEC;
  ioctl(heap,DMA_HEAP_IOCTL_ALLOC,&d); int dfd=(int)d.fd;
  int32_t* p=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0); p[0]=3;p[1]=5;p[2]=10;p[3]=100;
  r=GxpCapi_ImportBufferFromFd(dev,iopts,dfd,4096,&buf); if(r||!buf){P("import=%d\n",r);return 3;}
  GxpCapi_MapBufferAllCores(buf);

  /* Warmup */
  { void* req=0; GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn); GxpCapi_Request_AppendBuffer(req,buf);
    for(int i=0;i<10;i++) GxpCapi_RunSync(dev,es,req); if(relReq)relReq(req); }

  /* A) RunSync-only: eine Request-Struktur, N-mal ausfuehren */
  void* reqA=0; GxpCapi_CreateRequest(&reqA); GxpCapi_Request_SetFunction(reqA,fn); GxpCapi_Request_AppendBuffer(reqA,buf);
  double t0=now(); for(int i=0;i<N;i++) GxpCapi_RunSync(dev,es,reqA); double tA=now()-t0;
  if(relReq)relReq(reqA);

  /* B) Voll-Zyklus wie im Daemon: Request je Iteration neu bauen+freigeben */
  t0=now();
  for(int i=0;i<N;i++){ void* req=0; GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn);
    GxpCapi_Request_AppendBuffer(req,buf); GxpCapi_RunSync(dev,es,req); if(relReq)relReq(req); }
  double tB=now()-t0;

  P("N=%d\n",N);
  P("A) RunSync-only:   %.3f ms/Job  (%.0f Jobs/s)  = reiner Firmware-Roundtrip\n", tA/N*1e3, N/tA);
  P("B) Voll-Zyklus:    %.3f ms/Job  (%.0f Jobs/s)  = + CreateRequest/Append/ReleaseRequest\n", tB/N*1e3, N/tB);
  P("   Request-Overhead: %.3f ms/Job\n", (tB-tA)/N*1e3);
  P("   (Sozket+dma_buf-Sync im gxpd3-Pfad kommt beim vollen Job NOCH dazu; ~1.5-1.8 ms gemessen)\n");
  return 0;
}
