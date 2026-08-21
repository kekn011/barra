/* gxp-scale — prueft die Hypothese "DSP ist command-gebunden, nicht compute-gebunden".
 * Nutzt den sum-Kernel (ker_sumloop.elf, datenabhaengige Trip-Count: N in buf[0],
 * summiert buf[1..N]). Wir variieren N (Arbeit PRO Command) und messen die Pro-
 * Command-Latenz + den daraus folgenden Element-Durchsatz.
 *  - command-gebunden  => Latenz bleibt fuer kleine N ~flach (~1 ms), Elemente/s steigt ~linear mit N
 *  - compute-gebunden  => Latenz waechst sofort ~linear mit N
 * Bionic/NDK, root, Metrics-Stub zuerst.  gxp-scale [iters]  */
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
typedef void(*fSCI)(void*,int); typedef int(*fImpFd)(void*,void*,int,uint32_t,void**); typedef int(*fMap)(void*); typedef int(*fRel)(void*);

int main(int argc,char**argv){
  int IT=argc>1?atoi(argv[1]):100;
  const char* kf=argc>2?argv[2]:"/data/adb/hwbridge/ker_sumloop.elf";
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
  fSAT scache=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetCacheability");

  int r; void *rt=0,*spec=0,*dev=0,*wl=0,*lib=0,*fn=0,*es=0,*iopts=0,*buf=0;
  GxpCapi_Internal_Initialize(0); GxpCapi_CreateRuntime(&rt); GxpCapi_CreateDeviceSpec(&spec);
  r=GxpCapi_CreateDevice(rt,spec,&dev); if(r){P("CreateDevice=%d\n",r);return 2;}
  GxpCapi_CreateWakelockDescriptor(&wl); GxpCapi_AcquireWakelock(dev,wl);
  FILE* f=fopen(kf,"rb"); if(!f){P("kein %s\n",kf);return 2;}
  fseek(f,0,SEEK_END); long esz=ftell(f); fseek(f,0,SEEK_SET); void* elf=malloc(esz); if(fread(elf,1,esz,f)!=(size_t)esz)return 2; fclose(f);
  r=GxpCapi_OpenLibraryFromBuffer(dev,elf,(uint32_t)esz,&lib); if(r){P("OpenLib=%d\n",r);return 3;}
  r=GxpCapi_GetFunction(lib,"tpu_request_sync_submitter",&fn); if(r||!fn){P("GetFunction=%d\n",r);return 4;}
  GxpCapi_CreateBufferOptions(&iopts); if(sat)sat(iopts,2); if(sds)sds(iopts,1);
  const char* cv=getenv("GXP_CACHE"); if(cv&&scache){ scache(iopts,atoi(cv)); P("Cacheability(%s)\n",cv); }
  GxpCapi_CreateExecutionSpec(&es); GxpCapi_ExecutionSpec_SetCoreId(es,0); GxpCapi_ExecutionSpec_SetCoreCount(es,1);

  uint32_t BYTES=8u*1024*1024;            /* 8 MB -> bis ~2 Mio int32 pro Command */
  int heap=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC);
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=BYTES; d.fd_flags=O_RDWR|O_CLOEXEC;
  if(ioctl(heap,DMA_HEAP_IOCTL_ALLOC,&d)<0){P("alloc\n");return 5;} int dfd=(int)d.fd;
  int32_t* p=mmap(0,BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0); if(p==MAP_FAILED){P("mmap\n");return 5;}
  r=GxpCapi_ImportBufferFromFd(dev,iopts,dfd,BYTES,&buf); if(r||!buf){P("import=%d\n",r);return 3;}
  GxpCapi_MapBufferAllCores(buf);

  void* req=0; GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn); GxpCapi_Request_AppendBuffer(req,buf);

  /* Korrektheit bei N=4: [4,5,6,7,8] -> 26 */
  p[0]=4;p[1]=5;p[2]=6;p[3]=7;p[4]=8; GxpCapi_RunSync(dev,es,req);
  P("Korrektheit N=4: sum=%d (erw 26)  %s\n",p[0], p[0]==26?"OK":"?");

  int Ns[]={4,16,64,256,1024,4096,16384,65536,262144,1048576,2097100};
  P("%-9s  %-11s  %-13s  %-13s\n","N","ms/Command","Mio Elem/s","Commands/s");
  for(unsigned i=0;i<sizeof Ns/sizeof*Ns;i++){
    uint32_t N=Ns[i]; if((uint64_t)(N+1)*4>BYTES) break;
    p[0]=(int32_t)N; for(uint32_t j=1;j<=N && j<=64;j++) p[j]=1;   /* nur Anfang fuellen reicht fuer Timing */
    /* warmup 3 */ for(int w=0;w<3;w++) GxpCapi_RunSync(dev,es,req);
    double t0=now(); for(int it=0;it<IT;it++) GxpCapi_RunSync(dev,es,req); double dt=(now()-t0)/IT;
    double eps=N/dt; double cps=1.0/dt;
    P("%-9u  %-11.3f  %-13.2f  %-13.0f\n", N, dt*1e3, eps/1e6, cps);
  }
  if(relReq)relReq(req);
  P("\nDeutung: bleibt ms/Command fuer kleine N flach und steigen Mio Elem/s ~linear -> command-gebunden (Hypothese bestaetigt).\n");
  return 0;
}
