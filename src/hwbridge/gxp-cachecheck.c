/* gxp-cachecheck — prueft KORREKTHEIT (Cache-Kohaerenz) eines cacheable importierten
 * Puffers mit dem vscalen-Kernel: buf[0]=N, buf[1]=s, buf[2..N+1]*=s. Nach dem DSP-Lauf
 * MUSS die CPU buf[2+i] == (i+1)*s sehen. Vergleicht uncached vs cached.
 * Bionic/NDK, root, Metrics-Stub zuerst.  gxp-cachecheck [cache 0|1]  */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/types.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)
struct dma_heap_allocation_data { __u64 len; __u32 fd; __u32 fd_flags; __u64 heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0, struct dma_heap_allocation_data)
struct dma_buf_sync { __u64 flags; };
#define DMA_BUF_IOCTL_SYNC _IOW('b', 0, struct dma_buf_sync)
static void dbsync(int fd,__u64 f){ struct dma_buf_sync s={f}; ioctl(fd,DMA_BUF_IOCTL_SYNC,&s); }
static void* L;
#define G(t,n) t n=(t)dlsym(L,#n); if(!n){P("MISSING %s\n",#n);return 1;}
typedef int(*f1)(void**); typedef int(*fCD)(void*,void*,void**); typedef int(*fOLB)(void*,void*,uint32_t,void**);
typedef int(*fLL)(void*,const char*,void**); typedef void(*fSF)(void*,void*); typedef void(*fAB)(void*,void*);
typedef int(*fRun)(void*,void*,void*); typedef int(*fAW)(void*,void*); typedef void(*fSCI)(void*,int);
typedef int(*fImpFd)(void*,void*,int,uint32_t,void**); typedef int(*fMap)(void*); typedef void(*fSAT)(void*,int);

int main(int argc,char**argv){
  int cache=argc>1?atoi(argv[1]):1;
  setvbuf(stderr,0,_IONBF,0);
  L=dlopen("libgxp.so",RTLD_NOW|RTLD_GLOBAL); if(!L){P("dlopen FAIL\n");return 1;}
  G(f1,GxpCapi_Internal_Initialize)G(f1,GxpCapi_CreateRuntime)G(f1,GxpCapi_CreateDeviceSpec)
  G(fCD,GxpCapi_CreateDevice)G(fOLB,GxpCapi_OpenLibraryFromBuffer)G(fLL,GxpCapi_GetFunction)
  G(f1,GxpCapi_CreateRequest)G(fSF,GxpCapi_Request_SetFunction)G(fAB,GxpCapi_Request_AppendBuffer)
  G(f1,GxpCapi_CreateExecutionSpec)G(fRun,GxpCapi_RunSync)
  G(f1,GxpCapi_CreateWakelockDescriptor)G(fAW,GxpCapi_AcquireWakelock)
  G(fSCI,GxpCapi_ExecutionSpec_SetCoreId)G(fSCI,GxpCapi_ExecutionSpec_SetCoreCount)
  G(f1,GxpCapi_CreateBufferOptions)G(fImpFd,GxpCapi_ImportBufferFromFd)G(fMap,GxpCapi_MapBufferAllCores)
  fSAT sat=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAccessType");
  fSAT sds=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAllowDmaBufferSyncOps");
  fSAT scache=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetCacheability");

  int r; void *rt=0,*spec=0,*dev=0,*wl=0,*lib=0,*fn=0,*es=0,*iopts=0,*buf=0;
  GxpCapi_Internal_Initialize(0); GxpCapi_CreateRuntime(&rt); GxpCapi_CreateDeviceSpec(&spec);
  r=GxpCapi_CreateDevice(rt,spec,&dev); if(r){P("CreateDevice=%d\n",r);return 2;}
  GxpCapi_CreateWakelockDescriptor(&wl); GxpCapi_AcquireWakelock(dev,wl);
  FILE* f=fopen("/data/adb/hwbridge/ker_vscalen.elf","rb"); if(!f){P("kein ker_vscalen.elf\n");return 2;}
  fseek(f,0,SEEK_END); long esz=ftell(f); fseek(f,0,SEEK_SET); void* elf=malloc(esz); if(fread(elf,1,esz,f)!=(size_t)esz)return 2; fclose(f);
  GxpCapi_OpenLibraryFromBuffer(dev,elf,(uint32_t)esz,&lib); GxpCapi_GetFunction(lib,"tpu_request_sync_submitter",&fn);
  GxpCapi_CreateBufferOptions(&iopts); if(sat)sat(iopts,2); if(sds)sds(iopts,1);
  if(scache){ scache(iopts,cache); P("Cacheability(%d)\n",cache); }
  GxpCapi_CreateExecutionSpec(&es); GxpCapi_ExecutionSpec_SetCoreId(es,0); GxpCapi_ExecutionSpec_SetCoreCount(es,1);

  uint32_t N=1000, BYTES=(N+2)*4;
  int heap=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC);
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=BYTES; d.fd_flags=O_RDWR|O_CLOEXEC;
  ioctl(heap,DMA_HEAP_IOCTL_ALLOC,&d); int dfd=(int)d.fd;
  int32_t* p=mmap(0,BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0);
  r=GxpCapi_ImportBufferFromFd(dev,iopts,dfd,BYTES,&buf); if(r||!buf){P("import=%d\n",r);return 3;}
  GxpCapi_MapBufferAllCores(buf);
  void* req=0; GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn); GxpCapi_Request_AppendBuffer(req,buf);

  /* Eingabe: N=1000, s=3, buf[2..1001]=1..1000 -> erwartet buf[2+i]=(i+1)*3 */
  dbsync(dfd,2/*WRITE*/|0/*START*/);
  p[0]=N; p[1]=3; for(uint32_t i=0;i<N;i++) p[2+i]=(int32_t)(i+1);
  dbsync(dfd,2|4/*WRITE|END*/);
  r=GxpCapi_RunSync(dev,es,req);
  dbsync(dfd,1/*READ*/|0);
  int bad=0; for(uint32_t i=0;i<N;i++){ int32_t exp=(int32_t)((i+1)*3); if(p[2+i]!=exp){ if(bad<3) P("  FEHLER @%u: %d != %d\n",i,p[2+i],exp); bad++; } }
  dbsync(dfd,1|4);
  P("cache=%d RunSync=%d N=%u: buf[2..4]=[%d %d %d] (erw 3 6 9), Fehler=%d/%u -> %s\n",
    cache,r,N,p[2],p[3],p[4],bad,N, bad==0?"KORREKT (kohaerent)":"INKOHAERENT");
  return bad?1:0;
}
