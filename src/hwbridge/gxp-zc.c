/* gxp-zc — ZERO-COPY-Beweis auf dem Callisto-DSP.
 * Statt GxpCapi_CreateBuffer (GXP alloziert + GetHostPointer): wir allozieren
 * SELBST einen dmabuf aus /dev/dma_heap/system-uncached, mmappen ihn, schreiben
 * den Input, und geben den fd via GxpCapi_ImportBufferFromFd an den DSP. Der DSP
 * rechnet vscale (out[i]=in[i]*in[0]) IN unserem dmabuf; wir lesen das Ergebnis
 * aus DERSELBEN mmap -> kein Byte-Copy zwischen Host und Device.
 * Bionic/NDK; als root, LD_LIBRARY_PATH mit Metrics-Stub zuerst. */
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

typedef int (*f1)(void**);
typedef int (*fCD)(void*,void*,void**);
typedef int (*fOLB)(void*,void*,uint32_t,void**);
typedef int (*fLL)(void*,const char*,void**);
typedef void(*fSF)(void*,void*);
typedef void(*fAB)(void*,void*);
typedef int (*fRun)(void*,void*,void*);
typedef int (*fGRV)(void*,int64_t*);
typedef int (*fAW)(void*,void*);
typedef void(*fSCI)(void*,int);
typedef int (*fCB)(void*,void*,uint32_t,void**);        // CreateBuffer(dev,bufopts,size,&out)
typedef int (*fGBFD)(void*);                            // Buffer_GetBufferFd(buf) -> dmabuf-fd
typedef void*(*fGHP)(void*);                            // Buffer_GetHostPointer(buf)

static void* L;
#define G(t,n) t n=(t)dlsym(L,#n); if(!n){P("MISSING %s\n",#n);return 1;}

int main(void){
  setvbuf(stderr,0,_IONBF,0);
  L=dlopen("libgxp.so",RTLD_NOW|RTLD_GLOBAL); if(!L){P("dlopen FAIL %s\n",dlerror());return 1;}
  G(f1,GxpCapi_Internal_Initialize)G(f1,GxpCapi_CreateRuntime)G(f1,GxpCapi_CreateDeviceSpec)
  G(fCD,GxpCapi_CreateDevice)G(fOLB,GxpCapi_OpenLibraryFromBuffer)G(fLL,GxpCapi_GetFunction)
  G(f1,GxpCapi_CreateRequest)G(fSF,GxpCapi_Request_SetFunction)G(fAB,GxpCapi_Request_AppendBuffer)
  G(f1,GxpCapi_CreateExecutionSpec)G(fRun,GxpCapi_RunSync)G(fGRV,GxpCapi_Request_GetReturnValue)
  G(f1,GxpCapi_CreateWakelockDescriptor)G(fAW,GxpCapi_AcquireWakelock)
  G(fSCI,GxpCapi_ExecutionSpec_SetCoreId)G(fSCI,GxpCapi_ExecutionSpec_SetCoreCount)
  G(f1,GxpCapi_CreateBufferOptions)G(fCB,GxpCapi_CreateBuffer)
  G(fGBFD,GxpCapi_Buffer_GetBufferFd)G(fGHP,GxpCapi_Buffer_GetHostPointer)

  int r; void *rt=0,*spec=0,*dev=0,*wl=0,*lib=0,*fn=0,*es=0,*req=0,*buf=0;
  GxpCapi_Internal_Initialize(0);
  GxpCapi_CreateRuntime(&rt); GxpCapi_CreateDeviceSpec(&spec);
  r=GxpCapi_CreateDevice(rt,spec,&dev); if(r){P("CreateDevice=%d\n",r);return 2;}
  GxpCapi_CreateWakelockDescriptor(&wl); GxpCapi_AcquireWakelock(dev,wl);

  /* vscale.elf laden */
  const char* path="/data/adb/hwbridge/vscale.elf";
  FILE* f=fopen(path,"rb"); if(!f){P("open %s fail\n",path);return 2;}
  fseek(f,0,SEEK_END); long esz=ftell(f); fseek(f,0,SEEK_SET);
  void* elf=malloc(esz); if(fread(elf,1,esz,f)!=(size_t)esz){P("read elf\n");return 2;} fclose(f);
  r=GxpCapi_OpenLibraryFromBuffer(dev,elf,(uint32_t)esz,&lib); P("OpenLibraryFromBuffer=%d lib=%p\n",r,lib); if(r)return 3;
  r=GxpCapi_GetFunction(lib,"tpu_request_sync_submitter",&fn); P("GetFunction=%d fn=%p\n",r,fn); if(r||!fn)return 4;

  /* ===== ZERO-COPY per EXPORT: GXP alloziert dmabuf-Puffer, wir holen dessen fd,
   * mappen den fd SELBST (der Pfad, den ein Client nach SCM_RIGHTS nutzt) ===== */
  uint32_t nbytes=4096; void* bopts=0;
  GxpCapi_CreateBufferOptions(&bopts);
  r=GxpCapi_CreateBuffer(dev,bopts,nbytes,&buf); P("CreateBuffer(%u)=%d buf=%p\n",nbytes,r,buf); if(r||!buf)return 5;
  int dfd=GxpCapi_Buffer_GetBufferFd(buf);
  P("Buffer_GetBufferFd -> dmabuf-fd=%d (teilbar via SCM_RIGHTS)\n",dfd);
  if(dfd<0){ P("GetBufferFd fehlgeschlagen\n"); return 6; }
  /* den EXPORTIERTEN fd unabhaengig mappen (wie es ein anderer Prozess taete) */
  int32_t* p=mmap(0,nbytes,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0);
  if(p==MAP_FAILED){ P("mmap(bfd) fail\n"); return 6; }
  p[0]=3; p[1]=5; p[2]=10; p[3]=100;        /* Input via die fd-mmap schreiben */
  P("input (via exportierte fd-mmap):  [%d %d %d %d]\n",p[0],p[1],p[2],p[3]);

  GxpCapi_CreateExecutionSpec(&es); GxpCapi_ExecutionSpec_SetCoreId(es,0); GxpCapi_ExecutionSpec_SetCoreCount(es,1);
  GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn);
  GxpCapi_Request_AppendBuffer(req,buf);
  P("--- RunSync (DSP rechnet vscale IN unserem dmabuf) ---\n");
  r=GxpCapi_RunSync(dev,es,req); P("RunSync=%d\n",r);
  int64_t rv=-999; GxpCapi_Request_GetReturnValue(req,&rv); P("ReturnValue=%lld\n",(long long)rv);

  P("output (aus DERSELBEN mmap): [%d %d %d %d]\n",p[0],p[1],p[2],p[3]);
  P("erwartet vscale (in[i]*in[0]=x3):  [9 15 30 300]\n");
  if(p[0]==9 && p[1]==15 && p[2]==30 && p[3]==300)
    P(">>> ZERO-COPY OK: der DSP hat direkt in unseren dmabuf gerechnet, kein Copy. <<<\n");
  else P(">>> Ergebnis unerwartet (s.o.) <<<\n");
  munmap(p,4096); close(dfd);
  return 0;
}
