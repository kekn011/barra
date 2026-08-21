/* gxp-zc2 — DSP-ZERO-COPY, 2. Anlauf (Erkenntnis des GXP-Agenten):
 * ImportBufferFromHostAddress crasht bei ROHEM Host-Speicher (kein dmabuf-Backing),
 * ABER ein dmabuf-gemappter Zeiger IST dmabuf-backed -> Flush ok. Also: dmabuf
 * allozieren + mmappen + per HOST-ADRESSE importieren (nicht ByFd). Der DSP rechnet
 * vscale in unseren dmabuf; wir lesen aus DERSELBEN mmap. Bionic, root. */
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

static void* L;
#define G(t,n) t n=(t)dlsym(L,#n); if(!n){P("MISSING %s\n",#n);return 1;}
typedef int(*f1)(void**); typedef int(*fCD)(void*,void*,void**); typedef int(*fOLB)(void*,void*,uint32_t,void**);
typedef int(*fLL)(void*,const char*,void**); typedef void(*fSF)(void*,void*); typedef void(*fAB)(void*,void*);
typedef int(*fRun)(void*,void*,void*); typedef int(*fGRV)(void*,int64_t*); typedef int(*fAW)(void*,void*);
typedef void(*fSCI)(void*,int); typedef int(*fImpHA)(void*,void*,uint64_t,int,void**); /* ImportBufferFromHostAddress(dev,host,size,access,&out) */
typedef int(*fBio)(void*);  /* Buffer_Flush / Buffer_Invalidate (Cache-Maintenance) */

int main(void){
  setvbuf(stderr,0,_IONBF,0);
  L=dlopen("libgxp.so",RTLD_NOW|RTLD_GLOBAL); if(!L){P("dlopen FAIL\n");return 1;}
  G(f1,GxpCapi_Internal_Initialize)G(f1,GxpCapi_CreateRuntime)G(f1,GxpCapi_CreateDeviceSpec)
  G(fCD,GxpCapi_CreateDevice)G(fOLB,GxpCapi_OpenLibraryFromBuffer)G(fLL,GxpCapi_GetFunction)
  G(f1,GxpCapi_CreateRequest)G(fSF,GxpCapi_Request_SetFunction)G(fAB,GxpCapi_Request_AppendBuffer)
  G(f1,GxpCapi_CreateExecutionSpec)G(fRun,GxpCapi_RunSync)G(fGRV,GxpCapi_Request_GetReturnValue)
  G(f1,GxpCapi_CreateWakelockDescriptor)G(fAW,GxpCapi_AcquireWakelock)
  G(fSCI,GxpCapi_ExecutionSpec_SetCoreId)G(fSCI,GxpCapi_ExecutionSpec_SetCoreCount)
  G(fImpHA,GxpCapi_ImportBufferFromHostAddress)
  fBio bFlush=(fBio)dlsym(L,"GxpCapi_Buffer_Flush"); fBio bInval=(fBio)dlsym(L,"GxpCapi_Buffer_Invalidate");
  int r; void *rt=0,*spec=0,*dev=0,*wl=0,*lib=0,*fn=0,*es=0,*req=0,*buf=0;
  GxpCapi_Internal_Initialize(0);
  GxpCapi_CreateRuntime(&rt); GxpCapi_CreateDeviceSpec(&spec);
  r=GxpCapi_CreateDevice(rt,spec,&dev); if(r){P("CreateDevice=%d\n",r);return 2;}
  GxpCapi_CreateWakelockDescriptor(&wl); GxpCapi_AcquireWakelock(dev,wl);
  FILE* f=fopen("/data/adb/hwbridge/vscale.elf","rb"); if(!f){P("kein vscale.elf\n");return 2;}
  fseek(f,0,SEEK_END); long esz=ftell(f); fseek(f,0,SEEK_SET); void* elf=malloc(esz); if(fread(elf,1,esz,f)!=(size_t)esz)return 2; fclose(f);
  r=GxpCapi_OpenLibraryFromBuffer(dev,elf,(uint32_t)esz,&lib); if(r){P("OpenLib=%d\n",r);return 3;}
  r=GxpCapi_GetFunction(lib,"tpu_request_sync_submitter",&fn); if(r||!fn){P("GetFunction=%d\n",r);return 4;}

  /* dmabuf allozieren + mmappen (dmabuf-backed!) */
  int heap=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC); if(heap<0){P("open heap\n");return 5;}
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=4096; d.fd_flags=O_RDWR|O_CLOEXEC;
  if(ioctl(heap,DMA_HEAP_IOCTL_ALLOC,&d)<0){P("alloc\n");return 5;} int dfd=(int)d.fd;
  int32_t* p=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0); if(p==MAP_FAILED){P("mmap\n");return 6;}
  memset(p,0,4096); p[0]=3;p[1]=5;p[2]=10;p[3]=100;
  P("dmabuf fd=%d input (unsere mmap): [%d %d %d %d]\n",dfd,p[0],p[1],p[2],p[3]);

  /* PER HOST-ADRESSE importieren (Adresse ist dmabuf-backed) */
  r=GxpCapi_ImportBufferFromHostAddress(dev,p,16,2/*RW*/,&buf);
  P("ImportBufferFromHostAddress(dmabuf-mmap)=%d buf=%p\n",r,buf);
  if(r||!buf){P("Import fehlgeschlagen\n");return 7;}
  /* importierten Puffer explizit auf die DSP-Cores mappen (device-seitig sichtbar machen) */
  int(*mapAll)(void*,void*)=(int(*)(void*,void*))dlsym(L,"GxpCapi_MapBufferAllCores");
  int(*mapDev)(void*)=(int(*)(void*))dlsym(L,"GxpCapi_Buffer_Map");
  if(mapAll){ int mr=mapAll(dev,buf); P("MapBufferAllCores=%d\n",mr); }
  else if(mapDev){ int mr=mapDev(buf); P("Buffer_Map=%d\n",mr); }

  GxpCapi_CreateExecutionSpec(&es); GxpCapi_ExecutionSpec_SetCoreId(es,0); GxpCapi_ExecutionSpec_SetCoreCount(es,1);
  GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn); GxpCapi_Request_AppendBuffer(req,buf);
  if(bFlush){ int fr=bFlush(buf); P("Buffer_Flush(vor Lauf)=%d\n",fr); }
  P("--- RunSync (DSP rechnet vscale in unseren dmabuf) ---\n");
  r=GxpCapi_RunSync(dev,es,req); P("RunSync=%d\n",r);
  if(bInval){ int ir=bInval(buf); P("Buffer_Invalidate(nach Lauf)=%d\n",ir); }
  int64_t rv=0; GxpCapi_Request_GetReturnValue(req,&rv); P("rv=%lld\n",(long long)rv);
  void*(*gethp)(void*)=(void*(*)(void*))dlsym(L,"GxpCapi_Buffer_GetHostPointer");
  if(gethp){ int32_t* q=(int32_t*)gethp(buf); P("GetHostPointer=%p (unsere mmap=%p)\n",(void*)q,(void*)p);
    if((uintptr_t)q>0x10000) P("  via GetHostPointer: [%d %d %d %d]\n",q[0],q[1],q[2],q[3]); }
  P("output (unsere mmap): [%d %d %d %d]  erwartet [3 15 30 300]\n",p[0],p[1],p[2],p[3]);
  if(p[0]==3&&p[1]==15&&p[2]==30&&p[3]==300) P(">>> DSP-ZERO-COPY OK: der DSP rechnete direkt in unseren dmabuf. <<<\n");
  else P(">>> unerwartet <<<\n");
  return 0;
}
