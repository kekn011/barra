/* gxp-zc3 — DSP-ZERO-COPY, 3. Anlauf (bereinigt).
 * Erkenntnis aus gxp-zc2: ImportBufferFromHostAddress(dmabuf-mmap)=0 liefert ein
 * Buffer-Handle; der Crash kam vom experimentellen MapBufferAllCores-Aufruf
 * (falsche Signatur -> "number of gxp cores exceeds the maximum" + SIGSEGV).
 * Hier: KEIN Map-Aufruf — importieren, AppendBuffer, RunSync, aus derselben mmap lesen.
 * Modi: gxp-zc3 ha   = ImportBufferFromHostAddress(dev, mmap-ptr, size, RW)
 *       gxp-zc3 fd   = ImportBufferFromFd(dev, dmabuf-fd, size, RW)
 * Bionic/NDK, als root, LD_LIBRARY_PATH mit Metrics-Stub zuerst. */
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
/* dmabuf-Cache-Maintenance (CPU<->Device-Kohaerenz auf UNSEREM dmabuf) */
struct dma_buf_sync { __u64 flags; };
#define DMA_BUF_IOCTL_SYNC _IOW('b', 0, struct dma_buf_sync)
#define DBS_READ (1ULL<<0)
#define DBS_WRITE (1ULL<<1)
#define DBS_RW (DBS_READ|DBS_WRITE)
#define DBS_START 0ULL
#define DBS_END (1ULL<<2)
static void dbsync(int fd,__u64 flags){ struct dma_buf_sync s={flags}; ioctl(fd,DMA_BUF_IOCTL_SYNC,&s); }

static void* L;
#define G(t,n) t n=(t)dlsym(L,#n); if(!n){P("MISSING %s\n",#n);return 1;}
typedef int(*f1)(void**); typedef int(*fCD)(void*,void*,void**); typedef int(*fOLB)(void*,void*,uint32_t,void**);
typedef int(*fLL)(void*,const char*,void**); typedef void(*fSF)(void*,void*); typedef void(*fAB)(void*,void*);
typedef int(*fRun)(void*,void*,void*); typedef int(*fGRV)(void*,int64_t*); typedef int(*fAW)(void*,void*);
typedef void(*fSCI)(void*,int);
/* Echte Signatur (aus Crash-Backtrace CapiImportBuffer(dev&,opts&,int fd,uint size)):
 * das 2. Argument ist CapiBufferOptions*, NICHT der access-int. */
typedef int(*fImpHA)(void*,void*,void*,uint32_t,void**);  /* ImportBufferFromHostAddress(dev,opts,host,size,&out) */
typedef int(*fImpFd)(void*,void*,int,uint32_t,void**);    /* ImportBufferFromFd(dev,opts,fd,size,&out) */
typedef void(*fSAT)(void*,int);                           /* BufferOptions_SetAccessType(opts,type) */
typedef void(*fSDS)(void*,int);                           /* BufferOptions_SetAllowDmaBufferSyncOps(opts,bool) */
typedef int(*fBio)(void*);

int main(int argc, char** argv){
  const char* mode = (argc>1)? argv[1] : "ha";
  setvbuf(stderr,0,_IONBF,0);
  L=dlopen("libgxp.so",RTLD_NOW|RTLD_GLOBAL); if(!L){P("dlopen FAIL\n");return 1;}
  G(f1,GxpCapi_Internal_Initialize)G(f1,GxpCapi_CreateRuntime)G(f1,GxpCapi_CreateDeviceSpec)
  G(fCD,GxpCapi_CreateDevice)G(fOLB,GxpCapi_OpenLibraryFromBuffer)G(fLL,GxpCapi_GetFunction)
  G(f1,GxpCapi_CreateRequest)G(fSF,GxpCapi_Request_SetFunction)G(fAB,GxpCapi_Request_AppendBuffer)
  G(f1,GxpCapi_CreateExecutionSpec)G(fRun,GxpCapi_RunSync)G(fGRV,GxpCapi_Request_GetReturnValue)
  G(f1,GxpCapi_CreateWakelockDescriptor)G(fAW,GxpCapi_AcquireWakelock)
  G(fSCI,GxpCapi_ExecutionSpec_SetCoreId)G(fSCI,GxpCapi_ExecutionSpec_SetCoreCount)
  fImpHA impHA=(fImpHA)dlsym(L,"GxpCapi_ImportBufferFromHostAddress");
  fImpFd impFd=(fImpFd)dlsym(L,"GxpCapi_ImportBufferFromFd");
  int(*mkOpts)(void**)=(int(*)(void**))dlsym(L,"GxpCapi_CreateBufferOptions");
  fSAT setAccess=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAccessType");
  fSDS setDmaSync=(fSDS)dlsym(L,"GxpCapi_BufferOptions_SetAllowDmaBufferSyncOps");
  fBio bFlush=(fBio)dlsym(L,"GxpCapi_Buffer_Flush"); fBio bInval=(fBio)dlsym(L,"GxpCapi_Buffer_Invalidate");
  P("mode=%s impHA=%p impFd=%p flush=%p inval=%p\n",mode,(void*)impHA,(void*)impFd,(void*)bFlush,(void*)bInval);

  int r; void *rt=0,*spec=0,*dev=0,*wl=0,*lib=0,*fn=0,*es=0,*req=0,*buf=0;
  GxpCapi_Internal_Initialize(0);
  GxpCapi_CreateRuntime(&rt); GxpCapi_CreateDeviceSpec(&spec);
  r=GxpCapi_CreateDevice(rt,spec,&dev); if(r){P("CreateDevice=%d\n",r);return 2;}
  GxpCapi_CreateWakelockDescriptor(&wl); GxpCapi_AcquireWakelock(dev,wl);
  FILE* f=fopen("/data/adb/hwbridge/vscale.elf","rb"); if(!f){P("kein vscale.elf\n");return 2;}
  fseek(f,0,SEEK_END); long esz=ftell(f); fseek(f,0,SEEK_SET); void* elf=malloc(esz); if(fread(elf,1,esz,f)!=(size_t)esz)return 2; fclose(f);
  r=GxpCapi_OpenLibraryFromBuffer(dev,elf,(uint32_t)esz,&lib); if(r){P("OpenLib=%d\n",r);return 3;}
  r=GxpCapi_GetFunction(lib,"tpu_request_sync_submitter",&fn); if(r||!fn){P("GetFunction=%d\n",r);return 4;}

  /* dmabuf allozieren + mmappen */
  int heap=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC); if(heap<0){P("open heap\n");return 5;}
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=4096; d.fd_flags=O_RDWR|O_CLOEXEC;
  if(ioctl(heap,DMA_HEAP_IOCTL_ALLOC,&d)<0){P("alloc\n");return 5;} int dfd=(int)d.fd;
  int32_t* p=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0); if(p==MAP_FAILED){P("mmap\n");return 6;}
  dbsync(dfd,DBS_START|DBS_WRITE);
  memset(p,0,4096); p[0]=3;p[1]=5;p[2]=10;p[3]=100;
  dbsync(dfd,DBS_END|DBS_WRITE);   /* CPU-Schreiben nach DRAM sichtbar machen */
  P("dmabuf fd=%d input (unsere mmap): [%d %d %d %d]\n",dfd,p[0],p[1],p[2],p[3]);

  /* BufferOptions bauen (2. Import-Argument) */
  void* iopts=0;
  if(mkOpts) mkOpts(&iopts);
  if(setAccess) setAccess(iopts,2/*RW*/);
  if(setDmaSync) setDmaSync(iopts,1);
  P("BufferOptions=%p (setAccess=%p setDmaSync=%p)\n",iopts,(void*)setAccess,(void*)setDmaSync);

  if(!strcmp(mode,"fd")){
    if(!impFd){P("ImportBufferFromFd fehlt\n");return 7;}
    r=impFd(dev,iopts,dfd,4096,&buf);
    P("ImportBufferFromFd(dev,opts,fd=%d,4096)=%d buf=%p\n",dfd,r,buf);
  } else {
    if(!impHA){P("ImportBufferFromHostAddress fehlt\n");return 7;}
    r=impHA(dev,iopts,p,4096,&buf);
    P("ImportBufferFromHostAddress(dev,opts,%p,4096)=%d buf=%p\n",(void*)p,r,buf);
  }
  if(r||!buf){P("Import fehlgeschlagen\n");return 7;}

  /* importierten Puffer device-seitig mappen — Map-Funktionen nehmen EIN Arg (den Buffer),
   * NICHT (dev,buf): die drei Wrapper sind byte-identisch und pruefen nur x0. */
  int(*mapAll)(void*)=(int(*)(void*))dlsym(L,"GxpCapi_MapBufferAllCores");
  int(*mapOne)(void*)=(int(*)(void*))dlsym(L,"GxpCapi_MapBuffer");
  int(*bmap)(void*)=(int(*)(void*))dlsym(L,"GxpCapi_Buffer_Map");
  const char* mm=getenv("GXP_MAP"); if(!mm) mm="all";
  int mr=-1;
  if(!strcmp(mm,"all")&&mapAll){ mr=mapAll(buf); P("MapBufferAllCores(buf)=%d\n",mr); }
  else if(!strcmp(mm,"one")&&mapOne){ mr=mapOne(buf); P("MapBuffer(buf)=%d\n",mr); }
  else if(!strcmp(mm,"bm")&&bmap){ mr=bmap(buf); P("Buffer_Map(buf)=%d\n",mr); }
  else P("kein Map-Aufruf (GXP_MAP=%s)\n",mm);

  P("step: CreateExecutionSpec\n"); GxpCapi_CreateExecutionSpec(&es);
  P("step: SetCoreId/Count\n"); GxpCapi_ExecutionSpec_SetCoreId(es,0); GxpCapi_ExecutionSpec_SetCoreCount(es,1);
  P("step: CreateRequest\n"); GxpCapi_CreateRequest(&req);
  P("step: SetFunction\n"); GxpCapi_Request_SetFunction(req,fn);
  P("step: AppendBuffer\n"); GxpCapi_Request_AppendBuffer(req,buf);
  P("step: append ok\n");
  (void)bFlush;(void)bInval;   /* GXP-eigene Flush/Invalidate crashen auf importierten Puffern -> dmabuf-Sync nutzen */
  P("--- RunSync ---\n");
  r=GxpCapi_RunSync(dev,es,req); P("RunSync=%d\n",r);
  dbsync(dfd,DBS_START|DBS_READ);   /* Device-Schreiben in unsere CPU-Sicht holen */
  int64_t rv=0; GxpCapi_Request_GetReturnValue(req,&rv); P("rv=%lld\n",(long long)rv);
  P("output (unsere mmap): [%d %d %d %d]  vscale-Erwartung [3 15 30 300]\n",p[0],p[1],p[2],p[3]);
  dbsync(dfd,DBS_END|DBS_READ);
  int ok1 = (p[1]==15&&p[2]==30&&p[3]==300);
  if(ok1) P(">>> [A] DSP-ZERO-COPY OK: DSP rechnete direkt in unserem dmabuf. <<<\n");
  else { P(">>> [A] fehlgeschlagen <<<\n"); return 8; }

  /* [B] Gegenprobe echte Teilung: ZWEITE unabhaengige mmap desselben fd, neues
   * Muster reinschreiben, DSP erneut rechnen, Ergebnis via ERSTE mmap lesen. */
  int32_t* p2=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0);
  if(p2==MAP_FAILED){P("zweite mmap fail\n");return 8;}
  P("[B] zweite unabhaengige mmap @ %p (erste @ %p)\n",(void*)p2,(void*)p);
  dbsync(dfd,DBS_START|DBS_WRITE);
  p2[0]=2; p2[1]=7; p2[2]=9; p2[3]=11;   /* via ZWEITE mmap geschrieben */
  dbsync(dfd,DBS_END|DBS_WRITE);
  r=GxpCapi_RunSync(dev,es,req);
  dbsync(dfd,DBS_START|DBS_READ);
  P("[B] nach Schreiben via 2. mmap [2 7 9 11], RunSync=%d, gelesen via 1. mmap: [%d %d %d %d]  Erw. [2 14 18 22]\n",
    r,p[0],p[1],p[2],p[3]);
  dbsync(dfd,DBS_END|DBS_READ);
  int ok2 = (p[1]==14&&p[2]==18&&p[3]==22);
  if(ok2) P(">>> [B] BIDIREKTIONALE TEILUNG OK: DSP verarbeitete den ueber die 2. mmap geschriebenen Input, Ergebnis in der 1. mmap sichtbar. <<<\n");
  else P(">>> [B] Gegenprobe unerwartet <<<\n");
  munmap(p2,4096);

  P("\n=== FAZIT: DSP-Zero-Copy in externem, teilbarem dma_heap-dmabuf: %s ===\n", (ok1&&ok2)?"VERIFIZIERT":"NICHT bestanden");
  return (ok1&&ok2)?0:8;
}
