// gxpd2.c — gxpd erweitert um EIGENE Rechen-Kernel via OpenLibraryFromBuffer.
// Laedt zusaetzlich eine User-ELF (z.B. vscale.elf) und stellt deren Funktion
// unter einem Client-Namen bereit. System-Lib-Pfad (reverse_string) bleibt erhalten.
//
// Protokoll (wie gxpd): Request {magic,name_len,in_size,name,in}; Response {status,rv,exec_us,out_size,out}.
//   name "vscale" -> User-Kernel: out[i]=in[i]*in[0] (int32), rechnet auf dem Callisto-DSP.
//   sonstige Namen -> inbuilt system lib (reverse_string ...).
//
// Bauen (Bionic/NDK):  aarch64-linux-android31-clang gxpd2.c -o gxpd2 -ldl
// Start (root, Metrics-Stub zwingend):
//   GXPD_USERLIB=/data/adb/hwbridge/vscale.elf \
//   LD_LIBRARY_PATH=/data/adb/hwbridge:/system/lib64:/vendor/lib64 gxpd2 /opt/hwbridge/gxp.sock
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <errno.h>

#define MAGIC 0x47585044u
#define MAXNAME 128
#define MAXIO   (16u*1024*1024)
#define LOG(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)

typedef int   (*fI)(void*);   typedef int (*fO)(void**);
typedef int   (*fCD)(void*,void*,void**);
typedef int   (*fAW)(void*,void*);
typedef int   (*fLL)(void*,const char*,void**);
typedef int   (*fOLB)(void*,void*,uint32_t,void**);
typedef int   (*fCB)(void*,void*,uint32_t,void**);
typedef void* (*fGHP)(void*);
typedef void  (*fSC)(void*,int);
typedef void  (*fSFn)(void*,void*);
typedef void  (*fAB)(void*,void*);
typedef int   (*fRun)(void*,void*,void*);
typedef int   (*fGRV)(void*,int64_t*);
typedef int   (*fET)(void*,uint64_t*);
typedef int   (*fRel)(void*);

static void *L;
#define G(t,n) static t n;
G(fI,GxpCapi_Internal_Initialize) G(fO,GxpCapi_CreateRuntime) G(fO,GxpCapi_CreateDeviceSpec)
G(fCD,GxpCapi_CreateDevice) G(fO,GxpCapi_CreateWakelockDescriptor) G(fAW,GxpCapi_AcquireWakelock)
G(fLL,GxpCapi_LoadSystemLibrary) G(fLL,GxpCapi_GetFunction) G(fO,GxpCapi_CreateBufferOptions)
G(fOLB,GxpCapi_OpenLibraryFromBuffer)
G(fCB,GxpCapi_CreateBuffer) G(fGHP,GxpCapi_Buffer_GetHostPointer)
G(fO,GxpCapi_CreateExecutionSpec) G(fSC,GxpCapi_ExecutionSpec_SetCoreId) G(fSC,GxpCapi_ExecutionSpec_SetCoreCount)
G(fO,GxpCapi_CreateRequest) G(fSFn,GxpCapi_Request_SetFunction) G(fAB,GxpCapi_Request_AppendBuffer)
G(fRun,GxpCapi_RunSync) G(fGRV,GxpCapi_Request_GetReturnValue) G(fET,GxpCapi_Request_GetExecutionTimeUs)
G(fRel,GxpCapi_ReleaseBuffer) G(fRel,GxpCapi_ReleaseRequest)
#define GET(n) do{ n=(void*)dlsym(L,#n); if(!n){LOG("dlsym FAIL %s\n",#n); return -1;} }while(0)

static void *g_dev, *g_lib, *g_ulib, *g_bopts, *g_es;
static void *g_vscale_fn;  // user kernel handle

static void* slurp(const char* p, uint32_t* sz){
  FILE* f=fopen(p,"rb"); if(!f) return 0;
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  void* b=malloc(n); if(fread(b,1,n,f)!=(size_t)n){ free(b); fclose(f); return 0; }
  fclose(f); *sz=(uint32_t)n; return b;
}

static int gxp_init(void){
  L=dlopen("libgxp.so",RTLD_NOW|RTLD_GLOBAL); if(!L){ LOG("dlopen libgxp FAIL: %s\n",dlerror()); return -1; }
  GET(GxpCapi_Internal_Initialize); GET(GxpCapi_CreateRuntime); GET(GxpCapi_CreateDeviceSpec);
  GET(GxpCapi_CreateDevice); GET(GxpCapi_CreateWakelockDescriptor); GET(GxpCapi_AcquireWakelock);
  GET(GxpCapi_LoadSystemLibrary); GET(GxpCapi_GetFunction); GET(GxpCapi_CreateBufferOptions);
  GET(GxpCapi_OpenLibraryFromBuffer);
  GET(GxpCapi_CreateBuffer); GET(GxpCapi_Buffer_GetHostPointer);
  GET(GxpCapi_CreateExecutionSpec); GET(GxpCapi_ExecutionSpec_SetCoreId); GET(GxpCapi_ExecutionSpec_SetCoreCount);
  GET(GxpCapi_CreateRequest); GET(GxpCapi_Request_SetFunction); GET(GxpCapi_Request_AppendBuffer);
  GET(GxpCapi_RunSync); GET(GxpCapi_Request_GetReturnValue);
  GxpCapi_Request_GetExecutionTimeUs=(fET)dlsym(L,"GxpCapi_Request_GetExecutionTimeUs");
  GxpCapi_ReleaseBuffer=(fRel)dlsym(L,"GxpCapi_ReleaseBuffer");
  GxpCapi_ReleaseRequest=(fRel)dlsym(L,"GxpCapi_ReleaseRequest");

  void *rt=0,*spec=0,*wl=0; int r;
  GxpCapi_Internal_Initialize(0);
  GxpCapi_CreateRuntime(&rt); GxpCapi_CreateDeviceSpec(&spec);
  r=GxpCapi_CreateDevice(rt,spec,&g_dev); if(r){ LOG("CreateDevice=%d\n",r); return -1; }
  GxpCapi_CreateWakelockDescriptor(&wl); GxpCapi_AcquireWakelock(g_dev,wl);
  r=GxpCapi_LoadSystemLibrary(g_dev,"gxp_inbuilt_lib",&g_lib); if(r){ LOG("LoadSystemLibrary=%d\n",r); return -1; }
  GxpCapi_CreateBufferOptions(&g_bopts);
  GxpCapi_CreateExecutionSpec(&g_es);
  GxpCapi_ExecutionSpec_SetCoreId(g_es,0); GxpCapi_ExecutionSpec_SetCoreCount(g_es,1);

  // User-Kernel laden (best effort)
  const char* up=getenv("GXPD_USERLIB"); if(!up) up="/data/adb/hwbridge/vscale.elf";
  uint32_t usz=0; void* ubuf=slurp(up,&usz);
  if(ubuf){
    r=GxpCapi_OpenLibraryFromBuffer(g_dev,ubuf,usz,&g_ulib);
    if(r==0 && g_ulib){
      // Kernel ist in tpu_request_sync_submitter einpatchiert
      if(GxpCapi_GetFunction(g_ulib,"tpu_request_sync_submitter",&g_vscale_fn)!=0) g_vscale_fn=0;
      LOG("gxpd2: User-Lib '%s' geladen (lib=%p vscale_fn=%p)\n",up,g_ulib,g_vscale_fn);
    } else LOG("gxpd2: OpenLibraryFromBuffer('%s')=%d\n",up,r);
    free(ubuf);
  } else LOG("gxpd2: keine User-Lib unter %s (nur System-Lib aktiv)\n",up);
  LOG("gxpd2: DSP bereit (device=%p syslib=%p)\n",g_dev,g_lib);
  return 0;
}

static struct { char name[MAXNAME]; void* fn; } g_fns[16]; static int g_nfns;
static void* resolve_fn(const char* name){
  if(!strcmp(name,"vscale")) return g_vscale_fn;   // eigener DSP-Kernel
  for(int i=0;i<g_nfns;i++) if(!strcmp(g_fns[i].name,name)) return g_fns[i].fn;
  void* fn=0; int r=GxpCapi_GetFunction(g_lib,name,&fn);
  if(r||!fn){ LOG("GetFunction('%s')=%d\n",name,r); return 0; }
  if(g_nfns<16){ strncpy(g_fns[g_nfns].name,name,MAXNAME-1); g_fns[g_nfns].fn=fn; g_nfns++; }
  return fn;
}

static int gxp_run(const char* name, const uint8_t* in, uint32_t in_size,
                   uint8_t* out, int64_t* rv, uint32_t* exec_us){
  void* fn=resolve_fn(name); if(!fn) return 4;
  void* buf=0; int r=GxpCapi_CreateBuffer(g_dev,g_bopts,in_size,&buf);
  if(r||!buf){ LOG("CreateBuffer(%u)=%d\n",in_size,r); return 5; }
  void* hp=GxpCapi_Buffer_GetHostPointer(buf);
  if(!hp){ if(GxpCapi_ReleaseBuffer)GxpCapi_ReleaseBuffer(buf); return 6; }
  memcpy(hp,in,in_size);
  void* req=0; GxpCapi_CreateRequest(&req);
  GxpCapi_Request_SetFunction(req,fn);
  GxpCapi_Request_AppendBuffer(req,buf);
  r=GxpCapi_RunSync(g_dev,g_es,req);
  if(r==0){
    *rv=-1; GxpCapi_Request_GetReturnValue(req,rv);
    if(GxpCapi_Request_GetExecutionTimeUs){ uint64_t us=0; if(GxpCapi_Request_GetExecutionTimeUs(req,&us)==0) *exec_us=(uint32_t)us; }
    memcpy(out,hp,in_size);
  } else LOG("RunSync('%s')=%d\n",name,r);
  if(GxpCapi_ReleaseRequest)GxpCapi_ReleaseRequest(req);
  if(GxpCapi_ReleaseBuffer)GxpCapi_ReleaseBuffer(buf);
  return r?7:0;
}

static int read_n(int fd, void* p, size_t n){ uint8_t* b=p; size_t g=0; while(g<n){ ssize_t k=read(fd,b+g,n-g); if(k<=0) return -1; g+=k; } return 0; }
static int write_n(int fd, const void* p, size_t n){ const uint8_t* b=p; size_t s=0; while(s<n){ ssize_t k=write(fd,b+s,n-s); if(k<=0) return -1; s+=k; } return 0; }

static void serve(int cfd){
  uint32_t hdr[3];
  if(read_n(cfd,hdr,sizeof hdr)) return;
  if(hdr[0]!=MAGIC){ LOG("bad magic %08x\n",hdr[0]); return; }
  uint32_t nlen=hdr[1], in_size=hdr[2];
  if(nlen==0||nlen>=MAXNAME||in_size==0||in_size>MAXIO){ uint32_t st=1; write_n(cfd,&st,4); return; }
  char name[MAXNAME]; memset(name,0,sizeof name);
  if(read_n(cfd,name,nlen)) return; name[nlen]=0;
  uint8_t* in=malloc(in_size); uint8_t* out=malloc(in_size);
  if(!in||!out){ free(in);free(out); uint32_t st=8; write_n(cfd,&st,4); return; }
  if(read_n(cfd,in,in_size)){ free(in);free(out); return; }
  int64_t rv=0; uint32_t exec_us=0;
  int st=gxp_run(name,in,in_size,out,&rv,&exec_us);
  uint32_t status=(uint32_t)st;
  write_n(cfd,&status,4);
  if(st==0){ write_n(cfd,&rv,8); write_n(cfd,&exec_us,4); uint32_t osz=in_size; write_n(cfd,&osz,4); write_n(cfd,out,in_size); }
  free(in); free(out);
}

int main(int argc,char**argv){
  setvbuf(stderr,0,_IONBF,0);
  const char* sockpath=argc>1?argv[1]:"/opt/hwbridge/gxp.sock";
  if(gxp_init()){ LOG("gxpd2: Init fehlgeschlagen\n"); return 1; }
  int s=socket(AF_UNIX,SOCK_STREAM,0); if(s<0){ perror("socket"); return 1; }
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX;
  strncpy(a.sun_path,sockpath,sizeof a.sun_path-1);
  unlink(sockpath);
  if(bind(s,(struct sockaddr*)&a,sizeof a)<0){ perror("bind"); return 1; }
  chmod(sockpath,0666);
  if(listen(s,8)<0){ perror("listen"); return 1; }
  LOG("gxpd2: hoert auf %s\n",sockpath);
  for(;;){ int c=accept(s,0,0); if(c<0){ if(errno==EINTR)continue; perror("accept"); break; } serve(c); close(c); }
  return 0;
}
