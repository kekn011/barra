// gxpd3.c — Multi-Kernel DSP-Bruecke. Erweitert gxpd2 um MEHRERE eigene Callisto-
// Kernel (vscale, vadd, dot, matmul2x2, sum) + System-Lib (reverse_string).
// Jeder Kernel ist eine separat gepatchte inbuilt.elf (Body von
// tpu_request_sync_submitter ersetzt). Wir laden jede ELF via
// OpenLibraryFromBuffer in ein eigenes lib-Handle und mappen Client-Name->Funktion.
//
// Protokoll wie gxpd: Request {magic,name_len,in_size,name,in}; Resp {status,rv,exec_us,out_size,out}.
// Bauen: aarch64-linux-android31-clang gxpd3.c -o gxpd3 -ldl
// Start: LD_LIBRARY_PATH=$H:/system/lib64:/vendor/lib64 gxpd3 /opt/hwbridge/gxp.sock
//   Kernel-Verzeichnis via env GXPD_KDIR (default /data/adb/hwbridge).
// Selbsttest (kein Socket): gxpd3 x selftest
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <errno.h>

#define MAGIC    0x47585044u   /* 'GXPD' inline (Kopie, rueckwaerts-kompatibel) */
#define MAGIC_ZC 0x47585A32u   /* 'GXPZ' Zero-Copy-Session (Client schickt dmabuf-fds via SCM_RIGHTS) */
#define MAXNAME 128
#define MAXIO   (16u*1024*1024)
#define KFN     "tpu_request_sync_submitter"
#define ZC_MAXH   32           /* Handles pro Session */
#define ZC_MAXBUF 8            /* fds/Buffer pro Request */
#define LOG(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)

typedef int (*fI)(void*); typedef int (*fO)(void**);
typedef int (*fCD)(void*,void*,void**); typedef int (*fAW)(void*,void*);
typedef int (*fLL)(void*,const char*,void**); typedef int (*fOLB)(void*,void*,uint32_t,void**);
typedef int (*fCB)(void*,void*,uint32_t,void**); typedef void*(*fGHP)(void*);
typedef void (*fSC)(void*,int); typedef void (*fSFn)(void*,void*); typedef void (*fAB)(void*,void*);
typedef int (*fRun)(void*,void*,void*); typedef int (*fGRV)(void*,int64_t*);
typedef int (*fET)(void*,uint64_t*); typedef int (*fRel)(void*);

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

/* Zero-Copy-Import (optional; echte Signaturen aus Crash-Backtrace 20.8.):
 * ImportBufferFromFd(dev, CapiBufferOptions* opts, int fd, uint32 size, void** out);
 * MapBufferAllCores(buffer)  — EIN Argument (der Buffer), dispatcht per Ruecksprungadresse. */
typedef int (*fImpFd)(void*,void*,int,uint32_t,void**);
typedef int (*fMap)(void*);
typedef int (*fWait)(void*);
static fImpFd GxpCapi_ImportBufferFromFd;
static fMap   GxpCapi_MapBufferAllCores;
static fMap   GxpCapi_UnmapBufferAllCores;
static fRun   GxpCapi_RunAsync;        /* optional: async Submit (nicht-blockierend) */
static fWait  GxpCapi_Request_Wait;    /* optional: auf Completion warten */

static void *g_dev, *g_lib, *g_bopts, *g_es, *g_iopts, *g_iopts_c;   /* g_iopts=uncached, g_iopts_c=cacheable */

// Kernel-Tabelle: Client-Name -> ELF-Datei (im KDIR)
#define MAXKERN 64
static struct kern { const char* name; const char* file; void* lib; void* fn; }
g_kern[MAXKERN] = {
  {"vscale",    "vscale.elf",     0,0},
  {"vadd",      "ker_vadd.elf",   0,0},
  {"dot",       "ker_dot.elf",    0,0},
  {"matmul2x2", "ker_mm.elf",     0,0},
  {"sum",       "ker_sumloop.elf",0,0},
  {"softmul",   "ker_softmul2.elf",0,0},
  {"argmax",    "ker_argmax.elf", 0,0},   /* greedy Token-Auswahl (LLM-Decoding), 16.8. */
};
static int NKERN = 7;   /* feste Eintraege; danach Auto-Scan ker_*.elf im KDIR */

#include <dirent.h>
/* Auto-Scan: jede ker_<name>.elf im Kernel-Verzeichnis wird als Kernel <name> angeboten
 * (neue Kernel = Datei ablegen + gxpd neu starten, kein Rebuild). */
static void scan_kernels(const char* kdir){
  DIR* d=opendir(kdir); if(!d) return; struct dirent* e;
  while((e=readdir(d))&&NKERN<MAXKERN){
    const char* f=e->d_name; size_t n=strlen(f);
    if(strncmp(f,"ker_",4)||n<9||strcmp(f+n-4,".elf")) continue;
    int dup=0; for(int i=0;i<NKERN;i++) if(!strcmp(g_kern[i].file,f)) dup=1;
    if(dup) continue;
    char* name=strndup(f+4,n-8);            /* ker_<name>.elf -> <name> */
    g_kern[NKERN].name=name; g_kern[NKERN].file=strdup(f); g_kern[NKERN].lib=0; g_kern[NKERN].fn=0; NKERN++;
    LOG("gxpd3: Auto-Kernel '%s' aus %s\n",name,f);
  }
  closedir(d);
}

static struct { char name[MAXNAME]; void* fn; } g_sys[16]; static int g_nsys;

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
  /* Experiment 16.8.: Puffer-Optionen per env (der DSP ist auf dem Default-Puffer speicherlatenz-gebunden, ~0.8us/Element) */
  { typedef int (*fSet)(void*,int);
    const char* v;
    if((v=getenv("GXPD_CACHE"))){ fSet f=(fSet)dlsym(L,"GxpCapi_BufferOptions_SetCacheability"); int r=f?f(g_bopts,atoi(v)):-99; LOG("gxpd3: SetCacheability(%s)=%d\n",v,r); }
    if((v=getenv("GXPD_COH")))  { fSet f=(fSet)dlsym(L,"GxpCapi_BufferOptions_SetCoherence");    int r=f?f(g_bopts,atoi(v)):-99; LOG("gxpd3: SetCoherence(%s)=%d\n",v,r); }
    if((v=getenv("GXPD_SYNC"))) { fSet f=(fSet)dlsym(L,"GxpCapi_BufferOptions_SetAllowDmaBufferSyncOps"); int r=f?f(g_bopts,atoi(v)):-99; LOG("gxpd3: SetAllowDmaBufferSyncOps(%s)=%d\n",v,r); }
  }
  GxpCapi_CreateExecutionSpec(&g_es);
  GxpCapi_ExecutionSpec_SetCoreId(g_es,0); GxpCapi_ExecutionSpec_SetCoreCount(g_es,1);

  /* Zero-Copy-Import vorbereiten: eigene BufferOptions (RW + DmaBufferSyncOps) fuer importierte dmabufs */
  GxpCapi_ImportBufferFromFd=(fImpFd)dlsym(L,"GxpCapi_ImportBufferFromFd");
  GxpCapi_MapBufferAllCores=(fMap)dlsym(L,"GxpCapi_MapBufferAllCores");
  GxpCapi_UnmapBufferAllCores=(fMap)dlsym(L,"GxpCapi_UnmapBufferAllCores");
  GxpCapi_RunAsync=(fRun)dlsym(L,"GxpCapi_RunAsync");
  GxpCapi_Request_Wait=(fWait)dlsym(L,"GxpCapi_Request_Wait");
  { typedef void(*fSAT)(void*,int);
    fSAT sat=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAccessType");
    fSAT sds=(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetAllowDmaBufferSyncOps");
    fSAT sc =(fSAT)dlsym(L,"GxpCapi_BufferOptions_SetCacheability");
    /* uncached (Default, gut fuer viele kleine Async-Jobs) */
    GxpCapi_CreateBufferOptions(&g_iopts);   if(sat)sat(g_iopts,2);  if(sds)sds(g_iopts,1);
    const char* cv=getenv("GXPZ_CACHE");     if(cv&&sc)sc(g_iopts,atoi(cv));
    /* cacheable (per IMPORT-Flag waehlbar; ~30x fuer grosse Batch-Kernel) */
    GxpCapi_CreateBufferOptions(&g_iopts_c); if(sat)sat(g_iopts_c,2); if(sds)sds(g_iopts_c,1); if(sc)sc(g_iopts_c,1);
  }
  LOG("gxpd3: ZC import=%p mapAll=%p iopts=%p iopts_c=%p\n",(void*)GxpCapi_ImportBufferFromFd,(void*)GxpCapi_MapBufferAllCores,g_iopts,g_iopts_c);

  const char* kdir=getenv("GXPD_KDIR"); if(!kdir) kdir="/data/adb/hwbridge";
  scan_kernels(kdir);
  for(int i=0;i<NKERN;i++){
    char path[512]; snprintf(path,sizeof path,"%s/%s",kdir,g_kern[i].file);
    uint32_t sz=0; void* buf=slurp(path,&sz);
    if(!buf){ LOG("gxpd3: Kernel '%s' fehlt (%s) - uebersprungen\n",g_kern[i].name,path); continue; }
    r=GxpCapi_OpenLibraryFromBuffer(g_dev,buf,sz,&g_kern[i].lib); free(buf);
    if(r||!g_kern[i].lib){ LOG("gxpd3: Open('%s')=%d\n",g_kern[i].name,r); g_kern[i].lib=0; continue; }
    if(GxpCapi_GetFunction(g_kern[i].lib,KFN,&g_kern[i].fn)!=0) g_kern[i].fn=0;
    LOG("gxpd3: Kernel '%s' geladen (lib=%p fn=%p)\n",g_kern[i].name,g_kern[i].lib,g_kern[i].fn);
  }
  LOG("gxpd3: DSP bereit (device=%p syslib=%p)\n",g_dev,g_lib);
  return 0;
}

static void* resolve_fn(const char* name){
  for(int i=0;i<NKERN;i++) if(g_kern[i].fn && !strcmp(g_kern[i].name,name)) return g_kern[i].fn;
  for(int i=0;i<g_nsys;i++) if(!strcmp(g_sys[i].name,name)) return g_sys[i].fn;
  void* fn=0; int r=GxpCapi_GetFunction(g_lib,name,&fn);
  if(r||!fn){ LOG("GetFunction('%s')=%d\n",name,r); return 0; }
  if(g_nsys<16){ strncpy(g_sys[g_nsys].name,name,MAXNAME-1); g_sys[g_nsys].fn=fn; g_nsys++; }
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

/* n Bytes + evtl. SCM_RIGHTS-fds lesen (fds haengen am ersten Byte der Nachricht, wie bei tpud/TPZ2) */
static int recv_with_fds(int c, void* buf, long n, int* fds, int* nfd){
  char cbuf[CMSG_SPACE(sizeof(int)*ZC_MAXBUF)];
  struct iovec iov={.iov_base=buf,.iov_len=n};
  struct msghdr msg={.msg_iov=&iov,.msg_iovlen=1,.msg_control=cbuf,.msg_controllen=sizeof cbuf};
  *nfd=0;
  ssize_t r=recvmsg(c,&msg,MSG_WAITALL); if(r!=n) return -1;
  struct cmsghdr* cm=CMSG_FIRSTHDR(&msg);
  if(cm&&cm->cmsg_type==SCM_RIGHTS){ int k=(int)((cm->cmsg_len-CMSG_LEN(0))/sizeof(int));
    if(k>ZC_MAXBUF){ for(int i=0;i<k;i++) close(((int*)CMSG_DATA(cm))[i]); return -1; }
    memcpy(fds,CMSG_DATA(cm),k*sizeof(int)); *nfd=k; }
  return 0;
}

/* (A) Inline GXPD (Kopie) — Magic ist bereits gelesen; liest {nlen,in_size} und den Rest. */
static void serve_inline(int cfd){
  uint32_t two[2];
  if(read_n(cfd,two,sizeof two)) return;
  uint32_t nlen=two[0], in_size=two[1];
  if(nlen==0||nlen>=MAXNAME||in_size==0||in_size>MAXIO){ uint32_t st=1; write_n(cfd,&st,4); return; }
  char name[MAXNAME]; memset(name,0,sizeof name);
  if(read_n(cfd,name,nlen)) return; name[nlen]=0;
  uint8_t* in=malloc(in_size); uint8_t* out=malloc(in_size);
  if(!in||!out){ free(in);free(out); uint32_t st=8; write_n(cfd,&st,4); return; }
  if(read_n(cfd,in,in_size)){ free(in);free(out); return; }
  int64_t rv=0; uint32_t exec_us=0;
  int st=gxp_run(name,in,in_size,out,&rv,&exec_us);
  uint32_t status=(uint32_t)st; write_n(cfd,&status,4);
  if(st==0){ write_n(cfd,&rv,8); write_n(cfd,&exec_us,4); uint32_t osz=in_size; write_n(cfd,&osz,4); write_n(cfd,out,in_size); }
  free(in); free(out);
}

/* (B) Zero-Copy-Session GXPZ: der Client teilt seine dma_heap-dmabufs (SCM_RIGHTS), der DSP
 *     rechnet direkt darin — keine Datenkopie ueber den Socket. Cache-Kohaerenz macht der
 *     Client per DMA_BUF_IOCTL_SYNC auf dem eigenen fd (er besitzt die mmap).
 *   Header je Kommando (6*u32): magic, cmd, a, b, c, d
 *   cmd=1 IMPORT : a=nbuf + SCM_RIGHTS(nbuf fds); dann nbuf*u32 size -> u32 status[, nbuf*u32 handle]
 *   cmd=2 RUN    : a=name_len, b=nhandle; dann [name][nhandle*u32 handle]
 *                  -> u32 status, i64 rv, u32 exec_us   (Kernel rechnet in-place in den Puffern)
 *   cmd=3 RELEASE: a=nbuf; dann nbuf*u32 handle -> u32 status
 *   cmd=4 LIST   : -> u32 status=0, u32 nkern, dann je Kernel: u32 namelen,[name]
 *   cmd=5 SUBMIT : a=name_len, b=nhandle; dann [name][nhandle*u32 handle]  (RunAsync, nicht-blockierend)
 *                  -> u32 status, u32 token   (fuer WAIT)   -- ermoeglicht K Jobs gleichzeitig in-flight
 *   cmd=6 WAIT   : a=token -> u32 status, i64 rv, u32 exec_us  (Request_Wait auf den Token) */
#define ZC_INFLIGHT 32
typedef struct { int used; void* buf; int fd; uint32_t size; } ZBufG;
static void zc_release(ZBufG* z){
  if(!z->used) return;
  if(GxpCapi_UnmapBufferAllCores && z->buf) GxpCapi_UnmapBufferAllCores(z->buf);
  if(GxpCapi_ReleaseBuffer && z->buf) GxpCapi_ReleaseBuffer(z->buf);
  if(z->fd>=0) close(z->fd);
  memset(z,0,sizeof *z); z->fd=-1;
}
/* --- Abort-Recovery: der Firmware-Watchdog (zu langer Dispatch) invalidiert das Device;
 * danach schlaegt JEDE Operation fehl (auch IMPORT). gxpd3 erkennt das per Probe und
 * re-exect sich fuer eine frische Device-Initialisierung (Socket wird neu gebunden). --- */
static char** g_argv;
static int device_alive(void){
  void* b=0; if(GxpCapi_CreateBuffer(g_dev,g_bopts,16,&b)!=0||!b) return 0;
  void* hp=GxpCapi_Buffer_GetHostPointer(b); int ok=0;
  if(hp){ memcpy(hp,"0123456789ABCDEF",16); void* fn=resolve_fn("reverse_string");
    if(fn){ void* req=0; GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn); GxpCapi_Request_AppendBuffer(req,b);
      int r=GxpCapi_RunSync(g_dev,g_es,req); if(GxpCapi_ReleaseRequest)GxpCapi_ReleaseRequest(req); ok=(r==0); } }
  if(GxpCapi_ReleaseBuffer)GxpCapi_ReleaseBuffer(b);
  return ok;
}
static void guard_device(int rc){
  if(rc==0) return;
  if(device_alive()) return;                 /* nur ein Kernel-Fehler; Device lebt */
  /* Device ist abgestuerzt (Firmware-Watchdog). Prozess beenden -> alle fds schliessen
   * (Client bekommt sauberen Reset), der Supervisor startet einen FRISCHEN gxpd3, dessen
   * CreateDevice die Firmware zuverlaessig zuruecksetzt. Kein re-exec (fd-Leak/Loop-Gefahr). */
  LOG("[gxpd3] DSP-Device abgestuerzt (rc=%d) -> beende, Supervisor startet neu\n",rc);
  (void)g_argv;
  _exit(1);
}
static void serve_zc(int cfd, uint32_t* hdr, int* fds, int nfd){
  ZBufG h[ZC_MAXH]; memset(h,0,sizeof h); for(int i=0;i<ZC_MAXH;i++) h[i].fd=-1;
  void* inflight[ZC_INFLIGHT]; memset(inflight,0,sizeof inflight);   /* async: Token -> Request */
  long nrun=0;
  for(;;){
    uint32_t cmd=hdr[1], status=1;
    if(cmd==1){                                        /* IMPORT (hdr[3]!=0 -> cacheable) */
      uint32_t nbuf=hdr[2], sz[ZC_MAXBUF], hd[ZC_MAXBUF];
      void* useopts=(hdr[3]&&g_iopts_c)?g_iopts_c:g_iopts;
      if(!GxpCapi_ImportBufferFromFd||nbuf==0||nbuf>ZC_MAXBUF||(int)nbuf!=nfd||read_n(cfd,sz,nbuf*4)){ write_n(cfd,&status,4); goto next; }
      int ok=1;
      for(uint32_t i=0;i<nbuf;i++){
        int slot=-1; for(int k=0;k<ZC_MAXH;k++) if(!h[k].used){slot=k;break;}
        if(slot<0||sz[i]==0||sz[i]>MAXIO){ ok=0; break; }
        void* b=0; int dfd=dup(fds[i]);                /* die Lib bekommt ein dup; unser fd haelt den dmabuf am Leben */
        int ir=GxpCapi_ImportBufferFromFd(g_dev,useopts,dfd,sz[i],&b);
        if(ir||!b){ close(dfd); ok=0; LOG("[gxpd3] ImportBufferFromFd(size %u)=%d\n",sz[i],ir); break; }
        if(GxpCapi_MapBufferAllCores){ int mr=GxpCapi_MapBufferAllCores(b); if(mr){ if(GxpCapi_ReleaseBuffer)GxpCapi_ReleaseBuffer(b); ok=0; LOG("[gxpd3] MapBufferAllCores=%d\n",mr); break; } }
        h[slot].used=1; h[slot].buf=b; h[slot].fd=fds[i]; h[slot].size=sz[i]; fds[i]=-1;   /* fd uebernommen */
        hd[i]=(uint32_t)slot;
      }
      if(ok){ status=0; write_n(cfd,&status,4); write_n(cfd,hd,nbuf*4); }
      else   { write_n(cfd,&status,4); }
    } else if(cmd==2){                                 /* RUN name nhandle */
      uint32_t nlen=hdr[2], nh=hdr[3]; char name[MAXNAME]; uint32_t hd[ZC_MAXBUF];
      int64_t rv=0; uint32_t exec_us=0;
      if(nlen==0||nlen>=MAXNAME||nh==0||nh>ZC_MAXBUF||read_n(cfd,name,nlen)){ write_n(cfd,&status,4); write_n(cfd,&rv,8); write_n(cfd,&exec_us,4); goto next; }
      name[nlen]=0;
      if(read_n(cfd,hd,nh*4)){ write_n(cfd,&status,4); write_n(cfd,&rv,8); write_n(cfd,&exec_us,4); goto next; }
      void* fn=resolve_fn(name);
      int ok = (fn!=0);
      for(uint32_t i=0;ok&&i<nh;i++) if(hd[i]>=ZC_MAXH||!h[hd[i]].used) ok=0;
      if(ok){
        void* req=0; GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn);
        for(uint32_t i=0;i<nh;i++) GxpCapi_Request_AppendBuffer(req,h[hd[i]].buf);
        int r=GxpCapi_RunSync(g_dev,g_es,req);
        if(r==0){ status=0; GxpCapi_Request_GetReturnValue(req,&rv);
          if(GxpCapi_Request_GetExecutionTimeUs){ uint64_t us=0; if(GxpCapi_Request_GetExecutionTimeUs(req,&us)==0) exec_us=(uint32_t)us; } }
        else { status=7; LOG("[gxpd3] zc RunSync('%s')=%d\n",name,r); }
        if(GxpCapi_ReleaseRequest)GxpCapi_ReleaseRequest(req);
      } else status=4;
      nrun++; if(nrun<=3||nrun%1000==0) LOG("[gxpd3] zc run #%ld '%s' nh=%u status=%u %uus\n",nrun,name,nh,status,exec_us);
      write_n(cfd,&status,4); write_n(cfd,&rv,8); write_n(cfd,&exec_us,4);
      guard_device((int)status);
    } else if(cmd==3){                                 /* RELEASE */
      uint32_t nbuf=hdr[2], hd[ZC_MAXBUF];
      if(nbuf==0||nbuf>ZC_MAXBUF||read_n(cfd,hd,nbuf*4)){ write_n(cfd,&status,4); goto next; }
      for(uint32_t i=0;i<nbuf;i++) if(hd[i]<ZC_MAXH) zc_release(&h[hd[i]]);
      status=0; write_n(cfd,&status,4);
    } else if(cmd==4){                                 /* LIST */
      status=0; write_n(cfd,&status,4);
      uint32_t nk=0; for(int i=0;i<NKERN;i++) if(g_kern[i].fn) nk++;
      write_n(cfd,&nk,4);
      for(int i=0;i<NKERN;i++) if(g_kern[i].fn){ uint32_t l=(uint32_t)strlen(g_kern[i].name); write_n(cfd,&l,4); write_n(cfd,g_kern[i].name,l); }
    } else if(cmd==5){                                 /* SUBMIT name nhandle -> status, token */
      uint32_t nlen=hdr[2], nh=hdr[3]; char name[MAXNAME]; uint32_t hd[ZC_MAXBUF]; uint32_t token=0xffffffff;
      if(!GxpCapi_RunAsync||nlen==0||nlen>=MAXNAME||nh==0||nh>ZC_MAXBUF||read_n(cfd,name,nlen)){ write_n(cfd,&status,4); write_n(cfd,&token,4); goto next; }
      name[nlen]=0;
      if(read_n(cfd,hd,nh*4)){ write_n(cfd,&status,4); write_n(cfd,&token,4); goto next; }
      void* fn=resolve_fn(name); int ok=(fn!=0);
      for(uint32_t i=0;ok&&i<nh;i++) if(hd[i]>=ZC_MAXH||!h[hd[i]].used) ok=0;
      int slot=-1; if(ok){ for(int s=0;s<ZC_INFLIGHT;s++) if(!inflight[s]){slot=s;break;} if(slot<0) ok=0; }
      if(ok){
        void* req=0; GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn);
        for(uint32_t i=0;i<nh;i++) GxpCapi_Request_AppendBuffer(req,h[hd[i]].buf);
        int r=GxpCapi_RunAsync(g_dev,g_es,req);
        if(r==0){ status=0; inflight[slot]=req; token=(uint32_t)slot; }
        else { status=7; if(GxpCapi_ReleaseRequest)GxpCapi_ReleaseRequest(req); LOG("[gxpd3] RunAsync('%s')=%d\n",name,r); }
      } else status=4;
      write_n(cfd,&status,4); write_n(cfd,&token,4);
      guard_device((int)status);
    } else if(cmd==6){                                 /* WAIT token -> status, rv, exec_us */
      uint32_t token=hdr[2]; int64_t rv=0; uint32_t exec_us=0;
      if(!GxpCapi_Request_Wait||token>=ZC_INFLIGHT||!inflight[token]){ write_n(cfd,&status,4); write_n(cfd,&rv,8); write_n(cfd,&exec_us,4); goto next; }
      void* req=inflight[token]; int r=GxpCapi_Request_Wait(req);
      if(r==0){ status=0; GxpCapi_Request_GetReturnValue(req,&rv);
        if(GxpCapi_Request_GetExecutionTimeUs){ uint64_t us=0; if(GxpCapi_Request_GetExecutionTimeUs(req,&us)==0) exec_us=(uint32_t)us; } }
      else status=7;
      if(GxpCapi_ReleaseRequest)GxpCapi_ReleaseRequest(req); inflight[token]=0; nrun++;
      write_n(cfd,&status,4); write_n(cfd,&rv,8); write_n(cfd,&exec_us,4);
      guard_device((int)status);
    } else { write_n(cfd,&status,4); }
next:
    for(int i=0;i<nfd;i++) if(fds[i]>=0) close(fds[i]);
    nfd=0;
    if(recv_with_fds(cfd,hdr,24,fds,&nfd)) break;
    if(hdr[0]!=MAGIC_ZC) break;
  }
  for(int i=0;i<nfd;i++) if(fds[i]>=0) close(fds[i]);
  for(int s=0;s<ZC_INFLIGHT;s++) if(inflight[s]){ if(GxpCapi_Request_Wait)GxpCapi_Request_Wait(inflight[s]); if(GxpCapi_ReleaseRequest)GxpCapi_ReleaseRequest(inflight[s]); }
  int live=0; for(int k=0;k<ZC_MAXH;k++) if(h[k].used){ live++; zc_release(&h[k]); }
  LOG("[gxpd3] zc session ende: %ld runs, %d Handles freigegeben\n",nrun,live);
}

/* Dispatcher: erste 4 Bytes MIT Ancillary lesen (bei GXPZ haengen die IMPORT-fds am ersten Byte). */
static void serve(int cfd){
  uint32_t magic=0; int fds[ZC_MAXBUF]; int nfd=0;
  if(recv_with_fds(cfd,&magic,4,fds,&nfd)){ for(int i=0;i<nfd;i++) close(fds[i]); return; }
  if(magic==MAGIC){ for(int i=0;i<nfd;i++) close(fds[i]); serve_inline(cfd); return; }
  if(magic==MAGIC_ZC){
    uint32_t hdr[6]; hdr[0]=magic;
    if(read_n(cfd,&hdr[1],20)){ for(int i=0;i<nfd;i++) close(fds[i]); return; }
    serve_zc(cfd,hdr,fds,nfd); return;
  }
  LOG("bad magic %08x\n",magic);
  for(int i=0;i<nfd;i++) close(fds[i]);
}

// --- Selbsttest: laedt alle Kernel und rechnet je einen kanonischen Fall ---
static void put_i32(uint8_t* b,int i,int32_t v){ memcpy(b+4*i,&v,4); }
static int32_t get_i32(const uint8_t* b,int i){ int32_t v; memcpy(&v,b+4*i,4); return v; }
static void selftest(void){
  uint8_t in[64], out[64]; int64_t rv; uint32_t us; int st;
  // vadd [1,2,3,4]+[10,20,30,40]
  for(int i=0;i<4;i++){ put_i32(in,i,i+1); put_i32(in,i+4,10*(i+1)); }
  st=gxp_run("vadd",in,32,out,&rv,&us);
  LOG("[vadd]  st=%d -> %d %d %d %d (erw 11 22 33 44)\n",st,get_i32(out,0),get_i32(out,1),get_i32(out,2),get_i32(out,3));
  // dot [1..4].[5..8]=70
  for(int i=0;i<4;i++){ put_i32(in,i,i+1); put_i32(in,i+4,i+5); }
  st=gxp_run("dot",in,32,out,&rv,&us);
  LOG("[dot]   st=%d -> %d (erw 70)\n",st,get_i32(out,0));
  // matmul2x2 [[1,2],[3,4]]*[[5,6],[7,8]]
  int mv[8]={1,2,3,4,5,6,7,8}; for(int i=0;i<8;i++) put_i32(in,i,mv[i]); for(int i=8;i<12;i++) put_i32(in,i,0);
  st=gxp_run("matmul2x2",in,48,out,&rv,&us);
  LOG("[mm2x2] st=%d -> %d %d %d %d (erw 19 22 43 50)\n",st,get_i32(out,8),get_i32(out,9),get_i32(out,10),get_i32(out,11));
  // sum N=4 [4,5,6,7,8]=26
  put_i32(in,0,4); put_i32(in,1,5); put_i32(in,2,6); put_i32(in,3,7); put_i32(in,4,8);
  st=gxp_run("sum",in,20,out,&rv,&us);
  LOG("[sum]   st=%d -> %d (erw 26)\n",st,get_i32(out,0));
  // vscale [100,1,2,3]
  put_i32(in,0,100); put_i32(in,1,1); put_i32(in,2,2); put_i32(in,3,3);
  st=gxp_run("vscale",in,16,out,&rv,&us);
  LOG("[vscale]st=%d -> %d %d %d %d (erw 100 100 200 300)\n",st,get_i32(out,0),get_i32(out,1),get_i32(out,2),get_i32(out,3));
  // softmul: float 1.5 * 3.0 = 4.5 (out[2]); soft-float IEEE-754
  { uint32_t a=0x3FC00000u,b=0x40400000u,z=0; memcpy(in,&a,4);memcpy(in+4,&b,4);memcpy(in+8,&z,4);
    st=gxp_run("softmul",in,12,out,&rv,&us);
    uint32_t r; memcpy(&r,out+8,4); float fr; memcpy(&fr,&r,4);
    LOG("[softmul]st=%d -> 0x%08x (=%.4f, erw 0x40900000 =4.5)\n",st,r,fr); }
  // reverse_string (system lib)
  memcpy(in,"0123456789ABCDEF",16);
  st=gxp_run("reverse_string",in,16,out,&rv,&us);
  out[16]=0; LOG("[revstr]st=%d -> '%s'\n",st,(char*)out);
}

int main(int argc,char**argv){
  setvbuf(stderr,0,_IONBF,0);
  g_argv=argv;                               /* fuer re-exec bei Device-Abort */
  if(gxp_init()){ LOG("gxpd3: Init fehlgeschlagen\n"); return 1; }
  if(argc>2 && !strcmp(argv[2],"selftest")){ selftest(); return 0; }
  const char* sockpath=argc>1?argv[1]:"/opt/hwbridge/gxp.sock";
  int s=socket(AF_UNIX,SOCK_STREAM,0); if(s<0){ perror("socket"); return 1; }
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX;
  strncpy(a.sun_path,sockpath,sizeof a.sun_path-1);
  unlink(sockpath);
  if(bind(s,(struct sockaddr*)&a,sizeof a)<0){ perror("bind"); return 1; }
  chmod(sockpath,0666);
  if(listen(s,8)<0){ perror("listen"); return 1; }
  LOG("gxpd3: hoert auf %s\n",sockpath);
  for(;;){ int c=accept(s,0,0); if(c<0){ if(errno==EINTR)continue; perror("accept"); break; } serve(c); close(c); }
  return 0;
}
