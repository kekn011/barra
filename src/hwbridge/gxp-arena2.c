/* gxp-arena2 — VOLLER Zero-Copy-Beweis ueber die GXP-Arena.
 * Erkenntnis aus gxp-arena: der DSP-Puffer ist eine Sub-Allokation der 16-MB-
 * Shared-Heap-Arena, und die Arena IST ein echter dmabuf (/dmabuf:system, ~16 MB)
 * im Prozess. Dieser Test beweist Zero-Copy so, wie ihn ein FREMDER Chip/Prozess
 * saehe: wir finden den Arena-fd, mmappen ihn UNABHAENGIG (zweite Mapping), und
 * lesen den DSP-Output aus DIESER zweiten Mapping.
 *   1) Puffer anlegen, DSP vscale -> Muster [3,15,30,300] in die Arena schreiben
 *   2) Arena-fd finden (groesster /dmabuf:system, ~16 MB) + Offset des HostPointer
 *      in /proc/self/maps
 *   3) Arena-fd ein ZWEITES Mal mmappen (wie nach SCM_RIGHTS im Fremdprozess)
 *   4) an OFFSET das Muster [3,15,30,300] lesen  == DSP-Output  -> Read-Zero-Copy
 *   5) ueber die zweite Mapping neuen Input schreiben, DSP erneut rechnen,
 *      via HostPointer zuruecklesen                            -> Write-Zero-Copy
 * Bionic/NDK, root, Metrics-Stub zuerst im LD_LIBRARY_PATH. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)

static void* L;
#define G(t,n) t n=(t)dlsym(L,#n); if(!n){P("MISSING %s\n",#n);return 1;}
typedef int(*f1)(void**); typedef int(*fCD)(void*,void*,void**); typedef int(*fOLB)(void*,void*,uint32_t,void**);
typedef int(*fLL)(void*,const char*,void**); typedef void(*fSF)(void*,void*); typedef void(*fAB)(void*,void*);
typedef int(*fRun)(void*,void*,void*); typedef int(*fGRV)(void*,int64_t*); typedef int(*fAW)(void*,void*);
typedef void(*fSCI)(void*,int); typedef int(*fCB)(void*,void*,uint32_t,void**); typedef void*(*fGHP)(void*);

#define MAXDB 32
struct dbuf { int fd; long size; char tgt[64]; };
/* alle dmabuf-fds des Prozesses sammeln */
static int collect_dmabuf_fds(struct dbuf* out,int cap){
  DIR* d=opendir("/proc/self/fd"); if(!d) return 0;
  struct dirent* e; int n=0;
  while((e=readdir(d)) && n<cap){
    if(e->d_name[0]<'0'||e->d_name[0]>'9') continue;
    char path[64],tgt[256]; snprintf(path,sizeof path,"/proc/self/fd/%s",e->d_name);
    ssize_t k=readlink(path,tgt,sizeof tgt-1); if(k<=0) continue; tgt[k]=0;
    if(strstr(tgt,"dmabuf")){
      struct stat st; long sz=0; int fd=atoi(e->d_name); if(fstat(fd,&st)==0) sz=st.st_size;
      out[n].fd=fd; out[n].size=sz; snprintf(out[n].tgt,sizeof out[n].tgt,"%s",tgt); n++;
    }
  }
  closedir(d); return n;
}

int main(void){
  setvbuf(stderr,0,_IONBF,0);
  L=dlopen("libgxp.so",RTLD_NOW|RTLD_GLOBAL); if(!L){P("dlopen FAIL\n");return 1;}
  G(f1,GxpCapi_Internal_Initialize)G(f1,GxpCapi_CreateRuntime)G(f1,GxpCapi_CreateDeviceSpec)
  G(fCD,GxpCapi_CreateDevice)G(fOLB,GxpCapi_OpenLibraryFromBuffer)G(fLL,GxpCapi_GetFunction)
  G(f1,GxpCapi_CreateRequest)G(fSF,GxpCapi_Request_SetFunction)G(fAB,GxpCapi_Request_AppendBuffer)
  G(f1,GxpCapi_CreateExecutionSpec)G(fRun,GxpCapi_RunSync)G(fGRV,GxpCapi_Request_GetReturnValue)
  G(f1,GxpCapi_CreateWakelockDescriptor)G(fAW,GxpCapi_AcquireWakelock)
  G(fSCI,GxpCapi_ExecutionSpec_SetCoreId)G(fSCI,GxpCapi_ExecutionSpec_SetCoreCount)
  G(f1,GxpCapi_CreateBufferOptions)G(fCB,GxpCapi_CreateBuffer)G(fGHP,GxpCapi_Buffer_GetHostPointer)

  int r; void *rt=0,*spec=0,*dev=0,*wl=0,*lib=0,*fn=0,*es=0,*req=0,*buf=0,*bopts=0;
  GxpCapi_Internal_Initialize(0);
  GxpCapi_CreateRuntime(&rt); GxpCapi_CreateDeviceSpec(&spec);
  r=GxpCapi_CreateDevice(rt,spec,&dev); if(r){P("CreateDevice=%d\n",r);return 2;}
  GxpCapi_CreateWakelockDescriptor(&wl); GxpCapi_AcquireWakelock(dev,wl);
  FILE* f=fopen("/data/adb/hwbridge/vscale.elf","rb"); if(!f){P("kein vscale.elf\n");return 2;}
  fseek(f,0,SEEK_END); long esz=ftell(f); fseek(f,0,SEEK_SET); void* elf=malloc(esz); if(fread(elf,1,esz,f)!=(size_t)esz)return 2; fclose(f);
  r=GxpCapi_OpenLibraryFromBuffer(dev,elf,(uint32_t)esz,&lib); if(r){P("OpenLib=%d\n",r);return 3;}
  r=GxpCapi_GetFunction(lib,"tpu_request_sync_submitter",&fn); if(r||!fn){P("GetFunction=%d\n",r);return 4;}

  GxpCapi_CreateBufferOptions(&bopts);
  r=GxpCapi_CreateBuffer(dev,bopts,4096,&buf); if(r||!buf){P("CreateBuffer=%d\n",r);return 5;}
  int32_t* hp=(int32_t*)GxpCapi_Buffer_GetHostPointer(buf);
  P("HostPointer=%p\n",(void*)hp);
  hp[0]=3;hp[1]=5;hp[2]=10;hp[3]=100;

  GxpCapi_CreateExecutionSpec(&es); GxpCapi_ExecutionSpec_SetCoreId(es,0); GxpCapi_ExecutionSpec_SetCoreCount(es,1);
  GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn); GxpCapi_Request_AppendBuffer(req,buf);
  r=GxpCapi_RunSync(dev,es,req);
  P("[1] DSP RunSync=%d, HostPointer-Output=[%d %d %d %d]\n",r,hp[0],hp[1],hp[2],hp[3]);

  /* [1b] HostPointer (Tag-Byte abstreifen) in maps verorten -> welches Backing? */
  uintptr_t utag=(uintptr_t)hp & 0x00FFFFFFFFFFFFFFULL;
  P("[1b] HostPointer untagged=0x%lx — Backing laut /proc/self/maps:\n",(unsigned long)utag);
  { FILE* m=fopen("/proc/self/maps","r"); if(m){ char line[512]; uintptr_t a,b;
      while(fgets(line,sizeof line,m)){ if(sscanf(line,"%lx-%lx",&a,&b)!=2) continue;
        if(utag>=a&&utag<b){ char*nl=strchr(line,'\n'); if(nl)*nl=0;
          P("     %s\n     Offset in Mapping=0x%lx\n",line,(unsigned long)(utag-a)); break; } }
      fclose(m); } }

  /* [2] alle dmabuf-fds sammeln */
  struct dbuf dbs[MAXDB]; int ndb=collect_dmabuf_fds(dbs,MAXDB);
  P("[2] %d dmabuf-fds im Prozess:\n",ndb);
  for(int i=0;i<ndb;i++) P("    fd %-3d %-20s size %ld\n",dbs[i].fd,dbs[i].tgt,dbs[i].size);

  /* [3]+[4] jeden dmabuf UNABHAENGIG mappen und nach dem Muster [3,15,30,300] suchen */
  int hitfd=-1; long found=-1; uint8_t* arena=0; long asz=0;
  for(int i=0;i<ndb;i++){
    if(dbs[i].size<16) continue;
    uint8_t* m=mmap(0,dbs[i].size,PROT_READ|PROT_WRITE,MAP_SHARED,dbs[i].fd,0);
    if(m==MAP_FAILED) continue;
    for(long off=0; off+16<=dbs[i].size; off+=4){
      int32_t* q=(int32_t*)(m+off);
      if(q[0]==3&&q[1]==15&&q[2]==30&&q[3]==300){ hitfd=dbs[i].fd; found=off; arena=m; asz=dbs[i].size; break; }
    }
    if(hitfd>=0) break;
    munmap(m,dbs[i].size);
  }
  if(hitfd<0){ P("[4] Muster in KEINEM unabhaengig gemappten dmabuf gefunden -> KEIN Zero-Copy\n"); return 7; }
  P("[3] Treffer: DSP-Output liegt in dmabuf-fd %d (size %ld, %ld MB)\n",hitfd,asz,asz/1048576);
  P("[4] Muster @ Offset %ld (0x%lx) in der UNABHAENGIGEN Mapping: [%d %d %d %d]  == DSP-Output\n",
    found,(unsigned long)found,((int32_t*)(arena+found))[0],((int32_t*)(arena+found))[1],
    ((int32_t*)(arena+found))[2],((int32_t*)(arena+found))[3]);
  P("    >>> READ-ZERO-COPY OK: unabhaengige dmabuf-Mapping sieht den DSP-Output byte-identisch. <<<\n");

  /* [5] Rueckrichtung: neuen Input ueber die ZWEITE Mapping schreiben, DSP erneut rechnen */
  int32_t* w=(int32_t*)(arena+found); w[0]=2; w[1]=7; w[2]=9; w[3]=11;
  msync(arena,asz,MS_SYNC);
  r=GxpCapi_RunSync(dev,es,req);
  P("[5] nach Schreiben via Arena-Mapping [2 7 9 11], DSP RunSync=%d\n",r);
  P("    HostPointer-Output=[%d %d %d %d]  vscale-Erw. [2 14 18 22]\n",hp[0],hp[1],hp[2],hp[3]);
  if(hp[1]==14&&hp[2]==18&&hp[3]==22)
    P("    >>> WRITE-ZERO-COPY OK: DSP verarbeitete den ueber die Fremd-Mapping geschriebenen Input. <<<\n");
  else P("    >>> Rueckrichtung unerwartet <<<\n");

  munmap(arena,asz);
  return 0;
}
