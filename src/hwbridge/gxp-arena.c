/* gxp-arena — Arena-Export-Pfad fuer DSP-Zero-Copy.
 * GXP-Einzelpuffer sind Sub-Allokationen einer 16-MB-Arena; GetBufferFd liefert -1
 * (kein eigener fd). Die Arena selbst ist EIN dmabuf im Prozess. Dieses Tool:
 *  1) legt einen Puffer an, laesst den DSP vscale hineinrechnen (GetHostPointer),
 *  2) verortet den Host-Pointer in /proc/self/maps (welche Mapping-Range, Backing),
 *  3) listet /proc/self/fd nach dmabuf/dma_heap-fds + Groessen,
 *  4) meldet Offset = hp - mapping_start (Bruecken-Rezept: Arena-fd + Offset in GPU/TPU importieren).
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
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)

static void* L;
#define G(t,n) t n=(t)dlsym(L,#n); if(!n){P("MISSING %s\n",#n);return 1;}
typedef int(*f1)(void**); typedef int(*fCD)(void*,void*,void**); typedef int(*fOLB)(void*,void*,uint32_t,void**);
typedef int(*fLL)(void*,const char*,void**); typedef void(*fSF)(void*,void*); typedef void(*fAB)(void*,void*);
typedef int(*fRun)(void*,void*,void*); typedef int(*fGRV)(void*,int64_t*); typedef int(*fAW)(void*,void*);
typedef void(*fSCI)(void*,int); typedef int(*fCB)(void*,void*,uint32_t,void**); typedef void*(*fGHP)(void*);

static void locate(uintptr_t hp){
  FILE* m=fopen("/proc/self/maps","r"); if(!m){P("kein maps\n");return;}
  char line[512];
  while(fgets(line,sizeof line,m)){
    uintptr_t a,b; if(sscanf(line,"%lx-%lx",&a,&b)!=2) continue;
    if(hp>=a && hp<b){
      /* Pfad ist ab Spalte ~73; robust: letztes Feld */
      char* nl=strchr(line,'\n'); if(nl)*nl=0;
      P(">>> HostPointer %p liegt in Mapping %lx-%lx (len %lu KB)\n",(void*)hp,a,b,(b-a)/1024);
      P("    maps-Zeile: %s\n",line);
      P("    OFFSET in dieser Mapping: %lu bytes (0x%lx)\n",(unsigned long)(hp-a),(unsigned long)(hp-a));
      break;
    }
  }
  fclose(m);
}

static void list_dmabuf_fds(void){
  DIR* d=opendir("/proc/self/fd"); if(!d){P("kein fd-dir\n");return;}
  struct dirent* e;
  P("--- dmabuf/dma_heap-fds im Prozess ---\n");
  while((e=readdir(d))){
    if(e->d_name[0]<'0'||e->d_name[0]>'9') continue;
    char path[64],tgt[256]; snprintf(path,sizeof path,"/proc/self/fd/%s",e->d_name);
    ssize_t n=readlink(path,tgt,sizeof tgt-1); if(n<=0) continue; tgt[n]=0;
    if(strstr(tgt,"dmabuf")||strstr(tgt,"dma_heap")||strstr(tgt,"gxp")){
      struct stat st; off_t sz=0; if(fstat(atoi(e->d_name),&st)==0) sz=st.st_size;
      P("    fd %-3s -> %s  (size %ld)\n",e->d_name,tgt,(long)sz);
    }
  }
  closedir(d);
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
  P("CreateBuffer ok, HostPointer=%p\n",(void*)hp);
  hp[0]=3;hp[1]=5;hp[2]=10;hp[3]=100;

  GxpCapi_CreateExecutionSpec(&es); GxpCapi_ExecutionSpec_SetCoreId(es,0); GxpCapi_ExecutionSpec_SetCoreCount(es,1);
  GxpCapi_CreateRequest(&req); GxpCapi_Request_SetFunction(req,fn); GxpCapi_Request_AppendBuffer(req,buf);
  r=GxpCapi_RunSync(dev,es,req); P("RunSync=%d output=[%d %d %d %d] (vscale-Erw. [9 15 30 300])\n",r,hp[0],hp[1],hp[2],hp[3]);

  P("\n=== ARENA-LOKALISIERUNG ===\n");
  locate((uintptr_t)hp);
  list_dmabuf_fds();
  P("\n(Prozess bleibt 20 s offen fuer externe /proc/PID/-Inspektion, PID=%d)\n",getpid());
  sleep(20);
  return 0;
}
