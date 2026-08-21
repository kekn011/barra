/* tpu-zc — ZERO-COPY-Beweis auf der EdgeTPU (DarwinnApi2).
 * AllocateBuffer2(factory,opt,descr,size,fd,&out): fd=-1 alloziert, fd>=0 IMPORTIERT
 * einen dmabuf. Wir allozieren I/O-dmabufs selbst, importieren sie als Tensor-Puffer,
 * die TPU rechnet die Inferenz direkt hinein/heraus; wir lesen aus UNSERER mmap.
 * Gegenprobe: dieselbe Inferenz mit normalem AllocateBuffer -> Ergebnisse muessen gleich sein.
 * Bionic/NDK, als root. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/types.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)
struct dma_heap_allocation_data { __u64 len; __u32 fd; __u32 fd_flags; __u64 heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0, struct dma_heap_allocation_data)

static void* H;
#define S(n) dlsym(H,n)
static void* g_vdev,*g_factory,*g_opt,*g_graph; static long g_in,g_out;
/* Funktionszeiger */
static void*(*allocBuf)(void*,void*,void*,long,void*);
static void*(*impFd)(void*,void*,int,int,long,long,void*);  /* ImportBufferByFd(factory,obj,fd_type,fd,size,offset,&out) */
static int g_fdtype=0;
static void*(*mapHost)(void*,void*); static void*(*mapDev)(void*); static void*(*invCache)(void*);
static void*(*bufFree)(void*); static int(*hasFd)(void*,int*); static int(*getFd)(void*,int*);
static void*(*regGraph)(void*,void*,long,int,int,void*);
static int (*getInT)(void*,int,void**); static int(*getOutT)(void*,int,void**); static long(*tsize)(void*);
static void*(*createReq)(void*,void*,void*); static void*(*addIn)(void*,int,void*); static void*(*addOut)(void*,int,void*);
static void*(*submit)(void*,void*); static void*(*reqFree)(void*);

static int dmabuf(uint32_t len, int* fd, void** map){
  int h=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC); if(h<0) return -1;
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=(len+4095)&~4095u; d.fd_flags=O_RDWR|O_CLOEXEC;
  if(ioctl(h,DMA_HEAP_IOCTL_ALLOC,&d)<0){ close(h); return -1; } close(h);
  void* m=mmap(0,d.len,PROT_READ|PROT_WRITE,MAP_SHARED,d.fd,0); if(m==MAP_FAILED) return -1;
  *fd=(int)d.fd; *map=m; return 0;
}
/* normaler (allozierter) Puffer -> host-ptr */
static void* mkbuf(long sz,void** host){ long descr[4]={1,0,0,0}; void* b=0; allocBuf(g_factory,g_opt,descr,sz,&b);
  if(!b)return 0; void* hp=0; mapHost(b,&hp); if(host)*host=hp; return b; }
/* Zero-Copy: dmabuf-fd als Tensor-Puffer importieren */
static void* g_iobj=0;   /* Objekt an x1: opt (default) oder descr */
static void* impbuf(long sz,int fd){ void* b=0; impFd(g_factory,g_iobj,g_fdtype,fd,sz,0,&b); return b; }

static void infer(void* inBuf,void* outBuf){
  void* req=0; mapDev(inBuf); mapDev(outBuf); createReq(g_vdev,g_graph,&req);
  addIn(req,0,inBuf); addOut(req,0,outBuf); submit(g_vdev,req); usleep(400000); if(invCache) invCache(outBuf);
  if(reqFree) reqFree(req);
}

int main(int argc,char**argv){
  setvbuf(stderr,0,_IONBF,0);
  const char* model=argc>1?argv[1]:"/data/adb/hwbridge/test.package";
  const char* ft=getenv("TPU_FDTYPE"); if(ft) g_fdtype=atoi(ft);
  P("(fd_type=%d)\n",g_fdtype);
  H=dlopen("/vendor/lib64/libedgetpu_util.so",RTLD_NOW|RTLD_GLOBAL); if(!H){P("dlopen FAIL\n");return 1;}
  void*(*getSpec)(int*,unsigned char*,int*,int*,void**,unsigned char*)=(void*)S("DarwinnDelegate_GetDefaultDeviceSpec");
  void*(*createVD)(long,long,long,long,void*,long,void*)=(void*)S("DarwinnDelegate_CreateVirtualDevice");
  void*(*getVD)(void*)=(void*)S("DarwinnDelegate_EdgeTpuDevice_GetVirtualDevice");
  void*(*getBF)(void*)=(void*)S("DarwinnApi2_VirtualDevice_GetBufferFactory");
  void*(*getSupported)(void*,void*,int*)=(void*)S("DarwinnApi2_BufferFactory_GetSupportedOptions");
  regGraph=(void*)S("DarwinnApi2_VirtualDevice_RegisterGraphByBuffer");
  getInT=(void*)S("DarwinnApi2_Graph_GetInputTensor"); getOutT=(void*)S("DarwinnApi2_Graph_GetOutputTensor");
  tsize=(void*)S("DarwinnApi2_TensorInfo_SizeBytes");
  allocBuf=(void*)S("DarwinnApi2_BufferFactory_AllocateBuffer");
  impFd=(void*)S("DarwinnApi2_BufferFactory_ImportBufferByFd");
  mapHost=(void*)S("DarwinnApi2_Buffer_MapToHost"); mapDev=(void*)S("DarwinnApi2_Buffer_MapToDevice");
  invCache=(void*)S("DarwinnApi2_Buffer_InvalidateCache"); bufFree=(void*)S("DarwinnApi2_Buffer_Free");
  hasFd=(void*)S("DarwinnApi2_Buffer_HasFileDescriptor"); getFd=(void*)S("DarwinnApi2_Buffer_FileDescriptor");
  createReq=(void*)S("DarwinnApi2_VirtualDevice_CreateInferenceRequest");
  addIn=(void*)S("DarwinnApi2_InferenceRequest_AddInput"); addOut=(void*)S("DarwinnApi2_InferenceRequest_AddOutput");
  submit=(void*)S("DarwinnApi2_VirtualDevice_Submit"); reqFree=(void*)S("DarwinnApi2_InferenceRequest_Free");
  if(!impFd){P("ImportBufferByFd fehlt\n");return 1;}

  int s0,s2,s3; unsigned char s1,s5; void* s4=0; getSpec(&s0,&s1,&s2,&s3,&s4,&s5);
  void* edgeDev=0; createVD((long)s0,(long)s1,(long)s2,(long)s3,s4,(long)s5,&edgeDev);
  g_vdev=getVD(edgeDev); g_factory=getBF(g_vdev);
  void* arr=0; int cnt=0; getSupported(g_factory,&arr,&cnt); g_opt=((void**)arr)[0];
  g_iobj=g_opt;   /* x1-Objekt: die Buffer-Option (wie AllocateBuffer) */
  static long dd[4]={1,0,0,0};
  if(getenv("TPU_OBJ_DESCR")) g_iobj=dd;      /* alternativ: Descriptor */
  if(getenv("TPU_OBJ_ARR"))   g_iobj=arr;     /* alternativ: das Options-Array */

  /* Modell laden */
  FILE* f=fopen(model,"rb"); if(!f){P("kein Modell %s\n",model);return 1;}
  fseek(f,0,SEEK_END); long psz=ftell(f); fseek(f,0,SEEK_SET);
  void* pd=malloc(psz); if(fread(pd,1,psz,f)!=(size_t)psz){return 1;} fclose(f);
  void* ph=0; void* pbuf=mkbuf((psz+4095)&~4095L,&ph); memcpy(ph,pd,psz); mapDev(pbuf); free(pd);
  regGraph(g_vdev,pbuf,0,1,0,&g_graph); if(!g_graph){P("RegisterGraph FAIL\n");return 1;}
  void* it=0; getInT(g_graph,0,&it); g_in=tsize(it);
  void* ot=0; getOutT(g_graph,0,&ot); g_out=tsize(ot);
  P("Modell: in=%ld out=%ld Bytes\n",g_in,g_out);

  /* ---- Referenz: normale Puffer, Input 0x5A ---- */
  void *rih=0,*roh=0; void* rin=mkbuf(g_in,&rih); void* rout=mkbuf(g_out,&roh);
  memset(rih,0x5A,g_in);
  infer(rin,rout); infer(rin,rout);   /* 2x (warm) */
  unsigned char* ref=malloc(g_out); memcpy(ref,roh,g_out);
  P("Referenz-Output (alloziert): "); for(int i=0;i<8&&i<g_out;i++)P("%02x ",((unsigned char*)roh)[i]); P("...\n");

  /* ---- Zero-Copy: eigene dmabufs importieren ---- */
  int infd,outfd; void *inmap,*outmap;
  if(dmabuf(g_in,&infd,&inmap)||dmabuf(g_out,&outfd,&outmap)){P("dmabuf fail\n");return 1;}
  void* zin=impbuf(g_in,infd); void* zout=impbuf(g_out,outfd);
  if(!zin||!zout){P("AllocateBuffer2(fd) FAIL zin=%p zout=%p\n",zin,zout);return 1;}
  int hf=0,fdv=-1; if(hasFd)hasFd(zin,&hf); if(getFd)getFd(zin,&fdv);
  P("Import ok: zin HasFileDescriptor=%d FileDescriptor=%d (unser infd=%d)\n",hf,fdv,infd);
  memset(inmap,0x5A,g_in);   /* Input in UNSERE dmabuf-mmap */
  infer(zin,zout); infer(zin,zout);
  /* Output aus UNSERER dmabuf-mmap lesen */
  unsigned char* zc=(unsigned char*)outmap;
  P("Zero-Copy-Output (aus unserer mmap): "); for(int i=0;i<8&&i<g_out;i++)P("%02x ",zc[i]); P("...\n");

  int same=memcmp(ref,zc,g_out)==0;
  P("Vergleich Referenz vs Zero-Copy: %s\n", same?"IDENTISCH":"UNTERSCHIEDLICH");
  if(same) P(">>> TPU-ZERO-COPY OK: die TPU rechnete die Inferenz direkt in unseren dmabuf. <<<\n");
  else { int diff=0; for(long i=0;i<g_out;i++) if(ref[i]!=zc[i]) diff++; P("  (%d/%ld Bytes verschieden)\n",diff,g_out); }
  return 0;
}
