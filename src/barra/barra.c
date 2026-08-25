/* barra.c — Implementierung der Dispatch-Schicht. Spricht die drei Bruecken-
 * Protokolle ueber die Unix-Sockets unter /opt/hwbridge/ (Container-Sicht). */
#define _GNU_SOURCE
#include "barra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/types.h>
#include <errno.h>

#define SOCK_DIR "/opt/hwbridge"   /* Override: env BARRA_SOCK_DIR (z.B. Test-Daemons) */
static const char* sock_dir(void){ const char* d=getenv("BARRA_SOCK_DIR"); return (d&&*d)?d:SOCK_DIR; }

static int rn(int fd,void*p,size_t n){uint8_t*b=p;size_t g=0;while(g<n){ssize_t k=read(fd,b+g,n-g);if(k<0){if(errno==EINTR)continue;return -1;}if(k==0)return -1;g+=(size_t)k;}return 0;}
static int wn(int fd,const void*p,size_t n){const uint8_t*b=p;size_t s=0;while(s<n){ssize_t k=write(fd,b+s,n-s);if(k<0){if(errno==EINTR)continue;return -1;}if(k==0)return -1;s+=(size_t)k;}return 0;}

static int dial(const char* name){
  int s=socket(AF_UNIX,SOCK_STREAM,0); if(s<0) return -1;
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX;
  int pn=snprintf(a.sun_path,sizeof a.sun_path,"%s/%s",sock_dir(),name);
  if(pn<0||(size_t)pn>=sizeof a.sun_path){ fprintf(stderr,"barra: Socket-Pfad zu lang (BARRA_SOCK_DIR?)\n"); close(s); return -1; }
  if(connect(s,(struct sockaddr*)&a,sizeof a)<0){ close(s); return -1; }
  return s;
}

/* ---- DSP: gxp.sock, Protokoll GXPD ---------------------------------------
 * Req : magic 0x47585044, name_len u32, in_size u32, [name][input]
 * Resp: status u32, rv i64, exec_us u32, out_size u32, [output] */
static int run_dsp(const barra_op* op,const uint8_t* in,uint32_t in_size,uint8_t* out,uint32_t cap){
  if(!op->dsp_func) return -1;
  int s=dial("gxp.sock"); if(s<0){ fprintf(stderr,"barra: DSP-Socket nicht erreichbar\n"); return -1; }
  uint32_t nlen=strlen(op->dsp_func);
  uint32_t hdr[3]={0x47585044u,nlen,in_size};
  if(wn(s,hdr,sizeof hdr)||wn(s,op->dsp_func,nlen)||wn(s,in,in_size)){ close(s); return -1; }
  uint32_t status; if(rn(s,&status,4)){ close(s); return -1; }
  if(status!=0){ close(s); fprintf(stderr,"barra: DSP status=%u\n",status); return -1; }
  int64_t rv=0; uint32_t exec_us=0,osz=0;
  if(rn(s,&rv,8)||rn(s,&exec_us,4)||rn(s,&osz,4)){ close(s); return -1; }
  if(osz>cap) osz=cap;
  int r=rn(s,out,osz); close(s);
  return r?-1:(int)osz;
}

/* ---- TPU: tpu.sock, Protokoll TPD2 ---------------------------------------
 * Req : magic 0x54504432, model_id u32, in_size u32, [input]
 * Resp: status u32, out_size u32, [output]  (I/O-Groessen kennt der Daemon) */
static int run_tpu(const barra_op* op,const uint8_t* in,uint32_t in_size,uint8_t* out,uint32_t cap){
  int s=dial("tpu.sock"); if(s<0){ fprintf(stderr,"barra: TPU-Socket nicht erreichbar\n"); return -1; }
  uint32_t hdr[3]={0x54504432u,op->tpu_model_id,in_size};
  if(wn(s,hdr,sizeof hdr)||wn(s,in,in_size)){ close(s); return -1; }
  uint32_t status,osz=0; if(rn(s,&status,4)){ close(s); return -1; }
  if(status!=0){ close(s); fprintf(stderr,"barra: TPU status=%u\n",status); return -1; }
  if(rn(s,&osz,4)){ close(s); return -1; } if(osz>cap) osz=cap;
  int r=rn(s,out,osz); close(s);
  return r?-1:(int)osz;
}

/* ---- GPU: gpu.sock, Protokoll GPU1 (One-shot inline, EIN In/Out-Puffer) ---
 * Req: magic 0x47505531, spirv_len u32, gx,gy,gz u32, nbuf u32=1,
 *      {size u32, flags u32}, [spirv], [input]   Resp: [output(size)] */
static int run_gpu(const barra_op* op,const uint8_t* in,uint32_t in_size,uint8_t* out,uint32_t cap){
  if(!op->gpu_spirv||!op->gpu_spirv_len){ fprintf(stderr,"barra: GPU-Op ohne SPIR-V\n"); return -1; }
  int s=dial("gpu.sock"); if(s<0){ fprintf(stderr,"barra: GPU-Socket nicht erreichbar\n"); return -1; }
  uint32_t h[6]={0x47505531u,op->gpu_spirv_len,op->gx,op->gy,op->gz,1u};
  uint32_t buf[2]={in_size,0u};
  if(wn(s,h,sizeof h)||wn(s,buf,sizeof buf)||wn(s,op->gpu_spirv,op->gpu_spirv_len)||wn(s,in,in_size)){ close(s); return -1; }
  uint32_t osz=in_size>cap?cap:in_size;   /* in-place: gleiche Groesse zurueck */
  int r=rn(s,out,osz); close(s);
  return r?-1:(int)osz;
}

/* ---- CPU: in-process ------------------------------------------------------ */
static int run_cpu(const barra_op* op,const uint8_t* in,uint32_t in_size,uint8_t* out,uint32_t cap){
  if(!op->cpu_fn) return -1;
  /* Vertrag: *out_size traegt beim Aufruf die Ausgabe-KAPAZITAET; der Callback
   * darf hoechstens so viele Bytes schreiben und setzt *out_size auf die echte
   * Groesse. Das nachgelagerte Clamp ist nur eine zweite Verteidigungslinie —
   * ein Callback, der cap ignoriert und mehr schreibt, ueberlaeuft 'out'. */
  uint32_t osz=cap; op->cpu_fn(in,in_size,out,&osz);
  return (int)(osz>cap?cap:osz);
}

int barra_run(const barra_op* op,const uint8_t* in,uint32_t in_size,uint8_t* out,uint32_t cap){
  switch(op->device){
    case BARRA_CPU: return run_cpu(op,in,in_size,out,cap);
    case BARRA_TPU: return run_tpu(op,in,in_size,out,cap);
    case BARRA_GPU: return run_gpu(op,in,in_size,out,cap);
    case BARRA_DSP: return run_dsp(op,in,in_size,out,cap);
  }
  return -1;
}

#define BARRA_STAGE_CAP (1u<<20)
int barra_pipeline(const barra_op* ops,int n,const uint8_t* in,uint32_t in_size,uint8_t* out,uint32_t cap){
  /* Doppelpuffer: Ausgabe jeder Stufe wird Eingabe der naechsten (heute Kopie;
   * spaeter dmabuf-Handoff ohne Kopie). Puffer werden pro Aufruf allokiert, damit
   * parallele Pipelines sich nicht dieselben Stufenpuffer teilen (Thread-Safety). */
  uint8_t* a=malloc(BARRA_STAGE_CAP); uint8_t* b=malloc(BARRA_STAGE_CAP);
  if(!a||!b){ free(a); free(b); fprintf(stderr,"barra: Pipeline-Puffer OOM\n"); return -1; }
  const uint8_t* cur=in; uint32_t cur_sz=in_size; uint8_t* dst=a; int ret=-1;
  for(int i=0;i<n;i++){
    int r=barra_run(&ops[i],cur,cur_sz,dst,BARRA_STAGE_CAP);
    if(r<0){ fprintf(stderr,"barra: Stufe %d (%s @ %s) fehlgeschlagen\n",i,ops[i].label?ops[i].label:"",barra_devname(ops[i].device)); goto done; }
    if((uint32_t)r>=BARRA_STAGE_CAP){ fprintf(stderr,"barra: Stufe %d Ausgabe an Puffergrenze (%u B) — moegliche Trunkierung, Abbruch\n",i,(unsigned)BARRA_STAGE_CAP); goto done; }
    printf("  [%d] %-14s @ %-3s : %u B -> %u B\n", i, ops[i].label?ops[i].label:"op", barra_devname(ops[i].device), cur_sz, (unsigned)r);
    cur=dst; cur_sz=(uint32_t)r; dst=(dst==a)?b:a;
  }
  { uint32_t osz=cur_sz>cap?cap:cur_sz; memcpy(out,cur,osz); ret=(int)osz; }
done:
  free(a); free(b); return ret;
}

const char* barra_devname(barra_device d){
  switch(d){case BARRA_CPU:return"CPU";case BARRA_TPU:return"TPU";case BARRA_GPU:return"GPU";case BARRA_DSP:return"DSP";}
  return"?";
}

/* ================= Zero-Copy (GPU) ================= */
struct dma_heap_allocation_data { __u64 len; __u32 fd; __u32 fd_flags; __u64 heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0, struct dma_heap_allocation_data)
/* dma-buf CPU-Zugriffs-Klammern (linux/dma-buf.h) */
struct dma_buf_sync { __u64 flags; };
#define DMA_BUF_SYNC_READ  (1<<0)
#define DMA_BUF_SYNC_WRITE (2<<0)
#define DMA_BUF_SYNC_RW    (DMA_BUF_SYNC_READ|DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START (0<<2)
#define DMA_BUF_SYNC_END   (1<<2)
#define DMA_BUF_IOCTL_SYNC _IOW('b', 0, struct dma_buf_sync)
#define GPZC_MAGIC 0x47505A43u
#define GPZ2_MAGIC 0x47505A32u
#define ZC_MAXBUF 16
#define ZC_MAXSTAGE 32

int barra_zc_cpu_begin(barra_zbuf* z){ struct dma_buf_sync s={DMA_BUF_SYNC_START|DMA_BUF_SYNC_RW}; return (z&&z->fd>=0&&ioctl(z->fd,DMA_BUF_IOCTL_SYNC,&s)==0)?0:-1; }
int barra_zc_cpu_end(barra_zbuf* z){   struct dma_buf_sync s={DMA_BUF_SYNC_END  |DMA_BUF_SYNC_RW}; return (z&&z->fd>=0&&ioctl(z->fd,DMA_BUF_IOCTL_SYNC,&s)==0)?0:-1; }

int barra_zc_alloc(barra_zbuf* z, uint32_t size){
  if(z){ z->fd=-1; z->map=0; z->size=0; z->gpu_h=-1; z->tpu_h=-1; z->dsp_h=-1; }  /* fehlgeschlagene Allocs inert machen */
  int h=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC); if(h<0) return -1;
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=size; d.fd_flags=O_RDWR|O_CLOEXEC;
  if(ioctl(h,DMA_HEAP_IOCTL_ALLOC,&d)<0){ close(h); return -1; }
  close(h);
  void* m=mmap(0,size,PROT_READ|PROT_WRITE,MAP_SHARED,d.fd,0);
  if(m==MAP_FAILED){ close(d.fd); return -1; }
  z->fd=(int)d.fd; z->map=m; z->size=size; z->gpu_h=-1; z->tpu_h=-1; z->dsp_h=-1;
  barra_zc_cpu_begin(z);           /* Puffer startet im CPU-Zugriffs-Zustand */
  return 0;
}
void barra_zc_free(barra_zbuf* z){
  if(!z) return;
  /* Der lokale fd wird geschlossen, aber die Daemons halten ihre dup'ten fds bis
   * zum Session-Ende. Wer einen noch importierten Puffer freigibt, laesst den
   * dmabuf daemon-seitig gepinnt: erst barra_*_release rufen. */
  if(z->gpu_h>=0||z->tpu_h>=0||z->dsp_h>=0)
    fprintf(stderr,"barra: WARN barra_zc_free eines noch importierten Puffers (gpu_h=%d tpu_h=%d dsp_h=%d) — vorher barra_*_release rufen, sonst bleibt der dmabuf daemon-seitig gepinnt\n",z->gpu_h,z->tpu_h,z->dsp_h);
  if(z->map){ barra_zc_cpu_end(z); munmap(z->map,z->size); }
  if(z->fd>=0) close(z->fd);
  z->map=0; z->fd=-1; z->size=0; z->gpu_h=-1; z->tpu_h=-1; z->dsp_h=-1;
}

/* Header + fds in EINEM sendmsg (SCM_RIGHTS haengt am ersten Byte-Bereich) */
static int send_hdr_fds(int s, uint32_t hdr[6], const int* fds, int nfd){
  char cbuf[CMSG_SPACE(sizeof(int)*ZC_MAXBUF)]; memset(cbuf,0,sizeof cbuf);
  struct iovec iov={.iov_base=hdr,.iov_len=24};
  struct msghdr msg={.msg_iov=&iov,.msg_iovlen=1};
  if(nfd>0){
    msg.msg_control=cbuf; msg.msg_controllen=CMSG_SPACE(sizeof(int)*nfd);
    struct cmsghdr* cm=CMSG_FIRSTHDR(&msg);
    cm->cmsg_level=SOL_SOCKET; cm->cmsg_type=SCM_RIGHTS; cm->cmsg_len=CMSG_LEN(sizeof(int)*nfd);
    memcpy(CMSG_DATA(cm),fds,sizeof(int)*nfd);
  }
  return sendmsg(s,&msg,0)==24?0:-1;
}

/* --- v1 Einzel-Dispatch ------------------------------------------------------ */
int barra_gpu_zc(const uint8_t* spirv,uint32_t slen,uint32_t gx,uint32_t gy,uint32_t gz,barra_zbuf* bufs,int nbuf){
  if(nbuf<=0||nbuf>ZC_MAXBUF) return -1;
  int s=dial("gpuzc.sock"); if(s<0){ fprintf(stderr,"barra: gpuzc.sock nicht erreichbar\n"); return -1; }
  int fds[ZC_MAXBUF]; uint32_t sizes[ZC_MAXBUF];
  for(int i=0;i<nbuf;i++){ fds[i]=bufs[i].fd; sizes[i]=bufs[i].size; barra_zc_cpu_end(&bufs[i]); }
  uint32_t hdr[6]={GPZC_MAGIC,slen,gx,gy,gz,(uint32_t)nbuf};
  int rc=-1; uint32_t status=1;
  if(send_hdr_fds(s,hdr,fds,nbuf)==0 && !wn(s,sizes,nbuf*4) && !wn(s,spirv,slen) && !rn(s,&status,4) && status==0) rc=0;
  close(s);
  for(int i=0;i<nbuf;i++) barra_zc_cpu_begin(&bufs[i]);
  return rc;
}

/* --- v2 Session ---------------------------------------------------------------- */
int barra_gpu_open(barra_gpu* g){
  if(!g) return -1;
  g->sock=dial("gpuzc.sock"); g->nimported=0;
  if(g->sock<0){ fprintf(stderr,"barra: gpuzc.sock nicht erreichbar\n"); return -1; }
  return 0;
}
void barra_gpu_close(barra_gpu* g){ if(g&&g->sock>=0){ close(g->sock); g->sock=-1; } }

/* Nach einem Protokollfehler mitten in einer Nachricht ist der Stream desynchron:
 * Session schliessen (sock=-1), damit Folgeaufrufe deterministisch scheitern und
 * der Daemon die Handles beim Verbindungsabbruch aufraeumt. */
static int gpu_dead(barra_gpu* g){ if(g&&g->sock>=0){ close(g->sock); g->sock=-1; } return -1; }
static int tpu_dead(barra_tpu* t){ if(t&&t->sock>=0){ close(t->sock); t->sock=-1; } return -1; }
static int dsp_dead(barra_dsp* d){ if(d&&d->sock>=0){ close(d->sock); d->sock=-1; } return -1; }

int barra_gpu_import(barra_gpu* g, barra_zbuf** bufs, int n){
  if(!g||g->sock<0||n<=0||n>ZC_MAXBUF) return -1;
  int fds[ZC_MAXBUF]; uint32_t sizes[ZC_MAXBUF], hd[ZC_MAXBUF];
  for(int i=0;i<n;i++){ fds[i]=bufs[i]->fd; sizes[i]=bufs[i]->size; }
  uint32_t hdr[6]={GPZ2_MAGIC,1,(uint32_t)n,0,0,0};
  uint32_t status=1;
  if(send_hdr_fds(g->sock,hdr,fds,n)||wn(g->sock,sizes,n*4)||rn(g->sock,&status,4)) return gpu_dead(g);
  if(status!=0){ fprintf(stderr,"barra: GPU-Import status=%u\n",status); return -1; }
  if(rn(g->sock,hd,n*4)) return gpu_dead(g);
  for(int i=0;i<n;i++) bufs[i]->gpu_h=(int)hd[i];
  g->nimported+=n; return 0;
}
int barra_gpu_release(barra_gpu* g, barra_zbuf** bufs, int n){
  if(!g||g->sock<0||n<=0||n>ZC_MAXBUF) return -1;
  uint32_t hd[ZC_MAXBUF]; int k=0;
  for(int i=0;i<n;i++) if(bufs[i]->gpu_h>=0) hd[k++]=(uint32_t)bufs[i]->gpu_h;
  if(k==0) return 0;
  uint32_t hdr[6]={GPZ2_MAGIC,3,(uint32_t)k,0,0,0}; uint32_t status=1;
  if(send_hdr_fds(g->sock,hdr,0,0)||wn(g->sock,hd,k*4)||rn(g->sock,&status,4)) return gpu_dead(g);
  if(status!=0) return -1;   /* Daemon meldet Fehler, aber Stream ist synchron: Handles NICHT lokal loeschen */
  for(int i=0;i<n;i++) bufs[i]->gpu_h=-1;
  g->nimported-=k; return 0;
}
int barra_gpu_batch(barra_gpu* g, const barra_gpu_stage* st, int nstage){
  if(!g||g->sock<0||nstage<=0||nstage>ZC_MAXSTAGE) return -1;
  /* 1) alles Unimportierte importieren (einmalig, Handles bleiben) */
  for(int s=0;s<nstage;s++){
    if(st[s].nbuf<=0||st[s].nbuf>ZC_MAXBUF||!st[s].spirv||!st[s].slen||(st[s].slen&3)) return -1;
    barra_zbuf* need[ZC_MAXBUF]; int k=0;
    for(int i=0;i<st[s].nbuf;i++){ barra_zbuf* z=st[s].bufs[i]; if(z->gpu_h<0){ int dup_=0; for(int j=0;j<k;j++) if(need[j]==z) dup_=1; if(!dup_) need[k++]=z; } }
    if(k&&barra_gpu_import(g,need,k)) return -1;
  }
  /* 2) CPU-Zugriff schliessen (Cache-Flush), Batch senden */
  for(int s=0;s<nstage;s++) for(int i=0;i<st[s].nbuf;i++) barra_zc_cpu_end(st[s].bufs[i]);
  uint32_t hdr[6]={GPZ2_MAGIC,2,(uint32_t)nstage,0,0,0}; int rc=-1,desync=0; uint32_t status=1;
  if(send_hdr_fds(g->sock,hdr,0,0)==0){
    rc=0;
    for(int s=0;s<nstage&&rc==0;s++){
      uint32_t sh[5]={st[s].slen,st[s].gx,st[s].gy,st[s].gz,(uint32_t)st[s].nbuf}; uint32_t hd[ZC_MAXBUF];
      for(int i=0;i<st[s].nbuf;i++) hd[i]=(uint32_t)st[s].bufs[i]->gpu_h;
      if(wn(g->sock,sh,20)||wn(g->sock,hd,st[s].nbuf*4)||wn(g->sock,st[s].spirv,st[s].slen)){ rc=-1; desync=1; }
    }
    if(rc==0){ if(rn(g->sock,&status,4)){ rc=-1; desync=1; } else if(status!=0) rc=-1; }
  } else desync=1;
  /* 3) CPU-Zugriff wieder oeffnen (Cache-Invalidate) */
  for(int s=0;s<nstage;s++) for(int i=0;i<st[s].nbuf;i++) barra_zc_cpu_begin(st[s].bufs[i]);
  if(rc) fprintf(stderr,"barra: GPU-Batch fehlgeschlagen (status=%u)\n",status);
  if(desync) gpu_dead(g);   /* Stream desynchron: Session verwerfen */
  return rc;
}
int barra_gpu_dispatch(barra_gpu* g, const uint8_t* spirv, uint32_t slen, uint32_t gx, uint32_t gy, uint32_t gz, barra_zbuf** bufs, int nbuf){
  barra_gpu_stage st={.spirv=spirv,.slen=slen,.gx=gx,.gy=gy,.gz=gz,.bufs=bufs,.nbuf=nbuf};
  return barra_gpu_batch(g,&st,1);
}

/* ================= Zero-Copy (TPU) — Session TPZ2 ================= */
#define TPZ2_MAGIC 0x54505A32u
int barra_tpu_open(barra_tpu* t){
  if(!t) return -1;
  t->sock=dial("tpu.sock");
  if(t->sock<0){ fprintf(stderr,"barra: tpu.sock nicht erreichbar\n"); return -1; }
  return 0;
}
void barra_tpu_close(barra_tpu* t){ if(t&&t->sock>=0){ close(t->sock); t->sock=-1; } }
int barra_tpu_info(barra_tpu* t, uint32_t model_id, uint32_t* in_size, uint32_t* out_size, uint32_t* nmodels){
  if(!t||t->sock<0) return -1;
  uint32_t hdr[6]={TPZ2_MAGIC,4,model_id,0,0,0}; uint32_t r[4];
  if(send_hdr_fds(t->sock,hdr,0,0)||rn(t->sock,r,16)) return -1;
  if(nmodels)*nmodels=r[3];
  if(r[0]!=0) return -1;
  if(in_size)*in_size=r[1];
  if(out_size)*out_size=r[2];
  return 0;
}
int barra_tpu_import(barra_tpu* t, barra_zbuf** bufs, int n){
  if(!t||t->sock<0||n<=0||n>ZC_MAXBUF) return -1;
  int fds[ZC_MAXBUF]; uint32_t sizes[ZC_MAXBUF], hd[ZC_MAXBUF];
  for(int i=0;i<n;i++){ fds[i]=bufs[i]->fd; sizes[i]=bufs[i]->size; }
  uint32_t hdr[6]={TPZ2_MAGIC,1,(uint32_t)n,0,0,0}; uint32_t status=1;
  if(send_hdr_fds(t->sock,hdr,fds,n)||wn(t->sock,sizes,n*4)||rn(t->sock,&status,4)) return tpu_dead(t);
  if(status!=0){ fprintf(stderr,"barra: TPU-Import status=%u\n",status); return -1; }
  if(rn(t->sock,hd,n*4)) return tpu_dead(t);
  for(int i=0;i<n;i++) bufs[i]->tpu_h=(int)hd[i];
  return 0;
}
int barra_tpu_release(barra_tpu* t, barra_zbuf** bufs, int n){
  if(!t||t->sock<0||n<=0||n>ZC_MAXBUF) return -1;
  uint32_t hd[ZC_MAXBUF]; int k=0;
  for(int i=0;i<n;i++) if(bufs[i]->tpu_h>=0) hd[k++]=(uint32_t)bufs[i]->tpu_h;
  if(k==0) return 0;
  uint32_t hdr[6]={TPZ2_MAGIC,3,(uint32_t)k,0,0,0}; uint32_t status=1;
  if(send_hdr_fds(t->sock,hdr,0,0)||wn(t->sock,hd,k*4)||rn(t->sock,&status,4)) return tpu_dead(t);
  if(status!=0) return -1;
  for(int i=0;i<n;i++) bufs[i]->tpu_h=-1;
  return 0;
}
int barra_tpu_infer(barra_tpu* t, uint32_t model_id, barra_zbuf* in, barra_zbuf* out, uint32_t* exec_us){
  if(!t||t->sock<0||!in||!out||in==out) return -1;
  barra_zbuf* need[2]; int k=0;
  if(in->tpu_h<0) need[k++]=in;
  if(out->tpu_h<0) need[k++]=out;
  if(k&&barra_tpu_import(t,need,k)) return -1;
  barra_zc_cpu_end(in); barra_zc_cpu_end(out);
  uint32_t hdr[6]={TPZ2_MAGIC,2,model_id,(uint32_t)in->tpu_h,(uint32_t)out->tpu_h,0}; uint32_t r[2]={1,0};
  int io=(send_hdr_fds(t->sock,hdr,0,0)||rn(t->sock,r,8));
  int rc=(io||r[0]!=0)?-1:0;
  barra_zc_cpu_begin(in); barra_zc_cpu_begin(out);
  if(exec_us)*exec_us=r[1];
  if(rc) fprintf(stderr,"barra: TPU-Inferenz (zc) fehlgeschlagen (status=%u)\n",r[0]);
  if(io) tpu_dead(t);
  return rc;
}

int barra_tpu_submit(barra_tpu* t, uint32_t model_id, barra_zbuf* in, barra_zbuf* out){
  if(!t||t->sock<0||!in||!out||in==out) return -1;
  barra_zbuf* need[2]; int k=0;
  if(in->tpu_h<0) need[k++]=in;
  if(out->tpu_h<0) need[k++]=out;
  if(k&&barra_tpu_import(t,need,k)) return -1;
  barra_zc_cpu_end(in); barra_zc_cpu_end(out);
  uint32_t hdr[6]={TPZ2_MAGIC,2,model_id,(uint32_t)in->tpu_h,(uint32_t)out->tpu_h,0};
  if(send_hdr_fds(t->sock,hdr,0,0)){ barra_zc_cpu_begin(in); barra_zc_cpu_begin(out); return tpu_dead(t); }
  return 0;
}

int barra_tpu_wait(barra_tpu* t, barra_zbuf* in, barra_zbuf* out, uint32_t* exec_us){
  if(!t||t->sock<0||!in||!out) return -1;
  uint32_t r[2]={1,0};
  int io=rn(t->sock,r,8);
  int rc=(io||r[0]!=0)?-1:0;
  barra_zc_cpu_begin(in); barra_zc_cpu_begin(out);
  if(io) tpu_dead(t);
  if(exec_us)*exec_us=r[1];
  if(rc) fprintf(stderr,"barra: TPU-Inferenz (zc, split) fehlgeschlagen (status=%u)\n",r[0]);
  return rc;
}

/* ================= Zero-Copy (DSP) — Session GXPZ ================= */
#define GXPZ_MAGIC 0x47585A32u
int barra_dsp_open(barra_dsp* d){
  if(!d) return -1;
  d->sock=dial("gxp.sock");
  if(d->sock<0){ fprintf(stderr,"barra: gxp.sock nicht erreichbar\n"); return -1; }
  return 0;
}
void barra_dsp_close(barra_dsp* d){ if(d&&d->sock>=0){ close(d->sock); d->sock=-1; } }

int barra_dsp_import_ex(barra_dsp* d, barra_zbuf** bufs, int n, int cacheable){
  if(!d||d->sock<0||n<=0||n>ZC_MAXBUF) return -1;
  int fds[ZC_MAXBUF]; uint32_t sizes[ZC_MAXBUF], hd[ZC_MAXBUF];
  for(int i=0;i<n;i++){ fds[i]=bufs[i]->fd; sizes[i]=bufs[i]->size; }
  uint32_t hdr[6]={GXPZ_MAGIC,1,(uint32_t)n,(uint32_t)(cacheable?1:0),0,0}; uint32_t status=1;
  if(send_hdr_fds(d->sock,hdr,fds,n)||wn(d->sock,sizes,n*4)||rn(d->sock,&status,4)) return dsp_dead(d);
  if(status!=0){ fprintf(stderr,"barra: DSP-Import status=%u\n",status); return -1; }
  if(rn(d->sock,hd,n*4)) return dsp_dead(d);
  for(int i=0;i<n;i++) bufs[i]->dsp_h=(int)hd[i];
  return 0;
}
int barra_dsp_import(barra_dsp* d, barra_zbuf** bufs, int n){ return barra_dsp_import_ex(d,bufs,n,0); }
int barra_dsp_release(barra_dsp* d, barra_zbuf** bufs, int n){
  if(!d||d->sock<0||n<=0||n>ZC_MAXBUF) return -1;
  uint32_t hd[ZC_MAXBUF]; int k=0;
  for(int i=0;i<n;i++) if(bufs[i]->dsp_h>=0) hd[k++]=(uint32_t)bufs[i]->dsp_h;
  if(k==0) return 0;
  uint32_t hdr[6]={GXPZ_MAGIC,3,(uint32_t)k,0,0,0}; uint32_t status=1;
  if(send_hdr_fds(d->sock,hdr,0,0)||wn(d->sock,hd,k*4)||rn(d->sock,&status,4)) return dsp_dead(d);
  if(status!=0) return -1;
  for(int i=0;i<n;i++) bufs[i]->dsp_h=-1;
  return 0;
}
int barra_dsp_run(barra_dsp* d, const char* func, barra_zbuf** bufs, int n,
                  int64_t* rv, uint32_t* exec_us){
  if(!d||d->sock<0||!func||n<=0||n>ZC_MAXBUF) return -1;
  uint32_t nlen=(uint32_t)strlen(func); if(nlen==0||nlen>=128) return -1;
  /* auto-import (dedupliziert) */
  barra_zbuf* need[ZC_MAXBUF]; int k=0;
  for(int i=0;i<n;i++){ barra_zbuf* z=bufs[i]; if(z->dsp_h<0){ int dup_=0; for(int j=0;j<k;j++) if(need[j]==z) dup_=1; if(!dup_) need[k++]=z; } }
  if(k&&barra_dsp_import(d,need,k)) return -1;
  /* CPU-Zugriff schliessen (Flush) -> DSP rechnet -> CPU-Zugriff oeffnen (Invalidate) */
  for(int i=0;i<n;i++) barra_zc_cpu_end(bufs[i]);
  uint32_t hd[ZC_MAXBUF]; for(int i=0;i<n;i++) hd[i]=(uint32_t)bufs[i]->dsp_h;
  uint32_t hdr[6]={GXPZ_MAGIC,2,nlen,(uint32_t)n,0,0};
  uint32_t status=1; int64_t rrv=0; uint32_t us=0;
  int io=(send_hdr_fds(d->sock,hdr,0,0)||wn(d->sock,func,nlen)||wn(d->sock,hd,(uint32_t)n*4)
          ||rn(d->sock,&status,4)||rn(d->sock,&rrv,8)||rn(d->sock,&us,4));
  int rc=(io||status!=0)?-1:0;
  for(int i=0;i<n;i++) barra_zc_cpu_begin(bufs[i]);
  if(rv)*rv=rrv;
  if(exec_us)*exec_us=us;
  if(rc) fprintf(stderr,"barra: DSP-Kernel '%s' (zc) fehlgeschlagen (status=%u)\n",func,status);
  if(io) dsp_dead(d);
  return rc;
}
int barra_dsp_submit(barra_dsp* d, const char* func, barra_zbuf** bufs, int n, uint32_t* token){
  if(!d||d->sock<0||!func||n<=0||n>ZC_MAXBUF) return -1;
  uint32_t nlen=(uint32_t)strlen(func); if(nlen==0||nlen>=128) return -1;
  barra_zbuf* need[ZC_MAXBUF]; int k=0;
  for(int i=0;i<n;i++){ barra_zbuf* z=bufs[i]; if(z->dsp_h<0){ int dup_=0; for(int j=0;j<k;j++) if(need[j]==z) dup_=1; if(!dup_) need[k++]=z; } }
  if(k&&barra_dsp_import(d,need,k)) return -1;
  for(int i=0;i<n;i++) barra_zc_cpu_end(bufs[i]);          /* Flush: DSP liest */
  uint32_t hd[ZC_MAXBUF]; for(int i=0;i<n;i++) hd[i]=(uint32_t)bufs[i]->dsp_h;
  uint32_t hdr[6]={GXPZ_MAGIC,5,nlen,(uint32_t)n,0,0}; uint32_t status=1, tok=0xffffffff;
  int io=(send_hdr_fds(d->sock,hdr,0,0)||wn(d->sock,func,nlen)||wn(d->sock,hd,(uint32_t)n*4)
     ||rn(d->sock,&status,4)||rn(d->sock,&tok,4));
  if(io||status!=0){
    fprintf(stderr,"barra: DSP-Submit '%s' fehlgeschlagen (status=%u)\n",func,status);
    for(int i=0;i<n;i++) barra_zc_cpu_begin(bufs[i]);   /* CPU-Zugriff wiederherstellen (Sync-Klammer ausbalancieren) */
    return io?dsp_dead(d):-1; }
  if(token)*token=tok;
  return 0;
}
int barra_dsp_wait(barra_dsp* d, uint32_t token, barra_zbuf** bufs, int n, int64_t* rv, uint32_t* exec_us){
  if(!d||d->sock<0) return -1;
  uint32_t hdr[6]={GXPZ_MAGIC,6,token,0,0,0}; uint32_t status=1; int64_t rrv=0; uint32_t us=0;
  int io=(send_hdr_fds(d->sock,hdr,0,0)||rn(d->sock,&status,4)||rn(d->sock,&rrv,8)||rn(d->sock,&us,4));
  int rc=(io||status!=0)?-1:0;
  for(int i=0;i<n;i++) barra_zc_cpu_begin(bufs[i]);        /* Invalidate: CPU sieht Ergebnis */
  if(rv)*rv=rrv;
  if(exec_us)*exec_us=us;
  if(rc) fprintf(stderr,"barra: DSP-Wait token=%u fehlgeschlagen (status=%u)\n",token,status);
  if(io) dsp_dead(d);
  return rc;
}

void barra_devices(void){
  printf("barra Geraete (Capability-Tabelle):\n");
  printf("  %-3s  %-9s  %-11s  wofuer\n","DEV","transport","programmierb.");
  printf("  %-3s  %-9s  %-11s  %s\n","CPU","in-proc", "nativ",       "Verzweigtes, Tokenizing, Kleben");
  printf("  %-3s  %-9s  %-11s  %s\n","TPU","tpu.sock","vorkompil.",  "dichte Graphen: Attention/Matmul/Conv");
  printf("  %-3s  %-9s  %-11s  %s\n","GPU","gpu.sock","SPIR-V",      "parallele Arithmetik, grosse Matmuls");
  printf("  %-3s  %-9s  %-11s  %s\n","DSP","gxp.sock","Xtensa-ELF",  "kleine regulaere Integer/Vektor-Kernel");
  printf("  Datenfluss: Socket-Kopie ODER Zero-Copy (dmabuf): GPU- (barra_gpu_open/batch), TPU- (barra_tpu_open/infer) UND DSP-Session (barra_dsp_open/run); derselbe zbuf auf allen Chips = Cross-Chip-Handoff ueber CPU/GPU/TPU/DSP ohne Kopie.\n");
}
