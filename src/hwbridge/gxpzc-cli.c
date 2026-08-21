/* gxpzc-cli — Test-Client fuer das GXPZ-Zero-Copy-Protokoll von gxpd3.
 * Alloziert einen /dev/dma_heap/system-dmabuf, teilt dessen fd via SCM_RIGHTS mit
 * gxpd3 (IMPORT), laesst DSP-Kernel IN-PLACE darin rechnen (RUN) und liest das
 * Ergebnis aus der EIGENEN mmap — keine Datenkopie ueber den Socket.
 * Cache-Kohaerenz: DMA_BUF_IOCTL_SYNC auf dem eigenen fd (WRITE vor RUN, READ danach).
 *
 * Bauen: aarch64-linux-android31-clang gxpzc-cli.c -o gxpzc-cli
 * Lauf:  gxpzc-cli [socketpfad]   (default /data/local/tmp/gxpz.sock)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <linux/types.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)

#define MAGIC_ZC 0x47585A32u
struct dma_heap_allocation_data { __u64 len; __u32 fd; __u32 fd_flags; __u64 heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0, struct dma_heap_allocation_data)
struct dma_buf_sync { __u64 flags; };
#define DMA_BUF_IOCTL_SYNC _IOW('b', 0, struct dma_buf_sync)
#define DBS_READ (1ULL<<0)
#define DBS_WRITE (1ULL<<1)
#define DBS_START 0ULL
#define DBS_END (1ULL<<2)
static void dbsync(int fd,__u64 f){ struct dma_buf_sync s={f}; ioctl(fd,DMA_BUF_IOCTL_SYNC,&s); }

static int alloc_dmabuf(uint32_t len){
  int heap=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC); if(heap<0){P("open heap\n");return -1;}
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=len; d.fd_flags=O_RDWR|O_CLOEXEC;
  if(ioctl(heap,DMA_HEAP_IOCTL_ALLOC,&d)<0){P("alloc\n");close(heap);return -1;} close(heap);
  return (int)d.fd;
}
static int rd(int fd,void*p,size_t n){uint8_t*b=p;size_t g=0;while(g<n){ssize_t k=read(fd,b+g,n-g);if(k<=0)return -1;g+=k;}return 0;}
static int wr(int fd,const void*p,size_t n){const uint8_t*b=p;size_t s=0;while(s<n){ssize_t k=write(fd,b+s,n-s);if(k<=0)return -1;s+=k;}return 0;}

/* 24-Byte-Header (6*u32) senden, optional mit angehaengten fds (SCM_RIGHTS) */
static int send_hdr(int c,uint32_t cmd,uint32_t a,uint32_t b,uint32_t d,uint32_t e,int* fds,int nfd){
  uint32_t hdr[6]={MAGIC_ZC,cmd,a,b,d,e};
  struct iovec iov={.iov_base=hdr,.iov_len=sizeof hdr};
  char cbuf[CMSG_SPACE(sizeof(int)*8)]; memset(cbuf,0,sizeof cbuf);
  struct msghdr msg={.msg_iov=&iov,.msg_iovlen=1};
  if(nfd>0){ msg.msg_control=cbuf; msg.msg_controllen=CMSG_SPACE(sizeof(int)*nfd);
    struct cmsghdr* cm=CMSG_FIRSTHDR(&msg); cm->cmsg_level=SOL_SOCKET; cm->cmsg_type=SCM_RIGHTS;
    cm->cmsg_len=CMSG_LEN(sizeof(int)*nfd); memcpy(CMSG_DATA(cm),fds,sizeof(int)*nfd); }
  return sendmsg(c,&msg,0)==(ssize_t)sizeof hdr ? 0 : -1;
}

static int32_t gi(uint8_t*b,int i){int32_t v;memcpy(&v,b+4*i,4);return v;}
static void si(uint8_t*b,int i,int32_t v){memcpy(b+4*i,&v,4);}

#define MAGIC_INLINE 0x47585044u
static int connect_sock(const char* sp){
  int c=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sp,sizeof a.sun_path-1);
  if(connect(c,(struct sockaddr*)&a,sizeof a)<0){ close(c); return -1; }
  return c;
}
/* Rueckwaerts-Kompat: GXPD-inline (Kopie) reverse_string ueber eine eigene Verbindung. */
static int test_inline(const char* sp){
  int c=connect_sock(sp); if(c<0){P("inline connect\n");return 1;}
  const char* kn="reverse_string"; uint32_t nl=strlen(kn), insz=16;
  uint32_t hdr[3]={MAGIC_INLINE,nl,insz};
  if(wr(c,hdr,sizeof hdr)||wr(c,kn,nl)||wr(c,"0123456789ABCDEF",16)){P("inline send\n");close(c);return 1;}
  uint32_t st=1; if(rd(c,&st,4)){close(c);return 1;}
  if(st!=0){P("[inline] status=%u\n",st);close(c);return 1;}
  int64_t rv; uint32_t us,osz; char out[64];
  if(rd(c,&rv,8)||rd(c,&us,4)||rd(c,&osz,4)||osz>63||rd(c,out,osz)){close(c);return 1;}
  out[osz]=0; int ok=!strcmp(out,"EDCBA9876543210F"); /* reversiert n-1 Bytes, letztes Byte bleibt */
  P("[inline reverse_string] status=%u -> '%s' (erw 'EDCBA9876543210F') %s\n",st,out, ok?"OK":"?");
  close(c); return ok?0:1;
}

int main(int argc,char**argv){
  const char* sp = argc>1?argv[1]:"/data/local/tmp/gxpz.sock";
  test_inline(sp);
  int c=connect_sock(sp);
  if(c<0){P("connect %s fehlgeschlagen\n",sp);return 1;}
  P("verbunden mit %s\n",sp);

  uint32_t LEN=4096;
  int dfd=alloc_dmabuf(LEN); if(dfd<0) return 1;
  uint8_t* m=mmap(0,LEN,PROT_READ|PROT_WRITE,MAP_SHARED,dfd,0); if(m==MAP_FAILED){P("mmap\n");return 1;}
  memset(m,0,LEN);

  /* IMPORT: 1 Puffer, fd via SCM_RIGHTS am Header */
  int fds[1]={dfd};
  if(send_hdr(c,1,1,0,0,0,fds,1)){P("send IMPORT hdr\n");return 1;}
  uint32_t sz=LEN; if(wr(c,&sz,4)){P("send size\n");return 1;}
  uint32_t status=1, handle=0xffffffff;
  if(rd(c,&status,4)||status!=0){P("IMPORT status=%u\n",status);return 1;}
  if(rd(c,&handle,4)){P("IMPORT handle read\n");return 1;}
  P("IMPORT ok -> handle=%u\n",handle);

  int fail=0;
  /* --- vscale: [3,5,10,100] -> [3,15,30,300] (out[0]=Skalar bleibt) --- */
  { dbsync(dfd,DBS_START|DBS_WRITE); si(m,0,3);si(m,1,5);si(m,2,10);si(m,3,100); dbsync(dfd,DBS_END|DBS_WRITE);
    uint32_t hd[1]={handle}; const char* kn="vscale"; uint32_t nl=strlen(kn);
    if(send_hdr(c,2,nl,1,0,0,0,0)||wr(c,kn,nl)||wr(c,hd,4)){P("send RUN vscale\n");return 1;}
    int64_t rv=0; uint32_t us=0; if(rd(c,&status,4)||rd(c,&rv,8)||rd(c,&us,4)){P("RUN vscale resp\n");return 1;}
    dbsync(dfd,DBS_START|DBS_READ);
    int ok = status==0 && gi(m,0)==3 && gi(m,1)==15 && gi(m,2)==30 && gi(m,3)==300;
    P("[vscale] status=%u %uus -> [%d %d %d %d] (erw [3 15 30 300]) %s\n",status,us,gi(m,0),gi(m,1),gi(m,2),gi(m,3), ok?"OK":"FALSCH");
    dbsync(dfd,DBS_END|DBS_READ); if(!ok) fail=1;
  }
  /* --- vadd: [1,2,3,4]+[10,20,30,40] -> [11,22,33,44] (in-place) --- */
  { dbsync(dfd,DBS_START|DBS_WRITE); for(int i=0;i<4;i++){si(m,i,i+1);si(m,i+4,10*(i+1));} dbsync(dfd,DBS_END|DBS_WRITE);
    uint32_t hd[1]={handle}; const char* kn="vadd"; uint32_t nl=strlen(kn);
    if(send_hdr(c,2,nl,1,0,0,0,0)||wr(c,kn,nl)||wr(c,hd,4)){P("send RUN vadd\n");return 1;}
    int64_t rv=0; uint32_t us=0; if(rd(c,&status,4)||rd(c,&rv,8)||rd(c,&us,4)){P("RUN vadd resp\n");return 1;}
    dbsync(dfd,DBS_START|DBS_READ);
    int ok = status==0 && gi(m,0)==11 && gi(m,1)==22 && gi(m,2)==33 && gi(m,3)==44;
    P("[vadd]   status=%u %uus -> [%d %d %d %d] (erw [11 22 33 44]) %s\n",status,us,gi(m,0),gi(m,1),gi(m,2),gi(m,3), ok?"OK":"FALSCH");
    dbsync(dfd,DBS_END|DBS_READ); if(!ok) fail=1;
  }

  /* RELEASE */
  { uint32_t hd[1]={handle}; if(send_hdr(c,3,1,0,0,0,0,0)||wr(c,hd,4)){P("send RELEASE\n");return 1;}
    if(rd(c,&status,4)){P("RELEASE resp\n");return 1;} P("RELEASE status=%u\n",status); }

  munmap(m,LEN); close(dfd); close(c);
  P("\n=== GXPZ-Zero-Copy end-to-end: %s ===\n", fail?"FEHLGESCHLAGEN":"OK (DSP rechnete im geteilten dmabuf)");
  return fail;
}
