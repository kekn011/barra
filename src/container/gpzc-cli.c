/* gpzc-cli — Zero-Copy-Client fuer gpud-zc. Alloziert dmabufs, schreibt Input,
 * schickt die fds via SCM_RIGHTS + vadd-SPIR-V, liest das Ergebnis aus der EIGENEN
 * dmabuf-mmap. Kein Datenbyte geht ueber den Socket. Baut fuer glibc + Bionic. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/types.h>
struct dma_heap_allocation_data { __u64 len; __u32 fd; __u32 fd_flags; __u64 heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0, struct dma_heap_allocation_data)
#define MAGIC 0x47505A43u
#define N 64
#define PAGE 4096

static int dmabuf(uint32_t* outfd, void** outmap){
  int h=open("/dev/dma_heap/system",O_RDONLY|O_CLOEXEC); if(h<0) return -1;
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d); d.len=PAGE; d.fd_flags=O_RDWR|O_CLOEXEC;
  if(ioctl(h,DMA_HEAP_IOCTL_ALLOC,&d)<0){ close(h); return -1; }
  close(h);
  void* m=mmap(0,PAGE,PROT_READ|PROT_WRITE,MAP_SHARED,d.fd,0);
  if(m==MAP_FAILED) return -1;
  *outfd=d.fd; *outmap=m; return 0;
}
static int wfull(int fd,const void*b,long n){const char*p=b;while(n>0){ssize_t w=write(fd,p,n);if(w<=0)return -1;p+=w;n-=w;}return 0;}

int main(int argc,char**argv){
  const char* sock=argc>1?argv[1]:"/opt/hwbridge/gpuzc.sock";
  const char* spv=argc>2?argv[2]:"/opt/hwbridge/vadd.spv";
  /* 3 dmabufs: A,B (Input), C (Output) */
  uint32_t fdA,fdB,fdC; void *mA,*mB,*mC;
  if(dmabuf(&fdA,&mA)||dmabuf(&fdB,&mB)||dmabuf(&fdC,&mC)){ fprintf(stderr,"dmabuf alloc fail\n"); return 1; }
  for(int i=0;i<N;i++){ ((float*)mA)[i]=(float)i; ((float*)mB)[i]=1000.0f; }
  memset(mC,0,PAGE);
  printf("Client: 3 dmabufs (A,B,C) alloziert, A[i]=i, B[i]=1000, C=0\n");

  /* SPIR-V laden */
  FILE* f=fopen(spv,"rb"); if(!f){ fprintf(stderr,"spirv %s fehlt\n",spv); return 1; }
  fseek(f,0,SEEK_END); long slen=ftell(f); fseek(f,0,SEEK_SET);
  uint32_t* code=malloc(slen); if(fread(code,1,slen,f)!=(size_t)slen){return 1;} fclose(f);

  int s=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof a.sun_path-1);
  if(connect(s,(struct sockaddr*)&a,sizeof a)<0){ perror("connect"); return 1; }

  /* sendmsg: header + fds via SCM_RIGHTS */
  int fds[3]={(int)fdA,(int)fdB,(int)fdC};
  uint32_t hdr[6]={MAGIC,(uint32_t)slen,1,1,1,3};
  char cbuf[CMSG_SPACE(sizeof(int)*3)]; memset(cbuf,0,sizeof cbuf);
  struct iovec iov={.iov_base=hdr,.iov_len=sizeof hdr};
  struct msghdr msg={.msg_iov=&iov,.msg_iovlen=1,.msg_control=cbuf,.msg_controllen=CMSG_SPACE(sizeof(int)*3)};
  struct cmsghdr* cm=CMSG_FIRSTHDR(&msg);
  cm->cmsg_level=SOL_SOCKET; cm->cmsg_type=SCM_RIGHTS; cm->cmsg_len=CMSG_LEN(sizeof(int)*3);
  memcpy(CMSG_DATA(cm),fds,sizeof(int)*3);
  if(sendmsg(s,&msg,0)<0){ perror("sendmsg"); return 1; }
  /* per-buf sizes + spirv */
  uint32_t sizes[3]={PAGE,PAGE,PAGE};
  if(wfull(s,sizes,sizeof sizes)||wfull(s,code,slen)){ fprintf(stderr,"send fail\n"); return 1; }
  printf("Client: fds via SCM_RIGHTS + SPIR-V geschickt (KEINE Pufferdaten)\n");

  uint32_t status=99; if(read(s,&status,4)!=4){ fprintf(stderr,"keine Antwort\n"); return 1; }
  printf("gpud-zc status=%u\n",status);
  if(status!=0){ printf("Compute-Fehler\n"); return 1; }

  float* c=(float*)mC;   /* Output aus UNSERER mmap */
  printf("output (aus UNSERER dmabuf-mmap): c[0]=%.0f c[1]=%.0f c[10]=%.0f c[63]=%.0f\n",c[0],c[1],c[10],c[63]);
  printf("erwartet c[i]=a[i]*2+b[i]: [1000 1002 1020 1126]\n");
  if(c[0]==1000&&c[1]==1002&&c[10]==1020&&c[63]==1126)
    printf(">>> ZERO-COPY-TRANSPORT OK: GPU rechnete in unseren dmabufs, null Datenkopie ueber den Socket. <<<\n");
  else printf(">>> unerwartet <<<\n");
  return 0;
}
