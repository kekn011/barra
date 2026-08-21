// tiecli.c — minimal GXPD client (Bionic). Sends int32 args to a kernel, prints output.
// Build: aarch64-linux-android31-clang tiecli.c -o tiecli
// Use:   tiecli <sockpath> <kernelname> <i0> <i1> ...   (each iN is a signed int32)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#define MAGIC 0x47585044u
static int rd(int fd,void*p,size_t n){uint8_t*b=p;size_t g=0;while(g<n){ssize_t k=read(fd,b+g,n-g);if(k<=0)return -1;g+=k;}return 0;}
static int wr(int fd,const void*p,size_t n){const uint8_t*b=p;size_t s=0;while(s<n){ssize_t k=write(fd,b+s,n-s);if(k<=0)return -1;s+=k;}return 0;}
int main(int argc,char**argv){
  if(argc<3){fprintf(stderr,"usage: %s sock kernel i0 i1...\n",argv[0]);return 2;}
  const char*sock=argv[1];const char*kern=argv[2];
  int nv=argc-3; uint32_t insz=(nv?nv:1)*4;
  int32_t*in=calloc(nv?nv:1,4);
  for(int i=0;i<nv;i++) in[i]=(int32_t)strtol(argv[3+i],0,0);
  int fd=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof a.sun_path-1);
  if(connect(fd,(void*)&a,sizeof a)){perror("connect");return 1;}
  uint32_t hdr[3]={MAGIC,(uint32_t)strlen(kern),insz};
  wr(fd,hdr,12); wr(fd,kern,strlen(kern)); wr(fd,in,insz);
  uint32_t st; if(rd(fd,&st,4)){fprintf(stderr,"no resp\n");return 1;}
  printf("status=%u\n",st);
  if(st==0){
    int64_t rv; uint32_t us,osz; rd(fd,&rv,8); rd(fd,&us,4); rd(fd,&osz,4);
    uint8_t*out=malloc(osz); rd(fd,out,osz);
    printf("rv=%lld exec_us=%u out_size=%u\n",(long long)rv,us,osz);
    printf("in : "); for(int i=0;i<nv;i++) printf("%d ",in[i]); printf("\n");
    printf("out: "); for(uint32_t i=0;i+4<=osz;i+=4){int32_t v;memcpy(&v,out+i,4);printf("%d ",v);} printf("\n");
    printf("hex: "); for(uint32_t i=0;i<osz;i++) printf("%02x",out[i]); printf("\n");
  }
  return 0;
}
