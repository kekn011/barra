// gxpvcli — Client fuer den DSP-Rechen-Kernel "vscale" (out[i]=in[i]*k).
// Baut fuer Bionic (Android) UND glibc (Ubuntu-Container).
//   gxpvcli <sock> <k> <v1> <v2> ...   -> sendet [k,v1,v2,...] als int32, druckt Ergebnis
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#define MAGIC 0x47585044u
static int rn(int fd,void*p,size_t n){uint8_t*b=p;size_t g=0;while(g<n){ssize_t k=read(fd,b+g,n-g);if(k<=0)return -1;g+=k;}return 0;}
static int wn(int fd,const void*p,size_t n){const uint8_t*b=p;size_t s=0;while(s<n){ssize_t k=write(fd,b+s,n-s);if(k<=0)return -1;s+=k;}return 0;}
int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: %s <sock> <k> <v1> [v2 ...]\n",argv[0]); return 2; }
  const char* sock=argv[1];
  const char* fn="vscale"; uint32_t nlen=strlen(fn);
  int nv=argc-2; uint32_t in_size=nv*4;
  int32_t* buf=malloc(in_size);
  for(int i=0;i<nv;i++) buf[i]=(int32_t)strtol(argv[2+i],0,0);
  int s=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX;
  strncpy(a.sun_path,sock,sizeof a.sun_path-1);
  if(connect(s,(struct sockaddr*)&a,sizeof a)<0){ perror("connect"); return 1; }
  uint32_t hdr[3]={MAGIC,nlen,in_size};
  wn(s,hdr,sizeof hdr); wn(s,fn,nlen); wn(s,buf,in_size);
  uint32_t status; if(rn(s,&status,4)){ fprintf(stderr,"keine Antwort\n"); return 1; }
  if(status!=0){ printf("STATUS=%u (Fehler)\n",status); return 1; }
  int64_t rv; uint32_t exec_us, osz;
  rn(s,&rv,8); rn(s,&exec_us,4); rn(s,&osz,4);
  int32_t* out=malloc(osz); rn(s,out,osz);
  printf("OK rv=%lld exec=%uus  (k=%d)\n",(long long)rv,exec_us,buf[0]);
  printf("in :"); for(int i=0;i<nv;i++) printf(" %d",buf[i]); printf("\n");
  printf("out:"); for(uint32_t i=0;i<osz/4;i++) printf(" %d",out[i]); printf("\n");
  return 0;
}
