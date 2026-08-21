/* gxpcli - Test-Client fuer die Live-gxpd-Bruecke (GXPD-Protokoll ueber Unix-Socket).
 *   gxpcli [sockpath] [kernel]     default: gxp.sock, kernel softmul (1.5*3.0=4.5) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
static int rd(int fd,void*p,int n){int g=0;char*b=p;while(g<n){int k=read(fd,b+g,n-g);if(k<=0)return-1;g+=k;}return 0;}
int main(int argc,char**argv){
  const char* sp = argc>1?argv[1]:"/data/local/ubuntu/opt/hwbridge/gxp.sock";
  const char* name = argc>2?argv[2]:"softmul";
  int s=socket(AF_UNIX,SOCK_STREAM,0); if(s<0){perror("socket");return 1;}
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sp,sizeof a.sun_path-1);
  if(connect(s,(void*)&a,sizeof a)){perror("connect");return 1;}
  uint32_t in[3]={0x3FC00000u,0x40400000u,0u}; /* 1.5, 3.0, platz */
  uint32_t nlen=strlen(name), insz=12, hdr[3]={0x47585044u,nlen,insz};
  if(write(s,hdr,12)!=12||write(s,name,nlen)!=(ssize_t)nlen||write(s,in,insz)!=(ssize_t)insz){perror("write");return 1;}
  uint32_t st; if(rd(s,&st,4)){printf("keine Antwort\n");return 1;}
  printf("status=%u\n",st);
  if(st==0){ int64_t rv; uint32_t us,osz; if(rd(s,&rv,8)||rd(s,&us,4)||rd(s,&osz,4)){printf("kurze Antwort\n");return 1;}
    uint8_t out[64]={0}; if(osz>64)osz=64; if(rd(s,out,osz)){printf("kein out\n");return 1;}
    uint32_t r; memcpy(&r,out+8,4); float fr; memcpy(&fr,&r,4);
    printf("softmul out[2]=0x%08x (=%.4f)  exec_us=%u   [erwartet 0x40900000 = 4.5]\n",r,fr,us);
    return (r==0x40900000u)?0:2; }
  return 1;
}
