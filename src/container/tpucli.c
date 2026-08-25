/* tpucli - Client fuer Multi-Modell-tpud (Protokoll TPD2).
 *   Request:  magic 'TPD2', model_id, in_size, [input]
 *   Response: status, out_size, [output]
 * Aufruf: tpucli <sock> <in_size> [N=1] [model_id=0]
 * Verifiziert bei out_size==in_size die Identity (fuer pk_attn), sonst nur status+Groesse. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#define MAGIC 0x54504432u
static int rf(int fd,void*b,long n){char*p=b;while(n>0){long r=read(fd,p,n);if(r<=0)return -1;p+=r;n-=r;}return 0;}
static int wf(int fd,const void*b,long n){const char*p=b;while(n>0){long w=write(fd,p,n);if(w<=0)return -1;p+=w;n-=w;}return 0;}
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}

int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: tpucli <sock> <in_size> [N=1] [model_id=0]\n"); return 2; }
  const char* sock=argv[1]; long isz=atol(argv[2]);
  int N=argc>3?atoi(argv[3]):1; uint32_t mid=argc>4?(uint32_t)atoi(argv[4]):0;
  if(isz<=0){ fprintf(stderr,"in_size ungueltig\n"); return 2; }
  #define OUTCAP (16*1024*1024)
  unsigned char* in=malloc(isz); unsigned char* out=malloc(OUTCAP);
  if(!in||!out){ fprintf(stderr,"malloc fehlgeschlagen\n"); return 1; }
  for(long i=0;i<isz;i++) in[i]=(i+1)&0xff;

  int c=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
  if(connect(c,(struct sockaddr*)&a,sizeof a)<0){ printf("connect %s: %s\n",sock,strerror(errno)); return 1; }
  double t0=ms(); int badreq=0; uint32_t out_size=0;
  for(int k=0;k<N;k++){
    uint32_t hdr[3]={MAGIC,mid,(uint32_t)isz};
    if(wf(c,hdr,sizeof hdr)||wf(c,in,isz)){printf("send fail\n");return 1;}
    uint32_t st=999; if(rf(c,&st,4)||rf(c,&out_size,4)){printf("recv hdr fail\n");return 1;}
    if(st!=0){ printf("status=%u (Modell %u, in_size %ld abgelehnt?)\n",st,mid,isz); return 1; }
    if(out_size>OUTCAP){ printf("out_size=%u > Puffer %d — abgebrochen\n",out_size,OUTCAP); return 1; }
    if(rf(c,out,out_size)){printf("recv out fail\n");return 1;}
    if((long)out_size==isz){ long d=0; for(long i=0;i<isz;i++) if(out[i]!=in[i]) d++; if(d) badreq++; }
  }
  double dt=ms()-t0; close(c);
  printf("Modell %u: in=%ld out=%u | %d Req in %.1fms = %.2fms/req", mid,isz,out_size,N,dt,dt/N);
  if((long)out_size==isz) printf(" | %d/%d nicht-Identity", badreq,N);
  printf("\n");
  return 0;
}
