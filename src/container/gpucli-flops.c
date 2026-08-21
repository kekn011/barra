/* gpucli-flops - compute-bound FP32-Peak-Messung der Mali via gpud.
 * threads = groups*256, jeder Thread iters*16 FLOP. I/O minimal => misst rohe Rechenleistung.
 * Aufruf: gpucli-flops <sock> <flops.spv> [groups=4096] [iters=2000] [runs=3] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#define MAGIC 0x47505531u
#define F_INPUT 1u
#define F_OUTPUT 2u
static int rf(int fd,void*b,long n){char*p=b;while(n>0){long r=read(fd,p,n);if(r<=0)return -1;p+=r;n-=r;}return 0;}
static int wf(int fd,const void*b,long n){const char*p=b;while(n>0){long w=write(fd,p,n);if(w<=0)return -1;p+=w;n-=w;}return 0;}
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}

int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: gpucli-flops <sock> <flops.spv> [groups=4096] [iters=2000] [runs=3]\n"); return 2; }
  const char* sock=argv[1]; const char* spv=argv[2];
  uint32_t groups=argc>3?(uint32_t)atoi(argv[3]):4096;
  uint32_t iters=argc>4?(uint32_t)atoi(argv[4]):2000;
  int runs=argc>5?atoi(argv[5]):3;
  long threads=(long)groups*256; long szO=threads*4;
  FILE* f=fopen(spv,"rb"); if(!f){printf("SPIR-V?: %s\n",spv);return 1;}
  fseek(f,0,SEEK_END); long slen=ftell(f); fseek(f,0,SEEK_SET);
  void* spirv=malloc(slen); if(fread(spirv,1,slen,f)!=(size_t)slen){fclose(f);return 1;} fclose(f);
  float* O=malloc(szO); uint32_t P[4]={iters,0,0,0};

  double best=1e18;
  for(int r=0;r<runs;r++){
    int c=socket(AF_UNIX,SOCK_STREAM,0);
    struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
    if(connect(c,(struct sockaddr*)&a,sizeof a)<0){printf("connect %s: %s\n",sock,strerror(errno));return 1;}
    uint32_t hdr[6]={MAGIC,(uint32_t)slen,groups,1,1,2};
    uint32_t bd[4]={(uint32_t)szO,F_OUTPUT,16,F_INPUT};
    double t0=ms();
    if(wf(c,hdr,sizeof hdr)||wf(c,bd,sizeof bd)||wf(c,spirv,slen)||wf(c,P,16)){printf("send fail\n");return 1;}
    uint32_t status=999; if(rf(c,&status,4)){printf("recv status fail\n");return 1;}
    if(status!=0){printf("gpud status=%u\n",status);close(c);return 1;}
    if(rf(c,O,szO)){printf("recv fail\n");return 1;}
    double dt=ms()-t0; if(dt<best)best=dt;
    close(c);
  }
  double flop=(double)threads*iters*16.0;
  printf("O[0]=%.3f (nicht wegoptimiert) | threads=%ld iters=%u\n", O[0], threads, iters);
  printf("PEAK: %.0f MFLOP in %.1f ms (best) = %.1f GFLOPS\n", flop/1e6, best, flop/(best/1000.0)/1e9);
  return 0;
}
