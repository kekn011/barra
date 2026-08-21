/* gpucli - Test/Benchmark-Client fuer gpud. Schickt vadd-SPIR-V + 3 Buffer (A,B input, C output),
 * prueft c[i]=a[i]*2+b[i]. Baut sowohl als Bionic (Android) als auch glibc (Ubuntu-Container). Reines POSIX.
 * Aufruf: gpucli <sock> <vadd.spv> [N] [runs] */
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
  if(argc<3){ fprintf(stderr,"usage: gpucli <sock> <vadd.spv> [N=4096] [runs=1]\n"); return 2; }
  const char* sock=argv[1]; const char* spv=argv[2];
  int N=argc>3?atoi(argv[3]):4096; int runs=argc>4?atoi(argv[4]):1;
  if(N%64){ N=(N/64)*64; if(N==0)N=64; }

  FILE* f=fopen(spv,"rb"); if(!f){ printf("kann SPIR-V nicht oeffnen: %s\n",spv); return 1; }
  fseek(f,0,SEEK_END); long slen=ftell(f); fseek(f,0,SEEK_SET);
  void* spirv=malloc(slen); if(fread(spirv,1,slen,f)!=(size_t)slen){fclose(f);return 1;} fclose(f);

  long bytes=(long)N*4;
  float *A=malloc(bytes),*B=malloc(bytes),*C=malloc(bytes);
  for(int i=0;i<N;i++){ A[i]=(float)i; B[i]=(float)(3*i); }

  double t0=ms(); int last_bad=0;
  for(int r=0;r<runs;r++){
    int c=socket(AF_UNIX,SOCK_STREAM,0);
    struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
    if(connect(c,(struct sockaddr*)&a,sizeof a)<0){ printf("connect %s: %s\n",sock,strerror(errno)); return 1; }
    uint32_t hdr[6]={MAGIC,(uint32_t)slen,(uint32_t)(N/64),1,1,3};
    uint32_t bd[6]={(uint32_t)bytes,F_INPUT,(uint32_t)bytes,F_INPUT,(uint32_t)bytes,F_OUTPUT};
    if(wf(c,hdr,sizeof hdr)||wf(c,bd,sizeof bd)||wf(c,spirv,slen)||wf(c,A,bytes)||wf(c,B,bytes)){printf("send fail\n");return 1;}
    uint32_t status=999; if(rf(c,&status,4)){printf("recv status fail\n");return 1;}
    if(status!=0){ printf("gpud status=%u (Fehler)\n",status); close(c); return 1; }
    if(rf(c,C,bytes)){printf("recv C fail\n");return 1;}
    close(c);
    if(r==runs-1){ for(int i=0;i<N;i++){ float e=A[i]*2.0f+B[i]; if(C[i]!=e) last_bad++; } }
  }
  double dt=ms()-t0;
  printf("C[0..5] = %.0f %.0f %.0f %.0f %.0f %.0f (erwartet 0 5 10 15 20 25)\n",C[0],C[1],C[2],C[3],C[4],C[5]);
  printf("%s: %d/%d falsch | %d runs in %.1f ms = %.2f ms/req\n", last_bad?"!! FEHLER":"GPU-COMPUTE OK", last_bad, N, runs, dt, dt/runs);
  return last_bad?1:0;
}
