/* gpucli-mm - Matmul-Benchmark fuer gpud (GPU-Durchsatz/GFLOPS + grosse Workload).
 * C[MxN] = A[MxK] * B[KxN] auf der Mali via SPIR-V. Verifiziert Stichprobe gegen CPU, misst GFLOPS.
 * Aufruf: gpucli-mm <sock> <matmul.spv> [S=1024] [runs=3]   (M=N=K=S, Vielfaches von 16) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#define MAGIC 0x47505531u
#define F_INPUT 1u
#define F_OUTPUT 2u
static int rf(int fd,void*b,long n){char*p=b;while(n>0){long r=read(fd,p,n);if(r<=0)return -1;p+=r;n-=r;}return 0;}
static int wf(int fd,const void*b,long n){const char*p=b;while(n>0){long w=write(fd,p,n);if(w<=0)return -1;p+=w;n-=w;}return 0;}
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static float av(int i,int k,int K){ return (float)(((i*K+k)%97))/97.0f; }
static float bv(int k,int j,int N){ return (float)(((k*N+j)%89))/89.0f; }

int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: gpucli-mm <sock> <matmul.spv> [S=1024] [runs=3]\n"); return 2; }
  const char* sock=argv[1]; const char* spv=argv[2];
  int S=argc>3?atoi(argv[3]):1024; if(S%16)S=(S/16)*16; if(S<16)S=16; int runs=argc>4?atoi(argv[4]):3;
  int M=S,N=S,K=S;
  FILE* f=fopen(spv,"rb"); if(!f){ printf("SPIR-V?: %s\n",spv); return 1; }
  fseek(f,0,SEEK_END); long slen=ftell(f); fseek(f,0,SEEK_SET);
  void* spirv=malloc(slen); if(fread(spirv,1,slen,f)!=(size_t)slen){fclose(f);return 1;} fclose(f);

  long szA=(long)M*K*4, szB=(long)K*N*4, szC=(long)M*N*4;
  float *A=malloc(szA),*B=malloc(szB),*C=malloc(szC);
  for(int i=0;i<M;i++) for(int k=0;k<K;k++) A[i*K+k]=av(i,k,K);
  for(int k=0;k<K;k++) for(int j=0;j<N;j++) B[k*N+j]=bv(k,j,N);
  uint32_t dims[4]={(uint32_t)M,(uint32_t)N,(uint32_t)K,0};

  double t0=ms(); int last_bad=0;
  for(int r=0;r<runs;r++){
    int c=socket(AF_UNIX,SOCK_STREAM,0);
    struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
    if(connect(c,(struct sockaddr*)&a,sizeof a)<0){ printf("connect %s: %s\n",sock,strerror(errno)); return 1; }
    uint32_t hdr[6]={MAGIC,(uint32_t)slen,(uint32_t)(N/16),(uint32_t)(M/16),1,4};
    uint32_t bd[8]={(uint32_t)szA,F_INPUT,(uint32_t)szB,F_INPUT,(uint32_t)szC,F_OUTPUT,16,F_INPUT};
    if(wf(c,hdr,sizeof hdr)||wf(c,bd,sizeof bd)||wf(c,spirv,slen)||wf(c,A,szA)||wf(c,B,szB)||wf(c,dims,16)){printf("send fail\n");return 1;}
    uint32_t status=999; if(rf(c,&status,4)){printf("recv status fail\n");return 1;}
    if(status!=0){ printf("gpud status=%u\n",status); close(c); return 1; }
    if(rf(c,C,szC)){printf("recv C fail\n");return 1;}
    close(c);
    if(r==runs-1){ /* Stichprobe gegen CPU */
      for(int t=0;t<200;t++){ int i=(t*37)%M, j=(t*53)%N; double s=0; for(int k=0;k<K;k++) s+=(double)av(i,k,K)*bv(k,j,N);
        if(fabs((double)C[i*N+j]-s)>1e-2) last_bad++; }
    }
  }
  double dt=ms()-t0;
  double gflops = (2.0*M*N*K*runs)/(dt/1000.0)/1e9;
  printf("C[0]=%.3f C[M*N-1]=%.3f | Stichprobe %d/200 falsch\n", C[0], C[(long)M*N-1], last_bad);
  printf("%s: %dx%dx%d, %d runs in %.1f ms = %.2f ms/matmul, %.1f GFLOPS\n",
    last_bad?"!! FEHLER":"MATMUL OK", M,N,K, runs, dt, dt/runs, gflops);
  return last_bad?1:0;
}
