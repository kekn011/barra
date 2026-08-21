/* gpucli-persist - persistente GPU-Buffer (Command-Modus 'GPU2'): A,B einmal hochladen,
 * Matmul N-mal darauf rechnen, C einmal runterladen. Zeigt den Wegfall des Socket-I/O.
 * Aufruf: gpucli-persist <sock> <matmul.spv> [S=1024] [N=50] */
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
#define MAGIC_CMD 0x47505532u
#define CMD_ALLOC 1u
#define CMD_UPLOAD 3u
#define CMD_DOWNLOAD 4u
#define CMD_RUN 5u
static int rf(int fd,void*b,long n){char*p=b;while(n>0){long r=read(fd,p,n);if(r<=0)return -1;p+=r;n-=r;}return 0;}
static int wf(int fd,const void*b,long n){const char*p=b;while(n>0){long w=write(fd,p,n);if(w<=0)return -1;p+=w;n-=w;}return 0;}
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static float av(int i,int k,int K){ return (float)(((i*K+k)%97))/97.0f; }
static float bv(int k,int j,int N){ return (float)(((k*N+j)%89))/89.0f; }

static int u32(int fd,uint32_t v){ return wf(fd,&v,4); }
static int alloc_buf(int fd,uint32_t size,uint32_t*handle){
  if(u32(fd,CMD_ALLOC)||u32(fd,size)) return -1;
  uint32_t st=0,h=0; if(rf(fd,&st,4)||rf(fd,&h,4)) return -1; if(st) return -1; *handle=h; return 0;
}
static int upload(int fd,uint32_t h,const void*data,uint32_t size){
  if(u32(fd,CMD_UPLOAD)||u32(fd,h)||u32(fd,size)||wf(fd,data,size)) return -1;
  uint32_t st=1; if(rf(fd,&st,4)) return -1; return st?-1:0;
}
static int download(int fd,uint32_t h,void*data,uint32_t size){
  if(u32(fd,CMD_DOWNLOAD)||u32(fd,h)||u32(fd,size)) return -1;
  uint32_t st=1; if(rf(fd,&st,4)) return -1; if(st) return -1; return rf(fd,data,size);
}
static int run(int fd,const void*spirv,uint32_t slen,uint32_t gx,uint32_t gy,uint32_t*hnd,uint32_t nb){
  if(u32(fd,CMD_RUN)||u32(fd,slen)||u32(fd,gx)||u32(fd,gy)||u32(fd,1)||u32(fd,nb)) return -1;
  for(uint32_t i=0;i<nb;i++) if(u32(fd,hnd[i])) return -1;
  if(wf(fd,spirv,slen)) return -1;
  uint32_t st=1; if(rf(fd,&st,4)) return -1; return st?-1:0;
}

int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: gpucli-persist <sock> <matmul.spv> [S=1024] [N=50]\n"); return 2; }
  const char* sock=argv[1]; const char* spv=argv[2];
  int S=argc>3?atoi(argv[3]):1024; if(S%16)S=(S/16)*16; if(S<16)S=16; int N=argc>4?atoi(argv[4]):50;
  int M=S,NN=S,K=S;
  FILE* f=fopen(spv,"rb"); if(!f){printf("SPIR-V?: %s\n",spv);return 1;}
  fseek(f,0,SEEK_END); long slen=ftell(f); fseek(f,0,SEEK_SET);
  void* spirv=malloc(slen); if(fread(spirv,1,slen,f)!=(size_t)slen){fclose(f);return 1;} fclose(f);

  long szA=(long)M*K*4, szB=(long)K*NN*4, szC=(long)M*NN*4;
  float *A=malloc(szA),*B=malloc(szB),*C=malloc(szC);
  for(int i=0;i<M;i++) for(int k=0;k<K;k++) A[i*K+k]=av(i,k,K);
  for(int k=0;k<K;k++) for(int j=0;j<NN;j++) B[k*NN+j]=bv(k,j,NN);
  uint32_t dims[4]={(uint32_t)M,(uint32_t)NN,(uint32_t)K,0};

  int c=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
  if(connect(c,(struct sockaddr*)&a,sizeof a)<0){ printf("connect %s: %s\n",sock,strerror(errno)); return 1; }
  if(u32(c,MAGIC_CMD)){ printf("magic send fail\n"); return 1; }

  uint32_t hA,hB,hC,hD;
  if(alloc_buf(c,szA,&hA)||alloc_buf(c,szB,&hB)||alloc_buf(c,szC,&hC)||alloc_buf(c,16,&hD)){printf("ALLOC fail\n");return 1;}
  double tu0=ms();
  if(upload(c,hA,A,szA)||upload(c,hB,B,szB)||upload(c,hD,dims,16)){printf("UPLOAD fail\n");return 1;}
  double tup=ms()-tu0;
  uint32_t hnd[4]={hA,hB,hC,hD};
  double tr0=ms();
  for(int r=0;r<N;r++){ if(run(c,spirv,slen,NN/16,M/16,hnd,4)){printf("RUN %d fail\n",r);return 1;} }
  double trun=ms()-tr0;
  if(download(c,hC,C,szC)){printf("DOWNLOAD fail\n");return 1;}
  close(c);

  int bad=0; for(int t=0;t<200;t++){ int i=(t*37)%M,j=(t*53)%NN; double s=0; for(int k=0;k<K;k++) s+=(double)av(i,k,K)*bv(k,j,NN); if(fabs((double)C[i*NN+j]-s)>1e-2) bad++; }
  double gflops=(2.0*M*NN*K*N)/(trun/1000.0)/1e9;
  printf("%s: %dx%dx%d, upload(A,B) 1x=%.1fms, %d RUNs in %.1fms = %.2fms/run, %.1f GFLOPS | Stichprobe %d/200 falsch\n",
    bad?"!! FEHLER":"PERSIST OK", M,NN,K, tup, N, trun, trun/N, gflops, bad);
  return bad?1:0;
}
