/* barra_tpu_bench: Modell m aus dem laufenden tpud N-mal per Zero-Copy inferieren, Zeit pro Inferenz messen.
   usage: barra_tpu_bench <model_id> <in_bytes> <out_bytes> [N=20]   (BARRA_SOCK_DIR wie ueblich) */
#include "barra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_us(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
int main(int argc,char**argv){
  if(argc<4){fprintf(stderr,"usage: %s model_id in_bytes out_bytes [N]\n",argv[0]);return 1;}
  int m=atoi(argv[1]); uint32_t ib=atoi(argv[2]),ob=atoi(argv[3]); int N=argc>4?atoi(argv[4]):20;
  barra_tpu t; if(barra_tpu_open(&t)){fprintf(stderr,"tpu_open fail\n");return 1;}
  barra_zbuf ZI,ZO; if(barra_zc_alloc(&ZI,ib)||barra_zc_alloc(&ZO,ob)){fprintf(stderr,"alloc fail\n");return 1;}
  short*zi=ZI.map; for(uint32_t i=0;i<ib/2;i++)zi[i]=(short)((i*7919)%2001-1000);
  uint32_t us=0; barra_tpu_infer(&t,m,&ZI,&ZO,&us); /* warm */
  double best=1e18,sum=0; for(int i=0;i<N;i++){double a=now_us();if(barra_tpu_infer(&t,m,&ZI,&ZO,&us)){fprintf(stderr,"infer fail\n");return 1;}double d=now_us()-a;sum+=d;if(d<best)best=d;}
  printf("model %d: N=%d avg %.2f ms  best %.2f ms  (tpud exec_us letzte %u)\n",m,N,sum/N/1000,best/1000,us);
  if(getenv("DUMP_OUT")){ barra_zc_cpu_begin(&ZO); const signed char*o=(const signed char*)ZO.map;
    unsigned long long h=1469598103934665603ULL; long s2=0;
    for(uint32_t i=0;i<ob;i++){ h^=(unsigned char)o[i]; h*=1099511628211ULL; s2+=o[i]; }
    printf("OUT fnv=%016llx sum=%ld first=%d,%d,%d,%d\n",h,s2,o[0],o[1],o[2],o[3]); }
  return 0;
}
