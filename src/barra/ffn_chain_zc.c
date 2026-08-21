/* M2: FFN-Kette L0->L1 auf der TPU mit ZERO-COPY-Handoff (Zwischen-dmabuf geteilt). */
#include "barra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_us(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
int main(int argc,char**argv){
  const char* inpath=argv[1]; const char* outpath=argv[2];
  uint32_t B=argc>3?atoi(argv[3]):32, D=argc>4?atoi(argv[4]):896;
  uint32_t sz=B*D*2; /* int16 */
  barra_tpu t; if(barra_tpu_open(&t)){ fprintf(stderr,"tpu_open fail\n"); return 1; }
  barra_zbuf ZI,ZM,ZO;
  if(barra_zc_alloc(&ZI,sz)||barra_zc_alloc(&ZM,sz)||barra_zc_alloc(&ZO,sz)){ fprintf(stderr,"alloc fail\n"); return 1; }
  /* load x0_i16 into ZI */
  FILE* f=fopen(inpath,"rb"); if(!f){fprintf(stderr,"no input\n");return 1;} fread(ZI.map,1,sz,f); fclose(f);
  uint32_t us0=0,us1=0;
  /* warmup + chain */
  barra_tpu_infer(&t,0,&ZI,&ZM,&us0); barra_tpu_infer(&t,1,&ZM,&ZO,&us1);
  /* timed: chain with zero-copy handoff (ZM is L0-out AND L1-in, no copy) */
  int N=100; double t0=now_us();
  for(int i=0;i<N;i++){ barra_tpu_infer(&t,0,&ZI,&ZM,&us0); barra_tpu_infer(&t,1,&ZM,&ZO,&us1); }
  double per=(now_us()-t0)/N;
  fprintf(stderr,"[chain] L0 %u us + L1 %u us ; wall %.1f us/chain (%.2f ms/Tok @B=%u)\n",us0,us1,per,per/B/1000.0,B);
  /* write ZO out */
  FILE* g=fopen(outpath,"wb"); fwrite(ZO.map,1,sz,g); fclose(g);
  return 0;
}
