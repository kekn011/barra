/* barra_tpu_zcbench — TPU-Inferenz Zero-Copy vs Rohkopie fuer EIN Modell (Batch-Prefill-Fall).
 * Nutzt barra_tpu_infer (dmabuf, kein Socket-Transfer der Tensordaten). argv: model_id iters */
#include "barra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double ms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
int main(int argc,char**argv){
  uint32_t mid=argc>1?atoi(argv[1]):0; int iters=argc>2?atoi(argv[2]):20;
  barra_tpu t; if(barra_tpu_open(&t)){ fprintf(stderr,"tpu open fail\n"); return 1; }
  uint32_t isz=0,osz=0,nm=0; if(barra_tpu_info(&t,mid,&isz,&osz,&nm)){ fprintf(stderr,"info fail\n"); return 1; }
  printf("Modell %u: in=%u out=%u (%u Modelle)\n",mid,isz,osz,nm);
  barra_zbuf in,out; uint32_t ru=(isz+4095)/4096*4096, ro=(osz+4095)/4096*4096;
  if(barra_zc_alloc(&in,ru)||barra_zc_alloc(&out,ro)){ fprintf(stderr,"alloc fail\n"); return 1; }
  memset(in.map,3,isz);
  uint32_t us=0; if(barra_tpu_infer(&t,mid,&in,&out,&us)){ fprintf(stderr,"infer fail\n"); return 1; }  /* warm + import */
  double t0=ms(); for(int i=0;i<iters;i++) barra_tpu_infer(&t,mid,&in,&out,&us); double wall=(ms()-t0)/iters;
  printf("Zero-Copy: %.1f ms/Aufruf (Wand), TPU-Rechnung %u us\n",wall,us);
  barra_tpu_release(&t,(barra_zbuf*[]){&in,&out},2); barra_tpu_close(&t); barra_zc_free(&in); barra_zc_free(&out);
  return 0;
}
