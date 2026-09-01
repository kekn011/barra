/* gemv_bench — Mikrobench fuer die int-dot-GEMV-Kernel (1.9.2026), glibc, im Container.
 *   gemv_bench <shader.spv> <ne00> <ne01> <q6:0|1> [nstage=32] [nrep=5]
 * Alloziert W/XQ/D-dmabufs in Original-Layout, fuellt sie mit gutmuetigen Bytes (0x3C ->
 * f16 ~1.06, int8 60 - keine NaN/Inf-Verzerrung), dispatcht nstage identische GEMV-Stufen
 * je Batch (flags=1: ohne Zwischen-Barrieren = reiner Kernel-Durchsatz) und meldet GB/s
 * netto (nur Gewichtsbytes). Bau: gcc -O2 gemv_bench.c barra.c -o gemv_bench
 * Push-Constants = pc_gemv_iq aus ggml-gpud.cpp (20 u32). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "barra.h"
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint8_t* rdfile(const char* p, uint32_t* n){ FILE* f=fopen(p,"rb"); if(!f) return 0; fseek(f,0,SEEK_END); long l=ftell(f); fseek(f,0,SEEK_SET); uint8_t* b=malloc(l); if(fread(b,1,l,f)!=(size_t)l){ fclose(f); free(b); return 0; } fclose(f); *n=(uint32_t)l; return b; }
struct pc { uint32_t ne00, ne01, ne11, ne12, nb01u, nb02u, nb03u, nb11, nb12, nb13, nbd1, nbd2, nbd3, off0u, offxq, offd, r2, r3, n256, xqcols; };
int main(int argc, char** argv){
  if(argc<5){ fprintf(stderr,"gemv_bench <spv> <ne00> <ne01> <q6:0|1> [nstage] [nrep]\n"); return 2; }
  uint32_t slen=0; uint8_t* spv=rdfile(argv[1],&slen); if(!spv){ perror(argv[1]); return 1; }
  uint32_t ne00=atoi(argv[2]), ne01=atoi(argv[3]); int q6=atoi(argv[4]);
  int nstage=argc>5?atoi(argv[5]):32, nrep=argc>6?atoi(argv[6]):5;
  uint32_t n256=ne00/256, blkb=q6?210u:144u, unit=q6?2u:16u;
  uint64_t rowb=(uint64_t)n256*blkb, wbytes=rowb*ne01;
  uint32_t nb32=ne00/32, xqb=nb32*36u, db=ne01*4u;
  barra_gpu3 g; if(barra_gpu3_open(&g)){ fprintf(stderr,"gpuzc.sock?\n"); return 1; }
  int sh=barra_gpu3_load(&g,spv,slen,3,sizeof(struct pc)); if(sh<0){ fprintf(stderr,"LOAD fehlgeschlagen\n"); return 1; }
  barra_zbuf W,X,D; barra_zbuf* pb[3]={&W,&X,&D};
  if(barra_zc_alloc(&W,(uint32_t)wbytes)||barra_zc_alloc(&X,xqb<4096?4096:xqb)||barra_zc_alloc(&D,db<4096?4096:db)){ fprintf(stderr,"alloc\n"); return 1; }
  memset(W.map,0x3C,wbytes); memset(X.map,0x3C,xqb); memset(D.map,0,db);
  barra_zc_cpu_end(&W); barra_zc_cpu_end(&X); barra_zc_cpu_end(&D);
  if(barra_gpu3_import(&g,pb,3)){ fprintf(stderr,"import\n"); return 1; }
  barra_zbuf WN; memset(&WN,0,sizeof WN); WN.fd=-1; WN.tpu_h=-1; WN.dsp_h=-1;
  if(getenv("NATIVEW")){ int nh=barra_gpu3_alloc_native(&g,(uint32_t)wbytes);
    if(nh<0){ fprintf(stderr,"alloc_native\n"); return 1; }
    WN.size=(uint32_t)wbytes; WN.gpu_h=nh; pb[0]=&WN? &WN : pb[0]; /* W-Bind auf nativen Puffer umbiegen */ }
  struct pc pc={ .ne00=ne00,.ne01=ne01,.ne11=1,.ne12=1,
    .nb01u=(uint32_t)(rowb/unit),.nb02u=0,.nb03u=0,.nb11=ne00,.nb12=0,.nb13=0,
    .nbd1=ne01,.nbd2=0,.nbd3=0,.off0u=0,.offxq=0,.offd=0,.r2=1,.r3=1,.n256=n256,.xqcols=nb32 };
  uint32_t rows_wg=8; if(getenv("ROWSWG")) rows_wg=(uint32_t)atoi(getenv("ROWSWG"));
  uint32_t gx=(ne01+rows_wg-1)/rows_wg;
  barra_gpu3_bind bd[3]={{getenv("NATIVEW")?&WN:&W,0,0},{&X,0,0},{&D,0,0}};
  barra_gpu3_stage st[256]; if(nstage>256) nstage=256;
  for(int s=0;s<nstage;s++) st[s]=(barra_gpu3_stage){ sh,gx,1,1,bd,3,&pc,sizeof pc,1u };
  /* Warmup + Messung */
  if(barra_gpu3_batch(&g,st,4)){ fprintf(stderr,"Warmup-Batch fehlgeschlagen\n"); return 1; }
  double best=1e9, sum=0;
  for(int r=0;r<nrep;r++){ double t0=now(); if(barra_gpu3_batch(&g,st,nstage)){ fprintf(stderr,"Batch fehlgeschlagen\n"); return 1; } double dt=now()-t0; if(dt<best) best=dt; sum+=dt; }
  double gbs_best=(double)wbytes*nstage/best/1e9, gbs_avg=(double)wbytes*nstage/(sum/nrep)/1e9;
  printf("%s ne00=%u ne01=%u q6=%d W=%.1f MB rows/wg=%u: best %.2f GB/s (avg %.2f) | %.2f ms/Stufe best\n",
    argv[1],ne00,ne01,q6,wbytes/1e6,rows_wg,gbs_best,gbs_avg,best*1e3/nstage);
  barra_gpu3_release(&g,pb,3); barra_zc_free(&W); barra_zc_free(&X); barra_zc_free(&D);
  return 0;
}
