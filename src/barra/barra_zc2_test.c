/* barra_zc2_test — Verifikation + Benchmark der GPU-Zero-Copy-Pfade:
 *   (1) v1 Einzel-Dispatch (GPZC, Import je Aufruf)        - Kompatibilitaet
 *   (2) v2 Session: Import einmal, Batch aus 2 Stufen       - Korrektheit
 *   (3) grosse Puffer (1 MiB je Puffer, alle Elemente geprueft) - Cache-Sync real
 *   (4) Benchmark: us/Dispatch v1 vs v2 vs v2-Batch(2)
 * Shader: /opt/hwbridge/vadd.spv  c[i]=a[i]*2+b[i], local_size 64. */
#include "barra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint8_t* slurp(const char* p, uint32_t* n){
  FILE* f=fopen(p,"rb"); if(!f){ fprintf(stderr,"%s fehlt\n",p); return 0; }
  fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t* b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){return 0;} fclose(f); *n=(uint32_t)s; return b;
}
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int fails=0;
#define CHECK(c,msg) do{ if(c) printf("  ok   %s\n",msg); else { printf("  FAIL %s\n",msg); fails++; } }while(0)

int main(int argc,char**argv){
  int iters=argc>1?atoi(argv[1]):200;
  uint32_t slen=0; uint8_t* spv=slurp("/opt/hwbridge/vadd.spv",&slen); if(!spv) return 1;
  const uint32_t PAGE=4096, N=64;
  barra_zbuf A,B,C,D;
  if(barra_zc_alloc(&A,PAGE)||barra_zc_alloc(&B,PAGE)||barra_zc_alloc(&C,PAGE)||barra_zc_alloc(&D,PAGE)){ fprintf(stderr,"zc_alloc fehlgeschlagen\n"); return 1; }
  float *a=A.map,*b=B.map,*c=C.map,*d=D.map;
  for(uint32_t i=0;i<N;i++){ a[i]=(float)i; b[i]=1000.0f; c[i]=0; d[i]=0; }

  printf("[1] v1 Einzel-Dispatch (Kompatibilitaet)\n");
  barra_zbuf s1[3]={A,B,C}, s2[3]={C,A,D};
  int r1=barra_gpu_zc(spv,slen,1,1,1,s1,3); int r2=barra_gpu_zc(spv,slen,1,1,1,s2,3);
  CHECK(r1==0&&r2==0,"beide Dispatches status 0");
  CHECK(c[10]==1020&&d[0]==2000&&d[1]==2005&&d[10]==2050&&d[63]==2315,"C=2i+1000, D=5i+2000");

  printf("[2] v2 Session: Import einmal, Batch aus 2 Stufen (GPU-Barriere dazwischen)\n");
  memset(c,0,PAGE); memset(d,0,PAGE);
  barra_gpu g; CHECK(barra_gpu_open(&g)==0,"Session offen");
  barra_zbuf* all[4]={&A,&B,&C,&D};
  CHECK(barra_gpu_import(&g,all,4)==0,"4 Puffer importiert");
  printf("       Handles: A=%d B=%d C=%d D=%d\n",A.gpu_h,B.gpu_h,C.gpu_h,D.gpu_h);
  barra_zbuf* st1[3]={&A,&B,&C}; barra_zbuf* st2[3]={&C,&A,&D};
  barra_gpu_stage st[2]={{spv,slen,1,1,1,st1,3},{spv,slen,1,1,1,st2,3}};
  CHECK(barra_gpu_batch(&g,st,2)==0,"Batch(2) status 0");
  CHECK(c[10]==1020&&d[0]==2000&&d[1]==2005&&d[10]==2050&&d[63]==2315,"Ergebnis identisch zu v1 (Barriere korrekt)");
  /* nochmal mit geaenderten Eingaben: Handles bleiben gueltig, kein Re-Import */
  for(uint32_t i=0;i<N;i++){ a[i]=(float)(2*i); b[i]=0; }
  CHECK(barra_gpu_batch(&g,st,2)==0,"Batch(2) erneut (persistente Handles)");
  CHECK(c[5]==20&&d[5]==50,"neue Eingaben wirken: C=4i, D=10i");

  printf("[3] grosse Puffer: 1 MiB je Puffer, 262144 floats, alle Elemente geprueft\n");
  const uint32_t BIG=1u<<20, NB=BIG/4;
  barra_zbuf X,Y,Z; if(barra_zc_alloc(&X,BIG)||barra_zc_alloc(&Y,BIG)||barra_zc_alloc(&Z,BIG)){ fprintf(stderr,"zc_alloc(big) fehlgeschlagen\n"); return 1; }
  float *x=X.map,*y=Y.map,*z=Z.map;
  for(uint32_t i=0;i<NB;i++){ x[i]=(float)(i%1000); y[i]=(float)(i%7); z[i]=-1; }
  barra_zbuf* big[3]={&X,&Y,&Z};
  CHECK(barra_gpu_dispatch(&g,spv,slen,NB/64,1,1,big,3)==0,"Dispatch 1 MiB status 0");
  uint32_t bad=0; for(uint32_t i=0;i<NB;i++) if(z[i]!=x[i]*2+y[i]) bad++;
  printf("       falsche Elemente: %u / %u\n",bad,NB); CHECK(bad==0,"alle 262144 Elemente korrekt");
  /* zweiter Durchlauf mit veraenderten Daten (Cache-Sync muss neu greifen) */
  for(uint32_t i=0;i<NB;i++){ x[i]=(float)(i%13); y[i]=100; }
  CHECK(barra_gpu_dispatch(&g,spv,slen,NB/64,1,1,big,3)==0,"Dispatch 1 MiB #2 status 0");
  bad=0; for(uint32_t i=0;i<NB;i++) if(z[i]!=x[i]*2+y[i]) bad++;
  printf("       falsche Elemente: %u / %u\n",bad,NB); CHECK(bad==0,"alle Elemente korrekt (Durchlauf 2)");

  printf("[4] Benchmark (%d Iterationen, 4-KiB-Puffer, vadd 64 Elemente)\n",iters);
  double t0=now_us(); for(int i=0;i<iters;i++) barra_gpu_zc(spv,slen,1,1,1,s1,3); double tv1=(now_us()-t0)/iters;
  t0=now_us(); for(int i=0;i<iters;i++) barra_gpu_dispatch(&g,spv,slen,1,1,1,st1,3); double tv2=(now_us()-t0)/iters;
  t0=now_us(); for(int i=0;i<iters;i++) barra_gpu_batch(&g,st,2); double tb2=(now_us()-t0)/iters;
  printf("       v1 einzel (Import je Aufruf)     : %8.0f us/Dispatch\n",tv1);
  printf("       v2 Session (persistente Handles)  : %8.0f us/Dispatch  (x%.1f)\n",tv2,tv1/tv2);
  printf("       v2 Batch(2) (ein Roundtrip)       : %8.0f us/Batch = %.0f us/Stufe (x%.1f)\n",tb2,tb2/2,tv1/(tb2/2));
  CHECK(tv2<tv1,"v2 schneller als v1");

  CHECK(barra_gpu_release(&g,big,3)==0,"Release 3 Handles");
  barra_gpu_close(&g);
  barra_zc_free(&X); barra_zc_free(&Y); barra_zc_free(&Z);
  barra_zc_free(&A); barra_zc_free(&B); barra_zc_free(&C); barra_zc_free(&D);
  printf(fails?">>> %d FEHLER <<<\n":">>> ALLE TESTS OK <<<\n",fails);
  return fails?1:0;
}
