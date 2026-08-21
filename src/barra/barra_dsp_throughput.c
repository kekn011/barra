/* barra_dsp_throughput — zeigt beide Durchsatz-Hebel ueber libbarra:
 *  A) BATCH-Kernel "vscalen": ein Dispatch skaliert N Elemente (buf[0]=N, buf[1]=s,
 *     buf[2..N+1]*=s). Elemente/s ueber N -> Command-Overhead amortisiert.
 *  B) ASYNC-Pipeline: barra_dsp_submit/wait, K Jobs in-flight vs synchron barra_dsp_run.
 * Bauen mit barra.c. Lauf (root): BARRA_SOCK_DIR=/data/local/ubuntu/opt/hwbridge barra_dsp_throughput
 */
#include "barra.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

int main(void){
  barra_dsp d; if(barra_dsp_open(&d)){ P("dsp open\n"); return 1; }

  /* ===== A) BATCH-Kernel vscalen ===== */
  P("=== A) Batch-Kernel 'vscalen' (ein Dispatch skaliert N Elemente) ===\n");
  uint32_t CAP=8u*1024*1024; barra_zbuf z; if(barra_zc_alloc(&z,CAP)){ P("alloc\n"); return 1; }
  int32_t* m=(int32_t*)z.map; barra_zbuf* zb[1]={&z};
  int64_t rv; uint32_t us; int okall=1;
  /* Batch-Puffer CACHED vor-importieren (~30x); run re-importiert dann nicht */
  if(barra_dsp_import_ex(&d,zb,1,1)){ P("cached import\n"); return 1; }
  P("(Batch-Puffer cacheable importiert, dsp_h=%d)\n", z.dsp_h);
  /* Korrektheit N=8, s=3 */
  m[0]=8; m[1]=3; for(int i=0;i<8;i++) m[2+i]=i+1;
  if(barra_dsp_run(&d,"vscalen",zb,1,&rv,&us)){ P("vscalen run\n"); return 1; }
  int okc=(m[2]==3 && m[3]==6 && m[9]==24); P("Korrektheit N=8 s=3: buf[2..]=[%d %d %d ...] (erw 3 6 9 ...) %s\n",m[2],m[3],m[4], okc?"OK":"?"); if(!okc)okall=0;
  P("%-10s %-12s %-14s %-12s\n","N","ms/Dispatch","Mio Elem/s","Dispatch/s");
  int Ns[]={4,1024,16384,262144,1048576,2097100};   /* cached -> auch grosse N unter dem Watchdog */
  for(unsigned i=0;i<sizeof Ns/sizeof*Ns;i++){ uint32_t N=Ns[i]; if((uint64_t)(N+2)*4>CAP) break;
    m[0]=N; m[1]=2; for(uint32_t j=0;j<N && j<16;j++) m[2+j]=1;
    for(int w=0;w<2;w++) barra_dsp_run(&d,"vscalen",zb,1,&rv,&us);
    int IT= N>200000?40:200; double t0=now(); for(int it=0;it<IT;it++) barra_dsp_run(&d,"vscalen",zb,1,&rv,&us); double dt=(now()-t0)/IT;
    P("%-10u %-12.3f %-14.2f %-12.0f\n", N, dt*1e3, (N/dt)/1e6, 1.0/dt);
  }
  barra_zc_free(&z);

  /* ===== B) ASYNC-Pipeline vs synchron (kleiner vscale) ===== */
  P("\n=== B) Async-Pipeline (barra_dsp_submit/wait) vs synchron ===\n");
  enum{ KM=8 }; barra_zbuf zs[KM]; for(int k=0;k<KM;k++){ if(barra_zc_alloc(&zs[k],16)){P("alloc zs\n");return 1;}
    int32_t* p=zs[k].map; p[0]=3;p[1]=5;p[2]=10;p[3]=100; }
  /* alle Puffer EINMAL vor-importieren (<=ZC_MAXBUF=8 pro Import) -> kein Import mitten in der Async-Schleife */
  { barra_zbuf* all[KM]; for(int k=0;k<KM;k++) all[k]=&zs[k]; if(barra_dsp_import(&d,all,KM)){P("preimport\n");return 1;} }
  int N=3000;
  /* Baseline synchron */
  { barra_zbuf* one[1]={&zs[0]}; for(int w=0;w<10;w++) barra_dsp_run(&d,"vscale",one,1,&rv,&us);
    double t0=now(); for(int i=0;i<N;i++) barra_dsp_run(&d,"vscale",one,1,&rv,&us); double dt=now()-t0;
    P("synchron (barra_dsp_run):        %6.0f Jobs/s  (%.3f ms/Job)\n", N/dt, dt/N*1e3); }
  /* Async K in-flight */
  double base=0; { barra_zbuf* one[1]={&zs[0]}; double t0=now(); for(int i=0;i<500;i++) barra_dsp_run(&d,"vscale",one,1,&rv,&us); base=500/(now()-t0); }
  int Ks[]={1,2,4,8}; double best=0; int bK=0;
  for(unsigned i=0;i<sizeof Ks/sizeof*Ks;i++){ int K=Ks[i];
    uint32_t tok[KM];
    for(int k=0;k<K;k++){ barra_zbuf* one[1]={&zs[k]}; if(barra_dsp_submit(&d,"vscale",one,1,&tok[k])){P("submit\n");return 1;} }
    double t0=now();
    for(int i=0;i<N;i++){ int h=i%K; barra_zbuf* one[1]={&zs[h]};
      if(barra_dsp_wait(&d,tok[h],one,1,&rv,&us)){P("wait\n");return 1;}
      if(barra_dsp_submit(&d,"vscale",one,1,&tok[h])){P("resubmit\n");return 1;} }
    double dt=now()-t0;
    for(int k=0;k<K;k++){ barra_zbuf* one[1]={&zs[k]}; barra_dsp_wait(&d,tok[k],one,1,&rv,&us); }
    double jps=N/dt; P("async K=%2d:  %6.0f Jobs/s  (%.3f ms/Job)  x%.2f\n", K, jps, dt/N*1e3, jps/base);
    if(jps>best){best=jps;bK=K;}
  }
  P("=> Async best: K=%d, %.0f Jobs/s (x%.2f gegenueber synchron)\n", bK, best, best/base);
  for(int k=0;k<KM;k++) barra_zc_free(&zs[k]);
  barra_dsp_close(&d);
  P("\n=== FAZIT: Batch %s, Async-Pipeline verdrahtet ===\n", okall?"korrekt":"FEHLER");
  return okall?0:1;
}
