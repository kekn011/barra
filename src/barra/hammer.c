/* hammer — Dauerlast auf einem Chip fuer Last-Kalibrierung/Dashboard-Tests.
 * hammer <dsp|tpu|gpu> <sekunden> [add_const.spv]
 * Laeuft die jeweilige Operation in einer engen Schleife fuer N Sekunden. */
#include "barra.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static uint8_t* slurp(const char* p, uint32_t* l){ FILE* f=fopen(p,"rb"); if(!f)return 0; fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); uint8_t* b=malloc(n); if(fread(b,1,n,f)!=(size_t)n){free(b);fclose(f);return 0;} fclose(f); *l=n; return b; }
int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"hammer <dsp|tpu|gpu> <sek> [spv]\n"); return 2; }
  const char* mode=argv[1]; double T=atof(argv[2]); long it=0; double t0=now();
  if(!strcmp(mode,"dsp")){
    barra_dsp d; if(barra_dsp_open(&d))return 1; barra_zbuf z; barra_zc_alloc(&z,16); barra_zbuf* b[1]={&z};
    int32_t* m=z.map; m[0]=3;m[1]=5;m[2]=10;m[3]=100; int64_t rv; uint32_t us;
    while(now()-t0<T){ barra_dsp_run(&d,"vscale",b,1,&rv,&us); it++; }
    barra_zc_free(&z); barra_dsp_close(&d);
  } else if(!strcmp(mode,"tpu")){
    barra_tpu t; if(barra_tpu_open(&t))return 1; uint32_t in=0,out=0,n=0; barra_tpu_info(&t,0,&in,&out,&n);
    barra_zbuf zi,zo; barra_zc_alloc(&zi,in); barra_zc_alloc(&zo,out); uint32_t us;
    while(now()-t0<T){ barra_tpu_infer(&t,0,&zi,&zo,&us); it++; }
    barra_zc_free(&zi); barra_zc_free(&zo); barra_tpu_close(&t);
  } else if(!strcmp(mode,"gpu")){
    if(argc<4){ fprintf(stderr,"gpu braucht spv\n"); return 2; }
    uint32_t sl=0; uint8_t* spv=slurp(argv[3],&sl); if(!spv)return 1;
    barra_gpu g; if(barra_gpu_open(&g))return 1; barra_zbuf z; barra_zc_alloc(&z,4096); barra_zbuf* b[1]={&z};
    while(now()-t0<T){ barra_gpu_dispatch(&g,spv,sl,16,1,1,b,1); it++; }
    barra_zc_free(&z); barra_gpu_close(&g);
  } else { fprintf(stderr,"unbekannter mode\n"); return 2; }
  fprintf(stderr,"hammer %s: %ld Iterationen in %.1fs (%.0f/s)\n",mode,it,now()-t0,it/(now()-t0));
  return 0;
}
