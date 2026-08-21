/* barra_zc_demo — 2-Stufen-GPU-Pipeline mit ZERO-COPY-Handoff via barra.
 * Stufe1: vadd(A,B)->C   (C=A*2+B)
 * Stufe2: vadd(C,A)->D   (D=C*2+A)  -- C wird als GETEILTER dmabuf weitergereicht,
 *                                       KEINE Kopie zwischen den Stufen.
 * Die GPU rechnet in geteilten dmabufs; wir lesen alles aus unseren mmaps. */
#include "barra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 64
#define PAGE 4096

static uint8_t* slurp(const char* p, uint32_t* n){
  FILE* f=fopen(p,"rb"); if(!f){ fprintf(stderr,"%s fehlt\n",p); return 0; }
  fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t* b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){return 0;} fclose(f); *n=(uint32_t)s; return b;
}

int main(void){
  barra_devices();
  uint32_t slen=0; uint8_t* spv=slurp("/opt/hwbridge/vadd.spv",&slen); if(!spv) return 1;

  barra_zbuf A,B,C,D;
  if(barra_zc_alloc(&A,PAGE)||barra_zc_alloc(&B,PAGE)||barra_zc_alloc(&C,PAGE)||barra_zc_alloc(&D,PAGE)){
    fprintf(stderr,"zc_alloc fehlgeschlagen\n"); return 1; }
  float* a=A.map; float* b=B.map; float* c=C.map; float* dd=D.map;
  for(int i=0;i<N;i++){ a[i]=(float)i; b[i]=1000.0f; c[i]=0; dd[i]=0; }

  printf("\nZero-Copy-Pipeline (geteilte dmabufs, GPU rechnet in-place):\n");
  barra_zbuf s1[3]={A,B,C};
  if(barra_gpu_zc(spv,slen,1,1,1,s1,3)){ fprintf(stderr,"Stufe1 fehlgeschlagen\n"); return 1; }
  printf("  [1] vadd  @ GPU  A,B -> C   C[1]=%.0f C[10]=%.0f  (=i*2+1000)\n",c[1],c[10]);

  barra_zbuf s2[3]={C,A,D};   /* C ohne Kopie in die naechste Stufe */
  if(barra_gpu_zc(spv,slen,1,1,1,s2,3)){ fprintf(stderr,"Stufe2 fehlgeschlagen\n"); return 1; }
  printf("  [2] vadd  @ GPU  C,A -> D   (C wurde OHNE Kopie weitergereicht)\n");

  printf("\nErgebnis D[i]=C[i]*2+A[i]=(i*2+1000)*2+i=5i+2000:\n");
  printf("  D[0]=%.0f D[1]=%.0f D[10]=%.0f D[63]=%.0f  (erwartet 2000 2005 2050 2315)\n",dd[0],dd[1],dd[10],dd[63]);
  if(dd[0]==2000&&dd[1]==2005&&dd[10]==2050&&dd[63]==2315)
    printf(">>> ZERO-COPY-PIPELINE OK: 2 GPU-Stufen, geteilter dmabuf, null Kopie dazwischen. <<<\n");
  else printf(">>> unerwartet <<<\n");
  barra_zc_free(&A); barra_zc_free(&B); barra_zc_free(&C); barra_zc_free(&D);
  return 0;
}
