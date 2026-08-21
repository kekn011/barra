/* barra_dsp_zc_test — verifiziert das GXPZ-Zero-Copy-Backend von libbarra:
 * ein geteilter zbuf (dma_heap), DSP-Kernel rechnen IN-PLACE darin ueber
 * barra_dsp_open/run — keine Datenkopie ueber den Socket. Cache-Klammern setzt barra.
 * Bauen: mit barra.c zusammen (NDK Bionic ODER Container glibc).
 * Lauf (Android, root): BARRA_SOCK_DIR=/data/local/ubuntu/opt/hwbridge barra_dsp_zc_test
 */
#include "barra.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)

int main(void){
  barra_dsp d; if(barra_dsp_open(&d)){ P("barra_dsp_open fehlgeschlagen\n"); return 1; }
  barra_zbuf z; if(barra_zc_alloc(&z,4096)){ P("zc_alloc fehlgeschlagen\n"); barra_dsp_close(&d); return 1; }
  int32_t* m=(int32_t*)z.map; barra_zbuf* bufs[1]={&z};
  int fail=0; int64_t rv=0; uint32_t us=0;

  /* vscale: [3,5,10,100] -> [3,15,30,300] (out[0]=Skalar bleibt) */
  m[0]=3;m[1]=5;m[2]=10;m[3]=100;
  if(barra_dsp_run(&d,"vscale",bufs,1,&rv,&us)) fail=1;
  { int ok=(m[0]==3&&m[1]==15&&m[2]==30&&m[3]==300);
    P("[vscale] %uus -> [%d %d %d %d] (erw [3 15 30 300]) %s\n",us,m[0],m[1],m[2],m[3], ok?"OK":"FALSCH"); if(!ok)fail=1; }

  /* vadd: [1,2,3,4]+[10,20,30,40] -> [11,22,33,44] (in-place, Handle wird wiederverwendet) */
  for(int i=0;i<4;i++){ m[i]=i+1; m[i+4]=10*(i+1); }
  if(barra_dsp_run(&d,"vadd",bufs,1,&rv,&us)) fail=1;
  { int ok=(m[0]==11&&m[1]==22&&m[2]==33&&m[3]==44);
    P("[vadd]   %uus -> [%d %d %d %d] (erw [11 22 33 44]) %s\n",us,m[0],m[1],m[2],m[3], ok?"OK":"FALSCH"); if(!ok)fail=1; }

  P("dsp_h nach Laeufen=%d (Handle blieb ueber beide Kernel bestehen)\n", z.dsp_h);
  barra_dsp_release(&d,bufs,1);
  barra_zc_free(&z); barra_dsp_close(&d);
  P("\n=== libbarra DSP-Zero-Copy (GXPZ): %s ===\n", fail?"FEHLGESCHLAGEN":"OK");
  return fail;
}
