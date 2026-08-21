/* barra_crosschip_test — CROSS-CHIP-BEWEIS: EIN geteilter dmabuf, mehrere Chips
 * rechnen nacheinander IN-PLACE darin (Zero-Copy), jede Stufe sieht byte-exakt das
 * Ergebnis der vorigen. Verifiziert gegen eine unabhaengige CPU-Referenz.
 *
 *   Runde A: CPU-init -> GPU(+1000) -> DSP(vscale) -> CPU-Vergleich
 *   Runde B: CPU-init -> DSP(vscale) -> GPU(+1000) -> CPU-Vergleich
 * vscale: out[0]=in[0] (Skalar), out[i]=in[i]*in[0] fuer i>0 (int32).
 *
 * Beweiskraft: haette ein Chip NICHT im selben physischen Puffer gerechnet (oder
 * eine Kopie gesehen), waeche das Endergebnis von der Referenz ab.
 *
 * Bauen: mit barra.c (NDK Bionic ODER Container glibc). Braucht add_const.spv.
 * Lauf (root): BARRA_SOCK_DIR=/data/local/ubuntu/opt/hwbridge \
 *              barra_crosschip_test /data/local/tmp/add_const.spv
 */
#include "barra.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)
#define N 4   /* vscale-Kernel verarbeitet fest 4 int32 (wie gxpd3-Selftest); Puffer passend gross */

static uint8_t* slurp(const char* p, uint32_t* len){
  FILE* f=fopen(p,"rb"); if(!f){ P("kann %s nicht oeffnen\n",p); return 0; }
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t* b=malloc(n); if(fread(b,1,n,f)!=(size_t)n){ free(b); fclose(f); return 0; }
  fclose(f); *len=(uint32_t)n; return b;
}
static void show(const char* tag, const int32_t* m){
  P("  %-22s [",tag);
  for(int i=0;i<N;i++) P("%s%d", i?" ":"", m[i]);
  P("]\n");
}
/* Referenzmodelle der Stufen (in reinem C, unabhaengig von der HW) */
static void ref_gpu(int32_t* r){ for(int i=0;i<N;i++) r[i]+=1000; }
static void ref_dsp_vscale(int32_t* r){ int32_t s=r[0]; for(int i=1;i<N;i++) r[i]=r[i]*s; /* r[0] bleibt */ }

int main(int argc,char**argv){
  const char* spv_path = argc>1?argv[1]:"/data/local/tmp/add_const.spv";
  uint32_t slen=0; uint8_t* spv=slurp(spv_path,&slen); if(!spv) return 1;
  P("add_const.spv geladen (%u B)\n",slen);

  barra_gpu g; if(barra_gpu_open(&g)){ P("gpu open\n"); return 1; }
  barra_dsp d; if(barra_dsp_open(&d)){ P("dsp open\n"); return 1; }
  barra_zbuf z; if(barra_zc_alloc(&z, N*4)){ P("zc_alloc\n"); return 1; }
  int32_t* m=(int32_t*)z.map; barra_zbuf* zb[1]={&z};
  int fail=0; int64_t rv; uint32_t us;

  /* ---------- Runde A: CPU -> GPU -> DSP -> CPU ---------- */
  P("\n=== Runde A: CPU -> GPU(+1000) -> DSP(vscale) auf DEMSELBEN dmabuf ===\n");
  int32_t refA[N];
  for(int i=0;i<N;i++){ m[i]=i+1; refA[i]=i+1; }
  show("CPU init", m);
  if(barra_gpu_dispatch(&g,spv,slen,1,1,1,zb,1)){ P("gpu dispatch A\n"); fail=1; }
  show("nach GPU (+1000)", m);
  if(barra_dsp_run(&d,"vscale",zb,1,&rv,&us)){ P("dsp run A\n"); fail=1; }
  show("nach DSP (vscale)", m);
  ref_gpu(refA); ref_dsp_vscale(refA);
  { int ok=!memcmp(m,refA,sizeof refA); show("CPU-Referenz", refA);
    P("  => Runde A %s\n", ok?"OK (byte-exakt)":"ABWEICHUNG"); if(!ok)fail=1; }

  /* ---------- Runde B: CPU -> DSP -> GPU -> CPU (Gegenrichtung) ---------- */
  P("\n=== Runde B: CPU -> DSP(vscale) -> GPU(+1000) auf DEMSELBEN dmabuf ===\n");
  int32_t refB[N];
  for(int i=0;i<N;i++){ m[i]=i+2; refB[i]=i+2; }
  show("CPU init", m);
  if(barra_dsp_run(&d,"vscale",zb,1,&rv,&us)){ P("dsp run B\n"); fail=1; }
  show("nach DSP (vscale)", m);
  if(barra_gpu_dispatch(&g,spv,slen,1,1,1,zb,1)){ P("gpu dispatch B\n"); fail=1; }
  show("nach GPU (+1000)", m);
  ref_dsp_vscale(refB); ref_gpu(refB);
  { int ok=!memcmp(m,refB,sizeof refB); show("CPU-Referenz", refB);
    P("  => Runde B %s\n", ok?"OK (byte-exakt)":"ABWEICHUNG"); if(!ok)fail=1; }

  P("\nHandles auf demselben zbuf: gpu_h=%d dsp_h=%d (beide >=0 = beide Chips teilen genau diesen Puffer)\n", z.gpu_h, z.dsp_h);
  barra_gpu_release(&g,zb,1); barra_dsp_release(&d,zb,1);
  barra_zc_free(&z); barra_gpu_close(&g); barra_dsp_close(&d);
  P("\n=== CROSS-CHIP (CPU/GPU/DSP, ein dmabuf, Zero-Copy): %s ===\n", fail?"FEHLGESCHLAGEN":"OK");
  return fail;
}
