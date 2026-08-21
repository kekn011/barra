/* barra_crosschip4_test — VIER-CHIP-BEWEIS auf EINEM geteilten dmabuf.
 * Ein einziger dmabuf (Zin) wird nacheinander von CPU, GPU und DSP IN-PLACE
 * beschrieben und dann von der TPU als Eingabe gelesen (Zero-Copy, Zin->Zout).
 * Danach traegt Zin GLEICHZEITIG gpu_h, dsp_h UND tpu_h — dasselbe physische
 * Stueck Speicher, von allen drei Beschleunigern referenziert.
 *
 * Verifikation (ohne die TPU-Modellsemantik zu kennen): das Zero-Copy-Ergebnis
 * MUSS byte-identisch zu einer Inline-Referenz sein, bei der GENAU dieselben
 * (von GPU+DSP produzierten) Eingabebytes ueber den Kopie-Pfad an die TPU gehen.
 * Weicht es ab, hat die TPU NICHT die geteilten Bytes gelesen.
 *
 * Bauen: mit barra.c. Braucht add_const.spv + laufendes tpud/gpud-zc/gxpd3.
 * Lauf (root): BARRA_SOCK_DIR=/data/local/ubuntu/opt/hwbridge \
 *              barra_crosschip4_test /data/local/tmp/add_const.spv
 */
#include "barra.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)

static uint8_t* slurp(const char* p, uint32_t* len){
  FILE* f=fopen(p,"rb"); if(!f){ P("kann %s nicht oeffnen\n",p); return 0; }
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t* b=malloc(n); if(fread(b,1,n,f)!=(size_t)n){ free(b); fclose(f); return 0; }
  fclose(f); *len=(uint32_t)n; return b;
}

int main(int argc,char**argv){
  const char* spv_path = argc>1?argv[1]:"/data/local/tmp/add_const.spv";
  uint32_t slen=0; uint8_t* spv=slurp(spv_path,&slen); if(!spv) return 1;

  barra_gpu g; barra_tpu t; barra_dsp d;
  if(barra_gpu_open(&g)||barra_tpu_open(&t)||barra_dsp_open(&d)){ P("open (gpu/tpu/dsp)\n"); return 1; }
  uint32_t in_sz=0,out_sz=0,nm=0;
  if(barra_tpu_info(&t,0,&in_sz,&out_sz,&nm)||in_sz==0){ P("tpu_info fehlgeschlagen\n"); return 1; }
  P("TPU-Modell 0: in=%u out=%u bytes\n",in_sz,out_sz);

  barra_zbuf Zin, Zout;
  if(barra_zc_alloc(&Zin,in_sz)||barra_zc_alloc(&Zout,out_sz)){ P("zc_alloc\n"); return 1; }
  uint32_t nI=in_sz/4;                 /* int32-Elemente */
  int32_t* mi=(int32_t*)Zin.map;
  barra_zbuf* zin1[1]={&Zin};
  int fail=0; int64_t rv; uint32_t us;

  /* ---- 1) CPU fuellt Zin mit einem deterministischen Muster ---- */
  for(uint32_t i=0;i<nI;i++) mi[i]=(int32_t)(i*2654435761u ^ (i<<3));
  P("[1] CPU-Muster in Zin geschrieben (%u int32)\n",nI);

  /* ---- 2) GPU addiert 1000 auf JEDES Element (in-place, geteilter Puffer) ---- */
  uint32_t gx=(nI+63)/64;
  if(barra_gpu_dispatch(&g,spv,slen,gx,1,1,zin1,1)){ P("gpu dispatch\n"); fail=1; }
  P("[2] GPU (+1000, %u WG) fertig; Zin[0..3]=[%d %d %d %d]\n",gx,mi[0],mi[1],mi[2],mi[3]);

  /* ---- 3) DSP rechnet vscale in-place auf denselben Puffer (erste 4 int32) ---- */
  if(barra_dsp_run(&d,"vscale",zin1,1,&rv,&us)){ P("dsp run\n"); fail=1; }
  P("[3] DSP (vscale) fertig; Zin[0..3]=[%d %d %d %d]\n",mi[0],mi[1],mi[2],mi[3]);

  /* Zin haelt jetzt GPU+DSP-Ergebnis. Bytes fuer die Inline-Referenz sichern. */
  uint8_t* refin=malloc(in_sz); memcpy(refin,Zin.map,in_sz);

  /* ---- 4) TPU liest den GETEILTEN Puffer Zin (Zero-Copy) -> Zout ---- */
  if(barra_tpu_infer(&t,0,&Zin,&Zout,&us)){ P("tpu infer (zc)\n"); fail=1; }
  P("[4] TPU zero-copy infer fertig (%u us)\n",us);
  P("    Zin traegt jetzt gpu_h=%d dsp_h=%d tpu_h=%d  (alle >=0 = EIN Puffer, drei Beschleuniger)\n",
    Zin.gpu_h, Zin.dsp_h, Zin.tpu_h);

  /* ---- 5) Inline-Referenz: dieselben Bytes ueber den Kopie-Pfad an die TPU ---- */
  uint8_t* refout=malloc(out_sz);
  barra_op op; memset(&op,0,sizeof op); op.device=BARRA_TPU; op.tpu_model_id=0;
  int r=barra_run(&op,refin,in_sz,refout,out_sz);
  if(r!=(int)out_sz){ P("tpu inline ref rc=%d (erw %u)\n",r,out_sz); fail=1; }

  /* ---- 6) Byte-Vergleich: Zero-Copy-Ausgabe == Inline-Referenz? ---- */
  int same = (memcmp(Zout.map,refout,out_sz)==0);
  P("[6] Vergleich TPU-zero-copy vs Inline-Referenz (%u B): %s\n", out_sz, same?"BYTE-IDENTISCH":"ABWEICHUNG");
  if(!same){ fail=1;
    for(uint32_t i=0;i<out_sz;i++) if(((uint8_t*)Zout.map)[i]!=refout[i]){ P("    erste Abweichung @ Byte %u: zc=%u ref=%u\n",i,((uint8_t*)Zout.map)[i],refout[i]); break; } }

  free(refin); free(refout);
  barra_gpu_release(&g,zin1,1); barra_dsp_release(&d,zin1,1);
  barra_zc_free(&Zin); barra_zc_free(&Zout);
  barra_gpu_close(&g); barra_tpu_close(&t); barra_dsp_close(&d);
  P("\n=== VIER-CHIP CROSS-CHIP (CPU+GPU+DSP schreiben, TPU liest; EIN dmabuf, Zero-Copy): %s ===\n", fail?"FEHLGESCHLAGEN":"OK");
  return fail;
}
