/* gxp-abort-test — prueft die gxpd3-Abort-Recovery ueber libbarra:
 *  1) Baseline: kleiner vscale -> OK.
 *  2) Trigger: riesiger vscalen (N=2,5M, uncached ~4s) -> Firmware-Watchdog-Abort.
 *  3) Verify: erneut kleiner vscale (mit Reconnect-Retry) -> muss OHNE manuellen
 *     Neustart wieder gehen, wenn gxpd3 sich selbst re-exect hat.
 * Lauf (root): BARRA_SOCK_DIR=/data/local/ubuntu/opt/hwbridge gxp-abort-test  */
#include "barra.h"
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#define P(...) do{ fprintf(stderr,__VA_ARGS__); fflush(stderr);}while(0)

static int try_small(void){
  barra_dsp d; if(barra_dsp_open(&d)) return -1;
  barra_zbuf z; if(barra_zc_alloc(&z,16)){ barra_dsp_close(&d); return -1; }
  int32_t* m=z.map; m[0]=3;m[1]=5;m[2]=10;m[3]=100; barra_zbuf* one[1]={&z}; int64_t rv; uint32_t us;
  int r=barra_dsp_run(&d,"vscale",one,1,&rv,&us);
  int ok=(r==0 && m[1]==15 && m[2]==30 && m[3]==300);
  barra_zc_free(&z); barra_dsp_close(&d);
  return ok?0:-1;
}

int main(void){
  P("1) Baseline kleiner vscale: %s\n", try_small()==0?"OK":"FAIL");

  P("2) Trigger: riesiger vscalen N=2.5M (uncached, ~4s -> Watchdog-Abort erwartet)...\n");
  { barra_dsp d; if(!barra_dsp_open(&d)){
      barra_zbuf z; if(!barra_zc_alloc(&z,12u*1024*1024)){
        int32_t* m=z.map; m[0]=2500000; m[1]=1; barra_zbuf* one[1]={&z}; int64_t rv; uint32_t us;
        int r=barra_dsp_run(&d,"vscalen",one,1,&rv,&us);
        P("   trigger rc=%d (Fehler/Abort erwartet)\n", r);
        barra_zc_free(&z);
      }
      barra_dsp_close(&d);
  } }

  P("3) Verify Recovery (Reconnect-Retry bis ~25s, Supervisor-Respawn <=15s)...\n");
  int ok=-1;
  for(int i=0;i<100;i++){ if(try_small()==0){ ok=0; P("   -> wieder funktionsfaehig nach ~%.2fs\n", i*0.25); break; } usleep(250000); }
  P("\n=== gxpd3-Abort-Recovery: %s ===\n", ok==0?"OK (auto-recovered, kein manueller Neustart)":"FEHLGESCHLAGEN");
  return ok==0?0:1;
}
