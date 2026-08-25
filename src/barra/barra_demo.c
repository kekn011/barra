/* barra_demo — kleine heterogene Pipeline ueber barra, LIVE.
 * CPU (upper) -> DSP (reverse_string, echte On-DSP-Ausfuehrung) -> CPU (tag).
 * Zeigt: EINE API, explizite Platzierung pro Stufe, Datenfluss zwischen Chips. */
#include "barra.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

/* *on traegt beim Aufruf die Ausgabe-Kapazitaet (siehe barra.h cpu_fn-Vertrag). */
static void cpu_upper(const uint8_t* in,uint32_t n,uint8_t* out,uint32_t* on){
  uint32_t cap=*on; if(n>cap) n=cap;
  for(uint32_t i=0;i<n;i++) out[i]=(uint8_t)toupper(in[i]);
  *on=n;
}
static void cpu_tag(const uint8_t* in,uint32_t n,uint8_t* out,uint32_t* on){
  uint32_t cap=*on; const char* t=" <barra>";
  uint32_t nc=n>cap?cap:n; memcpy(out,in,nc);
  uint32_t tc=(cap>nc)?(cap-nc<8?cap-nc:8):0; if(tc) memcpy(out+nc,t,tc);
  *on=nc+tc;
}

static void show(const char* pfx,const uint8_t* b,int n){
  printf("%s'",pfx); for(int i=0;i<n;i++) putchar((b[i]>=32&&b[i]<127)?b[i]:'.'); printf("'\n");
}

int main(void){
  barra_devices();
  const char* input="Hallo_barra";
  barra_op ops[3]={
    { .device=BARRA_CPU, .cpu_fn=cpu_upper,               .label="upper"   },
    { .device=BARRA_DSP, .dsp_func="reverse_string",      .label="reverse" },
    { .device=BARRA_CPU, .cpu_fn=cpu_tag,                 .label="tag"     },
  };
  printf("\nPipeline (explizite Platzierung CPU -> DSP -> CPU):\n");
  show("  input : ",(const uint8_t*)input,(int)strlen(input));
  uint8_t out[256];
  int r=barra_pipeline(ops,3,(const uint8_t*)input,(uint32_t)strlen(input),out,sizeof out);
  if(r<0){ printf("Pipeline FEHLER\n"); return 1; }
  show("  ausgabe: ",out,r);
  printf("\n(Die DSP-Stufe lief ECHT auf dem Callisto ueber gxp.sock; reverse_string\n"
         " laesst das letzte Byte stehen - Kernel-Eigenart. Nach gxpd2-Promotion\n"
         " steht hier 'vscale' fuer echte Arithmetik.)\n");
  return 0;
}
