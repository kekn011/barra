/* barra_tpu_raw — TPD2-Inline-Client: liest in_size Bytes aus argv[2], schickt sie an Modell argv[1],
 * schreibt die Ausgabe nach argv[3]. Fuer den FFN-auf-TPU-Test (int16 rein/raus). */
#include "barra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char**argv){
  if(argc<4){ fprintf(stderr,"usage: %s <model_id> <in.bin> <out.bin>\n",argv[0]); return 2; }
  uint32_t mid=atoi(argv[1]);
  FILE* f=fopen(argv[2],"rb"); if(!f){ perror("in"); return 1; }
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t* in=malloc(n); if(fread(in,1,n,f)!=(size_t)n){return 1;} fclose(f);
  uint8_t* out=malloc(1<<20);
  barra_op op={.device=BARRA_TPU,.tpu_model_id=mid,.label="ffn"};
  int r=barra_run(&op,in,(uint32_t)n,out,1<<20);
  if(r<0){ fprintf(stderr,"barra_run fail\n"); return 1; }
  FILE* g=fopen(argv[3],"wb"); fwrite(out,1,r,g); fclose(g);
  fprintf(stderr,"ok: in %ld B -> out %d B (Modell %u)\n",n,r,mid);
  return 0;
}
