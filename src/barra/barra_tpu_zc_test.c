/* barra_tpu_zc_test — TPU-Zero-Copy + Cross-Chip-Handoff (GPU -> TPU) verifizieren.
 *   (1) INFO: Modellgroessen von tpud
 *   (2) Referenz: Inferenz inline (TPD2, Daten ueber Socket)
 *   (3) Zero-Copy: dieselben Eingabebytes in einem dmabuf, TPU rechnet direkt hinein
 *       -> Output aus unserer mmap muss BYTE-IDENTISCH zur Referenz sein
 *   (4) Wiederholung mit anderen Eingaben (persistente Handles, kein Re-Import)
 *   (5) Cross-Chip: die GPU (vadd) SCHREIBT den TPU-Input-dmabuf, die TPU rechnet
 *       darauf -> Vergleich mit Inline-Inferenz auf denselben Bytes (per memcpy geholt)
 *   (6) Benchmark: us/Inferenz inline vs zero-copy (inkl. Socket-Roundtrip) */
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
static uint32_t roundup(uint32_t v,uint32_t a){ return (v+a-1)/a*a; }
static void fill(uint8_t* p,uint32_t n,uint32_t seed){ for(uint32_t i=0;i<n;i++) p[i]=(uint8_t)((i*7+seed*13)^(i>>3)); }
static int inline_infer(uint32_t mid,const uint8_t* in,uint32_t isz,uint8_t* out,uint32_t osz){
  barra_op op={.device=BARRA_TPU,.tpu_model_id=mid,.label="ref"}; int r=barra_run(&op,in,isz,out,osz); return r==(int)osz?0:-1; }

int main(int argc,char**argv){
  int iters=argc>1?atoi(argv[1]):50; uint32_t mid=argc>2?(uint32_t)atoi(argv[2]):0;
  printf("[1] INFO\n");
  barra_tpu t; CHECK(barra_tpu_open(&t)==0,"TPU-Session offen");
  uint32_t isz=0,osz=0,nm=0; CHECK(barra_tpu_info(&t,mid,&isz,&osz,&nm)==0,"INFO ok");
  printf("       Modell %u: in=%u out=%u Bytes (%u Modelle geladen)\n",mid,isz,osz,nm);
  if(!isz||!osz) return 1;

  printf("[2] Referenz inline (TPD2)\n");
  uint8_t* in=malloc(isz); uint8_t* ref=malloc(osz); uint8_t* ref2=malloc(osz);
  fill(in,isz,1);
  CHECK(inline_infer(mid,in,isz,ref,osz)==0,"Inline-Inferenz #1");
  CHECK(inline_infer(mid,in,isz,ref2,osz)==0&&memcmp(ref,ref2,osz)==0,"Inline deterministisch (2x identisch)");
  int nz=0; for(uint32_t i=0;i<osz;i++) if(ref[i]) nz++; printf("       Referenz: %d/%u Bytes != 0\n",nz,osz);

  printf("[3] Zero-Copy: TPU rechnet direkt in unsere dmabufs\n");
  barra_zbuf ZI,ZO; uint32_t zin_sz=roundup(isz,4096), zout_sz=roundup(osz,4096);
  CHECK(barra_zc_alloc(&ZI,zin_sz)==0&&barra_zc_alloc(&ZO,zout_sz)==0,"2 dmabufs alloziert");
  memcpy(ZI.map,in,isz); memset(ZO.map,0xEE,zout_sz);
  uint32_t us=0; CHECK(barra_tpu_infer(&t,mid,&ZI,&ZO,&us)==0,"zc-Inferenz status 0 (auto-import)");
  printf("       Handles: in=%d out=%d, TPU-Zeit %.2f ms\n",ZI.tpu_h,ZO.tpu_h,us/1000.0);
  CHECK(memcmp(ZO.map,ref,osz)==0,"zc-Output BYTE-IDENTISCH zur Inline-Referenz");

  printf("[4] andere Eingaben, persistente Handles\n");
  fill(in,isz,2); memcpy(ZI.map,in,isz);
  CHECK(inline_infer(mid,in,isz,ref,osz)==0,"Inline-Referenz #2");
  CHECK(barra_tpu_infer(&t,mid,&ZI,&ZO,&us)==0,"zc-Inferenz #2 status 0");
  CHECK(memcmp(ZO.map,ref,osz)==0,"zc-Output #2 identisch (kein stale Ergebnis)");
  CHECK(memcmp(ZO.map,ref2,osz)!=0,"Output #2 unterscheidet sich von #1 (Eingabe wirkt)");

  printf("[5] Cross-Chip: GPU schreibt den TPU-Input-dmabuf, TPU rechnet darauf\n");
  uint32_t slen=0; uint8_t* spv=slurp("/opt/hwbridge/vadd.spv",&slen);
  if(spv){
    uint32_t nfl=zin_sz/4, gx=(nfl+63)/64;
    barra_zbuf A,B; CHECK(barra_zc_alloc(&A,zin_sz)==0&&barra_zc_alloc(&B,zin_sz)==0,"GPU-Eingabepuffer alloziert");
    float* a=A.map; float* b=B.map; for(uint32_t i=0;i<nfl;i++){ a[i]=(float)(i%251); b[i]=(float)(i%17); }
    memset(ZI.map,0,zin_sz);
    barra_gpu g; CHECK(barra_gpu_open(&g)==0,"GPU-Session offen");
    barra_zbuf* st[3]={&A,&B,&ZI};                       /* C = ZI: der TPU-Input wird von der GPU beschrieben */
    CHECK(barra_gpu_dispatch(&g,spv,slen,gx,1,1,st,3)==0,"GPU vadd -> TPU-Input-dmabuf");
    float* c=ZI.map; int badc=0; for(uint32_t i=0;i<nfl;i++) if(c[i]!=a[i]*2+b[i]) badc++;
    CHECK(badc==0,"GPU-Ergebnis im TPU-Input korrekt (CPU-Sicht)");
    printf("       ZI: gpu_h=%d tpu_h=%d (derselbe dmabuf auf beiden Chips)\n",ZI.gpu_h,ZI.tpu_h);
    /* Referenz: dieselben Bytes inline durch die TPU */
    memcpy(in,ZI.map,isz);
    CHECK(inline_infer(mid,in,isz,ref,osz)==0,"Inline-Referenz auf GPU-erzeugten Bytes");
    CHECK(barra_tpu_infer(&t,mid,&ZI,&ZO,&us)==0,"TPU zc-Inferenz auf GPU-erzeugtem Input (ohne Kopie)");
    CHECK(memcmp(ZO.map,ref,osz)==0,"Cross-Chip-Ergebnis BYTE-IDENTISCH zur Referenz");
    barra_gpu_close(&g); barra_zc_free(&A); barra_zc_free(&B);
  } else printf("  (vadd.spv fehlt - Cross-Chip uebersprungen)\n");

  printf("[6] Benchmark (%d Iterationen, Modell %u, in=%u out=%u)\n",iters,mid,isz,osz);
  fill(in,isz,3); memcpy(ZI.map,in,isz);
  double t0=now_us(); for(int i=0;i<iters;i++) inline_infer(mid,in,isz,ref,osz); double ti=(now_us()-t0)/iters;
  double tsum=0; t0=now_us(); for(int i=0;i<iters;i++){ barra_tpu_infer(&t,mid,&ZI,&ZO,&us); tsum+=us; } double tz=(now_us()-t0)/iters;
  printf("       inline (TPD2, Daten ueber Socket): %8.0f us/Inferenz\n",ti);
  printf("       zero-copy (TPZ2, nur Handles)    : %8.0f us/Inferenz  (davon TPU-Rechnung %.0f us)  x%.2f\n",tz,tsum/iters,ti/tz);
  CHECK(memcmp(ZO.map,ref,osz)==0,"nach Benchmark weiterhin identisch");

  barra_tpu_release(&t,(barra_zbuf*[]){&ZI,&ZO},2);
  barra_tpu_close(&t); barra_zc_free(&ZI); barra_zc_free(&ZO);
  printf(fails?">>> %d FEHLER <<<\n":">>> ALLE TESTS OK <<<\n",fails);
  return fails?1:0;
}
