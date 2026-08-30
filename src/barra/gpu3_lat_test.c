/* gpu3_lat_test — M0-Vorprobe fuer ggml-gpud (30.8.2026), glibc, im Container.
 *   gpu3_lat_test <shader.spv> [big_mb]      (Shader: vadd-artig, 3 Storage-Bindings, keine Push-Constants)
 * Misst: (1) Import grosser dmabufs (Gewichte: big_mb, Standard 3072 MB, notfalls in Haelften),
 *        (2) Roundtrip-Latenz je Stufenzahl (1, 16, 64, 256, 1024 Stufen je Batch),
 *        (3) Korrektheit mit Binding-Offsets (Sichten in EINEN Puffer),
 *        (4) v2 unangetastet (barra_gpu_batch mit demselben Shader laeuft weiter).
 * Bau: gcc -O2 gpu3_lat_test.c barra.c -o gpu3_lat_test */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "barra.h"
/* GX1=1 in der Umgebung: nur 1 Workgroup je Stufe (GPU-Grundpreis je Dispatch statt Rechenarbeit) */
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint8_t* rdfile(const char* p, uint32_t* n){ FILE* f=fopen(p,"rb"); if(!f) return 0; fseek(f,0,SEEK_END); long l=ftell(f); fseek(f,0,SEEK_SET); uint8_t* b=malloc(l); if(fread(b,1,l,f)!=(size_t)l){ fclose(f); free(b); return 0; } fclose(f); *n=(uint32_t)l; return b; }
int main(int argc, char** argv){
  if(argc<2){ fprintf(stderr,"gpu3_lat_test <shader.spv> [big_mb]\n"); return 2; }
  uint32_t slen=0; uint8_t* spv=rdfile(argv[1],&slen); if(!spv){ perror(argv[1]); return 1; }
  uint32_t big_mb=argc>2?(uint32_t)atoi(argv[2]):3072;
  barra_gpu3 g; if(barra_gpu3_open(&g)){ fprintf(stderr,"gpuzc.sock?\n"); return 1; }
  int sh=barra_gpu3_load(&g,spv,slen,3,0); if(sh<0){ fprintf(stderr,"LOAD fehlgeschlagen (v3 im Daemon?)\n"); return 1; }
  printf("[1] Shader geladen: handle %d (%u B SPIR-V)\n",sh,slen);
  if(getenv("NOBAR")){ barra_gpu3_flags(&g,1); printf("[1] Messmodus: keine Zwischen-Barrieren (Ergebnisse undefiniert)\n"); }

  /* (1) grosse dmabufs: Gewichte */
  { uint32_t mb=big_mb; barra_zbuf bz; double t0=now();
    while(mb>=256){ if(barra_zc_alloc(&bz,mb*1024u*1024u)==0) break; mb/=2; }
    if(mb<256){ printf("[2] FEHLER: kein dmabuf >= 256 MB allozierbar\n"); }
    else { double t1=now(); barra_zbuf* p=&bz; int r=barra_gpu3_import(&g,&p,1); double t2=now();
      printf("[2] dmabuf %u MB: alloc %.0f ms, GPU-Import %s (%.0f ms)%s\n",mb,(t1-t0)*1e3,r?"FEHLER":"ok",(t2-t1)*1e3,mb<big_mb?"  <-- kleiner als gewuenscht":"");
      if(r==0) barra_gpu3_release(&g,&p,1); barra_zc_free(&bz); }
    /* 6 x 512 MB = 3 GB in Stuecken (Rueckfallpfad Gewichte-Chunking) */
    barra_zbuf ch[6]; barra_zbuf* cp[6]; int nok=0; double t3=now();
    for(int i=0;i<6;i++){ if(barra_zc_alloc(&ch[i],512u*1024u*1024u)) break; cp[i]=&ch[i]; nok++; }
    int ri=nok?barra_gpu3_import(&g,cp,nok):-1; double t4=now();
    printf("[2] 6x512 MB: %d allokiert, Import %s (%.0f ms)\n",nok,ri?"FEHLER":"ok",(t4-t3)*1e3);
    if(ri==0) barra_gpu3_release(&g,cp,nok); for(int i=0;i<nok;i++) barra_zc_free(&ch[i]); }

  /* (2)+(3) Latenz und Offsets: ein Puffer, Sichten je 4096 floats (16 KB) */
  const uint32_t N=4096, VB=N*4; const int NV=64;            /* 64 Sichten a 16 KB in einem 1-MB-Puffer */
  barra_zbuf A; if(barra_zc_alloc(&A,VB*NV)){ printf("alloc\n"); return 1; }
  float* fa=(float*)A.map; for(uint32_t i=0;i<N*NV;i++) fa[i]=(float)(i%1000)*0.5f;
  int sizes[]={1,16,64,256,1024}; int ns=sizeof sizes/sizeof sizes[0];
  barra_gpu3_stage* st=calloc(1024,sizeof *st); barra_gpu3_bind* bd=calloc(1024*3,sizeof *bd);
  for(int k=0;k<ns;k++){ int n=sizes[k];
    for(int s=0;s<n;s++){ int va=s%(NV-2); /* out = 2*a + b, alle drei Sichten im selben Puffer */
      bd[s*3+0]=(barra_gpu3_bind){&A,(uint64_t)va*VB,VB}; bd[s*3+1]=(barra_gpu3_bind){&A,(uint64_t)(va+1)*VB,VB}; bd[s*3+2]=(barra_gpu3_bind){&A,(uint64_t)(NV-1)*VB,VB};
      st[s]=(barra_gpu3_stage){.sh=sh,.gx=getenv("GX1")?1:N/64,.gy=1,.gz=1,.binds=&bd[s*3],.nbind=3,.pc=0,.pcsize=0}; }
    double best=1e9; int reps=(n>=256)?5:20; int okc=1;
    for(int r=0;r<reps;r++){ double t0=now(); if(barra_gpu3_batch(&g,st,n)){ okc=0; break; } double dt=now()-t0; if(dt<best) best=dt; }
    /* Korrektheit der letzten Stufe: Sicht NV-1 = Sicht va + Sicht va+1 */
    int va=(n-1)%(NV-2); int bad=0; for(uint32_t i=0;i<N;i++){ float want=2.0f*fa[va*N+i]+fa[(va+1)*N+i];   /* vadd.spv: out = 2*in0 + in1 */ if(fa[(NV-1)*N+i]!=want){ bad++; } }
    printf("[3] %4d Stufen/Roundtrip: %s, best %.3f ms (%.1f us/Stufe), Offsets: %s\n",n,okc?"ok":"FEHLER",best*1e3,best*1e6/n,bad?"FALSCH":"korrekt");
  }
  barra_zc_free(&A); barra_gpu3_unload(&g,sh); barra_gpu3_close(&g);
  /* (4) v2 weiter intakt */
  { barra_gpu g2; barra_zbuf x,y,z; barra_zc_alloc(&x,VB); barra_zc_alloc(&y,VB); barra_zc_alloc(&z,VB);
    float* px=x.map,*py=y.map; for(uint32_t i=0;i<N;i++){ px[i]=i; py[i]=1; }
    barra_zbuf* bufs[3]={&x,&y,&z}; int r=barra_gpu_open(&g2)||barra_gpu_dispatch(&g2,spv,slen,N/64,1,1,bufs,3);
    float* pz=z.map; int bad=0; for(uint32_t i=0;i<N;i++) if(pz[i]!=2.0f*(float)i+1.0f) bad++;
    printf("[4] v2-Dispatch: %s (%d Abweichungen)\n",(r==0&&!bad)?"ok":"FEHLER",bad);
    barra_gpu_release(&g2,bufs,3); barra_gpu_close(&g2); barra_zc_free(&x); barra_zc_free(&y); barra_zc_free(&z); }
  return 0;
}
