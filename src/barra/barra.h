/* barra.h — schlanke heterogene Dispatch-Schicht ueber die Pixel-HW-Bruecken.
 *
 * Idee (ehrlich, KEIN Auto-Routing wie ein SYCL/TVM-Compiler): eine EINHEITLICHE
 * Submit-API ueber die drei laufenden Socket-Bruecken (TPU/GPU/DSP) + CPU, und der
 * Aufrufer legt EXPLIZIT fest, welche Pipeline-Stufe auf welchem Chip laeuft
 * (weil die Chips grundverschieden gefuettert werden). Daten fliessen als Byte-
 * Puffer; per Socket kopiert (Korrektheit) ODER als dmabuf Zero-Copy (GPU, s.u.).
 *
 * Chips & wofuer (Capability-Tabelle, s. barra_devices()):
 *   CPU  nativ, in-process        - Verzweigtes, Tokenizing, Kleben (kein Socket)
 *   TPU  tpu.sock (TPD2)          - dichte vorkompilierte Graphen (Attention/Matmul/Conv)
 *   GPU  gpu.sock (GPU1, SPIR-V)  - programmierbare parallele Arithmetik
 *   DSP  gxp.sock (GXPD)          - regulaere SIMD-Integer/Vektor-Kernel (Xtensa-ELF)
 */
#ifndef BARRA_H
#define BARRA_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define BARRA_VERSION "0.2.0"

typedef enum { BARRA_CPU=0, BARRA_TPU=1, BARRA_GPU=2, BARRA_DSP=3 } barra_device;

typedef struct {
  barra_device device;
  /* CPU: in-process-Funktion (out_size wird gesetzt) */
  void (*cpu_fn)(const uint8_t* in, uint32_t in_size, uint8_t* out, uint32_t* out_size);
  /* DSP: Funktionsname in der geladenen Lib ("reverse_string", spaeter "vscale") */
  const char* dsp_func;
  /* TPU: Modell-Index (der Daemon kennt den kompilierten Graphen + I/O-Groessen) */
  uint32_t tpu_model_id;
  /* GPU: SPIR-V-Shader + Dispatch-Dimensionen (ein In/Out-Puffer, in-place) */
  const uint8_t* gpu_spirv; uint32_t gpu_spirv_len; uint32_t gx, gy, gz;
  const char* label;         /* nur fuer die Ausgabe */
} barra_op;

/* eine Stufe ausfuehren: liefert out_size (>=0) oder -1 bei Fehler. */
int  barra_run(const barra_op* op, const uint8_t* in, uint32_t in_size,
               uint8_t* out, uint32_t out_cap);

/* Pipeline: n Stufen, Ausgabe jeder Stufe = Eingabe der naechsten (Handoff). */
int  barra_pipeline(const barra_op* ops, int n, const uint8_t* in, uint32_t in_size,
                    uint8_t* out, uint32_t out_cap);

const char* barra_devname(barra_device d);
void        barra_devices(void);   /* Capability-Tabelle ausgeben */

/* ================= Zero-Copy (GPU) =================
 * Ein barra_zbuf ist ein dmabuf (/dev/dma_heap/system), den mehrere Stufen TEILEN.
 * Der Client mappt ihn (z.map) und liest/schreibt direkt; die GPU rechnet in-place
 * ueber den gpud-zc-Transport (nur der fd geht ueber den Socket, SCM_RIGHTS).
 * Mali-G715 ist UMA: es gibt keinen "naeheren" Speicher als diesen dmabuf.
 *
 * CPU-Cache-Klammern (DMA_BUF_IOCTL_SYNC) setzt barra automatisch: nach alloc und
 * nach jedem Dispatch ist der Puffer im CPU-Zugriffs-Zustand; vor dem Dispatch
 * schliesst barra den CPU-Zugriff. Wer sehr lange nicht dispatcht und trotzdem
 * fremde DMA erwartet, kann barra_zc_cpu_end/_begin selbst rufen. */
typedef struct { int fd; void* map; uint32_t size; int gpu_h; int tpu_h; int dsp_h; } barra_zbuf;   /* gpu_h/tpu_h/dsp_h: Handle in einer GPU-/TPU-/DSP-Session, -1 = nicht importiert */
int  barra_zc_alloc(barra_zbuf* z, uint32_t size);   /* dmabuf + mmap; 0=ok */
void barra_zc_free(barra_zbuf* z);
int  barra_zc_cpu_begin(barra_zbuf* z);              /* DMA_BUF_SYNC_START|RW (CPU greift zu) */
int  barra_zc_cpu_end(barra_zbuf* z);                /* DMA_BUF_SYNC_END|RW   (CPU fertig, Geraet darf) */

/* --- Einzel-Dispatch (v1, GPZC): Import je Aufruf, eigene Verbindung. 0=ok. --- */
int  barra_gpu_zc(const uint8_t* spirv, uint32_t slen,
                  uint32_t gx, uint32_t gy, uint32_t gz,
                  barra_zbuf* bufs, int nbuf);

/* --- Session (v2, GPZ2): Verbindung bleibt offen, Puffer EINMAL importieren
 *     (persistente Handles), K Stufen in EINEM Roundtrip (Batch, GPU-Barrieren
 *     dazwischen, keine CPU-Beteiligung zwischen den Stufen). Das ist der schnelle
 *     Pfad fuer wiederholte Kernel / Ketten. */
typedef struct { int sock; int nimported; } barra_gpu;
int  barra_gpu_open(barra_gpu* g);                                 /* 0=ok */
void barra_gpu_close(barra_gpu* g);                                /* gibt alle Handles frei */
int  barra_gpu_import(barra_gpu* g, barra_zbuf** bufs, int n);     /* setzt bufs[i]->gpu_h; 0=ok */
int  barra_gpu_release(barra_gpu* g, barra_zbuf** bufs, int n);    /* Handles freigeben (dmabuf bleibt) */
typedef struct {
  const uint8_t* spirv; uint32_t slen;      /* Compute-Shader (main), Storage-Buffer bindings 0..nbuf-1 */
  uint32_t gx, gy, gz;                      /* Dispatch-Dimensionen */
  barra_zbuf** bufs; int nbuf;              /* Puffer der Stufe (werden bei Bedarf automatisch importiert) */
} barra_gpu_stage;
int  barra_gpu_batch(barra_gpu* g, const barra_gpu_stage* stages, int nstage);   /* 0=ok */
int  barra_gpu_dispatch(barra_gpu* g, const uint8_t* spirv, uint32_t slen,
                        uint32_t gx, uint32_t gy, uint32_t gz,
                        barra_zbuf** bufs, int nbuf);                              /* = batch mit 1 Stufe */

/* ================= Zero-Copy (TPU) =================
 * Dieselben dmabufs als TPU-Tensor-Puffer: tpud importiert sie einmal (ImportBufferByFd,
 * fd_type dmabuf), die TPU rechnet die Inferenz DIREKT hinein/heraus - keine Tensordaten
 * ueber den Socket. Ein zbuf kann gleichzeitig GPU- und TPU-Handle tragen -> Cross-Chip-
 * Handoff (GPU schreibt den TPU-Input / liest den TPU-Output) ohne Kopie. */
typedef struct { int sock; } barra_tpu;
int  barra_tpu_open(barra_tpu* t);                                 /* 0=ok */
void barra_tpu_close(barra_tpu* t);                                /* gibt alle Handles frei */
int  barra_tpu_info(barra_tpu* t, uint32_t model_id, uint32_t* in_size, uint32_t* out_size, uint32_t* nmodels);
int  barra_tpu_import(barra_tpu* t, barra_zbuf** bufs, int n);     /* setzt bufs[i]->tpu_h */
int  barra_tpu_release(barra_tpu* t, barra_zbuf** bufs, int n);
int  barra_tpu_infer(barra_tpu* t, uint32_t model_id, barra_zbuf* in, barra_zbuf* out, uint32_t* exec_us); /* auto-import; 0=ok */
/* Absenden und Warten getrennt: zwischen submit und wait gehoert CPU-Arbeit, die die Puffer NICHT anfasst
 * (sonst rechnet die TPU auf veraenderten Daten). Genau ein offener submit je barra_tpu. */
int  barra_tpu_submit(barra_tpu* t, uint32_t model_id, barra_zbuf* in, barra_zbuf* out); /* auto-import; 0=ok */
int  barra_tpu_wait(barra_tpu* t, barra_zbuf* in, barra_zbuf* out, uint32_t* exec_us);   /* 0=ok */

/* ================= Zero-Copy (DSP) — Session GXPZ =================
 * Dieselben dmabufs als DSP-Puffer: gxpd3 importiert sie einmal (ImportBufferFromFd +
 * MapBufferAllCores), der Xtensa-Kernel rechnet IN-PLACE hinein - keine Daten ueber den
 * Socket. Ein zbuf kann zugleich GPU-, TPU- UND DSP-Handle tragen -> Cross-Chip-Handoff
 * ueber alle vier Recheneinheiten auf DEMSELBEN Puffer ohne Kopie.
 * Cache-Kohaerenz macht barra automatisch (cpu_end vor dem Lauf, cpu_begin danach). */
typedef struct { int sock; } barra_dsp;
int  barra_dsp_open(barra_dsp* d);                                 /* 0=ok */
void barra_dsp_close(barra_dsp* d);                                /* gibt alle Handles frei (Verbindung zu) */
int  barra_dsp_import(barra_dsp* d, barra_zbuf** bufs, int n);     /* setzt bufs[i]->dsp_h (uncached); 0=ok */
/* Wie import, aber cacheable=1 macht die Puffer DSP-cacheable (~30x fuer grosse Batch-Kernel,
 * aber hoehere Fixkosten pro Dispatch -> nur fuer grosse N; kleine Async-Jobs uncached lassen).
 * Vor-importieren mit der gewuenschten Cacheability, dann rechnen (run/submit re-importieren nicht). */
int  barra_dsp_import_ex(barra_dsp* d, barra_zbuf** bufs, int n, int cacheable);
int  barra_dsp_release(barra_dsp* d, barra_zbuf** bufs, int n);
/* Kernel <func> (z.B. "vscale","vadd","argmax") IN-PLACE auf den geteilten Puffern rechnen.
 * bufs werden bei Bedarf automatisch importiert. rv/exec_us optional (koennen 0 sein). 0=ok. */
int  barra_dsp_run(barra_dsp* d, const char* func, barra_zbuf** bufs, int n,
                   int64_t* rv, uint32_t* exec_us);
/* Async-Split fuer maximalen Durchsatz: K Jobs gleichzeitig in-flight halten, damit sich die
 * ~1 ms Firmware-Latenz ueberlappt (~2,5x gemessen). submit gibt ein Token (RunAsync), wait
 * sammelt es ein (Request_Wait). Zwischen submit und wait die Puffer NICHT anfassen. submit setzt
 * die CPU-Cache-Klammer (Flush), wait loest sie (Invalidate) — daher braucht wait die bufs. */
int  barra_dsp_submit(barra_dsp* d, const char* func, barra_zbuf** bufs, int n, uint32_t* token);
int  barra_dsp_wait(barra_dsp* d, uint32_t token, barra_zbuf** bufs, int n, int64_t* rv, uint32_t* exec_us);

#ifdef __cplusplus
}
#endif
#endif
