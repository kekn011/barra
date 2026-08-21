# libbarra API reference

`#include <barra.h>`, link with `-lbarra` (or
`pkg-config --cflags --libs barra`). The header is the normative source; this
page is the guided tour. Everything is C, glibc, in the Ubuntu container.

libbarra is a **thin dispatch layer**, not an auto-scheduler: you say which
stage runs on which chip, and data moves between chips as shared dmabuf buffers.

## Devices

```c
typedef enum { BARRA_CPU, BARRA_TPU, BARRA_GPU, BARRA_DSP } barra_device;
const char* barra_devname(barra_device d);
void        barra_devices(void);   // print the capability table
```

| Device | Bridge | Runs |
|---|---|---|
| `BARRA_CPU` | in-process | branchy glue, tokenizing |
| `BARRA_TPU` | `tpu.sock` | precompiled graphs (`barrac` packages) |
| `BARRA_GPU` | `gpu.sock` | SPIR-V compute shaders |
| `BARRA_DSP` | `gxp.sock` | named Xtensa kernels |

## Zero-copy buffers (the core idea)

A `barra_zbuf` is one dmabuf (`/dev/dma_heap/system`), CPU-mappable and
importable by GPU/TPU/DSP **at the same time**. Chips hand data to each other by
computing on the same buffer.

```c
typedef struct { int fd; void* map; uint32_t size;
                 int gpu_h; int tpu_h; int dsp_h; } barra_zbuf;

int  barra_zc_alloc(barra_zbuf* z, uint32_t size);  // dmabuf + mmap; 0 = ok
void barra_zc_free (barra_zbuf* z);
int  barra_zc_cpu_begin(barra_zbuf* z);  // CPU is about to read/write
int  barra_zc_cpu_end  (barra_zbuf* z);  // CPU done, device may run
```

Cache coherency around a single dispatch is automatic. Call `cpu_begin`/`cpu_end`
yourself only around long CPU phases between dispatches. Read/write through
`z->map`.

## TPU

```c
typedef struct { int sock; } barra_tpu;
int  barra_tpu_open (barra_tpu* t);
void barra_tpu_close(barra_tpu* t);
int  barra_tpu_info (barra_tpu* t, uint32_t id, uint32_t* in, uint32_t* out, uint32_t* n);
int  barra_tpu_infer(barra_tpu* t, uint32_t id, barra_zbuf* in, barra_zbuf* out, uint32_t* us);
// pipelined: submit, do unrelated CPU work, then wait (don't touch the buffers between)
int  barra_tpu_submit(barra_tpu* t, uint32_t id, barra_zbuf* in, barra_zbuf* out);
int  barra_tpu_wait  (barra_tpu* t, barra_zbuf* in, barra_zbuf* out, uint32_t* us);
```

The **model id** is the position of the package in the tpud instance's load
order. `barra_tpu_info` tells you the expected I/O byte sizes per model. Point
libbarra at a specific tpud with the `BARRA_SOCK_DIR` environment variable.

```c
barra_tpu t; barra_tpu_open(&t);
barra_zbuf in, out;
barra_zc_alloc(&in, IN); barra_zc_alloc(&out, OUT);
memcpy(in.map, x, IN);
barra_tpu_infer(&t, 0, &in, &out, NULL);   // result in out.map, no copies
```

## GPU

```c
typedef struct { int sock; int nimported; } barra_gpu;
int  barra_gpu_open (barra_gpu* g);
void barra_gpu_close(barra_gpu* g);
int  barra_gpu_dispatch(barra_gpu* g, const uint8_t* spirv, uint32_t len,
                        uint32_t gx, uint32_t gy, uint32_t gz,
                        barra_zbuf** bufs, int nbuf);
// batch: K stages in one round-trip, GPU barriers between them, no CPU in the loop
int  barra_gpu_batch(barra_gpu* g, const barra_gpu_stage* stages, int nstage);
```

Buffers bind to `layout(binding=0..nbuf-1)` in the shader, in array order. For
repeated kernels, `barra_gpu_import` the buffers once and reuse the session.
There is also a one-shot `barra_gpu_zc()` that imports per call.

## DSP

```c
typedef struct { int sock; } barra_dsp;
int  barra_dsp_open (barra_dsp* d);
void barra_dsp_close(barra_dsp* d);
int  barra_dsp_run  (barra_dsp* d, const char* func, barra_zbuf** bufs, int n,
                     int64_t* rv, uint32_t* us);        // e.g. "vscale","vadd","argmax"
// async: keep K jobs in flight to hide ~1 ms firmware latency (~2.5x)
int  barra_dsp_submit(barra_dsp* d, const char* func, barra_zbuf** bufs, int n, uint32_t* tok);
int  barra_dsp_wait  (barra_dsp* d, uint32_t tok, barra_zbuf** bufs, int n, int64_t* rv, uint32_t* us);
// import with cacheable=1 before big batch kernels (~30x, higher per-dispatch cost)
int  barra_dsp_import_ex(barra_dsp* d, barra_zbuf** bufs, int n, int cacheable);
```

Kernels are named entries in the preloaded Xtensa kernel library. Writing your
own needs the Xtensa toolchain (advanced, v0.2); the bundled kernels do not.

## Cross-chip handoff

Because one `barra_zbuf` can hold a GPU, a TPU and a DSP handle simultaneously,
a buffer written by one chip is read by the next with no copy:

```c
// GPU writes bQ  ->  TPU reads bQ, writes bO  ->  CPU reads bO
barra_gpu_dispatch(&g, shader, len, gx,gy,gz, (barra_zbuf*[]){&bH,&bQ}, 2);
barra_tpu_infer(&t, MODEL_O, &bQ, &bO, NULL);
// bO.map now holds the result
```

This is exactly how the LLM attention offload works (see `src/ggml-barra/`): TPU
projections and GPU attention kernels pass tensors through shared dmabufs across
36 transformer layers without a single tensor copy.

## Higher-level pipeline helper

For simple byte-in/byte-out staging without managing buffers yourself:

```c
int barra_run     (const barra_op* op, const uint8_t* in, uint32_t n, uint8_t* out, uint32_t cap);
int barra_pipeline(const barra_op* ops, int nop, const uint8_t* in, uint32_t n, uint8_t* out, uint32_t cap);
```

See `barra_op` in the header and `examples/barra_demo.c`.

## Environment variables

| Variable | Meaning |
|---|---|
| `BARRA_SOCK_DIR` | directory holding the bridge sockets to use (default `/opt/hwbridge`) |

## Error convention

Functions return `0` on success and `-1` on error (or a non-negative size for
`barra_run`/`barra_pipeline`). Bridges must be running — they are started by the
node's bridge supervisor at boot.
