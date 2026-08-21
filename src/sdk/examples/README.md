# barra SDK examples

Small, self-contained C programs that show how to drive the Tensor G3
accelerators through **libbarra** — the barra driver API. Everything runs
directly on the node, no cross toolchain needed (the base image ships `gcc`).

## Build

```sh
sh build.sh          # compiles every example in this directory
```

or by hand:

```sh
gcc -O2 -Wall example.c $(pkg-config --cflags --libs barra) -o example
```

## Examples

| File                 | Shows                                                        |
|----------------------|--------------------------------------------------------------|
| `barra_demo.c`       | The basic pipeline API: CPU stage → DSP kernel → CPU stage   |
| `barra_zc_demo.c`    | Zero-copy buffers (dmabuf) shared between CPU and GPU        |
| `barra_tpu_zc_test.c`| TPU inference straight into a shared dmabuf (no data copies) |
| `barra_dsp_zc_test.c`| DSP kernels computing in-place on a shared dmabuf            |

## The one-page mental model

* `barra_zbuf` is a **dmabuf**: one buffer, mappable by the CPU and importable
  by GPU, TPU and DSP *at the same time*. Chips hand data to each other by
  simply computing on the same buffer — no copies.
* The TPU runs **precompiled graphs** (`.package` files, built on-device with
  `barrac` / `tpuc1`); `barra_tpu_infer()` executes one by model id.
* The GPU runs **SPIR-V compute shaders** via `barra_gpu_dispatch()` /
  `barra_gpu_batch()`.
* The DSP runs **named kernels** from the preloaded Xtensa kernel library
  (`vscale`, `vadd`, …) via `barra_dsp_run()` — with `submit`/`wait` for
  async pipelining.
* Cache coherency around device access is handled by libbarra; for long CPU
  phases between dispatches see `barra_zc_cpu_begin/_end` in `barra.h`.

The full API reference lives in `/usr/include/barra.h` — the header is the
documentation.
