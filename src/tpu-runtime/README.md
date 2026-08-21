# tpu-runtime — TPU-Fähigkeiten im Base-Image

| Baustein | Ort im Image | Quelle |
|---|---|---|
| `tpud` (Bionic-Daemon, Socket `/opt/hwbridge/tpu.sock`, Multi-Modell, Zero-Copy via ImportBufferByFd, Fence-Completion, CPU-Pinning) | `/data/adb/hwbridge/tpud` | `src/hwbridge/tpud.c`, `tpu-zc.c` |
| Supervisor + Boot-Hook | `/data/adb/hwbridge/hw-bridges.sh`, `/data/adb/service.d/60-hwbridges.sh` | `src/boot/` |
| TPU-Takt (performance + min 1,119 GHz, idle-gated) | `/data/adb/service.d/99-tpu-boost.sh`, `barra-perf tpu on|off` | `src/tpu-boost/`, `src/perf/barra-perf` |
| **On-Device-Compiler** tpuc1 + libcomp_std.so (binär-gepatchte G3-Compiler-Kopie, DGC0) + `compile-std.sh` / `compile-dir.sh` / `compile-verbose.sh` | `/data/adb/baseos/tpu/` | Binaries `tpu-toolchain/{tpuc1,libcomp_std.so}`, Skripte hier |
| Referenz-Package (Smoke-Test für tpud) | `/data/adb/hwbridge/test.package` | — |
| libbarra 0.2 (glibc, Container) + CLI-Werkzeuge (tpucli, barra_tpu_zc_test …) | `/usr/local/{lib,include,bin}` | `src/barra/` |
| ggml-barra (TPU-FFN-Backend für llama.cpp, Ausreißer-Auslagerung) | Build-Option `GGML_BARRA=ON` (Packages je Modell, nicht im Image) | `src/ggml-barra/` |

PC-Werkzeuge (bleiben im Repo): `tpu-toolchain/` (Modell→TFLite→Package-Generatoren, Batch-Patcher, DGC0-Dumper, Options-Injektion),
`src/experiments/roofline/` (Messungen, Templating, ISA-RE). Compile-Aufruf am Gerät: `sh /data/adb/baseos/tpu/compile-std.sh <model.tflite>` → `out.package`.
