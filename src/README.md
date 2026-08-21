# src/ — Quellen des Base-Images (Landkarte)

| Verzeichnis | Inhalt | Landet im Image unter |
|---|---|---|
| `boot/` | Boot-Hooks (service.d), Base-Boot, WLAN-Join/-Guard, Supervisor `hw-bridges.sh`, firstboot, prepare-image, barra-config, Dashboard-Steuerung | `/data/adb/service.d/`, `/data/adb/baseos/bin/`, Container `/usr/local/sbin` |
| `hwbridge/` | Bionic-Brücken: `tpud`, `gpud`, `gpud-zc`, `gxpd`, `audiod-alsa`, `btnd`, `dashd`, Zero-Copy-Transporte | `/data/adb/hwbridge/` |
| `barra/` | libbarra 0.2 (einheitliche Submit-API CPU/TPU/GPU/DSP, Zero-Copy) + Demos/Tests | Container `/usr/local/{lib,include,bin}` |
| `llama-mali/` | llama.cpp mit Mali-G715-Kerneln (Patch, Build, Installer, `llmserver.sh`, `llm-bench.sh`) | `/data/adb/baseos/llm/`, `bin/llmserver.sh` |
| `gpu-kernels/` | Mali-GEMV/GEMM-Handkernel + Harness (Referenz/Messung) | — (Forschung) |
| `perf/` | `barra-perf` (GPU-Pin, TPU-Boost, Status) | `/data/adb/baseos/bin/barra-perf` |
| `tpu-boost/` | TPU-Takt-Boot-Hook | `/data/adb/service.d/99-tpu-boost.sh` |
| `tpu-runtime/` | On-Device-TPU-Compiler-Skripte, Inventar der TPU-Bausteine | `/data/adb/baseos/tpu/` |
| `ggml-barra/` | TPU-FFN-Backend für llama.cpp (+ Ausreißer-Auslagerung) | Build-Option |
| `barra-llm/` | LLM über alle Chips (GPU-Decode + GXP-Argmax + TPU-FFN-Kette) | `/data/adb/baseos/llm/barra-llm` (geplant) |
| `dsp-kernels/` | Xtensa/Callisto-Kernel (XTMLD), TIE-Probes | `/data/adb/hwbridge/ker_*.elf` |
| `container/`, `assets/` | Container-Dienste, Fonts/Splash | Container |
| `experiments/` | Forschung (roofline, RE, Probes) — Ergebnisse in `experiments/roofline/RESULTS.md` | — |

Plan und Fortschritt der GPU/TPU-Konsolidierung: `docs/base-image-gpu-tpu-konsolidierung.md`.
