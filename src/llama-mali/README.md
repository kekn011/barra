# llama-mali — llama.cpp mit Mali-G715-Kerneln (Base-Komponente)

Das Mali-optimierte llama.cpp (Vulkan, Android/Bionic) ist Teil des Base-Images: **`/data/adb/baseos/llm/`**
(llama-server, llama-bench, llama-perplexity, llama-quantize, llama-cli + Libs, `env.sh`),
Start über **`/data/adb/baseos/bin/llmserver.sh start [gguf] [ctx] [ngl]`** (Port 8080), Messung über `llm-bench.sh`.

## Was der Patch enthält (`llama-cpp-mali-vulkan.patch`, gegen llama.cpp 8d274dd)
- Mali als Vendor (0x13b5) erkannt; **KHR_coopmat auf ARM standardmäßig aus** (GGML_VK_ARM_COOPMAT=1 schaltet an).
- **MMVQ-Pfad auf ARM nur für q4_K und q6_K** mit eigenen Mali-Kerneln (Integer-Dot, 8 bzw. 16 Lanes/Zeile, uvec4-Loads):
  q4_K 1,58× gegenüber dem generischen Pfad, q6_K über **Repack** der Gewichte auf 224-B-Stride (16-B-aligned) im Gerätespeicher —
  Host repackt in set/get_tensor **und in den async-Pfaden** (Loader-Upload, Scheduler), Alloc-Größe skaliert, Shader-Structs gepolstert.
- Ergebnis (tg64, GPU @890 MHz): Qwen2.5-1.5B q4_K_M 17,4 → **23,4 t/s**, Qwen3-4B q4_K_M 7,7 → **10,7 t/s**; PPL GPU ≈ CPU.
- Schalter für Vergleiche: `GGML_VK_DISABLE_MMVQ=1` (alter Pfad), `GGML_VK_ARM_NO_Q6K=1` (nur q4_K-Kernel), `GGML_VK_ARM_RM_KQ`.

## Bauen (WSL)
`./build.sh` — klont/setzt llama.cpp auf den Referenz-Commit, wendet den Patch an, hängt das **ggml-barra-TPU-FFN-Backend** ein (`GGML_BARRA=ON`, `libggml-barra.so`; aktiv nur mit `BARRA_META`/Packages, sonst passiv), baut mit NDK 27 + glslang 16 (`glslc-wrap.sh`, Integer-Dot) und legt die gestrippte Node-Kopie in `build-android-vulkan-idp/stage/` ab — inkl. `barra-llm` (GPU+DSP-Argmax, dynamisch gegen diesen Build) und `run-llm.sh`.
`patches/*.py` sind die Einzelschritte, aus denen der Patch entstand (Reihenfolge: q4k-3, q6k-repack, mmvq-q6k-v2, q6k-lpr16, q6k-async, arm-defaults).

## Auf den Node
`./install-node.sh [serial]` → `/data/adb/baseos/llm` + `bin/llmserver.sh` + `bin/llm-bench.sh`. Modelle nach `/data/local/ubuntu/home/<user>/models/`.

## Regeln am Gerät
- Kontext immer begrenzen (`-c`): Qwen3-4B meldet 40k Kontext = ~6 GB KV-Cache.
- Messen mit `llm-bench.sh` / llama-perplexity (ein Prozess, `</dev/null`); GPU-Takt pinnen mit `barra-perf gpu max`.
- Referenz/Forschung: `src/experiments/roofline/` (RESULTS.md), Handkernel in `src/gpu-kernels/`.
