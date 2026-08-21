# Base-Image: GPU + TPU vollständig eingebaut (Konsolidierung 19.8.2026)

Leitsatz (Kevin): **Alles, was wir uns in Sachen GPU und TPU erarbeitet haben, gehört ins Base-Image.**
Base-Image = der Zustand eines frisch geflashten Nodes (Golden Node → `barra-base.tar.zst`).
Nicht im Image: Modelle (GGUF, TPU-Packages) und reine PC-Werkzeuge (Compiler-Skripte, TF-Venv) — die bleiben im Repo bzw. werden nachgeladen.

## Ziel-Layout auf dem Node

| Pfad | Inhalt | Quelle im Repo |
|---|---|---|
| `/data/adb/hwbridge/` | Brücken: tpud, gpud, gpud-zc, gxpd, audiod-alsa, btnd, hw-bridges.sh (Supervisor) | `src/hwbridge/`, `src/boot/hw-bridges.sh` |
| `/data/adb/service.d/60-hwbridges.sh`, `99-tpu-boost.sh` | Boot-Hooks Brücken + TPU-Takt | `src/boot/`, `src/tpu-boost/` |
| `/data/adb/baseos/bin/` | Base-Programme + **`barra-perf`** (GPU-Pin / TPU-Boost / Status) + **`llmserver.sh`** | `src/boot/`, `src/perf/` |
| **`/data/adb/baseos/llm/`** | Mali-optimiertes llama.cpp (Vulkan, Int-Dot, q4_K/q6_K-Mali-Kernel; bionic): llama-server, llama-bench, llama-perplexity, llama-quantize, Libs, `env.sh` | `src/llama-mali/` (Patch + Build-Rezept) |
| **`/data/adb/baseos/tpu/`** | On-Device-TPU-Compiler: tpuc1 + libcomp_std.so (gepatcht) + compile-std.sh; Referenz-Package test.package | `tpu-toolchain/` (Binaries), `src/tpu-runtime/` |
| Container `/usr/local/{lib,include,bin}` | libbarra 0.2 (+ .h, pkgconfig), CLI-Werkzeuge (gpzc-cli, gxpcli, tpucli, barra_demo…) | `src/barra/` |
| Container `/opt/hwbridge/` | Sockets (tpu/gpu/gpuzc/gxp/audio), barra.h, dash2, cfg/ | `src/hwbridge/`, `src/boot/` |
| Modelle | `/data/local/ubuntu/home/<user>/models/` (nicht im Image) | — |

## Arbeitsschritte

A. **Quellen nach `src/`** (PC): `src/llama-mali/` (Patch, patch-*.py, glslc-wrap.sh, build.sh, README) · `src/gpu-kernels/` (Mali-GEMV/GEMM-Handkernel-Referenz) · `src/perf/barra-perf` · `src/tpu-runtime/` (compile-std.sh, Hinweise zu tpuc1/libcomp_std) · `src/README.md` als Landkarte.
B. **Golden Node** (akita 43251JEKB03751): `baseos/llm` einspielen, `llmserver.sh` auf Image-Pfade, `barra-perf` einbauen, `baseos/tpu` einspielen, `hwbridge/` aufräumen (.v1, Logs, Testkernel), Funktionsprobe (Bench 1.5B GPU, tpud-Ping, compile-std am Mini-Modell).
C. **Bake**: prepare-image (beide Seiten) → Staging-Kopie → `barra-base.tar.zst` v3 → SHA256SUMS → Flash-Test auf einem Node.
D. **Doku**: dieses Inventar aktuell halten; `README.md` des Repos auf Stand bringen.

Stand wird unten fortgeschrieben.

## Fortschritt
- 19.8. abends: Plan erstellt.
- **A erledigt**: `src/llama-mali/` (Patch gegen 8d274dd inkl. ARM-Defaults coopmat-aus, build.sh mit Strip-Stage, install-node.sh, env.sh, llmserver.sh, llm-bench.sh, README), `src/gpu-kernels/` (Handkernel + Harness + README), `src/perf/barra-perf`, `src/tpu-runtime/` (compile-std/-dir/-verbose auf Image-Pfade, README-Inventar), `src/README.md` (Landkarte).
- **B erledigt (Golden Node 43251JEKB03751)**: `/data/adb/baseos/llm/` (17 gestrippte Binaries/Libs, 72 MB, + env.sh), `bin/llmserver.sh`, `bin/llm-bench.sh`, `bin/barra-perf`, `/data/adb/baseos/tpu/` (tpuc1, libcomp_std.so 64 MB, compile-*.sh); `hwbridge/` bereinigt (.v1, altes llmserver.sh raus). **Funktionsprobe bestanden**: `llm-bench.sh` 1.5B = 23,63 t/s Vulkan ohne Env-Variablen; `compile-std.sh` Mini-FFN rc=0 (13,2 MB Package).
- **C erledigt**: `src/boot/barra-bake.sh` (auch im Image-bin) baut aus einer Staging-Kopie (ohne Logs/Modelle/Mountpoints, Benutzer zurueck auf ubuntu/ubuntu, Keys/machine-id/Creds raus, firstboot-Marker) das Payload; **`barra-setup/payload/barra-base.tar.gz` v3 = 272 MB, sha 70809b16…**, verifiziert (ubuntu uid 1001, Store ohne WLAN, llm/tpu/perf/hooks/Bruecken drin, 0 gguf/host-keys). Altes Payload als `.prev-20260816` daneben.
- **D erledigt**: `barra-llm` + `run-llm.sh` in `baseos/llm`; Repo-README neu (Stand 19.8.), alte Planung nach `docs/README-2026-08-09-planung.md`.
- 19.8.: `barra-flashkit/` (altes WSL-Kit) auf Kevins Wunsch geloescht; einziger Weg ist `barra-setup/`.
- 19.8. spaet: Mali-Build jetzt mit **ggml-barra** (`GGML_BARRA=ON`, CPU-Fallback via ggml-cpu statt Registry) und **barra-llm neu gegen den Shared-Build** (494 KB, rpath $ORIGIN) — liegt in `build-android-vulkan-idp/stage/` (20 Dateien, 72 MB), `install-node.sh` spielt alles nach `baseos/llm`. Golden Node wurde von Kevin gewiped (19.8.) -> Einspielen + **Bake v4** auf dem frisch aufgesetzten Node.
- **Flash-Test v3 BESTANDEN 19.8. 21:20** (Kevin, barra-setup auf 43251JEKB03751 -> `barra-e5d2`, Benutzer per Setup `barra`): Erst-Boot durch, alle 5 Sockets, tpud bereit, llm/tpu da.
- **Bake v4 vom frischen Node 19.8. 21:23**: vorher `install-node.sh` (llm mit ggml-barra + neuem barra-llm), Probe: llm-bench 0.5B q8_0 29 t/s mit Backend-Liste `Vulkan,BARRA`, compile-std rc=0. **`barra-base.tar.gz` v4 = 259 MB, sha 7f44d4ad…** -> **v4b ohne Root-Key: 259 MB, sha 60148f61…**, verifiziert (ubuntu uid 1001, Store ohne WLAN, 0 gguf/host-keys, firstboot-Marker, barra-perf-Fix, libggml-barra/barra-llm drin).
- Root-Key (`root/.ssh/authorized_keys`, seit 16.8. in jedem Payload) auf Kevins Entscheidung ENTFERNT (barra-bake.sh), der Setup legt den Nutzer-Key an.
- **Stand: Image voll auf unserem Stand.** Offen nur noch Kuer: barra-llm-Lauf mit DSP-Argmax am frischen Node, ggml-barra mit Packages.
- 19.8. 21:47 **Payload v5** (259 MB, sha fecdd8eb…): v4 + RAM-Tuning in `61-debloat.sh` (Stufe 6: Modem/Radio, NNAPI/edgetpu_app_service, keystore/citadel, drm, gpu, input.processor, installd, storageproxyd; watermark_scale_factor 10; drop_caches=2 nach Debloat + 90 s). Reboot-verifiziert auf barra-e5d2: MemAvailable 5,97 -> 6,62 GB, Systemverbrauch 1,7 -> ~1,1 GB; Bruecken/Dashboard intakt. RAM-Analyse: `src/experiments/memprobe.sh`, Befunde in den Notizen (akita-plattform).
