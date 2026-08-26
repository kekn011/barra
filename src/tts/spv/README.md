# gpudecd — Shader (SPIR-V-Header) und Bauanleitung

`gpudecd.c` bindet seine vier Compute-Shader als vorkompilierte SPIR-V-Header ein
(`#include "convfused16_spv.h"` usw.). Diese Header liegen hier und sind **Build-Eingang**:
ohne sie laesst sich gpudecd nicht uebersetzen.

## Bauen (im Container auf dem Node)

```
# Quellen + Header nach /data/local/tmp/gcbuild schieben, dann:
gcc -O2 gpudecd.c -o gpudecd -lbarra -lm
```
Verifiziert am 25.8.2026 auf barra-3d25 (Ubuntu-Container, glibc): uebersetzt sauber,
8/8 /say-Requests ohne Handle-Leak, David RTF 0,79.

## Shader-Quellen — vollstaendig (26.8.2026)

| Header | .comp-Quelle | was er tut |
|---|---|---|
| `convfused16_spv.h`  | `../shaders/conv_mm8.comp`               | fusionierter Conv1d, TM=128/MM=8 (Arbeitspferd) |
| `convfused16s_spv.h` | `../shaders/conv_gemm_fused_f16s.comp`   | schlanke Variante TM=32/MM=2 fuer kleine `Cout` |
| `shuffle16_spv.h`    | `../shaders/shuffle_f16.comp`            | Sub-Pixel-Shuffle nach dem Upsample-Conv |
| `mrf16_spv.h`        | `../shaders/mrf_f16.comp`                | Multi-Receptive-Field-Summe (Residual-Glue) |

Drei dieser Quellen galten als verloren (der alte Dev-Node wurde am 25.8. neu geflasht).
Sie sind am **26.8. aus dem Sitzungs-Scratchpad zurueckgeholt** und verifiziert:
`sh ../shaders/build-spv.sh -check` baut alle vier Header nach und vergleicht sie mit den
eingecheckten — **alle vier bitgleich**. Damit sind die Quellen bewiesen, nicht nur plausibel.

```
sh ../shaders/build-spv.sh          # baut nach spv/
sh ../shaders/build-spv.sh -check   # Waechter: Exitcode 1 bei Header-Drift
```

**glslang liegt in WSL, nicht in git-bash** — der Aufruf lautet also:
```
wsl.exe -e bash -lc "cd /mnt/c/.../src/tts/shaders && sh build-spv.sh -check"
```

`../shaders/conv_mm4.comp` (TM=64/MM=4) ist die Variante aus der Hebel-4-Untersuchung
(Registerdruck < 64/Thread, Fix-Kandidat fuer den Vokoder-Durchsatz); sie wird derzeit
nicht gebaut und ist nicht mit `conv_gemm_fused_f16s.comp` zu verwechseln.
