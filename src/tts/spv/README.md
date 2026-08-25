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

## Stand der Shader-Quellen (25.8.2026)

| Header | .comp-Quelle |
|---|---|
| `convfused16_spv.h` | `../shaders/conv_mm8.comp` (fusionierter Conv1d, TM128/MM8) |
| `convfused16s_spv.h` | **fehlt** — Variante mit tm=32 (`conv_gemm_fused_f16s`) |
| `shuffle16_spv.h` | **fehlt** |
| `mrf16_spv.h` | **fehlt** |

Die drei fehlenden Quellen existierten nur auf dem alten Dev-Node und im Scratchpad;
der Node wurde am 25.8. neu geflasht. Bis sie rekonstruiert sind, sind die Header hier
die einzige Bauwahrheit — deshalb sind sie eingecheckt (nicht generiert-und-weggeworfen).

`../shaders/conv_mm4.comp` ist die MM=4-Variante aus der Hebel-4-Untersuchung
(Registerdruck < 64/Thread, Fix-Kandidat fuer den Vokoder-Durchsatz; noch nicht gebaut).
