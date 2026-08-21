# ggml-barra — TPU-FFN-Backend für llama.cpp (M4), Stand 19.8.2026

**Im Base-Image** seit 19.8.: wird vom Mali-llama.cpp-Build (`src/llama-mali/build.sh`, `GGML_BARRA=ON`) als `libggml-barra.so` mitgeliefert; CPU-Fallback direkt über `ggml-cpu` (`ggml_backend_cpu_init`, kein Registry-Zugriff → kein Link-Zyklus im Shared-Build). Ohne `BARRA_META`/Packages bleibt das Backend passiv.

llama.cpp-Backend, das die Feed-Forward-Blöcke (`ffn_up`/`ffn_gate`/`swiglu`/`ffn_down` je Layer) als **ein**
vorkompiliertes Package auf der Tensor-G3-TPU rechnet (über `tpud` + libbarra, Zero-Copy dmabuf). Muster: BLAS-Backend
(Host-Buffer, `supports_op` selektiv). Nur Prefill/Batch (T ≥ `BARRA_MIN_BATCH`, Default 8); Decode bleibt CPU.

## Ergebnis (Qwen2.5-0.5B Q8_0, barra-5599, 5 Threads)
- **Korrekt:** Perplexity CPU 19,09 vs TPU-FFN **19,03**; Selbsttest je Layer 1–2 % rel. Fehler (= Host-Interpreter);
  Completion-Text identisch. Fallback (Ausreißerzeilen: BOS/Massive-Activation) exakt.
- **Speed:** llama-bench pp32/128/512 CPU **546/582/493 t/s** vs TPU-FFN **122/87/112 t/s** — CPU 4–6× schneller.
  TPU-Anteil: ~8,7 ms je 32-Zeilen-Kachel und Layer (Package-Floor), CPU-FFN ~1,5 ms/Layer für 19 Token.
- **4B-Einordnung (gemessen):** Qwen3-4B Q4_K_M CPU pp32/128 = 43/40 t/s (t=5) → CPU-FFN ≈ 17 ms je 32-Token-Kachel/Layer;
  TPU-FFN-Package 4B B32: int8 21,5 ms, 16x8 38 ms (75 MB Gewichte, ~3,5 GB/s Streaming-gebunden) → **bestenfalls Parität**.
  B=64/128 kompilieren nicht (auch als FF-Split-Versuch vorbereitet: `ffn4b_split.py`).
- **Fazit:** Reines FFN-Offload bringt auf diesem SoC keinen Prefill-Gewinn gegen llama.cpps int8-dotprod/i8mm-CPU-Pfad
  (~500 GFLOPS effektiv). Der TPU-Weg ist Gewichts-Streaming-gebunden (~3,5–4 GB/s) + ~1,2 ms Floor/Inferenz + B ≤ 32.
  Was es drehen könnte: (1) Nebenläufigkeit CPU‖TPU (async Backend + Chunk-Pipelining, ggml-Events) → bis ~1,5–2×;
  (2) schnellerer Gewichtspfad/größeres B (Compiler-Grenze, siehe tpu-dgc0-re); (3) Energie pro Token (TPU vs 5 Kerne @2,4 GHz)
  — nicht gemessen; (4) CPU-Entlastung für andere Arbeit.

## 4B Ende-zu-Ende (18.8. 14:00, Qwen3-4B Q4_K_M, t=5, 29/36 FFN-Layer auf TPU — IOMMU-Grenze ~2,2 GB Packages)
| | pp32 | pp128 | PPL (2 Chunks) | Fallback |
|---|---|---|---|---|
| CPU-only | 26,4 (±13; andere Läufe 43–44) | 37,1 t/s | 14,25 | – |
| TPU-FFN 16x8 | 14,8 | 16,5 t/s | 15,17 | 1050/7424 (14 %) |
| TPU-FFN int8 | 14,6 | 20,0 t/s | 15,03 | 6944/7424 (93 %, unbrauchbar: int8-Bereich saturiert) |
TPU-Zeit 38 ms je 32er-Kachel/Layer (16x8) — identisch zur isolierten Kachelmessung; CPU-Prefill 2,3× schneller.
Einordnung: unser Package-Pfad nutzt ~2,5 % der TPU (9,2 TOPS); Hebel = Codegen/Compiler (B>32, Array-Füllung).

## Dateien
- `ggml-barra.cpp/.h`, `CMakeLists.txt` (+ `src/barra/barra.c|h`), `install-into-llamacpp.sh <llama.cpp>` (kopiert,
  registriert `GGML_BARRA` in ggml/CMakeLists + ggml-backend-reg.cpp, idempotent).
- Build (WSL): `cmake -S . -B build-android-barra -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DGGML_BARRA=ON
  -DCMAKE_C_FLAGS=-march=armv8.2-a+dotprod+i8mm+fp16 (CXX ebenso)`; Targets llama-completion/llama-bench/llama-perplexity.
- Packages: `tpu-toolchain/gen_ffn_pkgs.py <gguf> <outdir> [B] [16x8|int8] [ntok]` → `ffn_L{i}.tflite` (f(h) ohne rmsnorm/Residual,
  Kalibrierung aus layer-streamendem numpy-Forward, Ausreißer raus, 25 % Headroom) + `ffn.meta` (D FF B bits NL + isc izp osc ozp).
  Prüfen: `verify_ffn_pkgs.py`. Gerät: `tpuc1` je tflite → `.package`; `tpud_pipe4 sock ffn_L0.package … ffn_L{NL-1}.package`.
- Gerät-Skripte: `tpu-toolchain/chain/m4_*.sh` (m4_dev = compile+start+Vergleich, m4_bench, m4_ppl4 = CHECK+PPL).

## Laufzeit-Env
`BARRA_FFN_META` (Pflicht), `BARRA_SOCK_DIR`, `BARRA_MIN_BATCH` (8), `BARRA_FFN_LOG=1` (WARN-Level: je Aufruf quant/infer/dequant/fbk),
`BARRA_CHECK=1` (Selbsttest gegen CPU-Referenz, erste 48 Aufrufe), `BARRA_FORCE_FB=1` (alles über CPU-Fallback), `BARRA_FB_THREADS` (4).

## Fallen (alle erlebt)
- Q8_0/Q4-Gewichte liegen in `CPU_REPACK`-Buffern (is_host=NULL) → `supports_buft` muss CPU-Device-Bufts akzeptieren, sonst
  fragt der Scheduler das Backend nie.
- `supports_op` NICHT für VIEW/RESHAPE/PERMUTE „ja" sagen (BLAS tut es): zieht Views in eigene Splits mit Kopien → CPU-Pfad 15× langsamer.
- **ggml-Allocator aliasiert `dst` (down-Ausgang) mit `h` (FFN-Eingang)** — h ist nach gate/up tot. Wer h nach dem ersten
  dst-Schreiben liest (Fallback, Check), liest dst → PPL 100. Lösung: h eingangs kopieren.
- Knoten einzeln auf CPU rechnen: `ggml_new_graph_custom(…,1)` + `ggml_graph_add_node`, NICHT `build_forward_expand`
  (zieht alle Vorgänger bis get_rows mit hinein → Assert/Doppelrechnung).
- llama-completion zeigt nur WARN-Logs der Bibliothek → Diagnose als GGML_LOG_WARN.
- Letzter Layer: llama.cpp reduziert vor dem FFN auf die Output-Zeilen (T=1 bei Completion) → korrekt CPU.
- Completion-Text-Gleichheit ist KEIN Korrektheitsbeweis (prüft nur die letzte Position) → Perplexity/CHECK.

## Ausreißer-Auslagerung (19.8., `BARRA_OUTLIER`)

LLM.int8()-Prinzip im Split-Pfad: die k größten **Eingangskanäle** werden im TPU-Eingang genullt und ihr Beitrag
exakt in float auf der CPU addiert — `y = TPU(h_rest) + CPU(h_out)`. Der Gewinn ist doppelt: der CPU-Anteil ist
exakt, **und** der TPU-Anteil hat ohne die Ausreißer einen 6–120× kleineren Zeilen-max, wodurch die
Zeilennormierung auf 127 die übrigen Kanäle entsprechend feiner trifft (gemessen mit `gen_outliers.py`, Qwen3-4B:
L1 h-max 47,5 → 1,8 ohne die obersten 5 %; L35 413 → 3,4).

- **Erzeugen:** `tpu-toolchain/gen_outliers.py <gguf> <out.bol> [frac] [ntok] [smoothfile]` → Format `BOL1`
  (Kanäle **nach Wichtigkeit sortiert**, deshalb ist jedes Präfix wieder gültig → ein 5-%-File taugt auch für 0,5 %).
  Arbeitet im gesmoothten Raum, wenn die `.smooth`-Datei mitgegeben wird — genau das, was das Backend der TPU vorlegt.
- **Nutzen:** `BARRA_OUTLIER=<datei.bol>` (+ optional `BARRA_OUTLIER_FRAC=0.005`).
- **Packages müssen dazu passen:** `gen_mm3_pkgs.py` mit `OL_BOL=<datei.bol> OL_FRAC=<anteil>` nullt die
  ausgelagerten Kanäle in den **Kalibrierzeilen**. Ohne das clippt der Ausgang massiv (mit den alten d3-Packages
  fielen bei 0,5 % Auslagerung 30–55 von 64 Zeilen in den CPU-Fallback), weil die verbleibenden Kanäle auf volle
  int8-Breite hochnormiert werden und die alte `osc` dafür zu klein ist.
- **Kernel:** `barra_rank_update` — Rang-k-Update spaltenblockweise über den Thread-Pool, **Kanal außen, Zeilen
  innen**. Zeilenweise wären es bei k=128/FF=9728/n=64 rund 320 MB Gewichtsverkehr je Kachel statt 5 MB.
  Gewichte bleiben f16 im Speicher und werden je Block einmal entpackt (bei n=64 ist das 1/64 der Arbeit).
