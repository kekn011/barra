# gpu-kernels — Mali-G715 Handkernel (Referenz, aus src/experiments/roofline)

Gemessene Bestwerte (GPU @890 MHz gepinnt, `barra-perf gpu max`):
- **GEMV q4 (Decode)**: `gemv.comp` Variante q40d/q4kd — signed Integer-Dot (`(q<<4)^0x80`, x int8 je 32er-Block), WG 64, 8–16 Lanes/Zeile,
  2–4 Zeilen/Thread, subgroupClusteredAdd: **27–31 GB/s** Gewichtsstrom (Leselatte 35 GB/s), exakt. q4_K-Layout real 25–27 GB/s.
  Eingeflossen in den llama.cpp-MMVQ-Pfad (`src/llama-mali`).
- **GEMM f16**: `gemm_best.comp` (vec4-A+B, K-Unroll ×4 explizit) **420 GFLOPS** = 6,5× llama.cpp-mul_mm, 18 % vom Peak.
- Harness: `gpugemv.c` / `gpugemm.c` laden beliebige .spv, `robustBufferAccess` + gepolsterte Puffer, Vergleich gegen CPU-Referenz.
  Neue Kernel IMMER zuerst hier messen (Kernel > ~3 ms/Op reißt in test-backend-ops den GPU-progress_timeout ~2,9 s).
Mali-Fakten: Subgroup 16, `VK_KHR_shader_integer_dot_product` beschleunigt nur signed/unsigned packed (mixed = 0), KHR_coopmat langsamer als mul_mm.
Build: `build.sh` / `sweep-build.sh` (glslang 16 via `../llama-mali/glslc-wrap.sh`, NDK 27), Lauf am Gerät `sweep-run.sh`.
