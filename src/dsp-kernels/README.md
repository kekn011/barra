# DSP-Kernel (GXP „Callisto“, Xtensa) — Bau & Test

- Toolchain: Espressif `xtensa-esp-elf` gcc/binutils 14.2 (WSL `/home/kevin/xtensa/xtensa-esp-elf/bin`), Basis-ISA windowed.
- Bau: `bash asmpatch2.sh k_<name>.S 14c ker_<name>.elf` (assembliert .S, spleisst .text in `inbuilt.elf` bei vaddr 0x14c = Body von `tpu_request_sync_submitter`, neutralisiert Relocs). Slot 0x14c..0x1d0 ≈ 132 B.
- Kernel-ABI: `kfn: entry a1,32; l32i a3,a3,0; l32i a3,a3,0; l32i a3,a3,8` → a3 = Host-Puffer (Ein-/Ausgabe in-place); Ende `memw; movi a2,0; retw.n`.
- gxpd3 laedt jede `ker_<name>.elf` im KDIR automatisch (Name = `<name>`); Aufruf aus Ubuntu: `barra_run(op{device=BARRA_DSP, dsp_func="<name>"}, in, n, out, cap)`.
- Test ohne Produktions-Brücke: `gxpd-test-daemon.sh <GXPD_CACHE> [GXPD_COH]` (adb-root) startet gxpd3 auf `/opt/hwbridge/test/gxp.sock` mit KDIR=/data/adb/hwbridge; Client: `BARRA_SOCK_DIR=/opt/hwbridge/test`.
- Erkenntnisse: Default-Puffer speicherlatenz-gebunden (~0,8 µs/Element), `GXPD_CACHE=1` → 6× schneller. Kein HW-Float (Coprozessor faultet), FLIX/TIE-Slots (VLIW/SIMD) mit esp-elf nicht erreichbar (Stand 16.8.) — siehe Memory `gxp-dsp-standalone`.
- Kernel: `k_argmax.S` (LLM-Token-Auswahl), `k_dot/vadd/mm/sumloop/softmul*.S`, `vscale_k.S`.
