#!/bin/bash
# flix-probe.sh <8-byte-bundle-hex-csv> <out.elf>  [marker|observe]
# Splice ONE verbatim Callisto FLIX bundle into the sync_submitter slot (0x14c) to test
# whether the DSP core executes it. Zeroes AR a2,a4..a15 before the bundle so any
# address-register use is deterministic (addr 0 -> clean fault, distinct from illegal-op 201).
#   marker  (default): after bundle, out[0]=in[0]+1  -> proves control-flow continued past bundle
#   observe          : no marker, buffer echoed as-is -> reveals memory side-effects of the bundle
# Result read via gxpd3 selftest by naming the elf ker_dot.elf/ker_vadd.elf/ker_sumloop.elf.
# Finding (16.8., this session): op0=0xf (2-slot 64-bit FLIX) bundles DECODE+EXECUTE on the
# Callisto core (RunSync=0, PC continues). op0=0xe (3-slot) sampled -> illegal (EXCCAUSE 201).
# Some op0=f encodings are still illegal (201) or control-transfers. Running != computing:
# these bundles' memory ops need surrounding register setup that is absent here.
set -e
XT=/home/kevin/xtensa/xtensa-esp-elf/bin
SP=$(cd "$(dirname "$0")" && pwd); cd "$SP"
BYTES="$1"; OUT="$2"; MODE="${3:-marker}"
{
echo '	.text'; echo '	.global kfn'; echo 'kfn:'
echo '	entry	a1, 32'
echo '	l32i	a3, a3, 0'; echo '	l32i	a3, a3, 0'; echo '	l32i	a3, a3, 8'
for r in 2 4 5 6 7 8 9 10 11 12 13 14 15; do echo "	movi	a$r, 0"; done
echo '	memw'
echo "	.byte $BYTES"
if [ "$MODE" = marker ]; then
  echo '	l32i	a4, a3, 0'; echo '	addi	a4, a4, 1'; echo '	s32i	a4, a3, 0'
fi
echo '	memw'; echo '	movi	a2, 0'; echo '	retw.n'
} > /tmp/flixprobe.S
bash "$SP/asmpatch2.sh" /tmp/flixprobe.S 14c "$OUT" | tail -1
