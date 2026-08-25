#!/bin/bash
# Whisper-Modell -> kompletter TPU-Package-Satz (PC-Teil): ref, Kern v5, Layer-Pakete, cross, conv.
#   gen-model.sh <name> <ggml-bin> <NL> <H>
set -e
NAME=$1; GGML=$2; NL=$3; H=$4
cd /mnt/c/Users/kevin/projects/pixel-cluster-base/src/whisper-mali/tpu
WD=/mnt/c/Users/kevin/projects/pixel-cluster-base/tpu-toolchain/whisper
OUT=$WD/${NAME}_pkgs
mkdir -p "$OUT"
FILT='WARNING|oneDNN|cuda|absl|fully_quantize|cpu_feature|interpreter.py|UserWarning|migration|deletion|LiteRT|XNNPACK|^\s*$|^INFO'
echo "== 1/5 Referenz ($NL Layer):"
python3 whisper_ref.py "$GGML" "$WD/test-de.wav" "$OUT" $NL 2>&1 | grep -vE "$FILT" | tail -4
echo "== 2/5 Kern v5 (H=$H):"
python3 whisper_core.py "$OUT/ref_base_l0.npz" "$OUT/wsp_core5_b0" 16x8 375 375 $H 64 0 v5 2>&1 | grep -vE "$FILT" | tail -4
echo "== 3/5 Layer-Pakete:"
python3 whisper_pkgs7.py "$GGML" "$OUT/ref_base_all.npz" "$OUT" $NL "$OUT/wsp_core5_b0.qparams.json" 2>&1 | grep -vE "$FILT" | grep -E "KETTE|WROTE l0_|DONE" | tail -8
echo "== 4/5 Cross:"
python3 gen_cross.py "$GGML" "$OUT/ref_base_all.npz" "$OUT/cross" 2>&1 | grep -vE "$FILT" | grep -E "CHECK|DONE" | tail -3
echo "== 5/5 Conv:"
python3 gen_conv.py "$GGML" "$WD/test-de.wav" "$OUT/conv" 2>&1 | grep -vE "$FILT" | grep -E "CHECK|DONE" | tail -2
echo "GENMODEL_DONE $NAME"
