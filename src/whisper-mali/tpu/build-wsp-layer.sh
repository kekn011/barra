#!/bin/bash
set -e
NDK=$(ls -d $HOME/android-sdk/ndk/* | sort -V | tail -1); TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
REPO=${BARRA_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
SRC=$REPO/src
OUT=${BARRA_WORK:-${TMPDIR:-/tmp}/barra}/wsp_layer; mkdir -p "$OUT"
$TC/aarch64-linux-android34-clang -O2 -I$SRC/experiments/gpu-attn $SRC/whisper-mali/tpu/wsp_layer.c $SRC/experiments/gpu-attn/barra.c -o $OUT -lm
echo LAYER_BUILD_OK
