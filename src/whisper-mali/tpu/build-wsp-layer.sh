#!/bin/bash
set -e
NDK=$(ls -d $HOME/android-sdk/ndk/* | sort -V | tail -1); TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
SRC=/mnt/c/Users/kevin/projects/pixel-cluster-base/src
OUT=/mnt/c/Users/kevin/AppData/Local/Temp/claude/C--Users-kevin-projects-barra/44767811-2ea3-4351-8e23-9384626c4a1b/scratchpad/wsp_layer
$TC/aarch64-linux-android34-clang -O2 -I$SRC/experiments/gpu-attn $SRC/whisper-mali/tpu/wsp_layer.c $SRC/experiments/gpu-attn/barra.c -o $OUT -lm
echo LAYER_BUILD_OK
