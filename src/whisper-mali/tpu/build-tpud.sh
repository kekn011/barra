#!/bin/bash
set -e
NDK=$(ls -d $HOME/android-sdk/ndk/* | sort -V | tail -1)
OUT=${1:-/mnt/c/Users/kevin/AppData/Local/Temp/claude/C--Users-kevin-projects-pixel-cluster-base/1e37acd5-72d0-4b77-85ca-4ac493ae284e/scratchpad/tpud_pipe4}
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang -O2 /mnt/c/Users/kevin/projects/pixel-cluster-base/src/hwbridge/tpud.c -o $OUT -ldl
echo BUILD_OK
