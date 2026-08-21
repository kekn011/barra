#!/bin/bash
set -e
SRC=/mnt/c/Users/kevin/projects/pixel-cluster-base/src/hwbridge/tpud.c
SP=/mnt/c/Users/kevin/AppData/Local/Temp/claude/C--Users-kevin-projects-pixel-cluster-base/087317f8-7a2e-477e-8ec5-ccbe33ea455c/scratchpad/gpu-attn
grep -n "MAXMODELS" "$SRC" | head -1
NDK=$(ls -d $HOME/android-sdk/ndk/* | sort -V | tail -1)
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang -O2 "$SRC" -o "$SP/out/tpud_pipe" -ldl
ls -la "$SP/out/tpud_pipe"
echo TPUD-BUILD-OK
