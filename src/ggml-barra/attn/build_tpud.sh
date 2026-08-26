#!/bin/bash
set -e
REPO=${BARRA_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
SRC=$REPO/src/hwbridge/tpud.c
SP=${BARRA_WORK:-${TMPDIR:-/tmp}/barra}/gpu-attn; mkdir -p "$SP"
grep -n "MAXMODELS" "$SRC" | head -1
NDK=$(ls -d $HOME/android-sdk/ndk/* | sort -V | tail -1)
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang -O2 "$SRC" -o "$SP/out/tpud_pipe" -ldl
ls -la "$SP/out/tpud_pipe"
echo TPUD-BUILD-OK
