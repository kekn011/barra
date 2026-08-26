#!/bin/bash
set -e
NDK=$(ls -d $HOME/android-sdk/ndk/* | sort -V | tail -1)
REPO=${BARRA_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
OUT=${1:-${BARRA_WORK:-${TMPDIR:-/tmp}/barra}/tpud_pipe4}
mkdir -p "$(dirname "$OUT")"
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang -O2 $REPO/src/hwbridge/tpud.c -o $OUT -ldl
echo BUILD_OK
