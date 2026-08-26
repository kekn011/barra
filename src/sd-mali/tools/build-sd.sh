#!/bin/bash
# Nur sd-cli inkrementell bauen + strippen ins Session-Scratchpad
set -e
NDK=${NDK:-$HOME/android-sdk/ndk/27.2.12479018}
S=$HOME/stable-diffusion.cpp
W=${BARRA_WORK:-${TMPDIR:-/tmp}/barra}; mkdir -p "$W"
cd $S
ninja -C build-android-vulkan sd-cli 2>&1 | grep -E "error|warning: unused|FAILED|Linking|^\[" | tail -5
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip -o $W/sd $S/build-android-vulkan/bin/sd-cli
ls -la $W/sd
echo SD-BUILD-FERTIG
