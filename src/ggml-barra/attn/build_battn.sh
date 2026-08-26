#!/bin/bash
set -e
SP=${BARRA_WORK:-${TMPDIR:-/tmp}/barra}/gpu-attn; mkdir -p "$SP"
python3 "$SP/battn_install.py" ~/llama.cpp
GLS=~/glslang-new/bin/glslang
mkdir -p "$SP/out"
$GLS -V --target-env vulkan1.1 "$SP/rope_qk_l.comp" -o "$SP/out/rope_qk_l.spv"
$GLS -V --target-env vulkan1.1 "$SP/rope_qk_l2.comp" -o "$SP/out/rope_qk_l2.spv"
$GLS -V --target-env vulkan1.1 "$SP/av_rows.comp" -o "$SP/out/av_rows.spv"
cd ~/llama.cpp
ninja -C build-android-vulkan-idp llama-cli llama-bench llama-perplexity llama-server 2>&1 | tail -5
NDK=$(ls -d $HOME/android-sdk/ndk/* | sort -V | tail -1)
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip -o "$SP/out/libggml-vulkan.so" build-android-vulkan-idp/bin/libggml-vulkan.so
ls -la "$SP/out/libggml-vulkan.so" "$SP/out/rope_qk_l.spv"
echo BATTN-BUILD-OK
