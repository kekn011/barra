#!/bin/bash
# ggml-barra GLM-Edge: .inc installieren, alle Shader (Qwen-Satz + GLM-Varianten) bauen,
# libggml-vulkan neu bauen + strippen. Ausgabe nach $SP/out.
set -e
SP=$(cd "$(dirname "$0")" && pwd)
python3 "$SP/battn_install.py" ~/llama.cpp
GLS=~/glslang-new/bin/glslang
mkdir -p "$SP/out"
for K in rope_qk_l5 rope_qk_l6 rope_qk_gen rope_qk_v8 qkt_f16 qkt_glm qkt_gen qkt_v8 softmax_causal av_rows3 av_glm av_gen av_v8; do
  $GLS -V --target-env vulkan1.1 "$SP/$K.comp" -o "$SP/out/$K.spv"
done
cd ~/llama.cpp
ninja -C build-android-vulkan-idp llama-cli llama-bench llama-perplexity llama-server 2>&1 | tail -3
NDK=$(ls -d $HOME/android-sdk/ndk/* | sort -V | tail -1)
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip -o "$SP/out/libggml-vulkan.so" build-android-vulkan-idp/bin/libggml-vulkan.so
ls -la "$SP/out/libggml-vulkan.so" "$SP/out/rope_qk_l6.spv" "$SP/out/qkt_glm.spv" "$SP/out/av_glm.spv"
echo BATTN-GLM-BUILD-OK
