#!/bin/bash
# llama.cpp (Android, Vulkan) neu bauen mit Integer-Dot-faehigem Shader-Compiler (glslang 16.5 via glslc-wrap.sh)
set -e
cp /mnt/c/Users/kevin/projects/pixel-cluster-base/src/experiments/roofline/gemv/glslc-wrap.sh ~/glslc-wrap.sh; chmod +x ~/glslc-wrap.sh
# Selbsttest des Wrappers am Feature-Test-Shader
cd ~/llama.cpp
~/glslc-wrap.sh -o - -fshader-stage=compute --target-env=vulkan1.3 ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/integer_dot.comp > /dev/null && echo "wrapper: integer_dot OK"
NDK=/home/kevin/android-sdk/ndk/27.2.12479018
B=build-android-vulkan-idp
cmake -S . -B $B -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR=$HOME/vk-inc \
  -DVulkan_GLSLC_EXECUTABLE=$HOME/glslc-wrap.sh -DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers \
  -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
  2>&1 | grep -iE 'integer_dot|glslc|error|Vulkan found' | head -20
ninja -C $B llama-cli llama-bench test-backend-ops 2>&1 | tail -5
ls -la $B/bin/llama-bench $B/bin/test-backend-ops $B/bin/llama-cli
