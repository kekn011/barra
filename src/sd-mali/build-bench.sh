#!/bin/bash
# Rebuild sd-cli (nach ggml-vulkan-Patches) + test-backend-ops als GEMM-Sweep-Werkzeug.
# In WSL: wsl bash /mnt/c/Users/kevin/projects/pixel-cluster-base/src/sd-mali/build-bench.sh
set -e
NDK=${NDK:-$HOME/android-sdk/ndk/27.2.12479018}
S=$HOME/stable-diffusion.cpp
W=/mnt/c/Users/kevin/AppData/Local/Temp/claude/C--Users-kevin-projects-pixel-cluster-base/3fe240e5-5b19-4c9f-929a-a40d4c151ed9/scratchpad
cd $S
echo "== sd-cli (inkrementell) =="
ninja -C build-android-vulkan sd-cli 2>&1 | tail -2
echo "== test-backend-ops (ggml standalone) =="
cmake -S ggml -B build-ggml-vk -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR=$HOME/vk-inc \
  -DVulkan_GLSLC_EXECUTABLE=$HOME/glslc-wrap.sh -DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=ON \
  2>&1 | grep -iE 'error' | head -8 || true
ninja -C build-ggml-vk test-backend-ops 2>&1 | tail -2
STRIP=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip
$STRIP -o $W/sd $S/build-android-vulkan/bin/sd-cli
TBO=$(find $S/build-ggml-vk -name test-backend-ops -type f | head -1)
$STRIP -o $W/test-backend-ops "$TBO"
ls -la $W/sd $W/test-backend-ops
echo BENCH-BUILD-FERTIG
