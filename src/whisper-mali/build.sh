#!/bin/bash
# whisper.cpp fuer akita (Android/Bionic, Vulkan auf Mali-G715) — Whisper-Stufe 2. In WSL ausfuehren:
#   wsl bash /mnt/c/Users/kevin/projects/pixel-cluster-base/src/whisper-mali/build.sh
# Rezept = src/llama-mali/build.sh ohne Mali-Patch/ggml-barra (Whisper-Modelle sind f16,
# der Mali-Patch beschleunigt nur q4_K/q6_K-Decode-GEMV). Ergebnis: stage/ mit Binaries+Libs.
set -e
W=${1:-$HOME/whisper.cpp}
NDK=${NDK:-$HOME/android-sdk/ndk/27.2.12479018}
B=build-android-vulkan
[ -d "$W/.git" ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$W"
cd "$W"
~/glslc-wrap.sh -o /dev/null -fshader-stage=compute --target-env=vulkan1.3 ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/integer_dot.comp && echo "glslc-wrap OK"
cmake -S . -B $B -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR=$HOME/vk-inc \
  -DVulkan_GLSLC_EXECUTABLE=$HOME/glslc-wrap.sh -DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON \
  2>&1 | grep -iE 'vulkan|error' | head -10 || true
ninja -C $B whisper-cli whisper-bench whisper-server 2>&1 | tail -3
STRIP=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip
mkdir -p $B/stage
for f in $B/bin/whisper-cli $B/bin/whisper-bench $B/bin/whisper-server $B/bin/*.so; do
  $STRIP -o $B/stage/$(basename "$f") "$f"
done
du -sh $B/stage; ls -l $B/stage
