#!/bin/bash
# stable-diffusion.cpp fuer akita (Android/Bionic, Vulkan auf Mali-G715) — Bild-Kit Stufe 1+2.
# In WSL ausfuehren:
#   wsl bash /mnt/c/Users/kevin/projects/pixel-cluster-base/src/sd-mali/build.sh
# Rezept = src/whisper-mali/build.sh (gleiche NDK-/glslc-wrap-Toolchain). sd.cpp baut das
# CLI-Binary "sd" statisch; wir bauen ZWEI Varianten: sd-cpu (Korrektheits-Baseline) und
# sd (Vulkan). Ergebnis: stage/ mit gestrippten Binaries.
set -e
S=${1:-$HOME/stable-diffusion.cpp}
NDK=${NDK:-$HOME/android-sdk/ndk/27.2.12479018}
[ -d "$S/.git" ] || git clone --recursive --depth 1 https://github.com/leejet/stable-diffusion.cpp "$S"
cd "$S"
COMMON="-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF"

echo "== 1) CPU-Referenz =="
cmake -S . -B build-android-cpu -G Ninja $COMMON 2>&1 | grep -iE 'error|warning: manually' | head -5 || true
ninja -C build-android-cpu 2>&1 | tail -2

echo "== 2) Vulkan =="
FT=$(ls ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/integer_dot.comp 2>/dev/null || true)
[ -n "$FT" ] && ~/glslc-wrap.sh -o /dev/null -fshader-stage=compute --target-env=vulkan1.3 "$FT" && echo "glslc-wrap OK"
cmake -S . -B build-android-vulkan -G Ninja $COMMON \
  -DSD_VULKAN=ON -DVulkan_INCLUDE_DIR=$HOME/vk-inc \
  -DVulkan_GLSLC_EXECUTABLE=$HOME/glslc-wrap.sh -DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  2>&1 | grep -iE 'vulkan|error' | head -10 || true
ninja -C build-android-vulkan 2>&1 | tail -2

echo "== 3) Stage =="
STRIP=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip
mkdir -p stage
CPU_BIN=$(find build-android-cpu -name sd -type f | head -1)
VK_BIN=$(find build-android-vulkan -name sd -type f | head -1)
[ -n "$CPU_BIN" ] && $STRIP -o stage/sd-cpu "$CPU_BIN"
[ -n "$VK_BIN" ] && $STRIP -o stage/sd "$VK_BIN"
for f in $(find build-android-vulkan -name '*.so' -type f 2>/dev/null); do $STRIP -o stage/$(basename "$f") "$f"; done
du -sh stage; ls -l stage
echo "BUILD-FERTIG"
