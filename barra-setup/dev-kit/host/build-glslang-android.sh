#!/bin/sh
# build-glslang-android.sh - reproduzierbarer aarch64-android Cross-Build von glslang (P2).
# IN WSL laufen. Ergebnis: ein bionic-glslang, das devbuild.sh 'spv' ON-DEVICE nutzt
# (der On-Device-Shaderbau-Unblocker). Danach ins Kit:
#   fetch-devtools.ps1 -Supply glslang -Path <out> ; dann pack-dev-kit.sh
#
# PIN (25.8. verifiziert, Ergebnis lief am Node): glslang @23076b376e..., NDK 27.2.12479018,
#   arm64-v8a / android-34, ENABLE_OPT=0. Gestripptes Binary sha256 1741431bded4f84a...bac,
#   NEEDED nur libm/libdl/libc (c++_static) -> laeuft standalone auf Bionic.
set -e
NDK=${ANDROID_NDK:-$HOME/android-sdk/ndk/27.2.12479018}
SRC=${GLSLANG_SRC:-$HOME/glslang-src}
OUT=${1:-$PWD/glslang-android}
PIN=23076b376e06a99b4c765df5c9836d127c8bbbfc
[ -d "$NDK" ] || { echo "NDK fehlt: $NDK (ANDROID_NDK setzen)"; exit 1; }

echo "== glslang @${PIN} holen =="
mkdir -p "$SRC"; cd "$SRC"
[ -d .git ] || git init -q
git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/KhronosGroup/glslang
git fetch --depth 1 origin "$PIN"
git checkout -q FETCH_HEAD

echo "== Android-Binary-Sperre aufheben (glslang baut den Compiler normal nicht FUER Android) =="
sed -i "s/^if (IOS OR ANDROID)/if (IOS)  # patched: on-device glslang/" CMakeLists.txt

echo "== cmake configure + build =="
rm -rf build-android
cmake -G Ninja -B build-android \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_OPT=0 -DGLSLANG_TESTS=OFF -DENABLE_GLSLANG_BINARIES=ON >/dev/null
ninja -C build-android >/dev/null

STRIP=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip
"$STRIP" build-android/StandAlone/glslang -o "$OUT"
echo "== fertig =="; file "$OUT"; sha256sum "$OUT"
echo "-> ins Kit: powershell fetch-devtools.ps1 -Supply glslang -Path '$OUT'  (dann pack-dev-kit.sh)"
