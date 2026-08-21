#!/bin/bash
# build.sh FMT WG TPR R  -> out/gemv_<fmt>_w<WG>_t<TPR>_r<R>.spv ; baut gpugemv wenn noetig
FMT=$1; WG=${2:-64}; TPR=${3:-16}; R=${4:-2}
cd /mnt/c/Users/kevin/projects/pixel-cluster-base/src/experiments/roofline/gemv
mkdir -p out
LC=$(echo $FMT | tr A-Z a-z); UC=$(echo $FMT | tr a-z A-Z)
NAME=gemv_${LC}_w${WG}_t${TPR}_r${R}
GLSL=${GLSL:-$HOME/glslang-new/bin/glslang}; [ -x $GLSL ] || GLSL=glslangValidator
$GLSL -V --target-env vulkan1.1 -DFMT_${UC}=1 -DWG=$WG -DTPR=$TPR -DR=$R gemv.comp -o out/$NAME.spv || { echo "GLSL FAIL"; exit 1; }
NDK=$(ls -d ~/android-sdk/ndk/* | sort -V | tail -1); TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
SYS=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/34
if [ ! -f out/gpugemv ] || [ gpugemv.c -nt out/gpugemv ]; then
  $TC/aarch64-linux-android34-clang -O2 gpugemv.c -o out/gpugemv -L$SYS -lvulkan -lm || exit 1; $TC/llvm-strip out/gpugemv; echo "gpugemv gebaut"
fi
echo "spv: $NAME"
