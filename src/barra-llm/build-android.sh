#!/bin/bash
# barra-llm fuer Android (Bionic) bauen — gegen den Mali-llama.cpp-Shared-Build (src/llama-mali/build.sh, build-android-vulkan-idp)
# + libbarra (src/barra/barra.c via NDK). WSL. Ergebnis landet gestrippt in $B/stage/barra-llm (neben den Libs -> baseos/llm).
set -e
NDK=${NDK:-$HOME/android-sdk/ndk/27.2.12479018}
TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
CC=$TC/aarch64-linux-android34-clang; CXX=$TC/aarch64-linux-android34-clang++
L=${1:-$HOME/llama.cpp}; B=$L/build-android-vulkan-idp
REPO=${BARRA_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
SRC=$REPO/src
O=~/barra-llm-build; mkdir -p $O $B/stage; cd $O
$CC -O2 -Wall -c $SRC/barra/barra.c -I$SRC/barra -o barra.o
$CXX -O2 -std=c++17 -I$SRC/barra -I$L/include -I$L/ggml/include -c $SRC/barra-llm/barra-llm.cpp -o barra-llm.o
$CXX -O2 barra-llm.o barra.o -o barra-llm -L$B/bin -lllama -lggml -lggml-base -lvulkan -llog -static-libstdc++ -Wl,-rpath,'$ORIGIN'
$TC/llvm-strip -o $B/stage/barra-llm barra-llm
ls -la $B/stage/barra-llm
