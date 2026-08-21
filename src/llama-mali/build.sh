#!/bin/bash
# llama.cpp fuer akita (Android/Bionic, Vulkan auf Mali-G715) mit unseren Mali-Kerneln bauen — in WSL ausfuehren.
#   ./build.sh [~/llama.cpp]          (Checkout wird bei Bedarf geklont und auf den Referenz-Commit gesetzt)
# Ergebnis: $LL/build-android-vulkan-idp/bin/{llama-server,llama-bench,llama-perplexity,llama-quantize,llama-cli,*.so}
# Voraussetzungen (einmalig): NDK 27 unter ~/android-sdk/ndk/27.2.12479018, Vulkan-Header in ~/vk-inc,
#   SPIRV-Headers (apt spirv-headers), glslang >= 16 via ./get-glslang.sh (Integer-Dot braucht glslang 16, NDK-shaderc 2022 kennt es nicht).
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
LL=${1:-$HOME/llama.cpp}
REF=8d274dd7c6233ed73c7509cc2a8be9960f7df7d5     # llama.cpp-Stand, gegen den der Patch erzeugt wurde (11.8.2026)
NDK=${NDK:-$HOME/android-sdk/ndk/27.2.12479018}
B=build-android-vulkan-idp
[ -d "$LL/.git" ] || git clone https://github.com/ggml-org/llama.cpp "$LL"
cd "$LL"
if ! git diff --quiet -- ggml/src/ggml-vulkan; then
  echo "ggml-vulkan hat lokale Aenderungen (Patch schon drin?) — lasse sie stehen."
else
  git checkout -q "$REF" 2>/dev/null || { git fetch -q origin; git checkout -q "$REF"; }
  git apply --check "$SRC/llama-cpp-mali-vulkan.patch" && git apply "$SRC/llama-cpp-mali-vulkan.patch" && echo "Mali-Patch angewendet"
fi
bash "$SRC/../ggml-barra/install-into-llamacpp.sh" "$LL" >/dev/null && echo "ggml-barra (TPU-FFN-Backend) eingehaengt"
cp "$SRC/glslc-wrap.sh" ~/glslc-wrap.sh; chmod +x ~/glslc-wrap.sh
~/glslc-wrap.sh -o - -fshader-stage=compute --target-env=vulkan1.3 ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/integer_dot.comp > /dev/null && echo "glslc-wrap: integer_dot OK"
cmake -S . -B $B -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR=$HOME/vk-inc \
  -DVulkan_GLSLC_EXECUTABLE=$HOME/glslc-wrap.sh -DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers \
  -DGGML_BARRA=ON -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
  2>&1 | grep -iE 'integer_dot|glslc|error|Vulkan found' | head -20 || true
ninja -C $B llama-server llama-bench llama-perplexity llama-quantize llama-cli test-backend-ops 2>&1 | tail -3
# gestrippte Kopie fuer den Node (~72 MB statt ~280 MB)
STRIP=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip; mkdir -p $B/stage
for f in llama-server llama-bench llama-perplexity llama-quantize llama-cli libggml-base.so libggml-cpu.so libggml-vulkan.so libggml-barra.so libggml.so          libllama.so libllama-common.so libmtmd.so libllama-server-impl.so libllama-bench-impl.so libllama-perplexity-impl.so libllama-quantize-impl.so libllama-cli-impl.so; do
  $STRIP -o $B/stage/$f $B/bin/$f; done
bash "$SRC/../barra-llm/build-android.sh" "$LL" >/dev/null && echo "barra-llm gebaut"
cp "$SRC/../barra-llm/run-llm.sh" $B/stage/
du -sh $B/stage; ls -la $B/stage/llama-server $B/stage/libggml-vulkan.so $B/stage/libggml-barra.so $B/stage/barra-llm
