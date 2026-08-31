#!/bin/bash
# Kopiert das ggml-gpud-Backend in einen llama.cpp-Checkout und registriert es (idempotent, Muster ggml-barra).
# usage: install-into-llamacpp.sh [~/llama.cpp]      (vorher: build-shaders.sh in WSL -> gpud_shaders.h)
set -e
LL=${1:-$HOME/llama.cpp}; SRC=$(cd "$(dirname "$0")" && pwd)
[ -f "$SRC/gpud_shaders.h" ] || { echo "gpud_shaders.h fehlt - erst build-shaders.sh (WSL)"; exit 1; }
mkdir -p "$LL/ggml/src/ggml-gpud"
cp "$SRC/ggml-gpud.cpp" "$SRC/CMakeLists.txt" "$SRC/gpud_shaders.h" "$LL/ggml/src/ggml-gpud/"
cp "$SRC/../barra/barra.c" "$SRC/../barra/barra.h" "$LL/ggml/src/ggml-gpud/"
cp "$SRC/ggml-gpud.h" "$LL/ggml/include/"
grep -q 'ggml_add_backend(GPUD)' "$LL/ggml/src/CMakeLists.txt" || sed -i 's/^ggml_add_backend(BLAS)$/ggml_add_backend(BLAS)\nggml_add_backend(GPUD)/' "$LL/ggml/src/CMakeLists.txt"
grep -q 'option(GGML_GPUD' "$LL/ggml/CMakeLists.txt" || sed -i 's/^option(GGML_BLAS .*$/&\noption(GGML_GPUD                           "ggml: use barra gpud-zc (Mali via dmabuf)"       OFF)/' "$LL/ggml/CMakeLists.txt"
R="$LL/ggml/src/ggml-backend-reg.cpp"
if ! grep -q 'GGML_USE_GPUD' "$R"; then
python3 - "$R" <<'PY' || perl -0pi -e 's/(#ifdef GGML_USE_BLAS\n#include "ggml-blas.h"\n#endif)/$1\n#ifdef GGML_USE_GPUD\n#include "ggml-gpud.h"\n#endif/; s/(#ifdef GGML_USE_BLAS\n        register_backend\(ggml_backend_blas_reg\(\)\);\n#endif)/$1\n#ifdef GGML_USE_GPUD\n        register_backend(ggml_backend_gpud_reg());\n#endif/' "$R"
import sys
p=sys.argv[1]; s=open(p).read()
a='#ifdef GGML_USE_BLAS\n#include "ggml-blas.h"\n#endif'
b='#ifdef GGML_USE_BLAS\n        register_backend(ggml_backend_blas_reg());\n#endif'
assert a in s and b in s
s=s.replace(a, a+'\n#ifdef GGML_USE_GPUD\n#include "ggml-gpud.h"\n#endif',1)
s=s.replace(b, b+'\n#ifdef GGML_USE_GPUD\n        register_backend(ggml_backend_gpud_reg());\n#endif',1)
open(p,'w').write(s); print("reg patched")
PY
fi
grep -c 'GPUD' "$LL/ggml/src/CMakeLists.txt" "$LL/ggml/CMakeLists.txt" "$R"
echo "installiert in $LL  (cmake -B build-gpud -DGGML_GPUD=ON -DGGML_VULKAN=OFF -DGGML_RPC=OFF -DGGML_NATIVE=ON)"
