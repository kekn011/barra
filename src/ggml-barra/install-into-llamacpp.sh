#!/bin/bash
# Kopiert das ggml-barra-Backend in einen llama.cpp-Checkout und registriert es (idempotent).
# usage: install-into-llamacpp.sh [~/llama.cpp]
set -e
LL=${1:-$HOME/llama.cpp}; SRC=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$LL/ggml/src/ggml-barra"
cp "$SRC/ggml-barra.cpp" "$SRC/CMakeLists.txt" "$LL/ggml/src/ggml-barra/"
cp "$SRC/../barra/barra.c" "$SRC/../barra/barra.h" "$LL/ggml/src/ggml-barra/"
cp "$SRC/ggml-barra.h" "$LL/ggml/include/"
grep -q 'ggml_add_backend(BARRA)' "$LL/ggml/src/CMakeLists.txt" || sed -i 's/^ggml_add_backend(BLAS)$/ggml_add_backend(BLAS)\nggml_add_backend(BARRA)/' "$LL/ggml/src/CMakeLists.txt"
grep -q 'option(GGML_BARRA' "$LL/ggml/CMakeLists.txt" || sed -i 's/^option(GGML_BLAS .*$/&\noption(GGML_BARRA                           "ggml: use barra TPU-FFN backend (Tensor G3)"     OFF)/' "$LL/ggml/CMakeLists.txt"
R="$LL/ggml/src/ggml-backend-reg.cpp"
if ! grep -q 'GGML_USE_BARRA' "$R"; then
python3 - "$R" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
a='#ifdef GGML_USE_BLAS\n#include "ggml-blas.h"\n#endif'
b='#ifdef GGML_USE_BLAS\n        register_backend(ggml_backend_blas_reg());\n#endif'
assert a in s and b in s
s=s.replace(a, a+'\n#ifdef GGML_USE_BARRA\n#include "ggml-barra.h"\n#endif',1)
s=s.replace(b, b+'\n#ifdef GGML_USE_BARRA\n        register_backend(ggml_backend_barra_reg());\n#endif',1)
open(p,'w').write(s); print("reg patched")
PY
fi
grep -c 'BARRA' "$LL/ggml/src/CMakeLists.txt" "$LL/ggml/CMakeLists.txt" "$R"
echo "installiert in $LL"
