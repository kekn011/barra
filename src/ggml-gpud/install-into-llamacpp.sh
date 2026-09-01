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
# common/arg.cpp: --override-tensor kennt nur die Default-Bufts der Devices; extra_bufts (GPUDnat)
# nachziehen, damit -ot "...=GPUDnat" funktioniert (idempotent).
A="$LL/common/arg.cpp"
if ! grep -q 'gpud-extra-bufts' "$A"; then
perl -0pi -e 's/(        if \(buft\) \{\n            buft_list\[ggml_backend_buft_name\(buft\)\] = buft;\n        \})/$1\n        \/\/ gpud-extra-bufts: auch die extra Buffer-Types des Devices anbieten (z.B. GPUDnat)\n        auto * ereg = ggml_backend_dev_backend_reg(dev);\n        if (ereg) {\n            auto get_extra = (ggml_backend_buffer_type_t * (*)(ggml_backend_dev_t))\n                ggml_backend_reg_get_proc_address(ereg, "ggml_backend_dev_get_extra_bufts");\n            if (get_extra) {\n                for (auto * eb = get_extra(dev); eb \&\& *eb; ++eb) {\n                    buft_list[ggml_backend_buft_name(*eb)] = *eb;\n                }\n            }\n        }/' "$A"
grep -q 'gpud-extra-bufts' "$A" && echo "arg.cpp gepatcht" || { echo "FEHLER: arg.cpp-Patch griff nicht"; exit 1; }
fi
# llama-bench hat einen EIGENEN -ot-Parser mit derselben Luecke (nur Default-Bufts) -> gleicher Patch.
B="$LL/tools/llama-bench/llama-bench.cpp"
if [ -f "$B" ] && ! grep -q 'gpud-extra-bufts' "$B"; then
perl -0pi -e 's/(                        auto \* buft = ggml_backend_dev_buffer_type\(dev\);\n                        if \(buft\) \{\n                            buft_list\[ggml_backend_buft_name\(buft\)\] = buft;\n                        \})/$1\n                        \/\/ gpud-extra-bufts: auch extra Buffer-Types anbieten (GPUDnat)\n                        auto * ereg = ggml_backend_dev_backend_reg(dev);\n                        if (ereg) {\n                            auto get_extra = (ggml_backend_buffer_type_t * (*)(ggml_backend_dev_t))\n                                ggml_backend_reg_get_proc_address(ereg, "ggml_backend_dev_get_extra_bufts");\n                            if (get_extra) {\n                                for (auto * eb = get_extra(dev); eb \&\& *eb; ++eb) {\n                                    buft_list[ggml_backend_buft_name(*eb)] = *eb;\n                                }\n                            }\n                        }/' "$B"
grep -q 'gpud-extra-bufts' "$B" && echo "llama-bench gepatcht" || { echo "FEHLER: llama-bench-Patch griff nicht"; exit 1; }
fi
grep -c 'GPUD' "$LL/ggml/src/CMakeLists.txt" "$LL/ggml/CMakeLists.txt" "$R"
echo "installiert in $LL  (cmake -B build-gpud -DGGML_GPUD=ON -DGGML_VULKAN=OFF -DGGML_RPC=OFF -DGGML_NATIVE=ON)"
