#!/bin/bash
# Kompiliert shaders/*.comp nach SPIR-V (Vulkan 1.1) und bettet sie als C-Arrays in gpud_shaders.h ein.
# Laeuft in WSL (glslangValidator; neueres unter ~/glslang-new/bin wird bevorzugt). Der Node hat keinen glslc.
set -e
cd "$(dirname "$0")"
G=$(ls ~/glslang-new/bin/glslangValidator 2>/dev/null || which glslangValidator)
[ -x "$G" ] || { echo "glslangValidator fehlt (src/llama-mali/get-glslang.sh)"; exit 1; }
OUT=gpud_shaders.h; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
{
  echo "// GENERIERT von build-shaders.sh ($(basename "$G")) - nicht von Hand aendern."
  echo "#pragma once"
  echo "#include <stdint.h>"
} > "$OUT"
for f in shaders/*.comp; do
  n=$(basename "$f" .comp)
  "$G" -V --target-env vulkan1.1 -Ishaders -o "$TMP/$n.spv" "$f" >"$TMP/$n.log" 2>&1 || { echo "FEHLER in $f:"; cat "$TMP/$n.log"; exit 1; }
  {
    echo "static const uint8_t gpud_spv_${n}[] = {"
    od -An -v -tu1 -w32 "$TMP/$n.spv" | sed 's/ \+/,/g; s/^,//; s/$/,/'
    echo "};"
    echo "static const uint32_t gpud_spv_${n}_len = $(stat -c %s "$TMP/$n.spv");"
  } >> "$OUT"
  echo "$n: $(stat -c %s "$TMP/$n.spv") B"
done
echo "-> $OUT ($(stat -c %s "$OUT") B)"
