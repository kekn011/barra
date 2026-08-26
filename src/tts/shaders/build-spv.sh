#!/bin/bash
# Baut die vier SPIR-V-Header, die gpudecd.c einbindet (src/tts/spv/*.h).
# Die Header sind eingecheckt, weil sie Build-Eingang von gpudecd sind; dieses Skript
# ist die Quelle-zu-Header-Kette und macht sie reproduzierbar.
#
#   sh build-spv.sh            baut nach ../spv/
#   sh build-spv.sh -check     baut in ein Temp-Verzeichnis und vergleicht mit ../spv/
#                              (Exitcode 1 bei Abweichung — Waechter gegen Header-Drift)
set -e
cd "$(dirname "$0")"
GLSL=${GLSL:-$(command -v glslangValidator || command -v glslang)}
[ -n "$GLSL" ] || { echo "glslangValidator/glslang nicht gefunden (GLSL=<pfad> setzen)"; exit 1; }

# Header-Symbol  <-  .comp-Quelle
#   convfused16   fusionierter Conv1d, TM=128/MM=8 (der Arbeitspferd-Kernel)
#   convfused16s  schlanke Variante TM=32/MM=2 fuer kleine Cout (weniger Padding-Verschwendung)
#   shuffle16     Sub-Pixel-Shuffle nach dem Upsample-Conv
#   mrf16         Multi-Receptive-Field-Summe (Residual-Glue)
SET="convfused16:conv_mm8.comp convfused16s:conv_gemm_fused_f16s.comp shuffle16:shuffle_f16.comp mrf16:mrf_f16.comp"

OUT=../spv
if [ "$1" = "-check" ]; then OUT=$(mktemp -d); fi

for pair in $SET; do
  sym=${pair%%:*}; src=${pair#*:}
  [ -f "$src" ] || { echo "FEHLT: $src"; exit 1; }
  "$GLSL" -V --target-env vulkan1.1 "$src" -o "$OUT/$sym.spv" >/dev/null
  # xxd -i benennt das Array nach dem Dateinamen -> im Temp-Verzeichnis bauen, damit der Name stimmt
  ( cd "$OUT" && xxd -i "$sym.spv" \
      | sed 's/unsigned char/const unsigned char/; s/unsigned int/const unsigned int/' \
      > "$sym""_spv.h" && rm -f "$sym.spv" )
done

if [ "$1" = "-check" ]; then
  rc=0
  for pair in $SET; do
    sym=${pair%%:*}
    if cmp -s "$OUT/${sym}_spv.h" "../spv/${sym}_spv.h"; then
      echo "  gleich    ${sym}_spv.h"
    else
      echo "  ABWEICHEND ${sym}_spv.h"; rc=1
    fi
  done
  rm -rf "$OUT"
  [ $rc -eq 0 ] && echo "SPV_CHECK_OK" || echo "SPV_CHECK_DRIFT"
  exit $rc
fi
echo "SPV_OK -> ../spv/"
