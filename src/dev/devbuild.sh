#!/system/bin/sh
# barra Dev-Kit Build-Helfer — die Edit->Build-Haelfte der Kernel-Schleife, GERAETELOKAL
# (ersetzt das host-gebundene gpu-kernels/build.sh mit seinen ~/glslang-/NDK-Pfaden).
#   devbuild.sh spv  <name.comp> <out.spv> [-DDEFINE=1 ...]   Shader -> SPIR-V (glslang)
#   devbuild.sh prog <src.c|.cpp> <out>                       libbarra-Programm (Container, gcc)
# Schneller GPU-Loop: nur die .spv neu erzeugen; die Harness-Binaries (gpugemv/gpugemm,
# NDK/bionic) laden sie zur Laufzeit -> kein Neubau, kein Daemon-Neustart.
# glslang kommt via fetch-or-supply (fetch-devtools.ps1) nach third-party/bin/.
DEV=${BARRA_DEV_DIR:-/data/adb/baseos/dev}
GLSL="$(command -v glslang || command -v glslangValidator || echo $DEV/third-party/bin/glslang)"
set -e
case "${1:-}" in
  spv)
    C="$2"; O="$3"; shift 3 2>/dev/null || { echo "usage: devbuild.sh spv <in.comp> <out.spv> [-D...]"; exit 1; }
    [ -f "$C" ] || { echo "Shader fehlt: $C"; exit 1; }
    [ -x "$GLSL" ] || { echo "glslang fehlt (fetch-devtools.ps1, oder in $DEV/third-party/bin/ ablegen)"; exit 2; }
    "$GLSL" -V --target-env vulkan1.1 "$@" "$C" -o "$O" && echo "OK -> $O";;
  prog)
    S="$2"; O="${3:-a.out}"; case "$S" in *.cpp|*.cc) CC=g++;; *) CC=gcc;; esac
    "$CC" -O2 "$S" -o "$O" -lbarra -lm && echo "OK -> $O";;
  *) echo "devbuild.sh spv <in.comp> <out.spv> [-D...] | prog <src> <out>";;
esac
