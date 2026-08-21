#!/bin/bash
# glslc-Ersatz auf Basis von glslang 16.5 (hat GL_EXT_integer_dot_product). Versteht die Aufrufe von
# llama.cpp vulkan-shaders-gen + dem CMake-Feature-Test:  glslc -fshader-stage=compute --target-env=vulkan1.2 [-O] [-DX=Y..] in -o out [-MD -MF dep]
GLSLANG=${GLSLANG:-$HOME/glslang-new/bin/glslang}
args=(); out=""; dep=""; stage="comp"; tenv="vulkan1.2"; in=""; tostdout=0
while [ $# -gt 0 ]; do
  case "$1" in
    -fshader-stage=*) s=${1#-fshader-stage=}; case $s in compute) stage=comp;; vertex) stage=vert;; fragment) stage=frag;; *) stage=$s;; esac; shift;;
    --target-env=*) tenv=${1#--target-env=}; shift;;
    -O|-O0|-Os) shift;;
    -o) out=$2; shift 2;;
    -MD) shift;;
    -MF) dep=$2; shift 2;;
    -D*) args+=("$1"); shift;;
    -I*) args+=("$1"); shift;;
    -*) shift;;
    *) in=$1; shift;;
  esac
done
if [ "$out" = "-" ]; then tostdout=1; out=$(mktemp /tmp/glslw.XXXXXX.spv); fi
depargs=(); [ -n "$dep" ] && depargs=(--depfile "$dep")
log=$("$GLSLANG" -V --target-env "$tenv" -S "$stage" -P"#extension GL_GOOGLE_include_directive : enable" -I"$(dirname "$in")" "${depargs[@]}" "${args[@]}" "$in" -o "$out" 2>&1)
rc=$?
# glslang druckt immer den Dateinamen -> nur bei Fehler ausgeben (vulkan-shaders-gen wertet jede Ausgabe als Fehler)
if [ $rc -ne 0 ]; then echo "$log" 1>&2; fi
# glslang druckt den Dateinamen auf stdout; stderr bekommt Fehler -> fuer den CMake-Test reicht die Fehlermeldung
# Depfile kommt von glslang (--depfile, echte Header-Abhaengigkeiten); Fallback falls leer
if [ $rc -eq 0 ] && [ -n "$dep" ] && [ ! -s "$dep" ]; then echo "$out: $in" > "$dep"; fi
if [ $tostdout -eq 1 ]; then [ $rc -eq 0 ] && cat "$out"; rm -f "$out"; fi
exit $rc
