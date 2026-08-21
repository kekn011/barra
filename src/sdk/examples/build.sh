#!/bin/sh
# Build all barra SDK examples in this directory (the base image has gcc but no make).
set -e
cd "$(dirname "$0")"
CC=${CC:-gcc}; CFLAGS=${CFLAGS:--O2 -Wall}
FLAGS=$(pkg-config --cflags --libs barra)
for f in *.c; do
  out=${f%.c}
  echo "  CC $f -> $out"
  $CC $CFLAGS "$f" $FLAGS -o "$out"
done
echo "done. run e.g.: ./barra_demo"
