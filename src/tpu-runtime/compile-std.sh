#!/system/bin/sh
# Standalone-TPU-Compile am Geraet (ohne frida) mit der gepatchten Compiler-Kopie aus dem Base-Image.
#   compile-std.sh <modell.tflite> [out.package]
B=/data/adb/baseos/tpu
export COMPILER_SO=$B/libcomp_std.so
export MODEL="$1"; OUT="${2:-$(dirname "$1")/out.package}"
[ -f "$MODEL" ] || { echo "usage: compile-std.sh <modell.tflite> [out.package]"; exit 1; }
rm -f /data/local/tmp/out.package /data/vendor/edgetpu/cache/_0 /data/vendor/edgetpu/cache/_0_checksum 2>/dev/null
cd /data/local/tmp && LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 $B/tpuc1 2>&1 | grep -iE "compile |rc=|PACKAGE|status=|chip_type|unrecognized|re-compile"
[ -f /data/local/tmp/out.package ] && { mv /data/local/tmp/out.package "$OUT"; ls -la "$OUT"; } || echo "kein Package"
