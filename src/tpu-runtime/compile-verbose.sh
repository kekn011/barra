#!/system/bin/sh
# compile-verbose.sh <modell.tflite> — Compiler-Ausgabe komplett (Diagnose)
B=/data/adb/baseos/tpu
rm -f /data/local/tmp/out.package /data/vendor/edgetpu/cache/_0 /data/vendor/edgetpu/cache/_0_checksum 2>/dev/null
cd /data/local/tmp && COMPILER_SO=$B/libcomp_std.so MODEL=$1 LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 $B/tpuc1 2>&1 | tail -30
