#!/system/bin/sh
# compile-dir.sh <verzeichnis> — alle *.tflite darin mit dem Base-Compiler kompilieren, je Modell rc/status/Package-Groesse ausgeben
B=/data/adb/baseos/tpu; D=$1; cd "$D" || exit 1
for f in *.tflite; do
  rm -f /data/local/tmp/out.package /data/vendor/edgetpu/cache/_0 /data/vendor/edgetpu/cache/_0_checksum 2>/dev/null
  R=$(cd /data/local/tmp && COMPILER_SO=$B/libcomp_std.so MODEL=$D/$f LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 $B/tpuc1 2>&1 | grep -oE 'rc=[0-9]+|status="[^"]*"|PACKAGE [0-9]+' | tr '\n' ' ')
  echo "$f : $R"; [ -f /data/local/tmp/out.package ] && mv /data/local/tmp/out.package "$D/${f%.tflite}.package"
done
