#!/system/bin/sh
# Bewachter On-Device-TPU-Compile (Lektion 21.8.: qsplit-Shapes trieben libcomp_std in
# Speicher-Runaway -> 2x Kernel-Panic). Watchdog killt tpuc1 unter 1,2 GB MemAvailable.
#   wsptpu-compile.sh <tflite> <out.package>
LOG=/data/local/tmp/wsptpu/compile.log
: > $LOG
echo "uid=$(id -u) compile $1" >> $LOG
( while :; do
    a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$a" -lt 1200000 ]; then
      echo "GUARD: MemAvailable ${a}kB -> pkill tpuc1" >> $LOG
      pkill -x tpuc1
      sleep 2
    fi
    sleep 0.5
  done ) &
G=$!
timeout 180 sh /data/adb/baseos/tpu/compile-std.sh "$1" "$2" >> $LOG 2>&1
RC=$?
kill $G 2>/dev/null
echo "COMPILE_RC=$RC" >> $LOG
tail -8 $LOG
