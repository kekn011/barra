#!/bin/sh
# Whisper-Stufe-2-Lauf auf dem Node (als root via su). $1 = Modellname (z.B. base), $2 = optionale Extra-Args.
# Ein Modell pro Lauf; Log /data/local/tmp/wspgpu/run-<modell>.log, Ende-Marker WSP_GPU_DONE/WSP_GPU_FAIL.
D=/data/local/tmp/wspgpu
M=/data/local/ubuntu/root/whisper.cpp/models/ggml-$1.bin
LOG=$D/run-$1.log
echo "uid=$(id -u) model=$M" > "$LOG"
[ -f "$M" ] || { echo WSP_GPU_FAIL_NOMODEL >> "$LOG"; exit 1; }
# GPU pinnen (890 MHz) — direkt über sysfs, unabhaengig von barra-perf
echo 890000 > /sys/class/misc/mali0/device/scaling_min_freq 2>>"$LOG"
echo 890000 > /sys/class/misc/mali0/device/hint_min_freq 2>>"$LOG"
cd "$D" || exit 1
export LD_LIBRARY_PATH=$D
( ./whisper-cli -m "$M" -f "$D/test-de.wav" -l de -t 8 $2 >> "$LOG" 2>&1 \
    && echo WSP_GPU_DONE >> "$LOG" || echo WSP_GPU_FAIL >> "$LOG"
  grep MemAvailable /proc/meminfo >> "$LOG" ) &
