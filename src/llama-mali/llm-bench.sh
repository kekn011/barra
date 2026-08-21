#!/system/bin/sh
# llm-bench.sh <modell.gguf> [ngl] — EIN llama-bench tg64 mit Speicher-Waechter (Abbruch bei <1,2 GB frei), protokolliert GPU-Takt + Peak.
. /data/adb/baseos/llm/env.sh
M=$1; NGL=${2:-99}; [ -f "$M" ] || { echo "Modell fehlt"; exit 1; }
L=/data/adb/baseos/run/llm-bench.log; : > $L
$LLM/llama-bench -m $M -ngl $NGL -p 0 -n 64 -r 2 -t 4 </dev/null > $L.raw 2>&1 &
P=$!; min=99999999; t=0
while kill -0 $P 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo); [ $a -lt $min ] && min=$a
  [ $a -lt 1228800 ] && { echo "WAECHTER: ${a}kB frei -> kill" | tee -a $L; kill -9 $P; break; }
  t=$((t+1)); [ $t -gt 300 ] && { echo "TIMEOUT" | tee -a $L; kill -9 $P; break; }
  sleep 1
done
wait $P 2>/dev/null; pkill -9 llama-bench 2>/dev/null
echo "min MemAvailable: $((min/1024)) MB, GPU-Takt: $(cat /sys/class/misc/mali0/device/cur_freq 2>/dev/null)" | tee -a $L
grep -aE "^\| |ggml_vulkan: [0-9]|error" $L.raw | tail -3 | tee -a $L
