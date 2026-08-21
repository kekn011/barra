#!/system/bin/sh
# idp-bench4b.sh [model] -> EIN llama-bench tg64 -ngl 99 mit Speicher-Waechter (kill bei MemAvailable < 1200 MB), Peak/Takt/CPU-Zeit protokolliert
D=/data/local/tmp/idp; L=$D/bench4b.log
export LD_LIBRARY_PATH=$D:/system/lib64:/vendor/lib64
export GGML_VK_DISABLE_COOPMAT=1
M=${1:-/data/local/tmp/models/qwen3-4b-q4_k_m.gguf}
cd $D; : > $L
./llama-bench -m $M -ngl 99 -p 0 -n 64 -r 2 -t 4 </dev/null > $L.raw 2>&1 &
P=$!
min=99999999; t=0
while kill -0 $P 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo); [ $a -lt $min ] && min=$a
  if [ $a -lt 1228800 ]; then echo "WAECHTER: MemAvailable ${a}kB -> kill" >> $L; kill -9 $P; pkill -9 llama-bench; break; fi
  t=$((t+1)); [ $t -gt 300 ] && { echo "TIMEOUT 300s -> kill" >> $L; kill -9 $P; pkill -9 llama-bench; break; }
  [ $((t%10)) -eq 0 ] && echo "t=${t}s avail=$((a/1024))MB gpu=$(cat /sys/class/misc/mali0/device/cur_freq) cpu=$(cat /proc/$P/stat 2>/dev/null | awk '{print ($14+$15)/100}')s" >> $L
  sleep 1
done
wait $P 2>/dev/null
echo "min MemAvailable: $((min/1024)) MB" >> $L
grep -aE "^\| qwen|error|llama_model_load|ggml_vulkan: [0-9]" $L.raw | tail -3 >> $L
pkill -9 llama-bench; ps -A -o PID,NAME | grep -a llama >> $L
echo DONE >> $L
