#!/system/bin/sh
T=/data/local/tmp; DF=/sys/class/devfreq/1a000000.rio
sh $T/mm/tpu-boost.sh on
echo "persistence: check governor over 20s idle..."
i=0; while [ $i -lt 4 ]; do sleep 5; echo "  +$((i*5+5))s: governor=$(cat $DF/governor) min=$(cat $DF/min_freq)"; i=$((i+1)); done
echo "--- run matmul, confirm high freq ---"
TPU_PIPE=8 TPU_PIPE_N=300 LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 $T/mm/tpud_pipe /dev/null $T/mm/peak/mm_D4096_B64.package >/tmp/o 2>&1 &
BG=$!; sleep 0.5; echo "  cur_freq under load: $(cat $DF/cur_freq)"; wait $BG 2>/dev/null
MS=$(grep -aoE '[0-9.]+ ms/Inf' /tmp/o | head -1 | grep -oE '[0-9.]+')
G=$(awk -v ms=$MS 'BEGIN{printf "%.0f", 2.0*64*4096*4096/(ms/1000)/1e9}')
echo "  D4096 B64: $MS ms -> $G GFLOPS (boosted)"
sh $T/mm/tpu-boost.sh off
echo "=== CPU topology (Tensor G3) ==="
for c in 0 1 2 3 4 5 6 7 8; do
  f=/sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq
  [ -e "$f" ] && echo "  cpu$c max=$(cat $f) Hz"
done
