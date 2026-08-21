#!/system/bin/sh
# pp512-Bench mit Attention-Offload (BARRA_ATTN_LAYERS optional via Env)
L=/data/adb/baseos/llm; B=/data/local/tmp/battn; M=/data/local/tmp/models/qwen3-4b.gguf
G=/sys/class/misc/mali0/device; MAX=$(tr ' ' '\n' < $G/available_frequencies|grep -v '^$'|sort -n|tail -1)
echo $MAX > $G/scaling_min_freq 2>/dev/null; echo $MAX > $G/hint_min_freq 2>/dev/null
export LD_LIBRARY_PATH=$B/lib:$L:/system/lib64:/vendor/lib64
export BARRA_SOCK_DIR=/data/local/tmp/e2ehw
export BARRA_ATTN_META=$B/attn3.meta
export BARRA_ATTN_DIR=$B
[ -n "$BARRA_ATTN_LAYERS" ] && export BARRA_ATTN_LAYERS
[ "$1" = "off" ] && export BARRA_ATTN_OFF=1
timeout 420 $B/lib/llama-bench -m $M -ngl 99 -p 512 -n 0 -r 2 2>/data/local/tmp/battn_bench.err
echo "RC=$?"
grep MemAvailable /proc/meminfo
echo BENCHDONE
