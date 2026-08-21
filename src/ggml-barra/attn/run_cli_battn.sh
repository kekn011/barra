#!/system/bin/sh
# Erster E2E-Test ggml-barra v2: llama-cli mit Attention-Offload, kurzer Prompt (Prefill>=min_batch), Text pruefen.
L=/data/adb/baseos/llm; B=/data/local/tmp/battn; M=/data/local/tmp/models/qwen3-4b.gguf
G=/sys/class/misc/mali0/device; MAX=$(tr ' ' '\n' < $G/available_frequencies|grep -v '^$'|sort -n|tail -1)
echo $MAX > $G/scaling_min_freq 2>/dev/null; echo $MAX > $G/hint_min_freq 2>/dev/null
export LD_LIBRARY_PATH=$B/lib:$L:/system/lib64:/vendor/lib64
export BARRA_SOCK_DIR=/data/local/tmp/e2ehw
export BARRA_ATTN_META=$B/attn.meta
export BARRA_ATTN_DIR=$B
export BARRA_ATTN_LOG=${BARRA_ATTN_LOG:-1}
[ -n "$BARRA_ATTN_LAYERS" ] && export BARRA_ATTN_LAYERS
[ "$1" = "off" ] && export BARRA_ATTN_OFF=1
PROMPT="Die Hauptstadt von Frankreich ist Paris. Die Hauptstadt von Italien ist Rom. Die Hauptstadt von Spanien ist Madrid. Die Hauptstadt von Deutschland ist"
timeout 300 $L/llama-cli -m $M -ngl 99 -c 1024 -n 24 --temp 0 -no-cnv -p "$PROMPT" </dev/null 2>/data/local/tmp/battn_cli.err
RC=$?
echo ""
echo "RC=$RC"
grep -aE 'barra-attn' /data/local/tmp/battn_cli.err | head -12
grep -aE 'barra-attn' /data/local/tmp/battn_cli.err | tail -4
grep MemAvailable /proc/meminfo
ps -A | grep -a llama | grep -v grep
echo CLIDONE
