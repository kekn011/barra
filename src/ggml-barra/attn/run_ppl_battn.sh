#!/system/bin/sh
# Perplexity-Vergleich (N Chunks a 512) mit/ohne Attention-Offload. $1=chunks (Default 4), $2=off fuer Baseline
L=/data/adb/baseos/llm; B=/data/local/tmp/battn; M=/data/local/tmp/models/qwen3-4b.gguf
N=${1:-4}
G=/sys/class/misc/mali0/device; MAX=$(tr ' ' '\n' < $G/available_frequencies|grep -v '^$'|sort -n|tail -1)
echo $MAX > $G/scaling_min_freq 2>/dev/null; echo $MAX > $G/hint_min_freq 2>/dev/null
export LD_LIBRARY_PATH=$B/lib:$L:/system/lib64:/vendor/lib64
export BARRA_SOCK_DIR=/data/local/tmp/e2ehw
export BARRA_ATTN_META=$B/attn3.meta
export BARRA_ATTN_DIR=$B
[ -n "$BARRA_ATTN_LAYERS" ] && export BARRA_ATTN_LAYERS
[ "$2" = "off" ] && export BARRA_ATTN_OFF=1
timeout 900 $B/lib/llama-perplexity -m $M -ngl 99 -f /data/local/tmp/wiki.test.raw --chunks $N -c 512 -b 512 -ub 512 2>/data/local/tmp/battn_ppl.err </dev/null | grep -aE 'perplexity|ETA|Final|estimate' | tail -6
grep -aE 'barra-attn' /data/local/tmp/battn_ppl.err | head -4
grep -acE 'barra-attn.*ok' /data/local/tmp/battn_ppl.err
grep -aE 'perplexity|Final' /data/local/tmp/battn_ppl.err | tail -4
echo "RC=$?"
echo PPLDONE
