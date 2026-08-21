#!/system/bin/sh
# Dump-Lauf: 1 PPL-Chunk, nur Layer $1 (Default 0) offloaded, alle Zwischenstufen nach battn/dump_*
B=/data/local/tmp/battn; M=/data/local/tmp/models/qwen3-4b.gguf
DL=${1:-0}
export LD_LIBRARY_PATH=$B/lib:/data/adb/baseos/llm:/system/lib64:/vendor/lib64
export BARRA_SOCK_DIR=/data/local/tmp/e2ehw
export BARRA_ATTN_META=$B/attn2.meta
export BARRA_ATTN_DIR=$B
export BARRA_ATTN_LAYERS=$DL
export BARRA_ATTN_DUMP=$B
export BARRA_ATTN_DUMP_L=$DL
timeout 600 $B/lib/llama-perplexity -m $M -ngl 99 -f /data/local/tmp/wiki.test.raw --chunks 1 -c 512 -b 512 -ub 512 2>/data/local/tmp/battn_dmp.err </dev/null >/dev/null
grep -a 'DUMP\|barra-attn' /data/local/tmp/battn_dmp.err | tail -3
chown shell:shell $B/dump_* 2>/dev/null
chcon u:object_r:shell_data_file:s0 $B/dump_* 2>/dev/null
ls $B/dump_* | wc -l
echo DUMPDONE
