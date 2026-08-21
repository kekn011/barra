#!/system/bin/sh
# CHECK-Lauf: 1 PPL-Chunk, alle Layer offloaded, je Layer cos unser-Block vs Vulkan-Referenz.
B=/data/local/tmp/battn; M=/data/local/tmp/models/qwen3-4b.gguf
export LD_LIBRARY_PATH=$B/lib:/data/adb/baseos/llm:/system/lib64:/vendor/lib64
export BARRA_SOCK_DIR=/data/local/tmp/e2ehw
export BARRA_ATTN_META=$B/attn2.meta
export BARRA_ATTN_DIR=$B
export BARRA_ATTN_CHECK=1
export BARRA_ATTN_LAYERS=${BARRA_ATTN_LAYERS:-0-35}
timeout 600 $B/lib/llama-perplexity -m $M -ngl 99 -f /data/local/tmp/wiki.test.raw --chunks 1 -c 512 -b 512 -ub 512 2>/data/local/tmp/battn_chk.err </dev/null >/dev/null
grep -a 'CHECK' /data/local/tmp/battn_chk.err
echo CHECKDONE
