#!/system/bin/sh
# 1-Step Perf-Logger, gibt FA-Zeilen + s/it aus; Envs GGML_VK_FA_TUNE / GGML_VK_FA_F16ACC vorher exportieren
cd /data/local/tmp
M=/data/local/barra-img
export GGML_VK_DISABLE_COOPMAT=1
export GGML_VK_MALI_GEMM=1
export GGML_VK_TILE_L=128,64,64,16,16,32,2,4,4,1,16
GGML_VK_PERF_LOGGER=1 ./sd-dbg -m $M/DreamShaper8_LCM.safetensors --taesd $M/taesd.safetensors -p "fox" --sampling-method lcm --steps 1 --cfg-scale 1.0 -W 512 -H 512 -s 42 --diffusion-fa -o /data/local/tmp/imgfa.png > /data/local/tmp/vkfa.log 2>&1
grep -E "barra: FA|FLASH_ATTN" /data/local/tmp/vkfa.log | sed -E 's/dst\([^)]*\), *q\(([^)]*)\), *k\(([^)]*)\), *v\([^)]*\), *m\([^)]*\)/q(\1) k(\2)/'
grep -E "s/it" /data/local/tmp/vkfa.log | tr -d '\r' | tail -c 12
