#!/system/bin/sh
cd /data/local/tmp
M=/data/local/barra-img
export GGML_VK_DISABLE_COOPMAT=1
export GGML_VK_MALI_GEMM=1
export GGML_VK_TILE_L=128,64,64,16,16,32,2,4,4,1,16
GGML_VK_PERF_LOGGER=1 ./sd -m $M/DreamShaper8_LCM.safetensors --taesd $M/taesd.safetensors -p "fox" --sampling-method lcm --steps 2 --cfg-scale 1.0 -W 512 -H 512 -s 42 --diffusion-fa -o /data/local/tmp/imgpall.png > /data/local/tmp/vkperfall.log 2>&1
grep -E "s/it|completed in" /data/local/tmp/vkperfall.log
