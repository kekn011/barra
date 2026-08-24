#!/system/bin/sh
cd /data/local/tmp
M=/data/local/barra-img
export GGML_VK_DISABLE_COOPMAT=1
export GGML_VK_MALI_GEMM=1
export GGML_VK_TILE_L=128,64,64,16,16,32,2,4,4,1,16
export BARRA_TIME=1
./sd-dbg -m $M/DreamShaper8_LCM.safetensors --taesd $M/taesd.safetensors -p "fox" --sampling-method lcm --steps 3 --cfg-scale 1.0 -W 512 -H 512 -s 42 --diffusion-fa -o /data/local/tmp/imgtime.png 2>&1 | grep -E "barra-(sd|vk)|s/it|completed in" | tr -d ''
