#!/system/bin/sh
cd /data/local/tmp
M=/data/local/barra-img
export GGML_VK_DISABLE_COOPMAT=1
export GGML_VK_MALI_GEMM=1
export GGML_VK_TILE_L=128,64,64,16,16,32,2,4,4,1,16
GGML_VK_PERF_LOGGER=1 ./sd-dbg -m $M/DreamShaper8_LCM.safetensors --taesd $M/taesd.safetensors -p "fox" --sampling-method lcm --steps 1 --cfg-scale 1.0 -W 512 -H 512 -s 42 --diffusion-fa -o /data/local/tmp/imgpmx.png > /data/local/tmp/vkperfmx.log 2>&1
grep -E "MUL_MAT" /data/local/tmp/vkperfmx.log | grep -E "m=320 n=4096 k=1280|m=1280 n=256 k=5120|m=640 n=1024 k=2560|m=320 n=4096 k=320 "
grep -E "^Total time" /data/local/tmp/vkperfmx.log | tail -3
