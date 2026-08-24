#!/system/bin/sh
cd /data/local/tmp
M=/data/local/barra-img
export GGML_VK_DISABLE_COOPMAT=1
export GGML_VK_MALI_GEMM=1
export GGML_VK_TILE_L=128,64,64,16,16,32,2,4,4,1,16
export GGML_VK_BARRA_DBG=1
./sd-dbg -m $M/DreamShaper8_LCM.safetensors --taesd $M/taesd.safetensors -p "fox" --sampling-method lcm --steps 1 --cfg-scale 1.0 -W 512 -H 512 -s 42 --diffusion-fa -o /data/local/tmp/imgdbg.png > /data/local/tmp/dbg.log 2>&1
echo RC=$?
