#!/system/bin/sh
# E2E 4-Step seed 42 mit sd-dbg; $1 = Ausgabename, weitere Envs vorher exportieren
cd /data/local/tmp
M=/data/local/barra-img
P="a red fox sitting in a snowy forest, winter morning light, detailed"
export GGML_VK_DISABLE_COOPMAT=1
export GGML_VK_MALI_GEMM=1
export GGML_VK_TILE_L=128,64,64,16,16,32,2,4,4,1,16
OUT=${1:-img-f16acc.png}
cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null
./sd-dbg -m $M/DreamShaper8_LCM.safetensors --taesd $M/taesd.safetensors -p "$P" --sampling-method lcm --steps 4 --cfg-scale 1.0 -W 512 -H 512 -s 42 --diffusion-fa -o /data/local/tmp/$OUT 2>&1 | grep -E "s/it|completed in|nan|NaN"
wc -c /data/local/tmp/$OUT
