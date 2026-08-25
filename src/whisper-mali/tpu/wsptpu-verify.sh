#!/system/bin/sh
# Genauigkeits-Verify eines Whisper-Packages via tpud-pipe_bench (echter Input + numpy-Referenz).
#   wsptpu-verify.sh <package> <in.bin> <ref.f32> <osc> <ozp> [odt]
T=/data/local/tmp; W=$T/wsptpu; D=/data/local/ubuntu/opt/hwbridge/pf; F=/sys/class/devfreq
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
echo performance > $F/1a000000.rio/governor 2>/dev/null
echo 1119000000 > $F/1a000000.rio/min_freq 2>/dev/null
pkill -f "tpud_pipe4 $D/tpu.sock" 2>/dev/null; sleep 0.6; rm -f $D/tpu.sock
TPU_CPU=8 TPU_PIPE=2 TPU_PIPE_N=16 TPU_PIPE_IN="$2" TPU_PIPE_REF="$3" \
  TPU_PIPE_ODT=${6:-1} TPU_PIPE_OSCALE="$4" TPU_PIPE_OZP="$5" \
  timeout 240 $W/tpud_pipe4 $D/tpu.sock "$1" 2>&1 | grep -aE "pipe|fence|FAIL|bereit"
pkill -f "tpud_pipe4 $D/tpu.sock" 2>/dev/null
echo WSPTPU_VERIFY_END
