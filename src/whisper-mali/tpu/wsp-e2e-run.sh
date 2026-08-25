#!/system/bin/sh
# M4-E2E: whisper-cli mit barra-TPU-Encoder (WHISPER_BARRA=1) auf der Test-WAV.
#   wsp-e2e-run.sh [modell]   (Default base)
T=/data/local/tmp; W=$T/wsptpu; P=$W/pkgs7; GG=$T/wspgpu; F=/sys/class/devfreq
DM=/data/local/ubuntu/opt/hwbridge/pf; DC=/data/local/ubuntu/opt/hwbridge/pfc
M=${1:-base}
mkdir -p $DC
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
echo performance > $F/1a000000.rio/governor 2>/dev/null
echo 1119000000 > $F/1a000000.rio/min_freq 2>/dev/null
echo performance > $F/17000010.devfreq_mif/governor 2>/dev/null
echo 890000 > /sys/class/misc/mali0/device/scaling_min_freq 2>/dev/null
echo 890000 > /sys/class/misc/mali0/device/hint_min_freq 2>/dev/null
pkill -x tpud_pipe4 2>/dev/null; sleep 0.6; rm -f $DM/tpu.sock $DC/tpu.sock
MODELS=""
for L in 0 1 2 3 4 5; do MODELS="$MODELS $P/l${L}_proj.package $P/l${L}_woc.package $P/l${L}_ffnresc.package"; done
TPU_CPU=8 TPU_ZC_BOUNCE=1 setsid $W/tpud_pipe4 $DM/tpu.sock $MODELS </dev/null >$W/tpud_e2m.tlog 2>&1 &
TPU_CPU=7 TPU_ZC_BOUNCE=1 setsid $W/tpud_pipe4 $DC/tpu.sock $W/wsp_core5_b0.package </dev/null >$W/tpud_e2c.tlog 2>&1 &
i=0; while [ $i -lt 120 ]; do grep -aq bereit $W/tpud_e2m.tlog && grep -aq bereit $W/tpud_e2c.tlog && break; sleep 1; i=$((i+1)); done
grep -aq bereit $W/tpud_e2m.tlog && grep -aq bereit $W/tpud_e2c.tlog || { echo TPUD_FAIL; exit 1; }
cd $GG
WHISPER_BARRA=1 WSP_PKG_DIR=$P WSP_SOCK_MAIN=$DM WSP_SOCK_CORE=$DC LD_LIBRARY_PATH=$GG \
  timeout 600 ./whisper-cli -m /data/local/ubuntu/root/whisper.cpp/models/ggml-$M.bin -f $GG/test-de.wav -l de -t 8 > $W/e2e-$M.log 2>&1
echo "RC=$?"
grep -aE "wsp-barra|^\[" $W/e2e-$M.log
grep -aE "total time|encode time" $W/e2e-$M.log
pkill -x tpud_pipe4 2>/dev/null
echo interactive > $F/17000010.devfreq_mif/governor 2>/dev/null
echo E2E_END
