#!/system/bin/sh
T=/data/local/tmp; W=$T/wsptpu; P=$W/turbo/pkgs; GG=$T/wspgpu; F=/sys/class/devfreq
DM=/data/local/ubuntu/opt/hwbridge/pf; DC=/data/local/ubuntu/opt/hwbridge/pfc
mkdir -p $DC
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
for CP in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > $CP/scaling_governor 2>/dev/null; done
echo performance > $F/1a000000.rio/governor 2>/dev/null
echo 1119000000 > $F/1a000000.rio/min_freq 2>/dev/null
echo performance > $F/17000010.devfreq_mif/governor 2>/dev/null
echo 890000 > /sys/class/misc/mali0/device/scaling_min_freq 2>/dev/null
echo 890000 > /sys/class/misc/mali0/device/hint_min_freq 2>/dev/null
pkill -x tpud_pipe4 2>/dev/null; sleep 0.6; rm -f $DM/tpu.sock $DC/tpu.sock
MODELS=""
for L in $(seq 0 31); do MODELS="$MODELS $P/l${L}_proj.package $P/l${L}_woc.package $P/l${L}_ffnresc.package"; done
for L in 0 1 2 3; do [ -f $P/l${L}_cross.package ] && MODELS="$MODELS $P/l${L}_cross.package"; done
[ -f $P/conv1.package ] && [ -f $P/conv2.package ] && MODELS="$MODELS $P/conv1.package $P/conv2.package"
TPU_CPU=8 setsid $W/tpud_pipe4 $DM/tpu.sock $MODELS </dev/null >$W/tpud_tm.tlog 2>&1 &
TPU_CPU=7 TPU_ZC_BOUNCE=${6:-1} setsid $W/tpud_pipe4 $DC/tpu.sock $W/turbo/wsp_core5t10_b0.package </dev/null >$W/tpud_tc.tlog 2>&1 &
i=0; while [ $i -lt 240 ]; do grep -aq bereit $W/tpud_tm.tlog && grep -aq bereit $W/tpud_tc.tlog && break; sleep 1; i=$((i+1)); done
grep -aq bereit $W/tpud_tm.tlog && grep -aq bereit $W/tpud_tc.tlog || { echo TPUD_FAIL; tail -3 $W/tpud_tm.tlog $W/tpud_tc.tlog; exit 1; }
grep MemAvailable /proc/meminfo
echo "RIO vor Lauf: gov=$(cat $F/1a000000.rio/governor 2>&1) min=$(cat $F/1a000000.rio/min_freq 2>&1) cur=$(cat $F/1a000000.rio/cur_freq 2>&1)"
cd $GG
EX=""; [ "$2" = "greedy" ] && EX="-bs 1 -bo 1"
MDL=/data/local/ubuntu/root/whisper.cpp/models/ggml-large-v3-turbo.bin
[ "$5" = "q5" ] && MDL=/data/local/ubuntu/root/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin
[ "$3" = "vkperf" ] && export GGML_VK_PERF_LOGGER=1
( i=0; while [ $i -lt 200 ]; do cat $F/1a000000.rio/cur_freq; sleep 0.3; i=$((i+1)); done > $W/riofreq.log 2>&1 ) &
RIOMON=$!
SY=0; [ "${6:-1}" = "2" ] && SY=1
WHISPER_BARRA=1 WSP_THREADS=${1:-4} WSP_PAIR=${4:-1} WSP_OVL=${7:-1} WSP_SYNC=$SY WSP_PKG_DIR=$P WSP_SOCK_MAIN=$DM WSP_SOCK_CORE=$DC LD_LIBRARY_PATH=$GG timeout 900 ./whisper-cli -m $MDL -f $GG/test-de.wav -l de -t 8 $EX > $W/e2e-turbo.log 2>&1
echo RC=$?
kill $RIOMON 2>/dev/null
echo "RIO-Takt-Histogramm im Lauf:"; sort $W/riofreq.log | uniq -c
grep -aE "wsp-barra|^\[" $W/e2e-turbo.log
grep -aE "total time|encode time" $W/e2e-turbo.log
pkill -x tpud_pipe4 2>/dev/null
for CP in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > $CP/scaling_governor 2>/dev/null; done
echo interactive > $F/17000010.devfreq_mif/governor 2>/dev/null
echo E2ET_END
