#!/system/bin/sh
# Blocker-Diagnose: rechnet tpud serverseitig Nullen oder verliert die Uebergabe?
T=/data/local/tmp; W=$T/wsptpu; DSOCK=/data/local/ubuntu/opt/hwbridge/pf; F=/sys/class/devfreq
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
echo performance > $F/1a000000.rio/governor 2>/dev/null
echo 1119000000 > $F/1a000000.rio/min_freq 2>/dev/null
pkill -x tpud_pipe4 2>/dev/null; sleep 0.6; rm -f $DSOCK/tpu.sock
TPU_CPU=8 TPU_SERVE_DBG=1 TPU_WARMUP=1 TPU_WARMUP_IN=$W/wsp_core5_b0.in.bin \
  setsid $W/tpud_pipe4 $DSOCK/tpu.sock $W/wsp_core5_b0.package </dev/null >$W/tpud_dbg.tlog 2>&1 &
i=0; while [ $i -lt 90 ]; do grep -aq bereit $W/tpud_dbg.tlog && break; sleep 1; i=$((i+1)); done
grep -aq bereit $W/tpud_dbg.tlog || { echo NOBEREIT; tail -5 $W/tpud_dbg.tlog; exit 1; }
DUMP_OUT=1 BARRA_SOCK_DIR=$DSOCK timeout 120 $W/barra_tpu_bench 0 3456000 390000 2 2>&1 | grep -aE "OUT|model"
sleep 0.5
echo "--- tpud-Log:"
grep -aE "warmup|serve M" $W/tpud_dbg.tlog
pkill -x tpud_pipe4 2>/dev/null
echo DBG_END
