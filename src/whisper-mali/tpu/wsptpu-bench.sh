#!/system/bin/sh
# TPU-Bench eines Whisper-Stufe-3-Packages (Muster = chain/bsweep2_dev.sh).
#   wsptpu-bench.sh <package> <in_bytes> <out_bytes> [iters]
T=/data/local/tmp; W=$T/wsptpu; D=/data/local/ubuntu/opt/hwbridge/pf; F=/sys/class/devfreq
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
PKG=$1; IB=$2; OB=$3; N=${4:-20}
R=$W/bench-$(basename "$PKG" .package).txt; : > $R
echo "uid=$(id -u)" >> $R
echo performance > $F/1a000000.rio/governor 2>/dev/null
echo 1119000000 > $F/1a000000.rio/min_freq 2>/dev/null
echo performance > $F/17000010.devfreq_mif/governor 2>/dev/null
pkill -f "tpud_pipe4 $D/tpu.sock" 2>/dev/null; sleep 0.6; rm -f $D/tpu.sock
TPU_CPU=8 setsid $W/tpud_pipe4 $D/tpu.sock "$PKG" </dev/null >$W/tpud.tlog 2>&1 &
i=0; while [ $i -lt 90 ]; do grep -aqE 'bereit|FAIL' $W/tpud.tlog && break; sleep 1; i=$((i+1)); done
if grep -aq bereit $W/tpud.tlog; then
  DUMP_OUT=1 BARRA_SOCK_DIR=$D timeout 300 $W/barra_tpu_bench 0 $IB $OB 4 >/dev/null 2>&1
  DUMP_OUT=1 BARRA_SOCK_DIR=$D timeout 600 $W/barra_tpu_bench 0 $IB $OB $N >> $R 2>&1
else
  echo TPUD_FAIL >> $R; tail -5 $W/tpud.tlog >> $R
fi
pkill -f "tpud_pipe4 $D/tpu.sock" 2>/dev/null
echo interactive > $F/17000010.devfreq_mif/governor 2>/dev/null
echo WSPTPU_BENCH_END >> $R; cat $R
