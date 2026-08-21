#!/system/bin/sh
# tpud_pipe mit 144 Split-Packages (q=4L, k=4L+1, v=4L+2, o=4L+3)
H=/data/adb/hwbridge; E=/data/local/tmp/e2ehw; BASE=/data/local/ubuntu/opt/hwbridge; A=/data/local/tmp/attn4b2
mount -t debugfs none /sys/kernel/debug 2>/dev/null
SUP=$(cat $H/supervisor.pid 2>/dev/null); [ -n "$SUP" ] && kill -STOP $SUP
pkill -9 -f tpud_pipe 2>/dev/null; pkill -9 -x tpud 2>/dev/null; sleep 1
mkdir -p $E; chmod 777 $E
for s in gpuzc.sock gxp.sock gpu.sock; do ln -sf $BASE/$s $E/$s; done
rm -f $E/tpu.sock $E/tpud.log
PKGS=""
i=0
while [ $i -lt 36 ]; do PKGS="$PKGS $A/q_L$i.package $A/k_L$i.package $A/v_L$i.package $A/o_L$i.package"; i=$((i+1)); done
TPU_CPU=8 TPU_WAKELOCK=1 TPU_FENCE=1 LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 nohup /data/local/tmp/tpud_pipe $E/tpu.sock $PKGS > $E/tpud.log 2>&1 &
i=0; while [ $i -lt 180 ]; do grep -aq bereit $E/tpud.log && break; sleep 1; i=$((i+1)); done
chmod 666 $E/tpu.sock 2>/dev/null
echo "tpud nach ${i}s: $(grep -ac Modell $E/tpud.log) Modelle"
tail -2 $E/tpud.log
grep MemAvailable /proc/meminfo
