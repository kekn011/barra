#!/system/bin/sh
# v2.3b: q1/q2/k/v/o-Packages in attn4b4 kompilieren (fehlende), tpud (MAXMODELS 224) mit 180 Packages starten
setenforce 0 2>/dev/null
T=/data/local/tmp; A=$T/attn4b4
cp /data/adb/baseos/tpu/tpuc1 $T/tpuc1 2>/dev/null; cp /data/adb/baseos/tpu/libcomp_std.so $T/libcomp_std.so 2>/dev/null
chmod 755 $T/tpuc1
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
pkill -9 -f tpud_pipe 2>/dev/null; pkill -9 -x tpud 2>/dev/null; sleep 1
NL=36; ok=0; fail=0
i=0
while [ $i -lt $NL ]; do
  for base in q1_L$i q2_L$i k_L$i v_L$i o_L$i; do
    if [ -s $A/$base.package ]; then ok=$((ok+1)); else
      rm -f $T/out.package /data/vendor/edgetpu/cache/_0 /data/vendor/edgetpu/cache/_0_checksum 2>/dev/null
      COMPILER_SO=$T/libcomp_std.so MODEL=$A/$base.tflite timeout 300 $T/tpuc1 >/dev/null 2>&1
      if [ -s $T/out.package ]; then mv $T/out.package $A/$base.package; ok=$((ok+1)); else echo "FEHLER: $base"; fail=$((fail+1)); fi
    fi
  done
  echo "L$i ($ok ok, $fail Fehler)"
  i=$((i+1))
done
echo "COMPILE ok=$ok fail=$fail"
setenforce 1 2>/dev/null
[ $fail -gt 0 ] && { echo ABBRUCH; exit 1; }
H=/data/adb/hwbridge; E=/data/local/tmp/e2ehw; BASE=/data/local/ubuntu/opt/hwbridge
mount -t debugfs none /sys/kernel/debug 2>/dev/null
SUP=$(cat $H/supervisor.pid 2>/dev/null); [ -n "$SUP" ] && kill -STOP $SUP
mkdir -p $E; chmod 777 $E
for s in gpuzc.sock gxp.sock gpu.sock; do ln -sf $BASE/$s $E/$s; done
rm -f $E/tpu.sock $E/tpud.log
chmod 755 /data/local/tmp/tpud_pipe224
PKGS=""
i=0
while [ $i -lt 36 ]; do PKGS="$PKGS $A/q1_L$i.package $A/q2_L$i.package $A/k_L$i.package $A/v_L$i.package $A/o_L$i.package"; i=$((i+1)); done
TPU_CPU=8 TPU_WAKELOCK=1 TPU_FENCE=1 LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 nohup /data/local/tmp/tpud_pipe224 $E/tpu.sock $PKGS > $E/tpud.log 2>&1 &
i=0; while [ $i -lt 240 ]; do grep -aq bereit $E/tpud.log && break; sleep 1; i=$((i+1)); done
chmod 666 $E/tpu.sock 2>/dev/null
echo "tpud nach ${i}s: $(grep -ac Modell $E/tpud.log) Modelle"
grep MemAvailable /proc/meminfo
echo ALLDONE
