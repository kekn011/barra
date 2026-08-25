#!/system/bin/sh
# Layer-Kette auf dem Node mit ZWEI tpud-Instanzen (Kern allein — Softmax-Package liefert
# Nullen neben anderen Graphen im selben tpud, Befund 21.8.).
#   wsp-layer-run.sh [iters]
T=/data/local/tmp; W=$T/wsptpu; P=$W/pkgs; F=/sys/class/devfreq
DM=/data/local/ubuntu/opt/hwbridge/pf; DC=/data/local/ubuntu/opt/hwbridge/pfc
mkdir -p $DC
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
echo performance > $F/1a000000.rio/governor 2>/dev/null
echo 1119000000 > $F/1a000000.rio/min_freq 2>/dev/null
echo performance > $F/17000010.devfreq_mif/governor 2>/dev/null
pkill -f "tpud_pipe4 $DM/tpu.sock" 2>/dev/null; pkill -f "tpud_pipe4 $DC/tpu.sock" 2>/dev/null
sleep 0.6; rm -f $DM/tpu.sock $DC/tpu.sock
TPU_CPU=8 setsid $W/tpud_pipe4 $DM/tpu.sock $P/l0_proj.package $P/l0_woc.package $P/l0_ffnresc.package </dev/null >$W/tpud_m.tlog 2>&1 &
TPU_CPU=7 setsid $W/tpud_pipe4 $DC/tpu.sock $W/wsp_core5_b0.package </dev/null >$W/tpud_c.tlog 2>&1 &
i=0; while [ $i -lt 90 ]; do grep -aq bereit $W/tpud_m.tlog && grep -aq bereit $W/tpud_c.tlog && break; sleep 1; i=$((i+1)); done
if grep -aq bereit $W/tpud_m.tlog && grep -aq bereit $W/tpud_c.tlog; then
  WSP_SOCK_MAIN=$DM WSP_SOCK_CORE=$DC timeout 300 $W/wsp_layer $P/l0_x0.in.bin $P/l0_out_ref.f32 $P/l0_params.txt ${1:-3}
else
  echo TPUD_FAIL; tail -3 $W/tpud_m.tlog $W/tpud_c.tlog
fi
pkill -f "tpud_pipe4 $DM/tpu.sock" 2>/dev/null; pkill -f "tpud_pipe4 $DC/tpu.sock" 2>/dev/null
echo interactive > $F/17000010.devfreq_mif/governor 2>/dev/null
