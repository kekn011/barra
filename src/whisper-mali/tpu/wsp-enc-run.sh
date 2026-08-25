#!/system/bin/sh
# Kompletter base-Encoder (6 Layer) auf dem Node: 2 tpud-Instanzen (main: 18 Packages,
# core: das geteilte Kern-Package), beide mit TPU_ZC_BOUNCE (LUT-dmabuf-Befund 21.8.).
#   wsp-enc-run.sh [iters]
T=/data/local/tmp; W=$T/wsptpu; P=$W/pkgs7; F=/sys/class/devfreq
DM=/data/local/ubuntu/opt/hwbridge/pf; DC=/data/local/ubuntu/opt/hwbridge/pfc
mkdir -p $DC
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
echo performance > $F/1a000000.rio/governor 2>/dev/null
echo 1119000000 > $F/1a000000.rio/min_freq 2>/dev/null
echo performance > $F/17000010.devfreq_mif/governor 2>/dev/null
pkill -x tpud_pipe4 2>/dev/null; sleep 0.6; rm -f $DM/tpu.sock $DC/tpu.sock
MODELS=""
for L in 0 1 2 3 4 5; do MODELS="$MODELS $P/l${L}_proj.package $P/l${L}_woc.package $P/l${L}_ffnresc.package"; done
TPU_CPU=8 TPU_ZC_BOUNCE=1 setsid $W/tpud_pipe4 $DM/tpu.sock $MODELS </dev/null >$W/tpud_em.tlog 2>&1 &
TPU_CPU=7 TPU_ZC_BOUNCE=1 setsid $W/tpud_pipe4 $DC/tpu.sock $W/wsp_core5_b0.package </dev/null >$W/tpud_ec.tlog 2>&1 &
i=0; while [ $i -lt 120 ]; do grep -aq bereit $W/tpud_em.tlog && grep -aq bereit $W/tpud_ec.tlog && break; sleep 1; i=$((i+1)); done
if grep -aq bereit $W/tpud_em.tlog && grep -aq bereit $W/tpud_ec.tlog; then
  WSP_SOCK_MAIN=$DM WSP_SOCK_CORE=$DC timeout 600 $W/wsp_enc $P/enc_x0.f32 $P/enc_out_ref.f32 $P/enc_params.txt $P/enc_ln.f32 ${1:-3}
else
  echo TPUD_FAIL; tail -3 $W/tpud_em.tlog $W/tpud_ec.tlog
fi
pkill -x tpud_pipe4 2>/dev/null
echo interactive > $F/17000010.devfreq_mif/governor 2>/dev/null
