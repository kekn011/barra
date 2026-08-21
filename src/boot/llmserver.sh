#!/system/bin/sh
# LLM-Server (llama-server, GPU-Vulkan mit Mali-Kerneln) — persistent, Modell/Kontext/Layer waehlbar.
#   llmserver.sh start [modell.gguf] [ctx] [ngl]   |  stop  |  status  |  log
# v2.4: TPU-Attention-Offload im Prefill (ggml-barra battn) — automatisch AN, wenn zum Modell ein
# Attention-Kit unter /data/local/barra-attn/<modellname> liegt (attn.meta + aux_attn.bin + *.package).
# Ohne Kit: unveraendert reiner GPU-Betrieb. BARRA_ATTN_LAYERS via Env uebersteuerbar (Default 1-35;
# L0 bleibt auf der GPU — Qwen3-4B ist auf L0-Attention-Stoerungen hypersensibel, gemessen 21.8.26).
. /data/adb/baseos/llm/env.sh
. /data/adb/baseos/bin/barra-i18n.sh
R=/data/adb/baseos/run; mkdir -p $R
MODELS=/data/local/ubuntu/home/barra/models
DEF=$(ls /data/local/ubuntu/home/*/models/*.gguf /data/local/ubuntu/home/*/*.gguf 2>/dev/null | head -1)
ATTN_BASE=/data/local/barra-attn
ASOCK=$R/attn-sock
BRIDGE=/data/local/ubuntu/opt/hwbridge

attn_stop() {
  pkill -f "$LLM/tpud-attn" 2>/dev/null
}
attn_start() {   # $1 = Modellpfad; setzt BARRA_* Env, wenn Kit vorhanden + tpud bereit
  MB=$(basename "$1" .gguf)
  A=$ATTN_BASE/$MB
  [ -f "$A/attn.meta" ] && [ -f "$A/aux_attn.bin" ] || return 1
  NL=$(head -1 "$A/attn.meta" | { read _ _ _ _ _ nl _ _; echo $nl; })
  [ -n "$NL" ] || return 1
  PKGS=""
  i=0
  while [ $i -lt $NL ]; do
    for ty in q1 q2 kv o; do
      [ -f "$A/${ty}_L$i.package" ] || { t llm.attn_pkg_missing "${ty}_L$i.package"; return 1; }
      PKGS="$PKGS $A/${ty}_L$i.package"
    done
    i=$((i+1))
  done
  ln -sf $LLM/battn/*.spv "$A/" 2>/dev/null
  mkdir -p $ASOCK; chmod 777 $ASOCK
  for s in gpuzc.sock gxp.sock gpu.sock; do ln -sf $BRIDGE/$s $ASOCK/$s; done
  if ! pgrep -f "$LLM/tpud-attn" >/dev/null; then
    rm -f $ASOCK/tpu.sock $R/tpud-attn.log
    TPU_CPU=8 TPU_WAKELOCK=1 TPU_FENCE=1 LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 \
      setsid $LLM/tpud-attn $ASOCK/tpu.sock $PKGS </dev/null >$R/tpud-attn.log 2>&1 &
    i=0; while [ $i -lt 300 ]; do grep -aq bereit $R/tpud-attn.log 2>/dev/null && break; sleep 1; i=$((i+1)); done
    grep -aq bereit $R/tpud-attn.log 2>/dev/null || { t llm.attn_tpud_fail; attn_stop; return 1; }
    chmod 666 $ASOCK/tpu.sock 2>/dev/null
  fi
  export BARRA_SOCK_DIR=$ASOCK
  export BARRA_ATTN_META=$A/attn.meta
  export BARRA_ATTN_DIR=$A
  export BARRA_ATTN_LAYERS=${BARRA_ATTN_LAYERS:-1-35}
  t llm.attn_active "$(grep -ac Modell $R/tpud-attn.log)" "$BARRA_ATTN_LAYERS"
  return 0
}

case "${1:-status}" in
  stop)   pkill -f "$LLM/llama-server"; attn_stop; sleep 1; t llm.stopped;;
  status) P=$(pgrep -f "$LLM/llama-server"|head -1); [ -n "$P" ] && { t llm.running "$P"; grep -a "model loaded\|loading model\|barra-attn" $R/llmserver.log|tail -3; } || t llm.off
          pgrep -f "$LLM/tpud-attn" >/dev/null && t llm.tpud_running || true;;
  log)    tail -50 $R/llmserver.log;;
  start)
    M="${2:-$DEF}"; CTX="${3:-4096}"; NGL="${4:-99}"
    pgrep -f "$LLM/llama-server" >/dev/null && { t llm.already_running; exit 0; }
    [ -f "$M" ] || { t llm.model_missing "$M" "$MODELS"; exit 1; }
    attn_start "$M" || t llm.attn_no_kit "$(basename "$M" .gguf)"
    setsid $LLM/llama-server -m "$M" -ngl $NGL --host 0.0.0.0 --port 8080 -c $CTX -t 4 </dev/null >$R/llmserver.log 2>&1 &
    t llm.started "$!" "$(basename $M)" "$CTX" "$NGL";;
  *) t llm.usage;;
esac
