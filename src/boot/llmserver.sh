#!/system/bin/sh
# LLM-Server (llama-server, GPU-Vulkan mit Mali-Kerneln) — persistent, Modell/Kontext/Layer waehlbar.
#   llmserver.sh start [modell.gguf] [ctx] [ngl]   |  stop  |  status  |  log
# TPU-Attention-Offload im Prefill (ggml-barra) — automatisch AN, wenn zum Modell ein Attention-
# Kit unter /data/local/barra-attn/<modellname> liegt (attn.meta + aux_attn.bin + *.package).
# Kit-Formate: v7/v8-Kits tragen pkglist.txt (exakte tpud-Reihenfolge) + optional layers.txt
# (Betriebspunkt, z.B. "1-39" bei GLM); Altbestand (v2.4-Qwen, q1/q2/kv/o je Layer) laeuft
# weiter ueber den Fallback mit Layers-Default 1-35 (L0-Hypersensibilitaet, gemessen 21.8.26).
# Ohne Kit: unveraendert reiner GPU-Betrieb. Waehrend des Offload-Betriebs sind CPU/TPU/MIF/GPU
# gepinnt (ohne Pins ist der Offload LANGSAMER als GPU-only, gemessen 23.8.26); LLM_NOPIN=1
# laesst die Pins aus, stop stellt die Governors zurueck.
. /data/adb/baseos/llm/env.sh
. /data/adb/baseos/bin/barra-i18n.sh
# Koexistenz-Waechter (gemeinsam fuer alle Kit-Dienste, src/boot/barra-guard.sh). Fehlt er auf
# einem aelteren Base, laufen die Aufrufe ins Leere statt ins Messer.
G=/data/adb/baseos/bin/barra-guard.sh
if [ -f "$G" ]; then . "$G"; else guard_check(){ :; }; guard_need(){ echo 0; }; guard_expendable(){ :; }; fi
R=/data/adb/baseos/run; mkdir -p $R
MODELS=/data/local/ubuntu/home/barra/models
DEF=$(ls /data/local/ubuntu/home/*/models/*.gguf /data/local/ubuntu/home/*/*.gguf 2>/dev/null | head -1)
ATTN_BASE=/data/local/barra-attn
ASOCK=$R/attn-sock
BRIDGE=/data/local/ubuntu/opt/hwbridge
F=/sys/class/devfreq

pins_on(){
  [ "$LLM_NOPIN" = "1" ] && return
  for CP in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > $CP/scaling_governor 2>/dev/null; done
  echo performance > $F/1a000000.rio/governor 2>/dev/null
  echo 1119000000 > $F/1a000000.rio/min_freq 2>/dev/null
  echo performance > $F/17000010.devfreq_mif/governor 2>/dev/null
  G=/sys/class/misc/mali0/device
  echo 890000 > $G/scaling_min_freq 2>/dev/null; echo 890000 > $G/hint_min_freq 2>/dev/null
}
pins_off(){
  for CP in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > $CP/scaling_governor 2>/dev/null; done
  echo simple_ondemand > $F/1a000000.rio/governor 2>/dev/null
  echo 0 > $F/1a000000.rio/min_freq 2>/dev/null
  echo interactive > $F/17000010.devfreq_mif/governor 2>/dev/null
  G=/sys/class/misc/mali0/device
  echo 150000 > $G/scaling_min_freq 2>/dev/null; echo 150000 > $G/hint_min_freq 2>/dev/null
}
attn_stop() {
  pkill -f "$LLM/tpud-attn" 2>/dev/null
  pins_off
}
attn_start() {   # $1 = Modellpfad; setzt BARRA_* Env, wenn Kit vorhanden + tpud bereit
  MB=$(basename "$1" .gguf)
  A=$ATTN_BASE/$MB
  [ -f "$A/attn.meta" ] && [ -f "$A/aux_attn.bin" ] || return 1
  PKGS=""
  if [ -f "$A/pkglist.txt" ]; then
    # v7/v8-Kit: exakte Paketreihenfolge aus der Liste (variable Paketzahl je Layer)
    while read -r base; do
      [ -n "$base" ] || continue
      [ -f "$A/$base.package" ] || { t llm.attn_pkg_missing "$base.package"; return 1; }
      PKGS="$PKGS $A/$base.package"
    done < "$A/pkglist.txt"
  else
    # Altbestand v2.4 (Qwen3-4B): festes q1/q2/kv/o-Schema
    NL=$(head -1 "$A/attn.meta" | { read _ _ _ _ _ nl _ _; echo $nl; })
    [ -n "$NL" ] || return 1
    i=0
    while [ $i -lt $NL ]; do
      for ty in q1 q2 kv o; do
        [ -f "$A/${ty}_L$i.package" ] || { t llm.attn_pkg_missing "${ty}_L$i.package"; return 1; }
        PKGS="$PKGS $A/${ty}_L$i.package"
      done
      i=$((i+1))
    done
    export BARRA_ATTN_LAYERS=${BARRA_ATTN_LAYERS:-1-35}
  fi
  [ -f "$A/layers.txt" ] && export BARRA_ATTN_LAYERS=${BARRA_ATTN_LAYERS:-$(head -1 "$A/layers.txt")}
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
  pins_on
  export BARRA_SOCK_DIR=$ASOCK
  export BARRA_ATTN_META=$A/attn.meta
  export BARRA_ATTN_DIR=$A
  export BARRA_ATTN_THREADS=${BARRA_ATTN_THREADS:-6}
  t llm.attn_active "$(grep -ac Modell $R/tpud-attn.log)" "${BARRA_ATTN_LAYERS:-alle}"
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
    # Waechter (29.8.): Sachverbot llm<->stt (TPU-Graph-Limit 145+104 > ~157) UND Speicherpruefung.
    # Aufschlag 1400 MB ueber der Modellgroesse (gemessen: 4031 MB bei 2654 MB Modell).
    guard_check llm "$(guard_need "$M" 1400)" || exit 1
    attn_start "$M" || t llm.attn_no_kit "$(basename "$M" .gguf)"
    setsid $LLM/llama-server -m "$M" -ngl $NGL --host 0.0.0.0 --port 8080 -c $CTX -t 4 </dev/null >$R/llmserver.log 2>&1 &
    P=$!
    # Ohne das erbt der Server oom_score_adj -1000 (su/adbd) und ist fuer den Kernel UNKILLBAR:
    # bei Speichermangel paniert dann der Kernel, statt den Server zu beenden (29.8. belegt).
    guard_expendable "$P"
    t llm.started "$P" "$(basename $M)" "$CTX" "$NGL";;
  *) t llm.usage;;
esac
