#!/system/bin/sh
# LLM-Server (llama-server, GPU-Vulkan mit Mali-Kerneln) — persistent, Modell/Kontext/Layer waehlbar.
#   llmserver.sh start [modell.gguf] [ctx] [ngl]   |  stop  |  status  |  log
. /data/adb/baseos/llm/env.sh
R=/data/adb/baseos/run; mkdir -p $R
MODELS=/data/local/ubuntu/home/barra/models
DEF=$(ls /data/local/ubuntu/home/*/models/*.gguf /data/local/ubuntu/home/*/*.gguf 2>/dev/null | head -1)
case "${1:-status}" in
  stop)   pkill -f "$LLM/llama-server"; sleep 1; echo "gestoppt";;
  status) P=$(pgrep -f "$LLM/llama-server"|head -1); [ -n "$P" ] && { echo "laeuft (PID $P) :8080"; grep -a "model loaded\|loading model" $R/llmserver.log|tail -2; } || echo "aus";;
  log)    tail -50 $R/llmserver.log;;
  start)
    M="${2:-$DEF}"; CTX="${3:-4096}"; NGL="${4:-99}"
    pgrep -f "$LLM/llama-server" >/dev/null && { echo "laeuft schon (erst stop)"; exit 0; }
    [ -f "$M" ] || { echo "Modell fehlt: $M  (GGUFs nach $MODELS legen)"; exit 1; }
    setsid $LLM/llama-server -m "$M" -ngl $NGL --host 0.0.0.0 --port 8080 -c $CTX -t 4 </dev/null >$R/llmserver.log 2>&1 &
    echo "gestartet (PID $!): $(basename $M) ctx=$CTX ngl=$NGL -> http://<node>:8080";;
  *) echo "llmserver.sh start [modell] [ctx] [ngl] | stop | status | log";;
esac
