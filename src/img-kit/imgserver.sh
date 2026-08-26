#!/system/bin/sh
# barra Bild-Dienst (stable-diffusion.cpp sd-server, Vulkan mit Mali-Kerneln: eigener GEMM + Flash-Attention).
# Wie llm/stt/pya/tts: manueller Start, KEIN Boot-Autostart. Modell bleibt geladen (1,9 GB RAM),
# 512x512 LCM 4 Schritte ~17 s je Bild (erstes Bild nach Start ~20 s: Shader-Compile).
#   su -c 'sh /data/adb/baseos/bin/imgserver.sh start [modell]'   (start | stop | status | log)
# HTTP: POST http://<node>:8096/sdapi/v1/txt2img  {"prompt":"...","seed":-1,"width":512,"height":512}
#       -> {"images":["<base64 png>"]}   (auch OpenAI-Route /v1/images/generations)
# Container-CLI: barra-img "prompt" [-o datei.png] [-s seed] [-W breite] [-H hoehe] [-n schritte]
K=/data/local/barra-img
. /data/adb/baseos/bin/barra-i18n.sh
R=/data/adb/baseos/run; mkdir -p $R
PORT=8096
MODELS=$K/models
DEF=$(ls $MODELS/*.safetensors $MODELS/*.gguf 2>/dev/null | grep -v taesd | head -1)

pins_on(){   # Mali-Mindesttakt anheben, solange der Dienst laeuft (wie tts)
  [ "$IMG_NOPIN" = "1" ] && return
  echo 890000 > /sys/class/misc/mali0/device/scaling_min_freq 2>/dev/null
  echo 890000 > /sys/class/misc/mali0/device/hint_min_freq 2>/dev/null
}
pins_off(){   # 'echo 0' ist auf diesem mali-devfreq ein No-op -> konkreten HW-Idle-Wert, BEIDE Regler
  echo 150000 > /sys/class/misc/mali0/device/scaling_min_freq 2>/dev/null
  echo 150000 > /sys/class/misc/mali0/device/hint_min_freq 2>/dev/null
}

case "${1:-status}" in
  stop)   pkill -f "$K/bin/sd-server"; pins_off; sleep 1; t img.stopped;;
  status) P=$(pgrep -f "$K/bin/sd-server"|head -1)
          if [ -n "$P" ]; then t img.running "$P" "$PORT"; grep -a "listening\|params memory" $R/imgserver.log | tail -2; else t img.off; fi;;
  log)    tail -50 $R/imgserver.log;;
  start)
    M="${2:-$DEF}"
    pgrep -f "$K/bin/sd-server" >/dev/null && { t img.already_running; exit 0; }
    [ -f "$M" ] || { t img.model_missing "$M" "$MODELS"; exit 1; }
    [ -f "$MODELS/taesd.safetensors" ] || { t img.taesd_missing "$MODELS"; exit 1; }
    pins_on
    # Mali-Betriebspunkt (24.8.26): KHR_coopmat aus (Emulationsbremse), eigener GEMM + FA, getunte Stock-Tiles
    export GGML_VK_DISABLE_COOPMAT=1
    export GGML_VK_MALI_GEMM=1
    export GGML_VK_TILE_L=128,64,64,16,16,32,2,4,4,1,16
    export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
    setsid $K/bin/sd-server -m "$M" --taesd $MODELS/taesd.safetensors --diffusion-fa \
      --sampling-method lcm --steps 4 --cfg-scale 1.0 -W 512 -H 512 \
      --listen-ip 0.0.0.0 --listen-port $PORT </dev/null >$R/imgserver.log 2>&1 &
    P=$!
    i=0; while [ $i -lt 90 ]; do grep -aq "listening on" $R/imgserver.log 2>/dev/null && break; kill -0 $P 2>/dev/null || break; sleep 1; i=$((i+1)); done
    if grep -aq "listening on" $R/imgserver.log 2>/dev/null; then
      t img.started "$P" "$(basename "$M")" "$PORT"
      # Hebel 5 (Lazy-Compile-Warmup): der erste txt2img zahlt ~5s Mali-Shader-Compile. Jetzt beim
      # Start abfangen statt beim Nutzer. Minimaler 1-Step-Gen (kompiliert dieselben Shader), via
      # toybox nc (Android-seitig kein curl), best-effort mit Timeout.
      WB='{"prompt":"warmup","seed":1,"width":512,"height":512,"steps":1}'
      printf 'POST /sdapi/v1/txt2img HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' "${#WB}" "$WB" | timeout 60 toybox nc localhost $PORT >/dev/null 2>&1
      t img.warm
    else
      t img.start_fail; tail -5 $R/imgserver.log; pins_off; exit 1
    fi;;
  *) t img.usage;;
esac
