#!/system/bin/sh
# ============================================================================
# Supervisor der Hardware-Bruecken: TPU (tpud) + GPU (gpud) + DSP (gxpd).
# Startet sie und zieht sie bei Absturz neu hoch. Sockets liegen im GETEILTEN
# Rootfs-Pfad, damit der Ubuntu-Container sie unter /opt/hwbridge/ sieht.
# Die Daemons laufen im Android/Bionic-Kontext (dort sind libedgetpu_util /
# libvulkan / libgxp erreichbar) und sprechen /dev/edgetpu-soc bzw. /dev/mali0
# bzw. /dev/gxp direkt an -> unabhaengig von Panel/WLAN/Framework.
#   manuell:  su -c 'sh /data/adb/hwbridge/hw-bridges.sh'
# ============================================================================
H=/data/adb/hwbridge
SHARED=/data/local/ubuntu/opt/hwbridge
LOG=$H/hwbridge.log
PIDF=$H/supervisor.pid
MODEL=$H/test.package                 # Default-TPU-Modell (spaeter via Config/App wechselbar)

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
[ "$(wc -c < "$LOG" 2>/dev/null || echo 0)" -gt 1048576 ] && : > "$LOG"

# Singleton: laeuft schon ein Supervisor, hier raus.
OLD=$(cat "$PIDF" 2>/dev/null)
if [ -n "$OLD" ] && grep -qa hw-bridges "/proc/$OLD/cmdline" 2>/dev/null; then
  log "Supervisor laeuft bereits (pid $OLD) - exit"; exit 0
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

mkdir -p "$SHARED" 2>/dev/null; chmod 755 "$SHARED" 2>/dev/null

start_tpu(){
  pgrep -f "$H/tpud" >/dev/null 2>&1 && return 0
  [ -f "$MODEL" ] || { log "tpud: Modell $MODEL fehlt - uebersprungen"; return 1; }
  log "starte tpud (Multi-Modell, Socket $SHARED/tpu.sock, Modell $MODEL)"
  # LD_LIBRARY_PATH noetig: ein /data-Binary bekommt einen Bionic-Namespace ohne
  # /vendor/lib64-Zugriff; ohne das scheitert dlopen(libedgetpu_util.so).
  # Multi-Modell-Arg-Reihenfolge: tpud <socket> <modell0> [modell1 ...] (Groessen auto-erkannt)
  LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 "$H/tpud" "$SHARED/tpu.sock" "$MODEL" >>"$LOG" 2>&1 &
}
start_btnd(){   # Seitentasten-Daemon (Power/Volume -> konfigurierbare Aktionen)
  pgrep -x btnd >/dev/null 2>&1 && return 0
  [ -x "$H/btnd" ] || return 1
  [ -f "$H/buttons.conf" ] || cp "$H/buttons.default.conf" "$H/buttons.conf" 2>/dev/null
  log "starte btnd (Seitentasten)"
  "$H/btnd" /dev/input/event0 >>"$LOG" 2>&1 &
}
start_audiod(){ # Audio-Ausgabe-Bruecke (Bionic raw-ALSA); Container schickt PCM ueber Socket
  pgrep -f "$H/audiod-alsa" >/dev/null 2>&1 && return 0
  [ -x "$H/audiod-alsa" ] || return 1
  log "starte audiod-alsa (Socket $SHARED/audio.sock)"
  LD_LIBRARY_PATH=/system/lib64 "$H/audiod-alsa" "$SHARED/audio.sock" >>"$LOG" 2>&1 &
}
start_cfgd(){   # Android-Config-Agent fuer pixel-config (charge/wlan-Bridge)
  pgrep -f "$H/baseos-cfgd" >/dev/null 2>&1 && return 0
  [ -f "$H/baseos-cfgd.sh" ] || return 1
  log "starte baseos-cfgd (Config-Bridge)"
  sh "$H/baseos-cfgd.sh" >>"$LOG" 2>&1 &
}
start_gpu(){
  pgrep -f "$H/gpud" >/dev/null 2>&1 && return 0
  log "starte gpud (Socket $SHARED/gpu.sock)"
  LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 "$H/gpud" "$SHARED/gpu.sock" >>"$LOG" 2>&1 &
}
start_gpuzc(){  # GPU Zero-Copy-Transport: Client schickt dmabuf-fds via SCM_RIGHTS (keine Datenkopie)
  pgrep -x gpud-zc >/dev/null 2>&1 && return 0
  [ -x "$H/gpud-zc" ] || return 1
  log "starte gpud-zc (Zero-Copy, Socket $SHARED/gpuzc.sock)"
  LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 "$H/gpud-zc" "$SHARED/gpuzc.sock" >>"$LOG" 2>&1 &
}
start_gxp(){    # DSP-Bruecke (GXP Callisto): gxpd3 = Multi-Kernel + GXPD-inline (Kopie) UND
                # GXPZ-Zero-Copy (Client teilt dma_heap-dmabufs via SCM_RIGHTS, DSP rechnet darin).
  pgrep -x gxpd3 >/dev/null 2>&1 && return 0
  BIN="$H/gxpd3"; [ -x "$BIN" ] || BIN="$H/gxpd"   # Fallback auf altes gxpd, falls gxpd3 fehlt
  [ -x "$BIN" ] || return 1
  [ -f "$H/gxp_metrics_logger.so" ] || { log "gxp: Metrics-Stub fehlt - uebersprungen"; return 1; }
  log "starte $(basename "$BIN") (DSP, Socket $SHARED/gxp.sock)"
  # $H MUSS zuerst im LD_LIBRARY_PATH stehen: unser Metrics-Stub ueberschattet den
  # echten gxp_metrics_logger.so (der sonst in AServiceManager_waitForService(Stats)
  # blockiert und CreateDevice ewig haengen laesst).
  # GXPD_CACHE=1: DSP-Kopie-Puffer cacheable -> 6x schnellere Schleifen (16.8.). GXPD_KDIR: Kernel-ELFs.
  GXPD_CACHE=1 GXPD_KDIR="$H" LD_LIBRARY_PATH="$H:/system/lib64:/vendor/lib64" "$BIN" "$SHARED/gxp.sock" >>"$LOG" 2>&1 &
}

log "=== hw-bridges Supervisor start (pid $$) ==="
# Audio-Routen einmal setzen (raw-ALSA Speaker+Mic, headless)
[ -f "$H/av-setup.sh" ] && { sh "$H/av-setup.sh" >>"$LOG" 2>&1; }
while :; do
  start_tpu
  start_gpu
  start_gpuzc
  start_gxp
  start_cfgd
  start_audiod
  start_btnd
  sleep 15
done
