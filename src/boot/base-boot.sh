#!/system/bin/sh
# ============================================================================
# Base-OS-Bringup nach jedem Neustart — HEADLESS-PROFIL (Image).
# Ablage: /data/adb/service.d/50-base-os.sh startet /data/local/tmp/base-boot.sh
# (Magisk late_start, root, globaler Mount-NS). Idempotent, manuell:
#   su -M -c 'sh /data/local/tmp/base-boot.sh'
#
# Ziel: nacktes Ubuntu mit Vollzugriff auf die Hardware, DISPLAY AUS.
# Erstzugang per adb (USB): 'adb shell' bzw. 'adb forward tcp:2222 tcp:22' +
# 'ssh -p2222 ubuntu@127.0.0.1' (ubuntu/ubuntu). Konfiguration via pixel-config.
# KEINE Oberflaeche/kein Panel/kein OSK/kein Wizard mehr am Boot — das Display
# schaltet der Nutzer erst in der Config ein (dispctl).
#
# Reihenfolge:
#   1. sys.boot_completed abwarten (sonst wertet Android den Start als Fehlschlag).
#   2. Grundsystem: Doze aus + Wakelock, zram, Kamera-HAL aus, DISPLAY AUS.
#   3. Ubuntu-Userland (systemd via pivot_root). Vor dem WLAN (DHCP-Client ist
#      systemd-networkd IM Container, teilt die netns).
#   4. Android stillstellen (sonst greift dessen Watchdog nach wlan0).
#   5. Eigenes WLAN (optional; adb bleibt der garantierte Zugang).
#
# NOTBREMSE: touch /data/adb/baseos/disable  -> normales Android beim naechsten Start.
# ============================================================================
T=/data/adb/baseos/bin      # Programme (Image-fest); Laufzeit-Files unter $D/run
D=/data/adb/baseos
. $T/barra-i18n.sh          # t <key> [args] — lokalisierte UI-Texte (Splash)
LOG=$D/boot.log
OFF=$D/disable
LOCK=$D/boot.lock
STATE=$D/state

mkdir -p "$D" "$D/run" "$D/bin"
[ "$(wc -c < "$LOG" 2>/dev/null || echo 0)" -gt 1048576 ] && : > "$LOG"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stage() { echo "$1" > "$STATE"; log "--- $1 ---"; sync; }
# Boot-Splash (bin/bootsplash, KMS): zeigt "barra" + Balken + Status. Startet erst nach dem
# Composer-Stop (framework-aus) - vorher hat Android das Panel. Status: "<pct>|<text>".
SPL=$D/run/splash.status
splash() { echo "$1|$2" > "$SPL" 2>/dev/null; }

if [ -f "$OFF" ]; then
  log "disable-Marker gesetzt - Base-OS-Bringup uebersprungen"; exit 0
fi
if ! mkdir "$LOCK" 2>/dev/null; then
  OLDPID=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$OLDPID" ] && grep -qa base-boot "/proc/$OLDPID/cmdline" 2>/dev/null; then
    log "laeuft bereits (pid $OLDPID) - exit"; exit 0
  fi
  log "verwaiste Sperre (pid ${OLDPID:-?}) - uebernehme"
fi
echo $$ > "$LOCK/pid"
trap 'rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null' EXIT

log "=== base-boot start (headless) ==="

# --- 1. Systemstart abwarten ------------------------------------------------
# Waehrend dieser Phase laeuft die normale Google-Bootanimation (die barra-
# Bootanimation wurde am 21.8. auf Kevins Wunsch wieder entfernt). Direkt nach
# boot_completed geht es zum framework-aus, wo unser bootsplash das Panel
# uebernimmt — kein langes Settle mehr (das hiess frueher: Lockscreen sichtbar
# + Android-Display-Timeout = lange Schwarzphase).
stage boot-wait
if [ "$(getprop sys.boot_completed)" != "1" ]; then
  i=0
  while [ "$(getprop sys.boot_completed)" != "1" ] && [ $i -lt 120 ]; do
    sleep 2; i=$((i+1))
  done
  sleep 3
fi
log "boot_completed=$(getprop sys.boot_completed) nach ${i:-0} Runden"

# --- 2. Grundsystem ----------------------------------------------------------
stage grundsystem
dumpsys deviceidle disable >>"$LOG" 2>&1
MP=$(command -v magiskpolicy || echo /system_ext/bin/magiskpolicy)
"$MP" --live "allow magisk sysfs_wake_lock file { open read write append getattr }" >/dev/null 2>&1
echo baseos >> /sys/power/wake_lock 2>/dev/null       # CPU wach trotz dunklem Schirm
echo "0 2147483647" > /proc/sys/net/ipv4/ping_group_range 2>/dev/null
echo 180 > /proc/sys/vm/swappiness 2>/dev/null
echo 0   > /proc/sys/vm/page-cluster 2>/dev/null
# zram 3,7GB->1GB: Android-Default kostet ~15MB vmalloc-Metadaten, Swap ist praktisch leer
# (RAM-Analyse 22.8.); beim Boot ist nichts geswappt -> Umbau gefahrlos, live verifiziert.
if [ -b /dev/block/zram0 ]; then
  swapoff /dev/block/zram0 2>/dev/null
  echo 1 > /sys/block/zram0/reset 2>/dev/null
  echo 1073741824 > /sys/block/zram0/disksize 2>/dev/null
  mkswap /dev/block/zram0 >/dev/null 2>&1
  swapon /dev/block/zram0 2>/dev/null
fi
setprop ctl.stop vendor.camera-provider-2-7-google
setprop ctl.stop cameraserver
# Display AN lassen: der Boot zeigt durchgehend barra (Bootanimation -> bootsplash ->
# Dashboard); dash2 blankt am Ende von selbst nach DISPLAY_TIMEOUT. (Frueher wurde hier
# hart abgeschaltet — Headless-Erbe; Folge war: Android sichtbar, dann minutenlang schwarz.)
echo 0 > /sys/class/backlight/panel0-backlight/bl_power 2>/dev/null
log "display bleibt an (Boot-Anzeige), wakelock=$(cat /sys/power/wake_lock 2>/dev/null)"

# --- 3. Android stillstellen + Splash uebernimmt -----------------------------
# BEWUSST VOR dem Ubuntu-Start (umsortiert 21.8.): so uebernimmt der barra-Splash
# das Panel schon ~25 s nach dem Einschalten und deckt Ubuntu-Start + WLAN ab —
# vorher lief hier erst der Ubuntu-Start (~40 s Lockscreen/Schwarzphase).
# boot-systemd.sh braucht nichts vom Framework (chroot/nsenter/Mounts).
stage framework-aus
sh $T/fw-quiet.sh off >>"$LOG" 2>&1
log "system_server=$(pidof system_server) netd=$(pidof netd)"
# Composer ist jetzt weg -> Panel frei -> Boot-Splash starten (laeuft bis Status 100)
splash 20 "$(t splash.userland)"
if [ -x $T/bootsplash ]; then
  setsid $T/bootsplash "$SPL" 300 </dev/null >>"$D/run/splash.log" 2>&1 &
  log "bootsplash gestartet"
fi

# --- 4. Ubuntu-Userland -----------------------------------------------------
stage ubuntu-userland
sh $T/boot-systemd.sh >>"$LOG" 2>&1
SDPID=$(cat $D/run/systemd.hostpid 2>/dev/null)
if [ -n "$SDPID" ] && [ -d "/proc/$SDPID" ]; then
  log "Ubuntu-Userland laeuft (host-pid $SDPID)"
  splash 55 "$(t splash.net_setup)"
else
  log "FEHLER: Ubuntu-Userland nicht hochgekommen - Framework zurueck, Abbruch (adb bleibt)"
  splash 100 ""
  echo 3000 > /sys/class/backlight/panel0-backlight/brightness 2>/dev/null
  sh $T/fw-quiet.sh on >>"$LOG" 2>&1
  exit 1
fi

# --- 5. Eigenes WLAN (optional) ---------------------------------------------
# adb ist der garantierte Zugang; WLAN ist Komfort fuers Netz-SSH. Ein
# fehlgeschlagener Join sperrt niemanden aus. (Fuer ein oeffentliches Image
# werden die WLAN-Creds spaeter aus wifi-join.sh entfernt und via pixel-config
# gesetzt - siehe Baking-Schritt.)
stage wlan
if [ "$(grep '^WLAN_ENABLED=' /data/adb/baseos/config 2>/dev/null | cut -d= -f2)" = "0" ]; then
  log "WLAN in Config deaktiviert (WLAN_ENABLED=0) - uebersprungen"
  splash 90 "$(t splash.wifi_off)"
else
  ip link set wlan0 up 2>/dev/null
  # FALLBACK (netzloser Boot): Der Boot blockiert nur KURZ auf den WLAN-Join. Frueher waren
  # es 2 Versuche a `timeout 320` = bis ~11 min Haenger ohne Netz (fuehlt sich endlos an).
  # Jetzt: EIN kurzer Boot-Join, dann sofort weiter ins Dashboard. Der wifi-guard retryt
  # danach endlos im Hintergrund und holt das WLAN nach, sobald das Netz da ist. adb (USB)
  # bleibt der garantierte Zugang. Timeout via Config-Store: WLAN_BOOT_TIMEOUT (Default 45s;
  # 0 = Boot-Join ganz ueberspringen, nur Hintergrund-Waechter).
  BT=$(grep '^WLAN_BOOT_TIMEOUT=' /data/adb/baseos/config 2>/dev/null | cut -d= -f2)
  case "$BT" in ''|*[!0-9]*) BT=45;; esac
  IP=""
  if [ "$BT" -gt 0 ]; then
    log "WLAN-Boot-Join (max ${BT}s, danach weiter)"
    splash 70 "$(t splash.wifi_connecting "$BT")"
    timeout "$BT" sh $T/wifi-join.sh >>"$LOG" 2>&1
    # Erfolg = ECHTE IPv4 auf wlan0. Die Lease holt der selbstheilende dhclient-Dienst.
    j=0; while [ $j -lt 5 ]; do
      IP=$(ip -4 addr show wlan0 2>/dev/null | awk '/inet /{print $2}')
      [ -n "$IP" ] && break; sleep 1; j=$((j+1))
    done
  else
    log "WLAN_BOOT_TIMEOUT=0 - Boot-Join uebersprungen, nur Hintergrund-Waechter"
  fi
  if [ -n "$IP" ]; then
    log "WLAN verbunden: $IP"; splash 90 "$(t splash.wifi_connected "${IP%/*}")"
  else
    log "WLAN beim Boot (noch) nicht verbunden - weiter ins Dashboard, Waechter retryt, adb bleibt"
    splash 90 "$(t splash.wifi_later)"
  fi
  # Waechter IMMER starten (retryt endlos im Hintergrund, bis WLAN steht)
  setsid sh $T/wifi-guard.sh start </dev/null >/dev/null 2>&1 &
  log "wifi-guard gestartet (retryt bis WLAN steht)"
fi

stage fertig
log "=== base-boot fertig (headless, Stufe: $(cat $STATE)) ==="

# --- 6. Boot-Uebergabe: Splash -> Dashboard --------------------------------------
# Splash auf 100 -> er beendet sich (gibt DRM-Master frei, blankt NICHT), dash2
# uebernimmt das bereits "warme" Panel nahtlos. dash2 blankt nach DISPLAY_TIMEOUT.
splash 98 "$(t splash.dashboard)"
sleep 2
splash 100 "$(t splash.done)"
i=0; while [ $i -lt 10 ] && pgrep -x bootsplash >/dev/null 2>&1; do sleep 0.3; i=$((i+1)); done
pkill -TERM -x bootsplash 2>/dev/null
sh /data/adb/hwbridge/dispctl.sh on >>"$LOG" 2>&1
log "Boot-Uebergabe: Splash -> Dashboard (dispctl on, Auto-Aus nach DISPLAY_TIMEOUT)"
