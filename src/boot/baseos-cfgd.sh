#!/system/bin/sh
# ============================================================================
# baseos-cfgd - Android-seitiger Config-Agent fuer pixel-config (im Ubuntu-Container).
# Der Container (pivot_root) sieht /data/adb + /sys/.../google,charger NICHT. Diese Bruecke
# ueber den GETEILTEN Pfad /opt/hwbridge/cfg/ nimmt Requests entgegen und wendet sie an.
#   Request:  eine Zeile in .../cfg/req, z.B. "charge 50 55"
#   Antwort:  .../cfg/resp, z.B. "OK charge 50 55" oder "ERR ..."
# Wird vom hw-bridges-Supervisor gestartet. Poll-Intervall 1s.
# ============================================================================
CFG=/data/local/ubuntu/opt/hwbridge/cfg
REQ=$CFG/req; RESP=$CFG/resp
CHG=/sys/devices/platform/google,charger
STORE=/data/adb/baseos/config
LOG=/data/adb/hwbridge/cfgd.log
PIDF=/data/adb/hwbridge/cfgd.pid

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >>"$LOG"; }
[ "$(wc -c < "$LOG" 2>/dev/null || echo 0)" -gt 524288 ] && : > "$LOG"

# Singleton
OLD=$(cat "$PIDF" 2>/dev/null)
if [ -n "$OLD" ] && grep -qa baseos-cfgd "/proc/$OLD/cmdline" 2>/dev/null; then exit 0; fi
echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT

# ACHTUNG Vertrauensgrenze: $CFG ist in den Container gemountet und welt-beschreibbar,
# damit jeder Container-uid Requests ablegen kann. Dieser Daemon laeuft als Android-root
# und fuehrt die Requests unauthentifiziert aus — jeder Container-Prozess kann also z.B.
# 'power reboot' ausloesen. 'exec:'-Aktionen sind auf ein root-eigenes Verzeichnis
# beschraenkt (s. apply_button). Fuer echte Isolation: $CFG auf 770 mit einer dem
# Container gemeinsamen Gruppe statt 777, oder Requests authentifizieren.
mkdir -p "$CFG" 2>/dev/null; chmod 777 "$CFG" 2>/dev/null

setkv(){ # key value -> in den root-Config-Store (robust gegen / & und Leerzeichen im Wert)
  k=$1; v=$2; touch "$STORE"; chmod 600 "$STORE"
  grep -v "^$k=" "$STORE" > "$STORE.tmp" 2>/dev/null; echo "$k=$v" >> "$STORE.tmp"
  mv "$STORE.tmp" "$STORE"; chmod 600 "$STORE"
}

apply_charge(){
  S=$1; E=$2
  case "$S$E" in ""|*[!0-9]*) echo "ERR charge: Zahlen erwartet"; return;; esac
  if [ "$S" -lt 5 ] || [ "$E" -gt 100 ] || [ "$S" -ge "$E" ]; then echo "ERR charge: Bereich (5<=start<stop<=100)"; return; fi
  setkv CHARGE_START "$S"; setkv CHARGE_STOP "$E"          # persistent (charge-limit.sh liest beim Boot)
  echo "$S" > "$CHG/charge_start_level" 2>/dev/null        # sofort wirksam
  echo "$E" > "$CHG/charge_stop_level" 2>/dev/null
  NS=$(cat "$CHG/charge_start_level" 2>/dev/null); NE=$(cat "$CHG/charge_stop_level" 2>/dev/null)
  echo "OK charge $NS $NE"
}

apply_button(){   # KEY AKTION -> in buttons.conf setzen (btnd liest sie frisch je Druck)
  K=$1; A=$2
  case "$K" in POWER|POWER_LONG|VOLUMEUP|VOLUMEDOWN) ;; *) echo "ERR button: Taste?"; return;; esac
  # WICHTIG: cfgd laeuft als Android-root und bedient unauthentifizierte Requests aus
  # dem container-sichtbaren, welt-beschreibbaren $CFG. 'exec:' laeuft spaeter als root
  # (btnd), also NUR Skripte aus einem root-eigenen, container-UNschreibbaren Verzeichnis
  # zulassen — sonst ist das ein Container->Android-root-Eskalationspfad.
  case "$A" in
    exec:*)
      P=${A#exec:}
      case "$P" in
        /data/adb/hwbridge/actions/*) ;;
        *) echo "ERR button: exec-Pfad nur unter /data/adb/hwbridge/actions/"; return;;
      esac
      case "$P" in *..*) echo "ERR button: exec-Pfad ungueltig"; return;; esac
      ;;
    none|log|shutdown|reboot|volup|voldown|display|displayon|displayoff) ;;
    *) echo "ERR button: Aktion?"; return;;
  esac
  BC=/data/adb/hwbridge/buttons.conf; touch "$BC"
  if grep -q "^$K=" "$BC" 2>/dev/null; then sed -i "s|^$K=.*|$K=$A|" "$BC"; else echo "$K=$A" >> "$BC"; fi
  echo "OK button $K=$A"
}
get_buttons(){ echo "OK buttons $(tr '\n' ' ' < /data/adb/hwbridge/buttons.conf 2>/dev/null)"; }

apply_display(){  # timeout N | on | off | toggle | status
  case "$1" in
    timeout) case "$2" in ''|*[!0-9]*) echo "ERR display: Sekunden (Zahl) erwartet"; return;; esac
             setkv DISPLAY_TIMEOUT "$2"; echo "OK display timeout=$2";;
    on|off|toggle) sh /data/adb/hwbridge/dispctl.sh "$1" >/dev/null 2>&1; echo "OK display $1";;
    status) echo "OK display $(sh /data/adb/hwbridge/dispctl.sh status)";;
    *) echo "ERR display: on|off|toggle|timeout N|status";;
  esac
}

apply_wlan(){   # on | off | status | setup <ssid> <psk>  (WLAN_ENABLED persistent; base-boot respektiert es)
  T=/data/adb/baseos/bin
  case "$1" in
    setup) # Creds in den root-Store (600). PSK darf Leerzeichen enthalten -> Rest der Zeile.
           SS=$2; PK=$3
           [ -z "$SS" ] && { echo "ERR wlan setup: SSID fehlt"; return; }
           setkv WIFI_SSID "$SS"; setkv WIFI_PSK "$PK"; setkv WLAN_ENABLED 1
           rm -f /data/local/tmp/wifi_ssid /data/local/tmp/wifi_psk /data/adb/baseos/run/wifi_ssid /data/adb/baseos/run/wifi_psk 2>/dev/null   # alte Entwicklungs-Dateien weg
           echo "OK wlan setup ssid=$SS (verbinde mit 'wlan on')";;
    off) setkv WLAN_ENABLED 0
         pkill -f wifi-guard 2>/dev/null
         pkill -f wpa_supplicant 2>/dev/null; killall wpa_supplicant 2>/dev/null
         ip addr flush dev wlan0 2>/dev/null
         ip link set wlan0 down 2>/dev/null
         echo "OK wlan off";;
    on)  setkv WLAN_ENABLED 1
         ip link set wlan0 up 2>/dev/null
         setsid sh "$T/wifi-join.sh" >/dev/null 2>&1 &
         setsid sh "$T/wifi-guard.sh" start </dev/null >/dev/null 2>&1 &
         echo "OK wlan on (Join gestartet - Verbindung dauert ~15-30s)";;
    status) OP=$(cat /sys/class/net/wlan0/operstate 2>/dev/null)
            IP=$(ip -4 addr show wlan0 2>/dev/null | awk '/inet /{print $2; exit}')
            EN=$(grep '^WLAN_ENABLED=' "$STORE" 2>/dev/null | cut -d= -f2)
            echo "OK wlan enabled=${EN:-1} operstate=${OP:-?} ip=${IP:-keine}";;
    *) echo "ERR wlan: on|off|status";;
  esac
}

apply_lang(){   # lang <code> -> UI-Sprache fuer Android-seitige Skripte (barra-i18n liest LANG_UI)
  case "$1" in
    ''|*[!a-z]*) echo "ERR lang: language code expected (e.g. de, en)"; return;;
  esac
  setkv LANG_UI "$1"
  echo "OK lang $1"
}

apply_power(){  # reboot | shutdown  (Geraet, nicht nur Container -> Android-powerctl)
  case "$1" in
    reboot)   log "power reboot"; echo "OK power reboot (Neustart in ~2s)";   ( sleep 2; setprop sys.powerctl reboot   ) & ;;
    shutdown) log "power shutdown"; echo "OK power shutdown (aus in ~2s)";     ( sleep 2; setprop sys.powerctl shutdown ) & ;;
    *) echo "ERR power: reboot|shutdown";;
  esac
}

get_state(){
  S=$(cat "$CHG/charge_start_level" 2>/dev/null); E=$(cat "$CHG/charge_stop_level" 2>/dev/null)
  CAP=$(cat /sys/class/power_supply/battery/capacity 2>/dev/null)
  ST=$(cat /sys/class/power_supply/battery/status 2>/dev/null)
  TMP=$(cat /sys/class/power_supply/battery/temp 2>/dev/null)
  echo "OK state charge=$S/$E cap=$CAP status=$ST temp=$TMP"
}

# ---- TPU-Compile-Worker (fuer barrac im Container) -------------------------------
# Queue: /opt/hwbridge/compile/<job>.tflite + <job>.req -> <job>.package + <job>.done (+ <job>.log)
# EIN Worker, seriell (tpuc1 nutzt feste Pfade /data/local/tmp/out.package + edgetpu-Cache).
CQ=/data/local/ubuntu/opt/hwbridge/compile
# libcomp_std.so ist ein GEPATCHTER VENDOR-BLOB und wird NIE mitgeliefert (er darf das Geraet
# nicht verlassen). Er wird beim ersten Compile-Auftrag aus der Vendor-Bibliothek erzeugt, die
# ohnehin auf dem Telefon liegt. Danach bleibt er liegen; der Nutzer merkt nur den ersten Lauf.
ensure_compiler(){
  T=/data/adb/baseos/tpu
  [ -s "$T/libcomp_std.so" ] && return 0
  [ -x "$T/extract-libedgetpu.sh" ] || [ -f "$T/extract-libedgetpu.sh" ] || return 1
  log "compile: libcomp_std.so fehlt - wird aus der Vendor-Bibliothek erzeugt"
  sh "$T/extract-libedgetpu.sh" "$T/libcomp_std.so" >>"$T/extract.log" 2>&1
  [ -s "$T/libcomp_std.so" ]
}
compile_worker(){
  T=/data/adb/baseos/tpu
  mkdir -p "$CQ" 2>/dev/null; chmod 777 "$CQ" 2>/dev/null
  while :; do
    [ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] || exit 0   # cfgd wurde ersetzt -> Worker beenden
    for r in "$CQ"/*.req; do
      [ -f "$r" ] || continue
      j="${r%.req}"; rm -f "$r"
      log "compile: $(basename "$j").tflite"
      if ! ensure_compiler; then
        echo "ERR TPU-Compiler nicht verfuegbar (libcomp_std.so liess sich nicht erzeugen; siehe $T/extract.log)" > "$j.done"
        log "compile: $(basename "$j") -> kein Compiler"
        continue
      fi
      rm -f /data/local/tmp/out.package /data/vendor/edgetpu/cache/_0 /data/vendor/edgetpu/cache/_0_checksum 2>/dev/null
      ( cd /data/local/tmp && COMPILER_SO=$T/libcomp_std.so MODEL="$j.tflite" LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 timeout 600 $T/tpuc1 ) > "$j.log" 2>&1
      if [ -s /data/local/tmp/out.package ]; then
        mv /data/local/tmp/out.package "$j.package"; chmod 666 "$j.package"
        echo "OK $(wc -c < "$j.package") bytes" > "$j.done"
      else
        echo "ERR compile failed (see $(basename "$j").log)" > "$j.done"
      fi
      chmod 666 "$j.done" "$j.log" 2>/dev/null
      log "compile: $(basename "$j") -> $(head -1 "$j.done")"
    done
    sleep 2
  done
}
compile_worker &

log "cfgd start (pid $$)"
while :; do
  if [ -f "$REQ" ]; then
    LINE=$(head -1 "$REQ" 2>/dev/null); rm -f "$REQ"
    case "$LINE" in "wlan setup"*) log "req: wlan setup <ssid> ****";; *) log "req: $LINE";; esac
    # shellcheck disable=SC2086
    set -f; set -- $LINE; set +f; CMD=$1   # set -f: kein Datei-Globbing beim Wort-Splitting (deterministisch)
    case "$CMD" in
      charge) R=$(apply_charge "$2" "$3");;
      button) R=$(apply_button "$2" "$3");;
      buttons) R=$(get_buttons);;
      display) R=$(apply_display "$2" "$3");;
      wlan)   if [ "$2" = "setup" ]; then
                # "wlan setup <ssid> <psk...>": SSID = 3. Wort, PSK = Rest (darf Leerzeichen haben)
                SS=$3; PK=$(echo "$LINE" | cut -d' ' -f4-)
                R=$(apply_wlan setup "$SS" "$PK")
              else R=$(apply_wlan "$2"); fi;;
      power)  R=$(apply_power "$2");;
      state)  R=$(get_state);;
      lang)   R=$(apply_lang "$2");;
      *)      R="ERR unknown command: $CMD";;
    esac
    echo "$R" > "$RESP" 2>/dev/null; chmod 666 "$RESP" 2>/dev/null
    log "resp: $R"
  fi
  sleep 1
done
