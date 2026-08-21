#!/system/bin/sh
# ============================================================================
# Haelt das WLAN des Base-OS am Leben.
#   wifi-guard.sh start | stop | status | once
#
# Warum es das braucht: ohne Android-Framework gibt es niemanden, der eine
# abgerissene Verbindung neu aufbaut. Der Supplicant laeuft unter `timeout` und
# beendet sich irgendwann selbst; die Assoziation ueberlebt das zwar (die
# Firmware haelt sie - das Geraet lief so 13 h weiter), aber ein echter Abriss
# (Router-Neustart, ausser Reichweite, Kanalwechsel) bleibt sonst fuer immer.
#
# Pruefkriterium ist bewusst NICHT operstate - das stand am 13.8. auf "up",
# waehrend gar kein Netz da war ([[wifi-headless-join]]). Geprueft wird, was
# zaehlt: IP vorhanden UND Gateway antwortet.
#
# Der Waechter haelt sich raus, solange Androids Framework laeuft - dann ist
# Android fuer wlan0 zustaendig und zwei Instanzen wuerden sich bekaempfen.
# ============================================================================
TMP=/data/adb/baseos/run   # Laufzeit-Files (wifi_result, Logs)
BIN=/data/adb/baseos/bin   # Programme (wlpm, wpactl, Skripte)
D=/data/adb/baseos
LOG=$D/wifi-guard.log
PIDF=$D/wifi-guard.pid
STOPF=$D/wifi-guard.stop

INTERVALL=60          # wie oft geprueft wird
FEHLER_BIS_JOIN=3     # erst nach 3 schlechten Runden (~3 min) neu verbinden
RUHE_NACH_JOIN=300    # nach einem Join-Versuch 5 min nicht nochmal ansetzen

mkdir -p "$D"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

netz_ok() {
  IP=$(ip -o -4 addr show wlan0 2>/dev/null | awk '{print $4}' | head -1)
  [ -z "$IP" ] && return 1
  GW=$(ip route show table main 2>/dev/null | awk '/^default/{print $3; exit}')
  [ -z "$GW" ] && return 1
  ping -c 1 -W 3 "$GW" >/dev/null 2>&1
}

case "$1" in
  start)
    OLD=$(cat "$PIDF" 2>/dev/null)
    if [ -n "$OLD" ] && [ -d "/proc/$OLD" ]; then echo "laeuft bereits (pid $OLD)"; exit 0; fi
    rm -f "$STOPF"; echo $$ > "$PIDF"
    [ "$(wc -c < "$LOG" 2>/dev/null || echo 0)" -gt 524288 ] && : > "$LOG"
    log "=== wifi-guard start (pid $$) ==="
    schlecht=0
    while [ ! -f "$STOPF" ]; do
      sleep $INTERVALL
      [ -f "$STOPF" ] && break
      if [ -n "$(pidof system_server)" ]; then
        log "Android-Framework laeuft - Waechter haelt sich raus"
        schlecht=0
        continue
      fi
      if netz_ok; then
        [ $schlecht -gt 0 ] && log "Netz wieder da ($IP via $GW)"
        schlecht=0
      else
        schlecht=$((schlecht+1))
        log "Netz nicht nutzbar (Runde $schlecht/$FEHLER_BIS_JOIN, ip=${IP:-keine} gw=${GW:-keins})"
        if [ $schlecht -ge $FEHLER_BIS_JOIN ]; then
          log "--- neuer Join-Versuch ---"
          timeout 320 sh "$BIN/wifi-join.sh" >>"$LOG" 2>&1
          log "Ergebnis: $(cat $TMP/wifi_result 2>/dev/null)"
          schlecht=0
          sleep $RUHE_NACH_JOIN
        fi
      fi
    done
    log "=== wifi-guard beendet ==="
    rm -f "$PIDF" "$STOPF"
    ;;
  stop)
    touch "$STOPF"
    P=$(cat "$PIDF" 2>/dev/null); [ -n "$P" ] && kill "$P" 2>/dev/null
    sleep 1; rm -f "$PIDF" "$STOPF"
    echo "wifi-guard gestoppt"
    ;;
  status)
    P=$(cat "$PIDF" 2>/dev/null)
    if [ -n "$P" ] && [ -d "/proc/$P" ]; then echo "laeuft (pid $P)"; else echo "laeuft nicht"; fi
    if netz_ok; then echo "Netz OK: $IP via $GW"; else echo "Netz NICHT nutzbar (ip=${IP:-keine} gw=${GW:-keins})"; fi
    tail -5 "$LOG" 2>/dev/null
    ;;
  once)
    if netz_ok; then echo "OK $IP via $GW"; else echo "FAIL ip=${IP:-keine} gw=${GW:-keins}"; exit 1; fi
    ;;
  *) echo "usage: wifi-guard.sh start|stop|status|once" ;;
esac
