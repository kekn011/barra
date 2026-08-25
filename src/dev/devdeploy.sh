#!/system/bin/sh
# barra Dev-Kit Deploy-Helfer (BASE-Seite, su noetig) — die Deploy-Haelfte der
# Edit-Build-Deploy-Schleife fuer die HW-Bruecken-Daemons.
#   su -c 'sh /data/adb/baseos/bin/devdeploy.sh <cmd> ...'
#
# Zwei Geschwindigkeiten (bewusst getrennt):
#  1) SCHNELL (Kernel iterieren) laeuft IM CONTAINER ueber Harnesses, die die .spv zur
#     Laufzeit laden - kein Daemon-Neustart, Sekunden. Siehe devbuild.sh + harness/.
#  2) LANGSAM (Kernel in den Daemon backen): Daemon wird auf dem HOST per NDK neu gebaut
#     (Shader sind als _spv.h einkompiliert), hierher gepusht und live getauscht. Das ist,
#     was dieses Skript macht - Binary tauschen, killen; der hw-bridges-Supervisor zieht
#     den Daemon automatisch neu hoch (pgrep-guarded start_*-Schleife).
H=/data/adb/hwbridge
SUP=$H/hw-bridges.sh
known="gpud gpud-zc tpud gxpd3 gxpd audiod-alsa audiod-record btnd"

running(){ pgrep -f "$H/$1" >/dev/null 2>&1 || pgrep -x "$1" >/dev/null 2>&1; }
cyc(){ pkill -f "$H/$1" 2>/dev/null; pkill -x "$1" 2>/dev/null; }
sup_up(){ pgrep -f hw-bridges >/dev/null 2>&1; }

case "${1:-status}" in
  daemon)                                    # neues Binary einspielen + live tauschen
    N="$2"; SRC="$3"
    case " $known " in *" $N "*) ;; *) echo "unbekannter Daemon: $N (bekannt: $known)"; exit 1;; esac
    [ -f "$SRC" ] || { echo "Quelle fehlt: $SRC"; exit 1; }
    cp "$SRC" "$H/$N" && chmod 755 "$H/$N" || { echo "Kopie fehlgeschlagen"; exit 1; }
    echo "eingespielt: $H/$N"
    sup_up || { echo "Supervisor laeuft nicht - starte ihn"; setsid sh "$SUP" </dev/null >/dev/null 2>&1 & sleep 2; }
    cyc "$N"; echo "gekillt - Supervisor respawnt $N"
    i=0; while [ $i -lt 15 ]; do running "$N" && { echo "$N laeuft wieder (PID $(pgrep -f "$H/$N"|head -1))"; exit 0; }; sleep 1; i=$((i+1)); done
    echo "WARN: $N nicht wieder oben - hw-bridges.log pruefen"; tail -5 "$H/hwbridge.log"; exit 2;;
  restart)                                   # nur zyklen (Supervisor respawnt, Loop braucht ein paar s)
    N="$2"; sup_up || { echo "Supervisor aus - starte"; setsid sh "$SUP" </dev/null >/dev/null 2>&1 & sleep 2; }
    cyc "$N"; echo "gekillt - warte auf Supervisor-Respawn ..."
    i=0; while [ $i -lt 15 ]; do sleep 1; running "$N" && { echo "$N neu gestartet (PID $(pgrep -f "$H/$N"|head -1))"; exit 0; }; i=$((i+1)); done
    echo "WARN: $N nach 15s nicht oben"; tail -3 "$H/hwbridge.log"; exit 2;;
  dspkernel)                                 # DSP-Kernel ker_<name>.elf nach KDIR + gxpd3 neu
    F="$2"; [ -f "$F" ] || { echo "ker-elf fehlt: $F"; exit 1; }
    case "$(basename "$F")" in ker_*.elf) ;; *) echo "Name muss ker_<name>.elf sein (gxpd3 keyt auf <name>)"; exit 1;; esac
    b=$(basename "$F"); cp "$F" "$H/$b" && chmod 644 "$H/$b" && echo "eingespielt: $H/$b"
    sup_up || { setsid sh "$SUP" </dev/null >/dev/null 2>&1 & sleep 2; }
    cyc gxpd3; echo "gxpd3 gekillt - Supervisor respawnt (laedt KDIR neu) ..."
    i=0; while [ $i -lt 15 ]; do sleep 1; running gxpd3 && { echo "gxpd3 wieder oben (PID $(pgrep -x gxpd3|head -1)) - $b geladen"; exit 0; }; i=$((i+1)); done
    echo "WARN: gxpd3 nicht oben"; tail -3 "$H/hwbridge.log"; exit 2;;
  status)
    echo "Supervisor: $(sup_up && echo laeuft || echo AUS)"
    for d in $known; do running "$d" && echo "  $d: laeuft (PID $(pgrep -f "$H/$d"|head -1))" || :; done;;
  *) echo "devdeploy.sh daemon <name> <binary> | dspkernel <ker_x.elf> | restart <name> | status";;
esac
