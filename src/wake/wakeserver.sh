#!/system/bin/sh
# barra Weckwort-Dienst-Steuerung ("Hey Barra"). Wie llm/stt/pya/tts: manueller Start, KEIN
# Boot-Autostart. Der Mic-Bridge audiod-record wird vom hw-bridges-Supervisor gestartet (sobald
# das Kit ihn nach /data/adb/hwbridge gelegt hat); die Erkennung ist der Container-systemd-Dienst
# barra-wake. Steuerung via enter-systemd.
#   su -M -c 'sh /data/adb/baseos/bin/wakeserver.sh start'   (start | stop | status | log)
ES=/data/adb/baseos/bin/enter-systemd.sh
U=/data/local/ubuntu
H=/data/adb/hwbridge
insys(){ printf %s "$1" > $U/root/.wakectl.sh; sh "$ES" $U/root/.wakectl.sh 2>&1; rm -f $U/root/.wakectl.sh; }

case "${1:-status}" in
  start)
    # audiod-record laeuft ueber den Supervisor (start_micrec), sobald das Binary da ist; anstossen falls noetig
    [ -x "$H/audiod-record" ] || { echo "audiod-record fehlt - Weckwort-Kit nicht installiert"; exit 1; }
    pgrep -f "$H/audiod-record" >/dev/null 2>&1 || LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 setsid "$H/audiod-record" "$U/opt/hwbridge/micrec.sock" </dev/null >/dev/null 2>&1 &
    insys "systemctl start barra-wake; sleep 3; systemctl is-active barra-wake"
    echo "Weckwort aktiv - sag \"Hey Barra\" (Trigger -> /run/barra-wake/trigger)";;
  stop)
    insys "systemctl stop barra-wake"
    echo "gestoppt (Mic-Bridge laeuft weiter, ohne Verbraucher passiv)";;
  status)
    insys "systemctl is-active barra-wake; systemctl show -p MainPID --value barra-wake"
    [ -S "$U/opt/hwbridge/micrec.sock" ] && echo "Mic-Socket bereit" || echo "Mic-Socket fehlt";;
  log)
    insys "journalctl -u barra-wake --no-pager 2>/dev/null | tail -40";;
  *) echo "wakeserver.sh start | stop | status | log";;
esac
