#!/system/bin/sh
# barra Weckwort-Dienst-Steuerung ("Hey Barra"). Wie llm/stt/pya/tts: manueller Start, KEIN
# Boot-Autostart. Der Mic-Bridge audiod-record wird vom hw-bridges-Supervisor gestartet (sobald
# das Kit ihn nach /data/adb/hwbridge gelegt hat); die Erkennung ist der Container-systemd-Dienst
# barra-wake. Steuerung via enter-systemd.
#   su -M -c 'sh /data/adb/baseos/bin/wakeserver.sh start'   (start | stop | status | log)
ES=/data/adb/baseos/bin/enter-systemd.sh
U=/data/local/ubuntu
H=/data/adb/hwbridge
. /data/adb/baseos/bin/barra-i18n.sh
insys(){ printf %s "$1" > $U/root/.wakectl.sh; sh "$ES" $U/root/.wakectl.sh 2>&1; rm -f $U/root/.wakectl.sh; }

case "${1:-status}" in
  start)
    # audiod-record laeuft ueber den Supervisor (start_micrec), sobald das Binary da ist; anstossen falls noetig
    [ -x "$H/audiod-record" ] || { t wake.no_kit; exit 1; }
    pgrep -f "$H/audiod-record" >/dev/null 2>&1 || LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 setsid "$H/audiod-record" "$U/opt/hwbridge/micrec.sock" </dev/null >/dev/null 2>&1 &
    insys "systemctl start barra-wake; sleep 3; systemctl is-active barra-wake"
    t wake.active;;
  stop)
    insys "systemctl stop barra-wake"
    t wake.stopped;;
  status)
    insys "systemctl is-active barra-wake; systemctl show -p MainPID --value barra-wake"
    [ -S "$U/opt/hwbridge/micrec.sock" ] && t wake.mic_ok || t wake.mic_missing;;
  log)
    insys "journalctl -u barra-wake --no-pager 2>/dev/null | tail -40";;
  *) t wake.usage;;
esac
