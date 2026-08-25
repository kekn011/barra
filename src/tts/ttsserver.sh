#!/system/bin/sh
# barra TTS-Dienst-Steuerung (David GPU-Vokoder + Piper de/en). Wie llm/stt/pya: manueller Start,
# KEIN Boot-Autostart. TTS rechnet im Container (glibc-Python + gpudecd) -> Steuerung ueber den
# Container-systemd-Dienst barra-tts via enter-systemd. GPU-Pin nur solange der Dienst laeuft.
#   su -M -c 'sh /data/adb/baseos/bin/ttsserver.sh start'   (start | stop | status | log)
#   -M (Mount-Namespace erhalten) ist fuer enter-systemd noetig.
ES=/data/adb/baseos/bin/enter-systemd.sh
U=/data/local/ubuntu

pins_on(){   # TTS nutzt den Grafikkern (David-Vokoder) -> Mali-Mindesttakt anheben
  [ "$TTS_NOPIN" = "1" ] && return
  echo 890000 > /sys/class/misc/mali0/device/scaling_min_freq 2>/dev/null
  echo 890000 > /sys/class/misc/mali0/device/hint_min_freq 2>/dev/null
}
pins_off(){
  # HW-Idle-Minimaltakt (150 MHz) zuruecksetzen. WICHTIG: "echo 0" ist auf diesem mali-devfreq
  # ein No-op (Treiber ignoriert 0) -> das alte pins_off hat NIE entpinnt (Mali blieb bis Reboot
  # auf 890). Konkreten Wert schreiben, BEIDE Regler (scaling + hint), sonst haelt der nicht
  # zurueckgesetzte den Takt oben. (Am Node am 25.8. verifiziert.)
  echo 150000 > /sys/class/misc/mali0/device/scaling_min_freq 2>/dev/null
  echo 150000 > /sys/class/misc/mali0/device/hint_min_freq 2>/dev/null
}
insys(){ printf %s "$1" > $U/root/.ttsctl.sh; sh "$ES" $U/root/.ttsctl.sh 2>&1; rm -f $U/root/.ttsctl.sh; }

case "${1:-status}" in
  start)
    pins_on
    insys "systemctl start barra-tts; sleep 3; systemctl is-active barra-tts"
    echo "TTS gestartet -> http://<node>:8095/say?voice=david|piper-de|piper-en&text=...";;
  stop)
    insys "systemctl stop barra-tts"
    pins_off
    echo "gestoppt";;
  status)
    insys "systemctl is-active barra-tts; systemctl show -p MainPID --value barra-tts";;
  log)
    insys "journalctl -u barra-tts --no-pager 2>/dev/null | tail -40; tail -20 /run/barra-tts/gpudecd.log 2>/dev/null";;
  *) echo "ttsserver.sh start | stop | status | log";;
esac
