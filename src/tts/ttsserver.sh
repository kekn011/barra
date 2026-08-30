#!/system/bin/sh
# barra TTS-Dienst-Steuerung (Piper de/en, Vokoder auf dem Grafikkern). Wie llm/stt/pya: manueller Start,
# KEIN Boot-Autostart. TTS rechnet im Container (glibc-Python + gpudecd) -> Steuerung ueber den
# Container-systemd-Dienst barra-tts via enter-systemd. GPU-Pin nur solange der Dienst laeuft.
#   su -M -c 'sh /data/adb/baseos/bin/ttsserver.sh start'   (start | stop | status | log)
#   -M (Mount-Namespace erhalten) ist fuer enter-systemd noetig.
ES=/data/adb/baseos/bin/enter-systemd.sh
U=/data/local/ubuntu

. /data/adb/baseos/bin/barra-i18n.sh
# Koexistenz-Waechter (gemeinsam fuer alle Kit-Dienste, src/boot/barra-guard.sh). Fehlt er auf
# einem aelteren Base, laufen die Aufrufe ins Leere statt ins Messer.
G=/data/adb/baseos/bin/barra-guard.sh
if [ -f "$G" ]; then . "$G"; else guard_check(){ :; }; guard_need(){ echo 0; }; guard_expendable(){ :; }; fi
pins_on(){   # TTS nutzt den Grafikkern (Vokoder) -> Mali-Mindesttakt anheben
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
    # Waechter (29.8.): gemessen 270 MB (ttsd 237 MB + zwei gpudecd)
    guard_check tts 300 || exit 1
    pins_on
    insys "systemctl start barra-tts; sleep 3; systemctl is-active barra-tts"
    # Hebel 5 (Lazy-Compile-Warmup): das erste /say zahlt den ~2,4s Mali-Shader-Compile.
    # Je GPU-Stimme laeuft ein eigener gpudecd, also jede einmal anstossen. play=0 heisst:
    # nur rechnen, NICHT abspielen - der Warmup darf nicht hoerbar sein.
    # Jetzt beim Start abfangen statt beim Nutzer. Kurzer Text, play=0, best-effort mit Retries.
    insys "for v in piper-de piper-en; do for i in 1 2 3; do curl -s -m 30 -o /dev/null \"http://localhost:8095/say?voice=\$v&play=0&text=warmup\" && break; sleep 2; done; done; echo warmup-done"
    # Die Container-Dienste erben oom_score_adj -1000 vom systemd des Containers -> killbar machen
    guard_expendable $(pgrep -f "barra-tts/bin/ttsd.py") $(pgrep -f "barra-tts/bin/gpudecd")
    t tts.started;;
  stop)
    insys "systemctl stop barra-tts"
    pins_off
    t tts.stopped;;
  status)
    insys "systemctl is-active barra-tts; systemctl show -p MainPID --value barra-tts";;
  log)
    insys "journalctl -u barra-tts --no-pager 2>/dev/null | tail -40; tail -20 /run/barra-tts/gpudecd.log 2>/dev/null";;
  *) t tts.usage;;
esac
