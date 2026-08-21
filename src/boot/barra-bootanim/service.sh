#!/system/bin/sh
# barra-bootanim service.sh (Magisk late_start): bootanim startet oft BEVOR das
# Modul /product overlayt und haelt dann die Stock-Zip offen -> einmal neu
# starten; die neue Instanz oeffnet die barra-Animation, noch bevor das Panel
# ueberhaupt erste Frames zeigt (~Sekunde 12).
i=0
while [ $i -lt 15 ]; do
  P=$(pidof bootanimation)
  [ -n "$P" ] && break
  sleep 1; i=$((i+1))
done
if [ -n "$P" ]; then
  setprop service.bootanim.exit 0
  setprop ctl.restart bootanim
fi
