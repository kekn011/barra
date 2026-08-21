#!/system/bin/sh
# Magisk service.d (late_start, root, globaler Mount-NS): startet die Hardware-
# Bruecken-Bruecken (TPU/GPU) nach dem Boot. Laeuft parallel zu 50-base-os.sh
# (base-boot) und ist davon entkoppelt - die Daemons brauchen kein Panel/WLAN.
# Notbremse teilt sich mit dem Base-OS: /data/adb/baseos/disable.
D=/data/adb/baseos
H=/data/adb/hwbridge
[ -f "$D/disable" ] && exit 0
(
  # auf Systemstart (Vendor-HALs/Grafik-Stack) und gemountetes Ubuntu-Volume warten
  i=0; while [ "$(getprop sys.boot_completed)" != "1" ] && [ $i -lt 120 ]; do sleep 2; i=$((i+1)); done
  i=0; while ! grep -q " /data/local/ubuntu " /proc/mounts && [ $i -lt 60 ]; do sleep 1; i=$((i+1)); done
  setsid sh "$H/hw-bridges.sh" </dev/null >/dev/null 2>&1 &
) </dev/null >/dev/null 2>&1 &
