#!/system/bin/sh
# Android-Framework fuer den Panel-/Headless-Betrieb ruhigstellen.
#   fw-quiet.sh off  -> class main stoppen (zygote/system_server/wificond) + WiFi-Stack aus der Hand nehmen
#   fw-quiet.sh on   -> zurueck zu Android (class main starten)
# Reversibel; adbd (class core) laeuft weiter. Beendet den system_server-Watchdog-Loop,
# der bei jedem Anlauf nach wlan0 greift.
case "$1" in
  off)
    echo "vorher: system_server=$(pidof system_server) zygote64=$(pidof zygote64) wificond=$(pidof wificond) suppl=$(pidof wpa_supplicant)"
    # WiFi-Stack zuerst, damit niemand mehr nach wlan0 greift
    setprop ctl.stop wpa_supplicant
    setprop ctl.stop wificond
    # bootanim (class core, von 'stop' nicht erfasst) laeuft seit base-boot durch —
    # jetzt beenden, direkt danach uebernimmt unser bootsplash das Panel.
    setprop ctl.stop bootanim
    stop                      # class main (zygote, system_server, ...)
    setprop ctl.stop zygote
    setprop ctl.stop zygote_secondary
    # HWComposer-HAL (class hal, von 'stop' NICHT erfasst) haelt sonst den DRM-Master
    # -> unser KMS-Dashboard (dash2) bekommt das Command-Mode-Panel nicht.
    setprop ctl.stop vendor.hwcomposer-3
    i=0
    while [ $i -lt 10 ]; do
      [ -z "$(pidof system_server)" ] && [ -z "$(pidof zygote64)" ] && break
      sleep 1; i=$((i+1))
    done
    echo "nachher: system_server=$(pidof system_server) zygote64=$(pidof zygote64) wificond=$(pidof wificond) suppl=$(pidof wpa_supplicant)"
    echo "init.svc.wpa_supplicant=$(getprop init.svc.wpa_supplicant) init.svc.zygote=$(getprop init.svc.zygote)"
    echo "setupwiz=$(pidof setupwiz)  backlight=$(cat /sys/class/backlight/panel0-backlight/brightness 2>/dev/null)"
    ;;
  on)
    start
    sleep 3
    echo "system_server=$(pidof system_server) zygote64=$(pidof zygote64)"
    ;;
  *) echo "usage: fw-quiet.sh off|on" ;;
esac
