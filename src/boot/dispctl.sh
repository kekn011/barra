#!/system/bin/sh
# dispctl - Display an/aus/umschalten. Zeigt das barra-Dashboard (dash2) auf dem Panel.
# dash2 ist ein glibc/FreeType-Programm und laeuft IM Ubuntu-Container. Der Start erfolgt
# NICHT per nsenter (das laesst sich nicht backgrounden), sondern ueber den persistenten
# Container-Dienst dash-ctl.service: dispctl schreibt nur ein Kommando ins Trigger-File.
#   dispctl.sh on|off|toggle|status
STORE=/data/adb/baseos/config
CMD=/data/local/ubuntu/opt/hwbridge/dash.cmd
DEF=60
get_timeout(){ t=$(grep '^DISPLAY_TIMEOUT=' "$STORE" 2>/dev/null | cut -d= -f2); case "$t" in ''|*[!0-9]*) t=$DEF;; esac; echo "$t"; }
put(){ echo "$1" > "$CMD"; }
# Androids HWComposer-HAL haelt sonst den DRM-Master -> dash2 bekommt das
# Command-Mode-Panel nicht. Vor jedem Einschalten stoppen (bleibt gestoppt).
comp_stop(){ setprop ctl.stop vendor.hwcomposer-3 2>/dev/null; }
case "$1" in
  on)     comp_stop; put "on:$(get_timeout)" ;;
  off)    put "off" ;;
  toggle) comp_stop; put "toggle:$(get_timeout)" ;;
  status) if pgrep -x dash2 >/dev/null 2>&1; then echo "on (timeout $(get_timeout)s)"; else echo "off (timeout $(get_timeout)s)"; fi ;;
  *) echo "usage: dispctl.sh on|off|toggle|status" ;;
esac
