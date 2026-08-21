#!/bin/bash
# barra Dashboard-Steuerung - laeuft IM Ubuntu-Container als systemd-Dienst (dash-ctl.service).
# Pollt /opt/hwbridge/dash.cmd (von dispctl/Android geschrieben) und schaltet dash2:
#   on:<T>   off   toggle:<T>       (T = Auto-Aus-Timeout in Sekunden, 0 = nie)
# dash2 laeuft container-nativ (kein nsenter noetig) und blankt das Panel bei SIGTERM.
CMD=/opt/hwbridge/dash.cmd
: > "$CMD" 2>/dev/null
launch(){ pgrep -x dash2 >/dev/null 2>&1 && return 0
  setsid /opt/hwbridge/dash2 "$1" </dev/null >>/opt/hwbridge/dash2.log 2>&1 & }
stop(){ pkill -TERM -x dash2 2>/dev/null; }
while true; do
  if [ -s "$CMD" ]; then
    c=$(head -n1 "$CMD" 2>/dev/null); : > "$CMD" 2>/dev/null
    case "$c" in
      on:*)     launch "${c#on:}";;
      off)      stop;;
      toggle:*) if pgrep -x dash2 >/dev/null 2>&1; then stop; else launch "${c#toggle:}"; fi;;
    esac
  fi
  sleep 1
done
