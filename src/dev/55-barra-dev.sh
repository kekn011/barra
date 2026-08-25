#!/system/bin/sh
# Magisk service.d (late_start, root, globaler Mount-NS) — barra Dev-Mode nach dem Boot
# wieder anwenden, FALLS er zuletzt aktiv war ($D/dev-mode.on). Laeuft nach 50-base-os.sh.
# Kein Effekt, wenn dev-mode aus ist. Reversibel: 'barra-dev-mode.sh off' loescht das Flag.
# Diese Datei wird von install-dev.ps1 abgelegt und beim Kit-Deinstall wieder entfernt.
( i=0; while [ $i -lt 120 ]; do [ "$(getprop sys.boot_completed 2>/dev/null)" = 1 ] && break; sleep 1; i=$((i+1)); done
  # Zusaetzlich auf base 'fertig' warten: base-boot fasst Mali NACH boot_completed nochmal an
  # (setzt hint_min_freq zurueck) -> sonst Race mit unserem Pin. Bounded (~3 min).
  j=0; while [ $j -lt 90 ]; do [ "$(cat /data/adb/baseos/state 2>/dev/null)" = fertig ] && break; sleep 2; j=$((j+1)); done
  setsid sh /data/adb/baseos/bin/barra-dev-mode.sh apply </dev/null >>/data/adb/baseos/dev-mode.log 2>&1 ) &
