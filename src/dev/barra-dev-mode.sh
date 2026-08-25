#!/system/bin/sh
# barra Dev-Mode — versetzt den Node in einen Kernel-Entwicklungs-Zustand und wieder zurueck.
#   barra-dev-mode.sh on | off | status | apply
# on   : SELinux permissive + Takt-Pinning (Mali/rio-TPU/MIF/CPU=performance) + Wakelock,
#        speichert den VORzustand in $SAVED und setzt das Persistenz-Flag $FLAG.
# off  : stellt exakt den gespeicherten Vorzustand wieder her, loescht Flag + Wakelock.
# apply: wie 'on', aber nur wenn $FLAG gesetzt ist (Boot-Hook 55-barra-dev.sh ruft das).
# status: zeigt getenforce, Taktzustaende, Flag, Boot-Hook.
#
# WARNUNG: permissive + Dauer-Pin sind NUR fuer Dev-Geraete. Nie auf einem Produktiv-Node.
# Persistenz: 'on' ueberdauert den Reboot (Flag + service.d-Hook). 'off' beendet das dauerhaft.
#
# Snapshot/Restore statt fester Defaults: die server.sh-pins_off raten Defaults (schedutil/
# interactive) — hier merken wir uns den echten Vorzustand und legen ihn 1:1 zurueck.
#
# HINWEIS: Helfer heissen rd/wr (NICHT r/w): Android-mksh hat 'r' als Alias fuer 'fc -s'
# (History-Redo), Aliase gehen vor Funktionen -> 'r' waere ueberschattet.
unalias rd wr 2>/dev/null
D=/data/adb/baseos
FLAG=$D/dev-mode.on          # existiert => Dev-Mode aktiv (Persistenz)
SAVED=$D/dev-mode.saved      # Vorzustand fuer sauberes 'off'
F=/sys/class/devfreq
MALI=/sys/class/misc/mali0/device
RIO=$F/1a000000.rio
MIF=$F/17000010.devfreq_mif
WLNAME=barra-dev

rd(){ cat "$1" 2>/dev/null; }                   # sicheres Lesen (leer wenn gesperrt)
wr(){ echo "$2" > "$1" 2>/dev/null; }           # best-effort Schreiben

snapshot(){                                     # Vorzustand einfrieren (nur einmal, bei 'on')
  : > "$SAVED"
  echo "SELINUX=$(getenforce 2>/dev/null)"                     >> "$SAVED"
  echo "MALI_MIN=$(rd $MALI/scaling_min_freq)"                 >> "$SAVED"
  echo "MALI_HINT=$(rd $MALI/hint_min_freq)"                   >> "$SAVED"
  echo "RIO_GOV=$(rd $RIO/governor)"                           >> "$SAVED"
  echo "RIO_MIN=$(rd $RIO/min_freq)"                           >> "$SAVED"
  echo "MIF_GOV=$(rd $MIF/governor)"                           >> "$SAVED"
  i=0; for CP in /sys/devices/system/cpu/cpufreq/policy*; do
    echo "CPU${i}=$(rd $CP/scaling_governor)" >> "$SAVED"; i=$((i+1)); done
}

pins_on(){
  setenforce 0 2>/dev/null
  wr $MALI/scaling_min_freq 890000; wr $MALI/hint_min_freq 890000
  wr $RIO/governor performance; wr $RIO/min_freq 1119000000
  wr $MIF/governor performance
  for CP in /sys/devices/system/cpu/cpufreq/policy*; do wr $CP/scaling_governor performance; done
  wr /sys/power/wake_lock $WLNAME
}

restore(){                                      # exakter Rueckbau aus $SAVED (Fallback: HW-Idle)
  [ -f "$SAVED" ] && . "$SAVED" 2>/dev/null
  case "$SELINUX" in Permissive) setenforce 0 2>/dev/null;; *) setenforce 1 2>/dev/null;; esac
  wr $MALI/scaling_min_freq "${MALI_MIN:-150000}"; wr $MALI/hint_min_freq "${MALI_HINT:-150000}"
  [ -n "$RIO_GOV" ] && wr $RIO/governor "$RIO_GOV"; [ -n "$RIO_MIN" ] && wr $RIO/min_freq "$RIO_MIN"
  [ -n "$MIF_GOV" ] && wr $MIF/governor "$MIF_GOV"
  i=0; for CP in /sys/devices/system/cpu/cpufreq/policy*; do
    eval "g=\$CPU${i}"; [ -n "$g" ] && wr $CP/scaling_governor "$g"; i=$((i+1)); done
  wr /sys/power/wake_unlock $WLNAME
}

case "${1:-status}" in
  on)
    [ -f "$SAVED" ] || snapshot          # nicht ueberschreiben, falls schon aktiv
    pins_on; touch "$FLAG"
    echo "dev-mode AN (permissive + Pins + Wakelock, persistent). 'off' zum Beenden.";;
  apply)                                  # Boot-Hook: nur wenn beim letzten Mal aktiviert
    [ -f "$FLAG" ] || { echo "dev-mode aus (kein Flag) - nichts zu tun"; exit 0; }
    [ -f "$SAVED" ] || snapshot
    pins_on
    echo "dev-mode re-applied (Boot).";;
  off)
    restore; rm -f "$FLAG" "$SAVED"
    echo "dev-mode AUS - Vorzustand wiederhergestellt (enforcing + Default-Takt).";;
  status)
    echo "SELinux : $(getenforce 2>/dev/null)"
    echo "Mali    : min=$(rd $MALI/scaling_min_freq) hint=$(rd $MALI/hint_min_freq) cur=$(rd $MALI/cur_freq)"
    echo "rio-TPU : gov=$(rd $RIO/governor) min=$(rd $RIO/min_freq)"
    echo "MIF-Bus : gov=$(rd $MIF/governor)"
    echo "Flag    : $([ -f "$FLAG" ] && echo gesetzt\ \(persistent\ AN\) || echo nicht\ gesetzt)"
    echo "Boot-Hk : $([ -f /data/adb/service.d/55-barra-dev.sh ] && echo installiert || echo fehlt)";;
  *) echo "barra-dev-mode.sh on | off | status | apply";;
esac
