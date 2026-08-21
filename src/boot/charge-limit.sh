#!/system/bin/sh
# Begrenzt das Laden auf einen akkuschonenden Bereich (Pixel-Ladesteuerung).
#
# Li-Ion altert am schnellsten bei hohem Ladestand und Waerme. Ein dauerhaft am
# Netz haengender Node wuerde sonst bei 100% stehen und den Akku aufblaehen.
# 65/60 als Vorgabe, weil der Akku zugleich als USV dient: bei Stromausfall
# laeuft der Node bei 60-65% noch mehrere Stunden weiter.
#
# Doppelrolle:
#   - manuell:   adb shell "/debug_ramdisk/su -c 'sh /sdcard/charge-limit.sh'"
#   - bei Boot:  liegt als /data/adb/service.d/charge-limit.sh (Magisk, als root)
#
# GEAENDERT 13.8.: die Werte standen frueher FEST in diesem Skript. Damit war
# Schritt 2 des Setup-Assistenten ueber einen Neustart hinaus wirkungslos - er
# schreibt die Schwellen nur ins sysfs, und hier kamen dann wieder 60/65. Jetzt
# kommen sie aus dem Config-Store; die alten Werte bleiben die Vorgabe, falls
# der Store fehlt (frisch geflashtes Geraet) oder Unsinn enthaelt.
CHG=/sys/devices/platform/google,charger
# Image-fester Pfad! Bis 21.8. stand hier /data/local/tmp/baseos-config.sh —
# Dev-Scratch, auf frisch geflashten Geraeten LEER -> get schlug still fehl und
# es galten IMMER die 60/65-Vorgaben, egal was das Setup eingestellt hatte.
CFGTOOL=/data/adb/baseos/bin/baseos-config.sh

START=$(sh "$CFGTOOL" get CHARGE_START 60 2>/dev/null)
STOP=$(sh  "$CFGTOOL" get CHARGE_STOP  65 2>/dev/null)

# Pruefen statt vertrauen: eine kaputte Config darf die Ladesteuerung nicht
# aushebeln. Unsinnige Werte werden NICHT zurechtgebogen, sondern durch die
# Vorgabe ersetzt - "999" auf 100 zu klemmen hiesse, den Akkuschutz abzuschalten,
# also genau das Gegenteil von dem, wofuer das Skript da ist.
echo "$START" | grep -qE '^[0-9]+$' || START=60
echo "$STOP"  | grep -qE '^[0-9]+$' || STOP=65
{ [ "$START" -lt 40 ] || [ "$START" -gt 90 ]; } && START=60
{ [ "$STOP"  -lt 45 ] || [ "$STOP"  -gt 100 ]; } && STOP=65
[ "$STOP" -le "$START" ] && { START=60; STOP=65; }

# Auf den Treiber warten (max ~40 s), falls beim Boot noch nicht da
i=0
while [ ! -w "$CHG/charge_stop_level" ] && [ $i -lt 40 ]; do
    sleep 2; i=$((i + 2))
done
[ -w "$CHG/charge_stop_level" ] || { echo "charger-sysfs nicht schreibbar"; exit 1; }

# Reihenfolge haengt von der RICHTUNG ab. Der Treiber lehnt jeden Schreibvorgang
# wortlos ab, der start >= stop ergaebe - auch nur voruebergehend. "start zuerst"
# stimmt deshalb nur beim Senken des Fensters; beim Anheben muss stop vorangehen.
# (Am 13.8. gemessen: 60/65 -> 70/75 uebernahm nur die 75, start blieb auf 60.)
CUR_STOP=$(cat "$CHG/charge_stop_level" 2>/dev/null)
if [ -n "$CUR_STOP" ] && [ "$STOP" -gt "$CUR_STOP" ]; then
    echo "$STOP"  > "$CHG/charge_stop_level"
    echo "$START" > "$CHG/charge_start_level"
else
    echo "$START" > "$CHG/charge_start_level"
    echo "$STOP"  > "$CHG/charge_stop_level"
fi

echo "charge_start_level = $(cat $CHG/charge_start_level)"
echo "charge_stop_level  = $(cat $CHG/charge_stop_level)"
echo "Ladestand          = $(cat /sys/class/power_supply/battery/capacity)%"
echo "Status             = $(cat /sys/class/power_supply/battery/status)"
