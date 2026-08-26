#!/bin/bash
# repack-base.sh - das Base-Image umpacken, ohne es kaputtzumachen.
#
# WARUM ES DIESES SKRIPT GIBT
# Am 26.8.2026 wurde barra-base.tar.gz von Hand umgepackt, um Vendor-Material fuer die
# Lizenzklaerung zu entfernen. Der Vorgang lief als normaler Benutzer - damit hat tar den
# Eigentuemer des Packenden eingetragen und ALLE setuid-Bits verworfen. Ergebnis im
# veroeffentlichten v0.1.0: das gesamte Container-Dateisystem gehoerte uid 1000, sudo, su,
# passwd und mount waren unbrauchbar, und der Benutzer kam nicht in sein eigenes Home -
# waehrend CONTRIBUTING.md genau "ssh you@your-node" plus sudo als Entwicklerweg nennt.
#
# Der Fehler war nur deshalb moeglich, weil der Vorgang nirgends aufgeschrieben war. Also:
# aufschreiben, als root ausfuehren, und danach pruefen statt hoffen.
#
#   sudo bash repack-base.sh <alt.tar.gz> <neu.tar.gz> [<datei-zum-entfernen> ...]
#
# Muss in WSL oder einem Linux laufen (auf NTFS gibt es keine Eigentuemer und keine
# setuid-Bits) und MUSS als root laufen, sonst bricht es sofort ab.
set -euo pipefail

ALT=${1:-}; NEU=${2:-}; shift 2 || true
[ -n "$ALT" ] && [ -n "$NEU" ] || { sed -n '2,20p' "$0"; exit 1; }
# Die Rechtefrage zuerst: sie ist die Ursache des Fehlers, den dieses Skript verhindern soll.
# Stuende die Dateipruefung davor, verdeckte eine fehlende Datei den eigentlichen Grund.
[ "$(id -u)" = "0" ] || { echo "FEHLER: als root ausfuehren - sonst gehen Eigentuemer und setuid-Bits verloren (siehe Kopf dieser Datei)."; exit 1; }
[ -f "$ALT" ] || { echo "FEHLER: $ALT nicht gefunden"; exit 1; }
case "$(uname -r)" in *icrosoft*|*WSL*) ;; *) [ -d /proc ] || { echo "FEHLER: Linux noetig"; exit 1; };; esac

# Diese Programme muessen in einem Ubuntu 24.04 setuid-root sein. Die Liste stammt aus
# einem echten Ubuntu 24.04 (find /usr/bin /usr/sbin -perm -4000), nicht aus dem Gedaechtnis.
PFLICHT_SETUID="usr/bin/sudo usr/bin/su usr/bin/passwd usr/bin/mount usr/bin/umount \
usr/bin/chfn usr/bin/chsh usr/bin/gpasswd usr/bin/newgrp usr/bin/fusermount3"

W=$(mktemp -d /tmp/repack-base.XXXX)
trap 'rm -rf "$W"' EXIT
cd "$W"

echo "== auspacken (Eigentuemer und Modi bleiben erhalten) =="
tar --same-owner --numeric-owner -xzpf "$ALT"
echo "   $(find . | wc -l) Eintraege"

for weg in "$@"; do
  if [ -e "$weg" ]; then
    echo "== entfernen: $weg"
    rm -rf "$weg"
  else
    echo "== nicht vorhanden, uebersprungen: $weg"
  fi
done

echo "== packen =="
tar --numeric-owner -czf "$NEU" ubuntu adb

echo "== Waechter: stimmen Eigentuemer und setuid-Bits im ERGEBNIS? =="
LISTE=$(tar -tzvf "$NEU")
fehler=0
for f in $PFLICHT_SETUID; do
  zeile=$(printf '%s\n' "$LISTE" | grep -E " ubuntu/$f\$" || true)
  if [ -z "$zeile" ]; then
    echo "   fehlt (nicht im Image): $f"
    continue
  fi
  rechte=$(printf '%s' "$zeile" | awk '{print $1}')
  eigner=$(printf '%s' "$zeile" | awk '{print $2}')
  if [ "$rechte" != "-rwsr-xr-x" ] || [ "$eigner" != "0/0" ]; then
    echo "   FEHLER: $f ist '$rechte $eigner', erwartet '-rwsr-xr-x 0/0'"
    fehler=1
  fi
done
if [ "$fehler" != "0" ]; then
  echo
  echo "ABBRUCH: das Archiv waere so unbrauchbar - genau der Fehler vom 26.8.2026."
  echo "Vermutlich lief das Auspacken nicht als root. $NEU wird geloescht."
  rm -f "$NEU"
  exit 1
fi
echo "   alle Pflicht-Programme sind setuid-root."

echo "== fertig =="
ls -l "$NEU" | awk '{printf "   %d Bytes\n", $5}'
sha256sum "$NEU"
echo
echo "Nicht vergessen: SHA256SUMS und models.psd1 (bytes + sha256) nachziehen,"
echo "sonst weist das Setup die Datei als beschaedigt zurueck."
