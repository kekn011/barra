#!/system/bin/sh
# WLAN einem neuen Netz beitreten (headless, ohne Android-WiFi-Framework).
# Wird vom Setup-Assistenten (Schritt 3 "Netzwerk") aufgerufen.
#
# SSID/PSK aus Dateien: /data/local/tmp/wifi_ssid  /data/local/tmp/wifi_psk
# Ergebnis -> /data/local/tmp/wifi_result :  "OK <ip>" | "FAIL <grund>"
#
# Verifiziert 2026-08-12 (Join in Kevins Heimnetz + DHCP + Internet). Die vier
# Dinge, an denen die erste Fassung scheiterte:
#  1. Conf MUSS in /data/vendor/wifi/wpa/ liegen (Kontext wpa_data_file). Aus
#     /data/local/tmp (shell_data_file) liest der Supplicant sie nicht - er
#     startet, meldet aber nichts und assoziiert nie.
#  2. ctrl_interface gehoert ins selbe Verzeichnis (dort legt er die Sockets an).
#     Nur in /data/local/tmp scheiterte es an chmod-EPERM - nicht generell.
#  3. country=DE: mit country=00 (World) sind alle 5-GHz-Kanaele NO_IR, d.h.
#     kein aktives Scannen -> 5-GHz-Netze bleiben unsichtbar.
#  4. Erfolg NICHT an logcat messen: der Supplicant loggt headless GAR NICHTS.
#     Messgroesse ist operstate (dormant -> up) und danach die DHCP-Lease.
#  Und: Verbindung != nutzbar. Ohne netd fehlen Default-Route und die Regel,
#  die normalen Traffic ueberhaupt nach 'main' schickt (sonst ENETUNREACH).
TMP=/data/adb/baseos/run   # Laufzeit-Files (wifi_result, Logs)
BIN=/data/adb/baseos/bin   # Programme (wlpm, wpactl, Skripte)
VDIR=/data/vendor/wifi/wpa
BAK=$TMP/wpa_supplicant.orig.conf
CONF=$VDIR/wpa_supplicant.conf
RES=$TMP/wifi_result
IFACE=wlan0
SUPPL=/vendor/bin/hw/wpa_supplicant
COUNTRY=DE

fail(){ echo "FAIL $1" > "$RES"; echo "FAIL $1"; exit 1; }

# Creds BEVORZUGT aus dem root-Config-Store (barra-config "WLAN einrichten"; 600, root),
# Rueckfall auf die alten Dateien in /data/local/tmp (Entwicklungsstand, gehoeren nicht ins Image).
STORE=/data/adb/baseos/config
SSID=$(sed -n 's/^WIFI_SSID=//p' "$STORE" 2>/dev/null | head -1)
PSK=$(sed -n 's/^WIFI_PSK=//p' "$STORE" 2>/dev/null | head -1)
if [ -z "$SSID" ]; then
  SSID=$(cat "$TMP/wifi_ssid" 2>/dev/null); PSK=$(cat "$TMP/wifi_psk" 2>/dev/null)
fi
[ -z "$SSID" ] && fail "kein WLAN eingerichtet (barra-config -> WLAN -> Netz einrichten)"
echo "verbinde mit $SSID ..." > "$RES"

# Restore-Punkt (Original-Conf ohne network-Bloecke) einmalig sichern
[ -f "$BAK" ] || cp "$CONF" "$BAK" 2>/dev/null

# Android-init-Supplicant stoppen, sonst respawnt init ihn und zwei Instanzen
# kaempfen um wlan0.
if [ "$(getprop init.svc.wpa_supplicant)" = "running" ]; then
  setprop ctl.stop wpa_supplicant
  i=0; while [ "$(getprop init.svc.wpa_supplicant)" = "running" ] && [ $i -lt 8 ]; do sleep 1; i=$((i+1)); done
fi
pkill -x wpa_supplicant 2>/dev/null; sleep 1

# Interface einmal zuruecksetzen -> die MAC ab Werk kommt zurueck.
# Warum (13.8. spaet geklaert): Androids Framework setzt beim Start seine
# eigene, pro Netzwerk gewuerfelte MAC auf wlan0. Unser Join fasst die
# Interface-Adresse nie an, uebernimmt sie also - und der Router vergibt bei
# jedem Neustart eine andere IP (.175 -> .177 -> ...). Fuer einen Server, den
# man unter fester Adresse erreichen will, unbrauchbar.
# `mac_addr=0` in der Conf ist NICHT schuld, das funktioniert: nach einem
# Interface-Reset steht dort e8:d5:2b:38:92:0d (einzige Adresse ohne das
# "lokal verwaltet"-Bit) und der Supplicant laesst sie in Ruhe.
# Achtung: waehrend des down meldet bcmdhd kurz "No such device" - das ist
# normal, das Interface kommt beim up zurueck.
ip link set "$IFACE" down 2>/dev/null; sleep 1
ip link set "$IFACE" up   2>/dev/null; sleep 1

mkdir -p "$VDIR/sockets" 2>/dev/null
chown wifi:wifi "$VDIR/sockets" 2>/dev/null
chmod 770 "$VDIR/sockets" 2>/dev/null

# Conf bauen: Chip-Header aus dem Backup + Land + unser Netz
{
  sed -e '/^ctrl_interface/d' -e '/^country=/d' -e '/^mac_addr/d' -e '/^preassoc_mac_addr/d' -e '/^network=/,$d' "$BAK"
  echo "ctrl_interface=$VDIR/sockets"
  echo "country=$COUNTRY"
  # MAC NICHT randomisieren: sonst wuerfelt jeder Supplicant-Start eine neue
  # lokal verwaltete MAC, der Router sieht jedesmal ein neues Geraet und vergibt
  # eine neue IP (beobachtet 12.8.: .173 -> .176). Feste Leases, Portfreigaben
  # und der ARP-Cache der Gegenstelle laufen dann ins Leere. Ein Server soll
  # unter derselben Adresse erreichbar bleiben.
  echo "mac_addr=0"
  echo "preassoc_mac_addr=0"
  echo ""
  echo "network={"
  echo "    ssid=\"$SSID\""
  echo "    mac_addr=0"
  if [ -n "$PSK" ]; then
    echo "    psk=\"$PSK\""
    echo "    key_mgmt=WPA-PSK WPA-PSK-SHA256 SAE"
    echo "    ieee80211w=1"
  else
    echo "    key_mgmt=NONE"
  fi
  echo "    scan_ssid=1"
  echo "}"
} > "$CONF"
chown wifi:wifi "$CONF"; chmod 660 "$CONF"

ip addr flush dev "$IFACE" 2>/dev/null       # alte statische IP weg: Erfolg haengt an DHCP
# Schnittstelle selbst hochziehen. Beim Aufruf aus dem Assistenten hatte das
# noch Androids Framework erledigt; im Boot-Ablauf (base-boot.sh) ist das
# Framework zu diesem Zeitpunkt schon gestoppt und es macht es NIEMAND mehr.
# Ohne UP scheitert der Join stumm (13.8. genau so passiert).
ip link set "$IFACE" up 2>/dev/null
# WIE der Supplicant gestartet wird, ist NICHT entscheidend - GEDULD ist es.
# 14 Laeufe am 13.8. gemessen: die Assoziation kam nach 58 s, 105 s, 114 s und
# 142 s. Alle Fehlschlaege waren Laeufe, die vorher aufgegeben haben. Auf dem
# Weg dahin sah es lange nach einem Muster aus (-B/setsid/nohup scheitern,
# "( timeout ... ) &" klappt) - das war ein Trugschluss aus zu kurzen Fenstern:
# derselbe Aufruf, der 4x "funktionierte", ist danach mit 150 s Fenster
# gescheitert, und die Form mit Conf-Neuschreibung hat es mit 142 s geschafft.
# Also nicht wieder an der Startform herumbauen - das Fenster ist die Stellschraube.
#
# Der Supplicant loggt dabei headless GAR NICHTS (auch mit -d keine Zeile), und
# ueber den ctrl-Socket kommen keine Antworten zurueck - man sieht nur operstate.
#
# KEIN timeout um den Supplicant! Hier stand mal "timeout 900" mit der Begruendung,
# die Verbindung ueberlebe sein Ende ohnehin (Firmware halte die Assoziation).
# WIDERLEGT am 13.8. abends im echten Netz: Der Join lief sauber (Lease .175,
# Gateway 10-59 ms), und ziemlich genau 15 Minuten spaeter war das Netz tot -
# Ping 100% Verlust, waehrend IP, Route, operstate=up und carrier=1 munter
# weiter "alles gut" meldeten. Beweisprobe: NUR den Supplicant zurueckgestartet,
# nichts sonst angefasst -> Gateway antwortete nach 1 s wieder, gleiche IP.
# Der Supplicant muss dauerhaft laufen: er haelt die Assoziation und macht die
# Key-Rekeys (die Fritzbox rekeyt den Gruppenschluessel im Minutenbereich).
# Die alte "13 h ohne Supplicant"-Beobachtung war wieder ein Buero-Artefakt -
# ein Geraet auf dem statischen Fallback .118 sieht genauso aus.
( "$SUPPL" -i"$IFACE" -Dnl80211 -c"$CONF" >/dev/null 2>&1 ) &
sleep 3
pgrep -x wpa_supplicant >/dev/null || fail "supplicant-start"

# CHIP AUFWECKEN - "frischer-Reboot-Falle" (14.8. abends, im echten Heimnetz gefunden).
# Nach dem Framework-Stop steht der bcmdhd auf country=US + mpc=1 (Minimum Power
# Consumption: Radio schlaeft im Nicht-Assoziiert-Zustand) + Suspend. Folge: der
# SCAN findet das Netz GAR NICHT ("kein Netz gefunden"), obwohl es in Reichweite
# ist. Bei langer Uptime faellt das nicht auf (Android hatte den Chip mal auf DE
# gesetzt und wach gehalten) - aber nach frischem Reboot scheitern Boot-Join UND
# wifi-guard reproduzierbar. Verifiziert: mit diesen drei Stellschrauben
# assoziiert der Chip SOFORT (RSSI meldet die SSID). country=US->DE, mpc 1->0.
# Muss VOR der Scan-Schleife stehen und der SCAN selbst haelt den Chip dann wach.
if [ -x "$BIN/wlpm" ]; then
  "$BIN/wlpm" cmd "COUNTRY $COUNTRY"  >/dev/null 2>&1
  "$BIN/wlpm" iovar mpc 0             >/dev/null 2>&1
  "$BIN/wlpm" cmd "SETSUSPENDMODE 0"  >/dev/null 2>&1
fi

# 1) Assoziation: operstate wechselt dormant -> up (logcat ist headless stumm).
#    ENTSCHEIDEND: Der Supplicant scannt hier von sich aus NICHT - er bleibt still
#    auf dormant stehen. Erst ein SCAN ueber den Control-Socket bringt ihn dazu,
#    das Netz zu suchen und zu assoziieren. Die Antwort des Sockets kommt nicht
#    zurueck (SELinux), das Kommando wird aber ausgefuehrt -> "-q" (fire&forget).
#    Fenster 240s statt 40s und SCAN alle 4s statt 8s.
#    ZAHLEN KORRIGIERT (13.8. abends, echtes Netz): die Assoziation kommt nach
#    8 s, der ganze Join ist nach 28 s durch. Die frueher notierten "58-142 s,
#    streut stark" waren gar keine Assoziationen, sondern das operstate-Flackern
#    ohne Netz in Reichweite. Das 240-s-Fenster bleibt trotzdem: es kostet nur
#    im Fehlerfall Zeit und puffert schwaches Signal und Bootlast ab.
i=0
while [ $i -lt 240 ]; do
  [ "$(cat /sys/class/net/$IFACE/operstate 2>/dev/null)" = "up" ] && break
  [ $((i % 4)) -eq 0 ] && "$BIN/wpactl" -q SCAN >/dev/null 2>&1
  sleep 1; i=$((i+1))
done
#    ACHTUNG: operstate=up ist NUR ein Hinweis, KEIN Beweis. Am 13.8. stand es
#    auf "up", waehrend gar kein Netz in Reichweite war und das Gateway stumm
#    blieb. Deshalb wird hier nicht mehr abgebrochen und vor allem nichts als
#    Erfolg gemeldet - das entscheidet erst Schritt 4.
if [ "$(cat /sys/class/net/$IFACE/operstate 2>/dev/null)" != "up" ]; then
  echo "keine Assoziation nach ${i}s, versuche trotzdem DHCP..." >> "$RES"
fi

# 2) DHCP via systemd-networkd im Container (teilt die netns mit dem Host)
#    Vorher flushen: sonst haelt man den statischen Offline-Fallback aus
#    boot-node.sh (192.168.178.118) fuer eine frische Lease.
ip addr flush dev "$IFACE" 2>/dev/null
sh "$BIN/enter-systemd.sh" "$BIN/inner-wifi-dhcp.sh" >"$TMP/wifi_dhcp.log" 2>&1

i=0; IP=""
while [ $i -lt 25 ]; do
  IP=$(ip -o -4 addr show $IFACE 2>/dev/null | awk '{print $4}' | head -1)
  [ -n "$IP" ] && break
  sleep 1; i=$((i+1))
done
[ -z "$IP" ] && fail "kein Netz gefunden (SSID in Reichweite? Passwort richtig?)"

# 3) Nutzbar machen: Default-Route + generelle main-Regel (ohne netd macht das niemand)
sh "$BIN/fix-default-route.sh" >/dev/null 2>&1
ip rule del pref 29000 2>/dev/null
ip rule add from all lookup main pref 29000 2>/dev/null

# 3b) ERREICHBARKEIT VON AUSSEN herstellen (13.8. abends gefunden).
#     Ohne das hier ist akita zwar online, aber fuer JEDEN unerreichbar: der
#     Treiber steht im Suspend-Zustand und laesst nur Unicast durch. ARP-Anfragen
#     sind Broadcasts -> kommen nie an -> niemand kann die MAC aufloesen -> kein
#     Ping, kein SSH. Nach aussen sieht alles gesund aus (Lease, Route, Gateway
#     antwortet, curl laeuft), weil das alles Antworten auf EIGENE Anfragen sind.
#     Gemessen: 0/40 ICMP und 0/10 auf Port 22 vom PC, `rcv_probes_mcast = 0`,
#     RX-Zaehler ruehrt sich nicht. Danach 8/8 und 5/5.
#     Sonst macht das Androids WiFi-HAL beim Bildschirm-An - der ist aus.
#     Isoliert bewiesen (frische IPs, damit kein ARP-Cache mitspielt):
#       SETSUSPENDMODE 1 -> 0/6 | SETSUSPENDOPT 0 -> 0/6 | SETSUSPENDMODE 0 -> 6/6
#     Warum der ioctl mal "Permission denied" liefert und mal nicht, ist NICHT
#     geklaert (Verdacht: der Treiber verweigert im Suspend selbst). Deshalb der
#     Umweg ueber setenforce - aber nur, wenn es anders nicht geht, und mit
#     garantierter Rueckstellung.
if [ -x "$BIN/wlpm" ]; then
  if ! "$BIN/wlpm" cmd "SETSUSPENDMODE 0" >/dev/null 2>&1; then
    VORHER=$(getenforce 2>/dev/null)
    setenforce 0 2>/dev/null
    "$BIN/wlpm" cmd "SETSUSPENDMODE 0" >/dev/null 2>&1
    [ "$VORHER" = "Enforcing" ] && setenforce 1 2>/dev/null
  fi
  if "$BIN/wlpm" cmd "SETSUSPENDMODE 0" >/dev/null 2>&1; then
    echo "  Suspend-Modus aus (von aussen erreichbar)" >> "$RES"
  else
    echo "  WARNUNG: Suspend-Modus liess sich nicht abschalten - Geraet ist" >> "$RES"
    echo "  moeglicherweise von aussen NICHT erreichbar (nur ausgehend)." >> "$RES"
  fi
else
  echo "  WARNUNG: wlpm fehlt - Erreichbarkeit von aussen ungeklaert" >> "$RES"
fi

# 4) DAS Erfolgskriterium: antwortet das Gateway?
#    Frueher galt schon die Lease als Erfolg und der stumme Gateway war eine
#    Randnotiz - so konnte "OK" gemeldet werden, obwohl gar nichts stand
#    (13.8.). Eine Lease ohne erreichbares Gateway ist kein nutzbares Netz,
#    also ist das hier ein FAIL. base-boot.sh haengt seinen Rueckfall daran.
GW=$(ip route show table main 2>/dev/null | awk '/^default/{print $3; exit}')
[ -z "$GW" ] && fail "keine Default-Route (IP ${IP%%/*})"
i=0
while [ $i -lt 3 ]; do
  if ping -c 1 -W 3 "$GW" >/dev/null 2>&1; then
    echo "OK ${IP%%/*}" > "$RES"; echo "OK ${IP%%/*}"; exit 0
  fi
  sleep 2; i=$((i+1))
done
fail "Gateway $GW antwortet nicht (IP ${IP%%/*} - Netz nicht nutzbar)"
