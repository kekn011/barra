#!/bin/bash
# Selbstheilendes DHCP fuer wlan0 UND USB-Ethernet (Container-Dienst wlan-dhcp.service).
# Sobald ein Interface UP ist aber keine IPv4 hat (oder sein dhclient fehlt), frisch dhclient
# (Daemon, renewt) starten. Faengt jede (Re-)Assoziation ab (wifi-join / wifi-guard / manuell),
# ohne bei Interface-Resets haengenzubleiben (dhclient re-discovert sonst nicht zuverlaessig).
#
# USB-Ethernet (30.8.: USB-C->RJ45-Adapter mit PD-Durchschleifen, r8152 fest im Kernel):
# eth*/enx*/usb* werden hochgefahren und per DHCP versorgt. Kabel vor Funk: die Default-Route
# ueber Ethernet bekommt Metrik 50, die von wlan0 wird auf 600 gesetzt — auch die metriklose
# onlink-Route, die fix-default-route.sh auf der Android-Seite anlegt. Ohne Adapter aendert
# sich nichts. Jeder dhclient bekommt eigene pid-/lease-Datei (zwei Instanzen mit dem Standard-
# pidfile wuerden sich gegenseitig abschiessen).
have_ip(){ ip -4 addr show "$1" 2>/dev/null | grep -q 'inet '; }
is_up(){ [ "$(cat /sys/class/net/$1/operstate 2>/dev/null)" = up ]; }
admin_up(){ [ $(( $(cat /sys/class/net/$1/flags 2>/dev/null || echo 0) & 1 )) -eq 1 ]; }
eth_ifs(){ ls /sys/class/net 2>/dev/null | grep -E '^(eth|enx|usb)[0-9a-z]*$'; }
dhcp_running(){ pgrep -f "dhclient.*[ ]$1\$" >/dev/null 2>&1; }
start_dhcp(){
  pkill -f "dhclient.*[ ]$1\$" 2>/dev/null; sleep 1
  dhclient -4 -nw -pf "/run/dhclient.$1.pid" -lf "/var/lib/dhcp/dhclient.$1.leases" "$1" 2>/dev/null
  logger -t barra-net "dhclient fuer $1 gestartet"
}
gw_of(){ awk '/option routers/{gsub(/[;,]/,"",$3); r=$3} END{print r}' "/var/lib/dhcp/dhclient.$1.leases" 2>/dev/null; }
prefer_cable(){
  local e gw r
  for e in $(eth_ifs); do
    is_up "$e" && have_ip "$e" || continue
    gw=$(gw_of "$e"); [ -n "$gw" ] || continue
    if ! ip -4 route show default dev "$e" 2>/dev/null | grep -q 'metric 50'; then
      ip -4 route del default dev "$e" 2>/dev/null
      ip -4 route replace default via "$gw" dev "$e" metric 50 2>/dev/null \
        && logger -t barra-net "Default-Route ueber $e via $gw (Metrik 50, Kabel vor Funk)"
    fi
    # ACHTUNG iproute2: 'ip route show ... dev X' laesst 'dev X' in der AUSGABE weg — beim
    # Wieder-Anlegen das Geraet explizit mitgeben (30.8.: add scheiterte still, del hatte gewirkt).
    r=$(ip -4 route show default dev wlan0 2>/dev/null | grep -v metric | head -1)
    if [ -n "$r" ]; then
      ip -4 route del $r dev wlan0 2>/dev/null; ip -4 route add $r dev wlan0 metric 600 2>/dev/null \
        && logger -t barra-net "wlan0-Default-Route auf Metrik 600 gesetzt"
    fi
    # Fallback selbstheilend: hat wlan0 eine IP, aber gar keine Default-Route mehr (z.B. nach dem
    # Fehler oben), aus der Lease eine mit Metrik 600 anlegen — Kabel ziehen darf nicht ins Leere fuehren.
    if have_ip wlan0 && ! ip -4 route show default dev wlan0 2>/dev/null | grep -q .; then
      gw=$(gw_of wlan0)
      [ -n "$gw" ] && ip -4 route add default via "$gw" dev wlan0 onlink metric 600 2>/dev/null \
        && logger -t barra-net "wlan0-Fallback-Route via $gw (Metrik 600) wiederhergestellt"
    fi
    # Auch die Netzroute von wlan0 nach hinten: beide Interfaces haengen im selben /24, sonst
    # antwortet der Node LAN-Nachbarn weiter ueber Funk, obwohl das Kabel steckt.
    c=$(ip -4 route show dev wlan0 scope link 2>/dev/null | grep -v -E 'metric|^default' | head -1)
    if [ -n "$c" ]; then
      ip -4 route del $c dev wlan0 2>/dev/null; ip -4 route add $c dev wlan0 metric 600 2>/dev/null \
        && logger -t barra-net "wlan0-Netzroute auf Metrik 600 gesetzt"
    fi
  done
}
reap_orphans(){  # dhclient, dessen Interface verschwunden ist (Adapter abgezogen)
  local pf i
  for pf in /run/dhclient.*.pid; do
    [ -f "$pf" ] || continue
    i=${pf#/run/dhclient.}; i=${i%.pid}
    [ -e "/sys/class/net/$i" ] || { kill "$(cat "$pf")" 2>/dev/null; rm -f "$pf"; logger -t barra-net "dhclient fuer $i beendet (Interface weg)"; }
  done
}
while true; do
  if is_up wlan0 && { ! have_ip wlan0 || ! dhcp_running wlan0; }; then
    start_dhcp wlan0; sleep 8
  fi
  for e in $(eth_ifs); do
    admin_up "$e" || { ip link set "$e" up 2>/dev/null && logger -t barra-net "$e hochgefahren"; }
    if is_up "$e" && { ! have_ip "$e" || ! dhcp_running "$e"; }; then
      start_dhcp "$e"; sleep 8
    fi
  done
  prefer_cable
  reap_orphans
  sleep 4
done
