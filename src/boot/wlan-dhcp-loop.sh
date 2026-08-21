#!/bin/bash
# Selbstheilendes DHCP fuer wlan0 (Container-Dienst wlan-dhcp.service).
# Sobald das Interface UP ist aber keine IPv4 hat, frisch dhclient (Daemon, renewt) starten.
# Faengt jede (Re-)Assoziation ab (wifi-join / wifi-guard / manuell), ohne bei
# Interface-Resets haengenzubleiben (dhclient re-discovert sonst nicht zuverlaessig).
have_ip(){ ip -4 addr show wlan0 2>/dev/null | grep -q 'inet '; }
is_up(){ [ "$(cat /sys/class/net/wlan0/operstate 2>/dev/null)" = up ]; }
while true; do
  if is_up && ! have_ip; then
    pkill -x dhclient 2>/dev/null; sleep 1
    dhclient -4 -nw wlan0 2>/dev/null
    sleep 8
  fi
  sleep 4
done
