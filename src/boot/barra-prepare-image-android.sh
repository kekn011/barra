#!/system/bin/sh
# barra-prepare-image-android.sh - ANDROID-seitiges Gegenstueck zu barra-prepare-image (Container).
# Als root (su) ausfuehren, unmittelbar bevor das Image gezogen wird. Entfernt alles
# Node-Spezifische/Geheime der Android-Seite. Idempotent.
D=/data/adb/baseos; S=$D/config; H=/data/adb/hwbridge
echo "== barra-prepare-image (android) =="
# 1) WLAN-Zugangsdaten aus dem Store (Heimnetz-Passwort darf NICHT ins Image)
if [ -f "$S" ]; then
  grep -v '^WIFI_SSID=\|^WIFI_PSK=' "$S" > "$S.tmp"; mv "$S.tmp" "$S"; chmod 600 "$S"; chown 0:0 "$S"
  echo "WLAN-Creds aus Store entfernt; Rest: $(cut -d= -f1 "$S" | tr '\n' ' ')"
fi
# 2) alte Entwicklungs-Creds/-Dateien
rm -f /data/local/tmp/wifi_ssid /data/local/tmp/wifi_psk /data/local/tmp/wifi_result 2>/dev/null
# 3) Laufzeit + Zustand + Logs
rm -rf "$D/run"; mkdir -p "$D/run"; chmod 700 "$D/run"
rm -f "$D/state" "$D/disable" 2>/dev/null; rm -rf "$D/boot.lock" 2>/dev/null
: > "$D/boot.log" 2>/dev/null
for f in hwbridge.log cfgd.log btnd.log dash2.log disphold.log supervisor.pid cfgd.pid; do : > "$H/$f" 2>/dev/null; done
rm -f "$H/dash-run.sh" 2>/dev/null
# 4) wpa-Conf zurueck auf Original (ohne unser Netz), Sockets weg
[ -f "$D/run/wpa_supplicant.orig.conf" ] || true
rm -rf /data/vendor/wifi/wpa/sockets 2>/dev/null
# 5) Boot-Anim/Restbilder etc. gibt es nicht mehr; Magisk-Module-Reste pruefen
[ -d /data/adb/modules/bootanimbar ] && rm -rf /data/adb/modules/bootanimbar && echo "bootanimbar-Modul entfernt"
echo "Android-Seite ist image-bereit. (Container: sudo barra-prepare-image)"
