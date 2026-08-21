#!/system/bin/sh
# Magisk service.d: stoppt nach dem Boot die fuer einen headless Compute-Node
# unnoetigen Android-HAL-Dienste (Media/Kamera/Biometrie/BT/NFC/DRM/GPS/Sensoren/
# SIM-SE + Telemetrie/OTA). GEPRUEFT SICHER 15.8. (Gerät blieb stabil, alle
# Beschleuniger + Brücken intakt). Reversibel: Datei entfernen -> Reboot stellt alles her.
# Stufe 6 (19.8., live verifiziert: Bruecken intakt, MemAvailable +~150 MB): Modem/Telefonie, Keystore/
# Keymint/Citadel, NNAPI-darwinn + edgetpu_app_service (tpud spricht /dev/edgetpu-soc direkt), drm, gpu (gpuservice),
# input.processor, installd, storageproxyd. Weiter NICHT angefasst: twoshay, Thermal/Battery, Audio-AoC, vold, adbd.
# Dazu RAM-Tuning: vm.watermark_scale_factor 200 -> 10 (+~400 MB MemAvailable, reine Reserve-Buchhaltung) und
# drop_caches=2 nach dem Debloat (leert die dma-heap-Seitenpools, +~200 MB; fuellen sich nur bei HAL-Aktivitaet).
LOG=/data/adb/hwbridge/debloat.log
[ -f /data/adb/baseos/disable ] && exit 0

avail(){ awk '/MemAvailable/{printf "%.0f",$2/1024}' /proc/meminfo; }

# auf Boot warten
i=0; while [ "$(getprop sys.boot_completed)" != "1" ] && [ $i -lt 150 ]; do sleep 2; i=$((i+1)); done
sleep 10   # HALs erst hochkommen lassen (kurzer Peak ist ok, wir stoppen gleich)

SVCS="
android-hardware-media-c2-hal-1-2
android-hardware-media-c2-hal-2-0-google
media
media.swcodec
mediaextractor
mediametrics
vendor.cas-default
vendor.drm-clearkey-service
vendor.drm-widevine-hal
cameraserver
vendor.face-hal
vendor.fingerprint-goodix
bcmbtlinux
nfc_hal_service
pixel.gnss-default
slsi_gnss_service
gnssd
vendor.vibrator.cs40l26
wireless_charger_AIDL
vendor.contexthub-default
vendor.sensors-hal-multihal
gto_secure_element_aidl_service
secure_element_uicc_hal_service-sim1
secure_element_uicc_hal_service-sim2
statsd
traced
traced_probes
trusty_metricsd
incidentd
vendor.sscoredump
vendor.pixelstats_vendor
update_engine
vendor.dumpstate-default
credstore
mdnsd
ril-daemon
init.shared_modem_platform
vendor.modem_ml_svc_sit
radio_ext
vendor.google.radioext@1.0
hal_neuralnetworks_darwinn
edgetpu_app_service
edgetpu_logging
drm
gpu
vendor.input.processor
installd
keystore2
vendor.citadeld
vendor.keymint-citadel
vendor.keymint.rust-trusty
vendor.authsecret_hal_aidl
vendor.oemlock_hal_aidl
vendor.weaver_hal_aidl
storageproxyd
"
B=$(avail)
# zwei Durchlaeufe gegen etwaige Restart-Trigger
for pass in 1 2; do
  for s in $SVCS; do setprop ctl.stop "$s"; done
  sleep 4
done
# RAM-Tuning: Reserve-Watermarks auf Kernel-Default, dma-heap-Pools leeren
echo 10 > /proc/sys/vm/watermark_scale_factor 2>/dev/null
sync; echo 2 > /proc/sys/vm/drop_caches 2>/dev/null; sleep 2
# Pools fuellen sich in der Boot-Nachphase (Audio-HAL) teils nach -> spaeter noch einmal leeren
( sleep 90; echo 2 > /proc/sys/vm/drop_caches 2>/dev/null ) </dev/null >/dev/null 2>&1 &
A=$(avail)
echo "[$(date '+%m-%d %H:%M:%S')] debloat: MemAvailable $B -> $A MB (+$((A-B))), watermark_scale_factor=$(cat /proc/sys/vm/watermark_scale_factor), ION_heap_pool=$(awk '/ION_heap_pool/{print $2}' /proc/meminfo)kB" >> "$LOG"
