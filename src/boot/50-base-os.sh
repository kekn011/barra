#!/system/bin/sh
# Magisk service.d (late_start, root, globaler Mount-NS):
# bringt nach dem Neustart das Base-OS "barra" hoch (Ubuntu-Userland + WLAN + Dashboard-Uebergabe).
# Detached, damit der service.d-Ablauf nicht blockiert.
# Alle Base-Programme liegen Image-fest unter /data/adb/baseos/bin (nicht in /data/local/tmp).
#
# Notbremse: touch /data/adb/baseos/disable  -> normales Android beim naechsten Start.
setsid sh /data/adb/baseos/bin/base-boot.sh </dev/null >/dev/null 2>&1 &
