#!/bin/bash
# barra-kernel-update.sh - Kernel-Update eines laufenden Nodes OHNE Kabel (im Container, als root).
#   barra-kernel-update.sh <dir>   dir enthaelt: boot-lz4.img, wifi-rfkill.ko, optional *.ko (k8s-Module)
#
# 1. boot-lz4.img per dd in den aktiven Slot (Sicherung der Partition nach /var/lib/barra/boot_<slot>.bak,
#    Rueckleseprobe per SHA-256).
# 2. wifi-rfkill.ko auf die ANDROID-Seite (/data/adb/wifi-rfkill.ko): der Container sieht /data/adb
#    nicht, aber die userdata-Partition ist als Blockgeraet da. Ein zweites mount(2) desselben
#    Geraets (rw - ro gaebe EBUSY, weil die Flags vom bestehenden Superblock abweichen) liefert
#    denselben Superblock unter /mnt/data -> /mnt/data/adb ist lesbar und schreibbar (30.8. belegt).
#    util-linux' mount verweigert das ("already mounted on /") -> eigener Mini-Aufruf via C.
#    Noetig, weil rfkill.ko mit dem Schluessel des Kernel-Builds signiert sein muss (GKI protected
#    exports); sonst laedt bcmdhd nicht -> Node ohne WLAN.
# 3. Weitere .ko (ipset/xt_set/vxlan fuer MicroK8s) nach /opt/barra-k8s/modules.
# Danach: `reboot` (ueber barra-power = echter Geraete-Neustart).
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
D=${1:?Verzeichnis mit boot-lz4.img und wifi-rfkill.ko}
log(){ echo "[$(date +%T)] $*"; }
die(){ log "FEHLER: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "als root ausfuehren (sudo)"
IMG="$D/boot-lz4.img"; KO="$D/wifi-rfkill.ko"
[ -f "$IMG" ] || die "$IMG fehlt"; [ -f "$KO" ] || die "$KO fehlt"

# ---- 1) Kernel --------------------------------------------------------------------------------
SLOT=$(grep slot_suffix /proc/bootconfig 2>/dev/null | sed 's/.*"\(_[ab]\)".*/\1/'); [ -n "$SLOT" ] || die "Slot unklar"
DEV=/dev/block/by-name/boot$SLOT; [ -b "$DEV" ] || die "Partition fehlt: $DEV"
SZ=$(stat -c %s "$IMG"); PSZ=$(blockdev --getsize64 "$DEV")
[ "$SZ" -le "$PSZ" ] || die "Image ($SZ) groesser als Partition ($PSZ)"
WANT=$(sha256sum "$IMG" | cut -d' ' -f1)
CUR=$(head -c "$SZ" "$DEV" | sha256sum | cut -d' ' -f1)
if [ "$CUR" = "$WANT" ]; then
  log "Kernel bereits drauf (sha $(echo "$WANT" | cut -c1-16))"
else
  mkdir -p /var/lib/barra
  [ -f "/var/lib/barra/boot$SLOT.bak" ] || { dd if="$DEV" of="/var/lib/barra/boot$SLOT.bak" bs=4M 2>/dev/null; log "Sicherung: /var/lib/barra/boot$SLOT.bak"; }
  dd if="$IMG" of="$DEV" bs=4M conv=fsync 2>/dev/null; sync
  RB=$(head -c "$SZ" "$DEV" | sha256sum | cut -d' ' -f1)
  [ "$RB" = "$WANT" ] || die "Rueckleseprobe weicht ab ($RB)"
  log "Kernel geschrieben: boot$SLOT = $(echo "$WANT" | cut -c1-16)"
fi

# ---- 2) rfkill auf die Android-Seite ------------------------------------------------------------
ROOTDEV=$(df --output=source / | tail -1)
[ -b "$ROOTDEV" ] || die "Root-Blockgeraet unklar: $ROOTDEV"
if [ ! -x /usr/local/sbin/barra-mount2 ]; then
  cat > /tmp/barra-mount2.c <<'EOF'
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/mount.h>
/* zweites mount(2) eines bereits eingehaengten Blockgeraets (gleicher Superblock); rw, weil
 * abweichende ro/rw-Flags EBUSY geben. Aufruf: barra-mount2 <dev> <dir> [fstype] */
int main(int c, char **v){ if(c<3){fprintf(stderr,"barra-mount2 <dev> <dir> [fstype]\n");return 2;}
  if(mount(v[1],v[2],c>3?v[3]:"f2fs",0,NULL)){ fprintf(stderr,"mount: %s\n",strerror(errno)); return 1;} return 0; }
EOF
  gcc -O2 -o /usr/local/sbin/barra-mount2 /tmp/barra-mount2.c || die "barra-mount2 bauen"
fi
mkdir -p /mnt/data
mountpoint -q /mnt/data || /usr/local/sbin/barra-mount2 "$ROOTDEV" /mnt/data || die "userdata nicht einhaengbar"
[ -d /mnt/data/adb ] || { umount /mnt/data; die "/mnt/data/adb fehlt (falsches Geraet?)"; }
OLD=$(sha256sum /mnt/data/adb/wifi-rfkill.ko 2>/dev/null | cut -c1-16)
NEW=$(sha256sum "$KO" | cut -c1-16)
if [ "$OLD" = "$NEW" ]; then
  log "wifi-rfkill.ko bereits aktuell ($NEW)"
else
  cp /mnt/data/adb/wifi-rfkill.ko "/mnt/data/adb/wifi-rfkill.ko.prev-$(date +%Y%m%d-%H%M)" 2>/dev/null
  cp "$KO" /mnt/data/adb/wifi-rfkill.ko.new && chmod 644 /mnt/data/adb/wifi-rfkill.ko.new && mv /mnt/data/adb/wifi-rfkill.ko.new /mnt/data/adb/wifi-rfkill.ko || { umount /mnt/data; die "rfkill schreiben"; }
  sync; log "wifi-rfkill.ko: $OLD -> $NEW"
fi
umount /mnt/data

# ---- 3) weitere Module (MicroK8s) -------------------------------------------------------------
n=0; for f in "$D"/*.ko; do b=$(basename "$f"); [ "$b" = wifi-rfkill.ko ] && continue; mkdir -p /opt/barra-k8s/modules; install -m 644 "$f" /opt/barra-k8s/modules/"$b"; n=$((n+1)); done
[ $n -gt 0 ] && log "$n Kernel-Module nach /opt/barra-k8s/modules"
log "fertig - jetzt: sudo reboot (echter Neustart ueber Android)"
