#!/bin/sh
# build-gki.sh - GKI-Kernel (akita/Pixel 8a, 6.1.157) bauen -> boot-lz4.img (P2 GKI-Loop).
# Wired das kernel/-Rezept (barra ist Wahrheit: kernel/README.md, PINS.txt, patches/aosp.patch).
#
# VORAUSSETZUNG (einmalig, schwerer Vorlauf ~30 GB Sync - NICHT von diesem Skript gemacht):
#   mkdir akita-kernel && cd akita-kernel
#   repo init -u https://android.googlesource.com/kernel/manifest -b android-gs-akita-android16
#   repo sync -j4
#   # dann die exakten Revisionen aus kernel/PINS.txt auschecken (nur aosp/ noetig fuers boot-Image)
#
# DANN:  build-gki.sh <akita-kernel-tree>
# Ergebnis: <tree>/out/dist/boot-lz4.img -> reflash via barra-setup (Kernel-Schritt) ODER
#   dev-schnell: adb reboot bootloader && fastboot flash boot boot-lz4.img && fastboot reboot
#   (RISIKO: schlechter Kernel = kein Boot; Rueckweg = Stock-boot via barra-setup neu flashen).
set -e
TREE=${1:?"Pfad zum gesyncten akita-kernel-Baum (s. kernel/README.md fuer den repo-sync)"}
KDIR=$(cd "$(dirname "$0")/../../../kernel" && pwd)   # barra kernel/ (Patch + Pins)
[ -d "$TREE/aosp" ] || { echo "kein aosp/ in $TREE -> erst 'repo sync' (kernel/README.md)"; exit 1; }
[ -f "$KDIR/patches/aosp.patch" ] || { echo "Patch fehlt: $KDIR/patches/aosp.patch"; exit 1; }
echo "PINS (auf diese Revisionen pinnen, s. kernel/PINS.txt):"; sed -n "1,40p" "$KDIR/PINS.txt"
cd "$TREE"
echo "== Patch anwenden =="
if ( cd aosp && git apply --check "$KDIR/patches/aosp.patch" ) 2>/dev/null; then
  ( cd aosp && git apply "$KDIR/patches/aosp.patch" ) && echo "aosp.patch angewandt"
else
  echo "aosp.patch schon drin oder Konflikt -> pruefen (git -C aosp status)"
fi
echo "== GKI-Build (bazel; dauert lange) =="
tools/bazel run //aosp:kernel_aarch64_dist -- --dist_dir=out/dist
IMG="$TREE/out/dist/boot-lz4.img"
echo "OK -> $IMG"; [ -f "$IMG" ] && sha256sum "$IMG"
echo "Referenz (ausgeliefert): 01938bbe72dc1f282b671673a0d01dd9f10a00f6674c6bef7b8c245e9da42949"
echo "Reflash: barra-setup Kernel-Schritt, oder fastboot flash boot '$IMG'"
