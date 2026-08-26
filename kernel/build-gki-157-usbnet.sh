#!/bin/bash
# knews-Kernel Stufe 4: k8s-Basis (userns + Netfilter-Module) + USB-Ethernet =y.
# Grund: Dock-LAN (Anker, RTL8153). Lose usbnet-Module scheitern am
# GKI-Symbolschutz (mii exportiert generic_mii_ioctl nicht an Module).
# Fix: Treiber built-in (=y) -> alle Symbole in vmlinux, kein KMI-Thema.
# Additive Code-Options, kein Struktur-Bruch (gleiche Klasse wie VXLAN=y/IP_VS=y).
set -e
T="$HOME/akita-kernel"
DIST="${DIST:-$T/out/gki-157-usbnet-dist}"
LOG="${LOG:-$HOME/build-gki-157-usbnet.log}"
CFG="$T/aosp/arch/arm64/configs/gki_defconfig"
GKICFG="$T/aosp/build.config.gki"

cd "$T/aosp"
git checkout -q -- arch/arm64/configs/gki_defconfig build.config.gki BUILD.bazel
cd "$T"

# check_defconfig aus (nicht-kanonische defconfig zulassen)
sed -i 's/^POST_DEFCONFIG_CMDS="check_defconfig"/POST_DEFCONFIG_CMDS="true"/' "$GKICFG"

echo "== Optionen: PID_NS + USER_NS + Netfilter-Module (wie k8s-Build) =="
sed -i '/^# CONFIG_PID_NS is not set$/d' "$CFG"        # default y
setopt() { sed -i "/^$1[= ]/d;/^# $1 is not set\$/d" "$CFG"; echo "$1=$2" >> "$CFG"; }
setopt CONFIG_USER_NS y
setopt CONFIG_IP_SET m
setopt CONFIG_IP_SET_HASH_IP m
setopt CONFIG_IP_SET_HASH_NET m
setopt CONFIG_IP_SET_HASH_IPPORT m
setopt CONFIG_IP_SET_HASH_IPPORTIP m
setopt CONFIG_IP_SET_HASH_IPPORTNET m
setopt CONFIG_IP_SET_HASH_NETPORT m
setopt CONFIG_IP_SET_HASH_NETNET m
setopt CONFIG_IP_SET_HASH_NETPORTNET m
setopt CONFIG_IP_SET_HASH_NETIFACE m
setopt CONFIG_IP_SET_LIST_SET m
setopt CONFIG_NETFILTER_XT_SET m
setopt CONFIG_NETFILTER_XT_MATCH_ADDRTYPE m
setopt CONFIG_NETFILTER_XT_MATCH_RPFILTER m
setopt CONFIG_VXLAN m

echo "== NEU: USB-Ethernet built-in (=y) fuer Dock-LAN =="
setopt CONFIG_MII y
setopt CONFIG_USB_NET_DRIVERS y
setopt CONFIG_USB_USBNET y
setopt CONFIG_USB_RTL8152 y
setopt CONFIG_USB_NET_AX8817X y
setopt CONFIG_USB_NET_AX88179_178A y
setopt CONFIG_USB_NET_CDCETHER y
setopt CONFIG_USB_NET_CDC_NCM y
setopt CONFIG_USB_NET_AQC111 y
setopt CONFIG_USB_RTL8153_ECM y

echo "== NEU: NFS-Server + -Client + FHANDLE (SSD-Freigabe im Cluster) =="
# NFSD: Kernel-NFS-Server (Export der ext4-SSD aus dem Ubuntu-Chroot).
# NFS_FS: Client, damit ANDERE Nodes (gleiches Image) die Freigabe mounten.
# FHANDLE: name_to/open_by_handle_at — braucht nfs-utils/Ganesha.
# Alles additiver Code (fs/nfsd, fs/nfs, net/sunrpc), kein KMI-Struktur-Bruch.
setopt CONFIG_FHANDLE y
setopt CONFIG_NFSD y
setopt CONFIG_NFSD_V4 y
setopt CONFIG_NFS_FS y
setopt CONFIG_NFS_V3 y
setopt CONFIG_NFS_V4 y
setopt CONFIG_NFS_V4_1 y
setopt CONFIG_NFS_V4_2 y

echo "== module_outs in BUILD.bazel deklarieren =="
# Bazel verlangt, dass neue =m-Module im kernel_aarch64-Target deklariert sind
# (sonst: "kernel modules are built but not copied").
python3 - "$T/aosp/BUILD.bazel" <<'PYEOF'
import sys
p = sys.argv[1]
mods = [
    "net/netfilter/ipset/ip_set.ko",
    "net/netfilter/ipset/ip_set_hash_ip.ko",
    "net/netfilter/ipset/ip_set_hash_net.ko",
    "net/netfilter/ipset/ip_set_hash_ipport.ko",
    "net/netfilter/ipset/ip_set_hash_ipportip.ko",
    "net/netfilter/ipset/ip_set_hash_ipportnet.ko",
    "net/netfilter/ipset/ip_set_hash_netport.ko",
    "net/netfilter/ipset/ip_set_hash_netnet.ko",
    "net/netfilter/ipset/ip_set_hash_netportnet.ko",
    "net/netfilter/ipset/ip_set_hash_netiface.ko",
    "net/netfilter/ipset/ip_set_list_set.ko",
    "net/netfilter/xt_set.ko",
    "net/netfilter/xt_addrtype.ko",
    "drivers/net/vxlan/vxlan.ko",
]
# Diese GKI-Listen-Module sind jetzt =y (built-in) -> werden nicht mehr als
# .ko gebaut; blieben sie deklariert, bricht der Build mit "Unable to find".
gone = [
    "drivers/net/mii.ko",
    "drivers/net/usb/usbnet.ko",
    "drivers/net/usb/cdc_ether.ko",
    "drivers/net/usb/cdc_ncm.ko",
    "drivers/net/usb/r8152.ko",
    "drivers/net/usb/r8153_ecm.ko",
    "drivers/net/usb/asix.ko",
    "drivers/net/usb/ax88179_178a.ko",
    "drivers/net/usb/aqc111.ko",
]
s = open(p).read()
anchor = '"module_implicit_outs": get_gki_modules_list("arm64"),'
assert anchor in s, "module_implicit_outs-Zeile nicht gefunden"
ins = '"module_implicit_outs": [m for m in get_gki_modules_list("arm64") if m not in [' \
    + ", ".join('"%s"' % g for g in gone) + ']] + [' \
    + ", ".join('"%s"' % m for m in mods) + '],'
s = s.replace(anchor, ins, 1)
open(p, "w").write(s)
print("module_implicit_outs erweitert: %d Module" % len(mods))
PYEOF

echo "== Build =="
set +e
tools/bazel run --config=stamp --lto=none \
    //aosp:kernel_aarch64_dist -- --dist_dir="$DIST" >> "$LOG" 2>&1
RC=$?
set -e
echo "rc=$RC"
[ "$RC" -eq 0 ] || { tail -25 "$LOG"; exit $RC; }

IK=/tmp/ikconfig-usbnet.txt
"$T/aosp/scripts/extract-ikconfig" "$DIST/Image" > "$IK"
for K in CONFIG_PID_NS CONFIG_USER_NS CONFIG_RFKILL CONFIG_IP_SET CONFIG_NETFILTER_XT_SET \
         CONFIG_MII CONFIG_USB_USBNET CONFIG_USB_RTL8152 CONFIG_USB_NET_AX88179_178A \
         CONFIG_USB_NET_CDCETHER CONFIG_USB_NET_CDC_NCM CONFIG_USB_NET_AQC111 \
         CONFIG_FHANDLE CONFIG_NFSD CONFIG_NFS_FS CONFIG_NFS_V4; do
    printf '  %-36s %s\n' "$K" "$(grep -e "^$K=" -e "^# $K is not set" "$IK" || echo '(fehlt)')"
done
REPO=${BARRA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${BARRA_OUT:-$REPO/out/gki-6.1.157-usbnet}
mkdir -p "$OUT"
cp "$DIST/boot-lz4.img" "$OUT/"
cp "$DIST"/*.ko "$OUT/" 2>/dev/null || true
ls -la "$OUT"
echo "boot-lz4.img + Module -> images/gki-6.1.157-usbnet/"
