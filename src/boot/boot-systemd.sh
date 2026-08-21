#!/system/bin/sh
# Bootet systemd als PID 1 in eigenem PID+Mount-NS via ECHTEM pivot_root
# (nicht chroot!). Grund: bei chroot ist der Mount-NS-Root weiterhin das
# Android-/ -> crun exec in Pods loest Binaries im falschen Root auf
# ("executable file not found"). Mit pivot_root ist der NS-Root = ubuntu-Rootfs.
#
# STATUS 2026-08-11: VOLLTEST BESTANDEN, zur Standard-Variante promotet
# (Vorgaenger: boot-systemd-chroot.bak.sh). Cluster komplett hoch, Node Ready,
# alle Pods 1/1 (exec-Probes gruen - erstmals!), kubectl exec funktioniert.
# ACHTUNG: nach containerd/squashfuse-Neustarts koennen STALE SHIMS "failed to
# start exec: not found" liefern -> betroffene Pods neu erzeugen.
#
# Historie des Raetsels: systemd startete nicht, weil der fruehere
# WIP-Stand $ROOTFS im Kind-NS nochmal PLAIN auf sich selbst gebunden hat -
# ein nicht-rekursiver Bind VERSCHATTET alle vererbten Submounts (/dev, /sys,
# cgroup2, bpf). Nach dem Pivot: leeres /dev (kein /dev/kmsg -> deshalb auch
# keine kmsg-Logs vom Sterben, /dev/null war eine normale Datei), kein cgroup2.
# FIX: den inneren Re-Bind WEGLASSEN - $ROOTFS ist durch den globalen Bind
# bereits Mountpoint, nach nsprep-rprivate pivot-tauglich, und die vererbten
# Submounts bleiben sichtbar. Mit Fix: "Startup finished in 486ms",
# emergency.target erreicht (Volltest mit default-Target steht aus - kollidiert
# mit laufender chroot-Instanz, gehoert in die naechste Reboot-Session).
#
# VOR der Promotion zu boot-systemd.sh noch zu klaeren:
# 1. enter-systemd.sh: nutzt `nsenter --mount ... chroot $ROOTFS` - im Pivot-NS
#    existiert /data/local/ubuntu nicht mehr. Kandidat: OHNE --mount nsenter'n
#    (nur --pid --uts --ipc) und `chroot /proc/$SDPID/root` (Magic-Link kreuzt
#    den Mount-NS, klassisches Container-Inspektions-Idiom).
# 2. Readiness-Check: $ROOTFS/run/... auf Platte ist post-pivot STALE (systemd
#    mountet /run als privates tmpfs). Hier schon umgestellt auf
#    /proc/$SDPID/root/run/... (loest im Ziel-NS auf, geht fuer beide Varianten).
# 3. sd_notify-Meldungen ("Received notify message without valid credentials")
#    im emergency-Test beobachtet - vermutlich PID-NS-Uebersetzung; pruefen, ob
#    Type=notify-Units (snapd!) darunter leiden oder wie im chroot-Fall laufen.
#
# toybox kann keine private Mount-Propagation setzen -> Helfer /data/adb/nsprep
# (statisch: MS_REC|MS_PRIVATE + pivot_root-Syscall).
# Aufruf im globalen Mount-NS:  su -M -c 'sh /data/local/tmp/boot-systemd-pivot.wip.sh'
ROOTFS=/data/local/ubuntu
LOG=/data/adb/baseos/run/systemd-boot.log
PIDF=/data/adb/baseos/run/systemd.hostpid
NSPREP=/data/adb/nsprep

is_mounted() { grep -q " $1 " /proc/mounts 2>/dev/null; }
do_mount() { is_mounted "$2" && return 0; mkdir -p "$2"; mount $1 "$2" || echo "WARN mount $2"; }

[ -x "$NSPREP" ] || { echo "FEHLER: $NSPREP fehlt (statischer pivot-Helfer)"; exit 1; }

# Basismounts im globalen NS (werden in den Kind-NS vererbt, wandern durch pivot)
if ! is_mounted "$ROOTFS"; then
    mount -o bind "$ROOTFS" "$ROOTFS"
    mount -o remount,bind,suid,dev "$ROOTFS"
fi
do_mount "-t sysfs sysfs"   "$ROOTFS/sys"
do_mount "-o bind /dev"     "$ROOTFS/dev"
do_mount "-t devpts devpts" "$ROOTFS/dev/pts"
do_mount "-t tmpfs tmpfs"   "$ROOTFS/dev/shm"
do_mount "-t cgroup2 none"  "$ROOTFS/sys/fs/cgroup"
# netd pinnt BPF-Programme, die Androids filter-Table referenziert; der Kernel
# oeffnet die Pins beim Regel-Commit im Mount-NS des Aufrufers. Ohne diesen
# Bind schlaegt im Container JEDE Aenderung der filter-Table mit ENOENT fehl.
do_mount "-o bind /sys/fs/bpf" "$ROOTFS/sys/fs/bpf"

# Laeuft schon? NUR akzeptieren, wenn die PID WIRKLICH der Container-systemd ist.
# Die $PIDF-Datei liegt in /data/local/tmp und ueberlebt Reboots; nach einem
# Neustart kann die alte PID auf einen fremden Android-Prozess zeigen (PID-Reuse).
# Deshalb cmdline UND die systemd-run-dir im Ziel-NS pruefen, sonst frisch starten.
OLDPID=$(cat "$PIDF" 2>/dev/null)
if [ -n "$OLDPID" ] && [ -d "/proc/$OLDPID" ]; then
    OC=$({ tr '\0' ' ' < "/proc/$OLDPID/cmdline"; } 2>/dev/null)
    case "$OC" in
        "/lib/systemd/systemd --system"*|"/usr/lib/systemd/systemd --system"*)
            if [ -d "/proc/$OLDPID/root/run/systemd/system" ]; then
                echo "systemd laeuft bereits (host-pid $OLDPID)"
                exit 0
            fi ;;
    esac
    echo "stale PIDF: host-pid $OLDPID ist NICHT der Container-systemd - starte frisch"
fi
rm -f "$PIDF"

echo "=== $(date) systemd-Start (pivot_root) ===" >> "$LOG"
# Neues PID+Mount-NS. Ablauf im Kind:
#   1) nsprep rprivate  -> / rekursiv privat (pivot_root-Voraussetzung)
#   2) KEIN Re-Bind von $ROOTFS (verschattet sonst die vererbten Submounts!)
#   3) oldroot + frisches proc anlegen, pivot_root
#   4) altes Android-/ (nach /oldroot) lazy detachen
#   5) exec systemd (ubuntu-Binary; /dev,/sys,cgroup2 aus dem globalen NS geerbt)
setsid env -i \
    container=lxc \
    HOME=/root TERM=linux \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    /system/bin/unshare -m -p -f /system/bin/sh -c "
        $NSPREP rprivate || exit 41
        mkdir -p $ROOTFS/oldroot
        mount -t proc proc $ROOTFS/proc
        cd $ROOTFS || exit 42
        $NSPREP pivot . oldroot || exit 43
        /usr/bin/umount -l /oldroot 2>/dev/null || /bin/umount -l /oldroot 2>/dev/null
        exec /lib/systemd/systemd --system
    " >> "$LOG" 2>&1 &

sleep 6
# Host-PID der Container-PID1: cmdline-Scan (ps zeigt argv nicht zuverlaessig)
SDPID=""
for p in /proc/[0-9]*; do
    c=$({ tr '\0' ' ' < "$p/cmdline"; } 2>/dev/null)
    case "$c" in
        "/lib/systemd/systemd --system"*|"/usr/lib/systemd/systemd --system"*) SDPID=${p##*/}; break;;
    esac
done
echo "$SDPID" > "$PIDF"
echo "systemd host-pid: $SDPID"

# Readiness ueber /proc/PID/root (loest im Ziel-NS auf - NICHT $ROOTFS/run,
# das ist post-pivot stale, weil systemd /run als privates tmpfs mountet)
i=0
while [ $i -lt 30 ]; do
    [ -n "$SDPID" ] && [ -d "/proc/$SDPID/root/run/systemd/system" ] && break
    sleep 1; i=$((i + 1))
done
if [ -n "$SDPID" ] && [ -d "/proc/$SDPID/root/run/systemd/system" ]; then
    echo "OK: systemd bereit nach ${i}s"
else
    echo "systemd noch nicht bereit - Log pruefen: $LOG"
    tail -5 "$LOG"
fi
