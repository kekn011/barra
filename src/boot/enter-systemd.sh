#!/system/bin/sh
# Betritt die laufende systemd-Umgebung (PID+Mount+UTS+IPC-NS).
#   ohne Argument: interaktive bash
#   mit Argument (Pfad auf Android-Seite): fuehrt das Skript im Container aus
# Aufruf: su -M -c 'sh /data/local/tmp/enter-systemd.sh [/data/local/tmp/x.sh]'
#
# Funktioniert fuer BEIDE Varianten:
#   chroot-Instanz (boot-systemd.sh):      NS-Root = Android-/  -> chroot noetig
#   pivot-Instanz (boot-systemd-pivot.*):  NS-Root = ubuntu     -> setns(-m)
#     setzt root/cwd automatisch auf den NS-Root, chroot entfaellt (der Pfad
#     /data/local/ubuntu existiert dort gar nicht mehr)
# Erkennung: readlink /proc/PID/root ist "/" beim pivot, sonst der chroot-Pfad.
ROOTFS=/data/local/ubuntu
PATH_IN=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# systemd-Host-PID: cmdline BEGINNT mit /lib/systemd/systemd (nicht der unshare-Wrapper).
# Nach einem daemon-reexec heisst argv[0] /usr/lib/systemd/systemd (--deserialize=N) - beide matchen.
sd_match() {
    # timeout 2: cmdline-Read haengt an gerade sterbenden Prozessen (mmap_lock)
    c=$({ timeout -s KILL 2 tr '\0' ' ' < "/proc/$1/cmdline"; } 2>/dev/null)
    case "$c" in
        "/lib/systemd/systemd --system"*|"/usr/lib/systemd/systemd --system"*|"systemd --system"*) return 0;;
    esac
    return 1
}
SDPID=""
# Schnellweg: boot-systemd.sh hinterlegt die Host-PID in run/systemd.hostpid.
# Identitaet pruefen (PID-Reuse nach Reboot), dann ist KEIN /proc-Vollscan noetig —
# der Scan forkt pro Eintrag eine Subshell+tr (~600 Prozesse, ueber 30 s Boot-CPU).
SP=$(cat /data/adb/baseos/run/systemd.hostpid 2>/dev/null)
if [ -n "$SP" ] && sd_match "$SP"; then
    SDPID=$SP
else
    for p in /proc/[0-9]*; do
        # Prozesse koennen zwischen Glob und Lesen verschwinden - sd_match schluckt den Fehler
        if sd_match "${p#/proc/}"; then SDPID=${p#/proc/}; break; fi
    done
fi
[ -n "$SDPID" ] || { echo "systemd laeuft nicht - erst boot-systemd.sh"; exit 1; }

NSROOT=$(readlink "/proc/$SDPID/root" 2>/dev/null)

run_inner() {
    if [ "$NSROOT" = "/" ]; then
        # ZWEISTUFIG: toybox-nsenter setzt Namespaces sequenziell und liest
        # /proc erst beim jeweiligen Open - nach dem mount-Join zeigt /proc
        # ins Container-proc (Host-PID existiert dort nicht) und die
        # restlichen ns-Opens scheitern. Daher: erst pid/uts/ipc (Android-
        # /proc noch sichtbar), dann separat mount.
        nsenter -t "$SDPID" --pid --uts --ipc -- \
            /system/bin/nsenter -t "$SDPID" --mount -- \
            /usr/bin/env PATH="$PATH_IN" HOME=/root TERM="${TERM:-xterm}" \
            /bin/bash -l "$@"
    else
        nsenter -t "$SDPID" --mount --pid --uts --ipc -- \
            chroot "$ROOTFS" /usr/bin/env PATH="$PATH_IN" HOME=/root TERM="${TERM:-xterm}" \
            /bin/bash -l "$@"
    fi
}

if [ -n "$1" ]; then
    IN="_inner.$$.sh"
    cp "$1" "$ROOTFS/root/$IN"
    chmod +x "$ROOTFS/root/$IN"
    run_inner "/root/$IN"
    RC=$?
    rm -f "$ROOTFS/root/$IN"
    exit $RC
else
    run_inner
fi
