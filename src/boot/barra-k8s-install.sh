#!/bin/bash
# barra-k8s-install.sh - MicroK8s 1.28 auf einem barra-Node einrichten (im Container, als root).
# Idempotent: mehrfach ausfuehren = nachziehen. Reproduziert den Laborweg vom knews-Cluster
# (Aug. 2026) auf einem frisch geflashten Node:
#   0. snapd: /snap anlegen (der Bake wirft /var/lib/snapd weg -> snapd meldet sonst
#      "unexpected snap mount directory"), Dienste entmaskieren, Snap laden.
#   1. Kernel-Module ip_set/xt_set/xt_addrtype/vxlan aus dem Kernel-Build (/opt/barra-k8s/modules,
#      mit dem Schluessel unseres Kernels signiert - fremde .ko lehnt der Kernel ab).
#   2. Snaps entsquashen: der Kernel hat kein squashfs, snapd haengt sie per squashfuse ein -
#      das bricht bei jedem Neustart. Kopie nach /var/lib/snapd/unpacked + Bind-Mount-Drop-in.
#   3. kubelite-Drop-in: Androids filter-Table traegt quota2/bpf-Regeln, die das Snap-iptables
#      (1.8.6) nicht lesen kann -> Chroot-iptables 1.8.10 mit libxt_quota2.so drueberbinden.
#   4. Argumente: Kubelet ohne cgroup-Zwang (Androids cgroup-Hierarchie ist nicht delegiert),
#      Node-IP = Kabel-Interface, kube-proxy nur auf der Node-IP, containerd mit crun ohne
#      pivot_root und ohne cgroups, native Snapshotter.
#   5. Daemons in Reihenfolge starten, auf API + Node Ready warten.
#   6. Policy-Routing (Pod-CIDR ueber main) + FORWARD/MASQUERADE vor Androids tether-Chain,
#      als systemd-Unit barra-k8s-net.service, damit es jeden Boot wieder gilt.
set -u
export PATH=/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
MDIR=${MDIR:-/opt/barra-k8s/modules}
DEST=/var/lib/snapd/unpacked
POD_CIDR=${POD_CIDR:-10.1.0.0/16}
SVC_CIDR=${SVC_CIDR:-10.152.183.0/24}
CHANNEL=${CHANNEL:-1.28/stable}
XT=/usr/lib/aarch64-linux-gnu/xtables
log(){ echo "[$(date +%T)] $*"; }
die(){ log "FEHLER: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "als root ausfuehren (sudo)"

LAN=$(ip -4 route show default 2>/dev/null | awk '{m=1e9; for(i=1;i<=NF;i++) if($i=="metric") m=$(i+1); print m, $5}' | sort -n | head -1 | awk '{print $2}')
[ -n "$LAN" ] || die "keine Default-Route"
NODE_IP=$(ip -4 -o addr show "$LAN" | awk '{print $4}' | cut -d/ -f1 | head -1)
[ -n "$NODE_IP" ] || die "keine IPv4 auf $LAN"
log "Node $(hostname): IP $NODE_IP ueber $LAN"

# ---- 0) snapd + Snap -----------------------------------------------------------------------
if [ ! -f /snap/microk8s/current/meta/snap.yaml ]; then
  [ -L /snap ] && rm -f /snap
  mkdir -p /snap /var/lib/snapd/cache
  systemctl unmask snapd.socket snapd.service snapd.seeded.service >/dev/null 2>&1
  systemctl enable --now snapd.socket snapd.service >/dev/null 2>&1
  timeout 300 snap wait system seed.loaded >/dev/null 2>&1 || true
  log "snap install microk8s --classic --channel=$CHANNEL (dauert einige Minuten)"
  snap install microk8s --classic --channel="$CHANNEL" 2>&1 | tail -3
  [ -f /snap/microk8s/current/meta/snap.yaml ] || die "microk8s-Snap nicht installiert"
fi
log "microk8s: $(snap list microk8s 2>/dev/null | tail -1 | awk '{print $2, "rev", $3}')"

# ---- 1) Kernel-Module ----------------------------------------------------------------------
for m in ip_set ip_set_hash_ip ip_set_hash_net ip_set_hash_ipport ip_set_hash_ipportip ip_set_hash_ipportnet \
         ip_set_hash_netport ip_set_hash_netnet ip_set_hash_netportnet ip_set_hash_netiface ip_set_list_set \
         xt_set xt_addrtype vxlan; do
  grep -q "^$m " /proc/modules && continue
  [ -f "$MDIR/$m.ko" ] || die "Modul fehlt: $MDIR/$m.ko"
  insmod "$MDIR/$m.ko" || die "insmod $m"
done
log "Module geladen: $(lsmod | grep -c -E '^(ip_set|xt_set|xt_addrtype|vxlan)')"

# ---- 2) Daemons anhalten, Snaps entsquashen ------------------------------------------------
for d in apiserver-kicker cluster-agent kubelite containerd k8s-dqlite; do systemctl stop snap.microk8s.daemon-$d 2>/dev/null; done
mkdir -p "$DEST"
changed=0
for d in /snap/*/[0-9]*; do
  [ -f "$d/meta/snap.yaml" ] || continue
  name=$(basename "$(dirname "$d")"); rev=$(basename "$d"); tgt="$DEST/$name-$rev"
  if [ ! -f "$tgt/meta/snap.yaml" ]; then
    rm -rf "$tgt.tmp"; cp -a "$d/." "$tgt.tmp" && mv "$tgt.tmp" "$tgt" || die "Kopie $name-$rev"
    log "entpackt $name-$rev ($(du -sh "$tgt" | cut -f1))"; changed=1
  fi
  unit="snap-$(systemd-escape "$name")-$rev.mount"; u=/etc/systemd/system/$unit.d
  if [ ! -f "$u/zz-bind.conf" ]; then
    mkdir -p "$u"
    printf '[Mount]\nWhat=%s\nType=none\nOptions=bind,x-gdu.hide,x-gvfs-hide\nLazyUnmount=yes\n' "$tgt" > "$u/zz-bind.conf"
    changed=1
  fi
done
if [ $changed = 1 ] || grep -q ' fuse.squashfuse' /proc/mounts; then
  systemctl daemon-reload
  systemctl stop snapd.service snapd.socket 2>/dev/null
  umount -l /snap/microk8s/current/sbin/xtables-legacy-multi 2>/dev/null
  for d in /snap/*/[0-9]*; do umount -l "$d" 2>/dev/null; done
  pkill -9 squashfuse 2>/dev/null; sleep 1
  systemctl reset-failed 'snap-*.mount' 2>/dev/null
  for unit in $(systemctl list-unit-files 'snap-*.mount' --no-legend | awk '{print $1}'); do systemctl restart "$unit" 2>/dev/null; done
  systemctl start snapd.socket snapd.service
  sleep 3
fi
[ -f /snap/microk8s/current/meta/snap.yaml ] || die "/snap/microk8s nach dem Umhaengen nicht erreichbar"
log "Snap-Mounts: $(grep -c ' /snap/[a-z0-9_-]*/[0-9]* ' /proc/mounts) (squashfuse: $(grep -c fuse.squashfuse /proc/mounts))"

# ---- 3) kubelite-Drop-in: Chroot-iptables mit quota2 -----------------------------------------
[ -f "$XT/libxt_quota2.so" ] || die "$XT/libxt_quota2.so fehlt (AOSP-Extension, s. src/boot/README)"
mkdir -p /etc/systemd/system/snap.microk8s.daemon-kubelite.service.d
cat > /etc/systemd/system/snap.microk8s.daemon-kubelite.service.d/xtables.conf <<EOF
[Service]
# Snap-iptables 1.8.6 kann Androids filter-Table (quota2/bpf) nicht lesen -> Chroot-iptables 1.8.10 mit libxt_quota2.
Environment=XTABLES_LIBDIR=$XT
ExecStartPre=/bin/sh -c 'mountpoint -q /snap/microk8s/current/sbin/xtables-legacy-multi || mount --bind /usr/sbin/xtables-legacy-multi /snap/microk8s/current/sbin/xtables-legacy-multi'
EOF

# ---- 4) Argumente ----------------------------------------------------------------------------
A=/var/snap/microk8s/current/args
setarg(){ local f="$A/$1"; shift; for a in "$@"; do local k=${a%%=*}; if grep -q -- "^$k\(=\|$\)" "$f" 2>/dev/null; then sed -i "s|^$k\(=.*\)\?$|$a|" "$f"; else echo "$a" >> "$f"; fi; done; }
setarg kubelet --cgroups-per-qos=false --enforce-node-allocatable= --fail-swap-on=false --node-ip=$NODE_IP
setarg kube-proxy --nodeport-addresses=$NODE_IP/32
T=$A/containerd-template.toml
# Snapshotter kommt in dieser Snap-Version aus args/containerd-env (${SNAPSHOTTER})
E=$A/containerd-env
if grep -q '^SNAPSHOTTER=' "$E" 2>/dev/null; then sed -i 's|^SNAPSHOTTER=.*|SNAPSHOTTER=native|' "$E"; else echo 'SNAPSHOTTER=native' >> "$E"; fi
if ! grep -q 'BinaryName = "/usr/bin/crun"' "$T"; then
  [ -f "$T.orig" ] || cp "$T" "$T.orig"
  # disable_cgroup direkt hinter die CRI-Sektion; crun + NoPivotRoot als runc.options-Untersektion
  # direkt hinter dem runc-Block (die Vorlage hat keine options-Sektion fuer runc).
  awk '
    /^\[plugins\."io\.containerd\.grpc\.v1\.cri"\]$/ && !c { print; print "  # barra: Androids cgroup-Hierarchie ist nicht delegiert"; print "  disable_cgroup = true"; c=1; next }
    /^[ 	]*\[plugins\."io\.containerd\.grpc\.v1\.cri"\.containerd\.runtimes\.runc\]$/ { print; inrunc=1; next }
    inrunc && /^[ 	]*\[/ { print "      [plugins.\"io.containerd.grpc.v1.cri\".containerd.runtimes.runc.options]"; print "        # barra: crun statt runc, kein pivot_root im Container"; print "        BinaryName = \"/usr/bin/crun\""; print "        NoPivotRoot = true"; print ""; inrunc=0 }
    { print }' "$T.orig" > "$T"
fi
# Der Snap-Wrapper (run-containerd-with-args) berechnet ${SNAPSHOTTER} selbst (native nur bei ZFS,
# sonst overlayfs) und ignoriert containerd-env -> der Wert muss fest in die Vorlage. overlayfs
# scheitert hier mit "invalid argument" (Container-Rootfs auf f2fs unter pivot_root).
sed -i 's|snapshotter = "\${SNAPSHOTTER}"|snapshotter = "native"|; s|snapshotter = "overlayfs"|snapshotter = "native"|' "$T"
grep -q 'disable_cgroup = true' "$T" || die "containerd-template: disable_cgroup nicht gesetzt"
grep -q 'BinaryName = "/usr/bin/crun"' "$T" || die "containerd-template: crun nicht gesetzt"
grep -q 'snapshotter = "native"' "$T" || die "containerd-template: Snapshotter nicht auf native"

# ---- 5) Netz-Unit (jeden Boot): Module, Policy-Routing, FORWARD/MASQ -----------------------
mkdir -p /opt/barra-k8s
cat > /opt/barra-k8s/barra-k8s-net.sh <<EOF
#!/bin/bash
# barra-k8s-net.sh - Kernel-Module, Policy-Routing und Forwarding fuer MicroK8s (jeden Boot, idempotent)
export PATH=/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export XTABLES_LIBDIR=$XT
POD_CIDR=$POD_CIDR; SVC_CIDR=$SVC_CIDR
for m in ip_set ip_set_hash_ip ip_set_hash_net ip_set_hash_ipport ip_set_hash_ipportip ip_set_hash_ipportnet ip_set_hash_netport ip_set_hash_netnet ip_set_hash_netportnet ip_set_hash_netiface ip_set_list_set xt_set xt_addrtype vxlan; do
  grep -q "^\$m " /proc/modules || insmod $MDIR/\$m.ko 2>/dev/null
done
rule(){ ip rule show | grep -q "\$1" || ip rule add \$2; }
rule "to \$POD_CIDR lookup main"   "to \$POD_CIDR table main pref 90"
rule "to 169.254.1.1 lookup main"   "to 169.254.1.1/32 table main pref 91"
rule "from \$POD_CIDR lookup main" "from \$POD_CIDR table main pref 92"
rule "to \$SVC_CIDR lookup main"   "to \$SVC_CIDR table main pref 93"
echo 1 > /proc/sys/net/ipv4/ip_forward
EG=\$(ip -4 route show default | awk '{m=1e9; for(i=1;i<=NF;i++) if(\$i=="metric") m=\$(i+1); print m, \$5}' | sort -n | head -1 | awk '{print \$2}')
IPT=\$(command -v iptables-legacy || command -v iptables)
top(){ \$IPT -t filter -C FORWARD \$1 \$POD_CIDR -m comment --comment "microk8s pod fwd (pre-tether)" -j ACCEPT 2>/dev/null && \$IPT -t filter -D FORWARD \$1 \$POD_CIDR -m comment --comment "microk8s pod fwd (pre-tether)" -j ACCEPT; \$IPT -t filter -I FORWARD 1 \$1 \$POD_CIDR -m comment --comment "microk8s pod fwd (pre-tether)" -j ACCEPT; }
top -d; top -s
for dev in \$(ip -4 -o link show | awk -F': ' '{print \$2}' | grep -E '^(wlan0|eth|enx|usb)'); do
  \$IPT -t nat -C POSTROUTING -s \$POD_CIDR ! -d \$POD_CIDR -o \$dev -m comment --comment "microk8s pod egress" -j MASQUERADE 2>/dev/null \
    || \$IPT -t nat -I POSTROUTING 1 -s \$POD_CIDR ! -d \$POD_CIDR -o \$dev -m comment --comment "microk8s pod egress" -j MASQUERADE
done
echo "k8s-net: egress \$EG, rules \$(ip rule | grep -c -E '\$POD_CIDR|169.254.1.1|\$SVC_CIDR'), forward top: \$(\$IPT -t filter -S FORWARD | sed -n 2p)"
EOF
chmod 755 /opt/barra-k8s/barra-k8s-net.sh
cat > /etc/systemd/system/barra-k8s-net.service <<'EOF'
[Unit]
Description=barra: Kernel-Module, Policy-Routing und Forwarding fuer MicroK8s
Before=snap.microk8s.daemon-containerd.service snap.microk8s.daemon-kubelite.service
After=network.target
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/opt/barra-k8s/barra-k8s-net.sh
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload; systemctl enable barra-k8s-net.service >/dev/null 2>&1
systemctl restart barra-k8s-net.service; systemctl status barra-k8s-net.service --no-pager -n 3 | tail -3

# ---- 5b) netd-cleanup (Dauerdienst): Androids bw_/fw_-Ketten mit quota2/bpf leeren ----------
# Der Chroot-iptables-Trick aus Schritt 3 fixt nur kubelite/kube-proxy. Der CALICO-Container
# bringt sein eigenes, aelteres iptables (1.8.4) mit, das Androids netd-Matches quota2 und
# "bpf --object-pinned" NICHT parsen kann -> felix crasht beim save/restore der GESAMTEN Tabellen
# ("Can't find library for match quota2" / "bpf: failed to get bpf object") in einer Panic-Schleife.
# netd programmiert diese Ketten beim Boot neu (live kommen sie nicht wieder) -> wir halten sie leer.
cat > /opt/barra-k8s/barra-netd-cleanup.sh <<'EOF'
#!/bin/bash
# barra-netd-cleanup.sh - haelt Androids netd-Ketten (bw_/fw_) mit quota2/bpf-Matches leer,
# damit die aeltere iptables im Calico-Container sie lesen kann (sonst felix-Panic-Loop).
while true; do
  for t in raw mangle nat filter; do
    for c in $(iptables-legacy-save -t "$t" 2>/dev/null | grep -E ' -m (bpf|quota2)' | sed -n 's/^-A \([^ ]*\) .*/\1/p' | sort -u); do
      iptables-legacy -t "$t" -F "$c" 2>/dev/null
    done
  done
  sleep 20
done
EOF
chmod 755 /opt/barra-k8s/barra-netd-cleanup.sh
cat > /etc/systemd/system/barra-netd-cleanup.service <<'EOF'
[Unit]
Description=barra: Androids netd bpf/quota2 iptables-Ketten leeren (Calico-Kompatibilitaet)
After=network.target
[Service]
ExecStart=/opt/barra-k8s/barra-netd-cleanup.sh
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload; systemctl enable --now barra-netd-cleanup.service >/dev/null 2>&1
log "netd-cleanup: $(systemctl is-active barra-netd-cleanup.service)"

# ---- 6) Daemons in Reihenfolge, warten ------------------------------------------------------
B=/var/snap/microk8s/current/var/kubernetes/backend
for d in k8s-dqlite containerd kubelite cluster-agent apiserver-kicker; do systemctl reset-failed snap.microk8s.daemon-$d 2>/dev/null; done
systemctl restart snap.microk8s.daemon-k8s-dqlite
i=0; while [ $i -lt 30 ]; do ls "$B"/kine.sock* >/dev/null 2>&1 && break; sleep 2; i=$((i+1)); done
log "dqlite: $(systemctl is-active snap.microk8s.daemon-k8s-dqlite), kine.sock nach ${i}x2s"
systemctl restart snap.microk8s.daemon-containerd; sleep 4
log "containerd: $(systemctl is-active snap.microk8s.daemon-containerd)"
systemctl restart snap.microk8s.daemon-kubelite
systemctl restart snap.microk8s.daemon-cluster-agent snap.microk8s.daemon-apiserver-kicker 2>/dev/null
i=0; while [ $i -lt 60 ]; do c=$(curl -sk -o /dev/null -w '%{http_code}' -m 3 https://127.0.0.1:16443/readyz 2>/dev/null); case "$c" in 200|401|403) break;; esac; sleep 5; i=$((i+1)); done
log "API: HTTP $c nach $((i*5))s"
i=0; while [ $i -lt 60 ]; do st=$(microk8s kubectl get node "$(hostname)" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = Ready ] && break; sleep 5; i=$((i+1)); done
log "Node: ${st:-unbekannt} nach $((i*5))s"
# Calico felix fest auf iptables-legacy: dieser GKI-Kernel hat KEIN nf_tables
# (iptables-nft-save = "Could not fetch rule set generation id") -> nft-Backend crasht felix.
# Auto-Detect waehlt zwar legacy, explizit ist sicherer. Idempotent.
microk8s kubectl set env ds/calico-node -n kube-system FELIX_IPTABLESBACKEND=Legacy FELIX_IPTABLESLOCKTIMEOUTSECS=10 >/dev/null 2>&1 \
  && log "calico: FELIX_IPTABLESBACKEND=Legacy gesetzt" || log "calico: felix-env noch nicht setzbar (DaemonSet nicht bereit)"
/opt/barra-k8s/barra-k8s-net.sh
microk8s kubectl get nodes -o wide 2>&1 | tail -2
microk8s kubectl get pods -A 2>&1 | tail -6
log "fertig - Kontrolle: microk8s status; microk8s kubectl get pods -A"
