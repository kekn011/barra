#!/system/bin/sh
# barra Flash-Kit - laeuft AUF dem Geraet als root (von install-base.sh aufgerufen).
# Entpackt den Base-Payload und richtet Hooks/Rechte ein. Idempotent (nochmal ausfuehren = Update).
set -e
K=/data/local/tmp/barra-kit
# Klartext-Credentials (Passwort/PSK) garantiert entfernen — auch bei Fehlerabbruch (set -e).
trap 'rm -f "$K/preconfig.env" 2>/dev/null' EXIT
TB=$K/barra-base.tar.gz
MODE=${1:-install}     # install (Standard) | preconfig (nur Pre-Einrichtung auf vorhandene Base anwenden)
if [ "$MODE" = preconfig ]; then
  echo "-- Modus: nur Pre-Einrichtung (Base bleibt) --"
  [ -d /data/local/ubuntu ] || { echo "Base fehlt (/data/local/ubuntu)"; exit 1; }
  # F9: In diesem Modus MUSS preconfig.env vorhanden sein — sonst wuerde nichts angewendet,
  # aber unten trotzdem "OK - Base installiert" gemeldet (GUI meldet faelschlich Erfolg).
  [ -f "$K/preconfig.env" ] || { echo "FEHLER: preconfig.env fehlt - nichts angewendet"; exit 1; }
  # Container NICHT anhalten (ein Container-Neustart reisst den USB-Gadget/adb mit) - usermod/chpasswd/
  # Hostname gehen auch im laufenden Rootfs (barra-config macht es genauso); wirksam ab dem Neustart in Schritt 5.
else
[ -f "$TB" ] || { echo "Payload fehlt: $TB"; exit 1; }
echo "-- entpacken nach /data/local/barra-stage (dann umziehen) --"
rm -rf /data/local/barra-stage; mkdir -p /data/local/barra-stage
(cd /data/local/barra-stage && tar -xpzf "$TB")
rm -f "$TB"
echo "-- Container-Rootfs -> /data/local/ubuntu --"
# Vorhandenen Container stoppen (falls Update auf laufendem Node)
pkill -f 'lib/systemd/systemd' 2>/dev/null || true; sleep 1
[ -d /data/local/ubuntu ] && mv /data/local/ubuntu /data/local/ubuntu.old
mv /data/local/barra-stage/ubuntu /data/local/ubuntu
mkdir -p /data/local/ubuntu/proc /data/local/ubuntu/sys /data/local/ubuntu/dev /data/local/ubuntu/run /data/local/ubuntu/tmp /data/local/ubuntu/oldroot
chmod 1777 /data/local/ubuntu/tmp
rm -rf /data/local/ubuntu.old &
echo "-- /data/adb-Teile --"
mkdir -p /data/adb/service.d /data/adb/post-fs-data.d
cp -a /data/local/barra-stage/adb/baseos /data/adb/
cp -a /data/local/barra-stage/adb/hwbridge /data/adb/
cp -a /data/local/barra-stage/adb/service.d/. /data/adb/service.d/
cp -a /data/local/barra-stage/adb/post-fs-data.d/. /data/adb/post-fs-data.d/
cp -a /data/local/barra-stage/adb/usbnet-modules /data/adb/
# Magisk-Modul barra-bootanim (Boot-Branding), falls im Payload
if [ -d /data/local/barra-stage/adb/modules/barra-bootanim ]; then
  mkdir -p /data/adb/modules
  cp -a /data/local/barra-stage/adb/modules/barra-bootanim /data/adb/modules/
fi
cp /data/local/barra-stage/adb/nsprep /data/adb/nsprep; chmod 755 /data/adb/nsprep
cp /data/local/barra-stage/adb/wifi-rfkill.ko /data/adb/wifi-rfkill.ko
chmod 755 /data/adb/service.d/*.sh /data/adb/post-fs-data.d/*.sh /data/adb/baseos/bin/* /data/adb/hwbridge/*.sh 2>/dev/null || true
chmod 600 /data/adb/baseos/config; chown 0:0 /data/adb/baseos/config
mkdir -p /data/adb/baseos/run; chmod 700 /data/adb/baseos/run
# SELinux-Kontexte fuer Magisk-Hooks (sonst startet init sie nicht)
chcon -R u:object_r:adb_data_file:s0 /data/adb/service.d /data/adb/post-fs-data.d 2>/dev/null || true
restorecon -R /data/adb/baseos /data/adb/hwbridge 2>/dev/null || true
rm -rf /data/local/barra-stage
fi  # MODE=install
# ---- Pre-Einrichtung (aus der Setup-GUI): direkt ins Rootfs, VOR dem ersten Boot ---------------
PRE=$K/preconfig.env
if [ -f "$PRE" ]; then
  echo "-- Pre-Einrichtung anwenden --"
  U=/data/local/ubuntu
  getv(){ sed -n "s/^$1=//p" "$PRE" | head -1; }
  P_USER=$(getv USER); P_PASS=$(getv PASS); P_HOST=$(getv HOST); P_TZ=$(getv TZ); P_SSID=$(getv SSID); P_PSK=$(getv PSK)
  # SSH-Key(s) base64-kodiert transportiert (mehrzeilig + injektionssicher); Fallback: alte SSHKEY-Zeile
  P_KEY_B64=$(getv SSHKEY_B64); if [ -n "$P_KEY_B64" ]; then P_KEY=$(printf '%s' "$P_KEY_B64" | base64 -d 2>/dev/null); else P_KEY=$(getv SSHKEY); fi
  P_CS=$(getv CHARGE_START); P_CE=$(getv CHARGE_STOP); P_DT=$(getv DISPLAY_TIMEOUT); P_LANG=$(getv LANG_UI)
  # chroot ins Rootfs fuer usermod/chpasswd (Container laeuft noch nicht). /proc/dev/sys kurz binden.
  # (bei laufendem Container sind proc/dev dort schon gemountet -> mount/umount duerfen scheitern)
  MNT_P=0; MNT_D=0
  if ! mountpoint -q $U/proc 2>/dev/null; then mount -t proc proc $U/proc 2>/dev/null && MNT_P=1 || true; fi
  if ! mountpoint -q $U/dev  2>/dev/null; then mount -o bind /dev $U/dev 2>/dev/null && MNT_D=1 || true; fi
  unmnt(){ [ "$MNT_D" = 1 ] && umount $U/dev 2>/dev/null; [ "$MNT_P" = 1 ] && umount $U/proc 2>/dev/null; return 0; }
  # PATH explizit: Androids su-Shell vererbt /system/bin -> im chroot faende bash sonst kein usermod/sed
  CH="chroot $U /usr/bin/env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /bin/bash -c"
  # Benutzer: ubuntu -> P_USER (Home mit), Passwort setzen, KEIN chage-Zwang
  if [ -n "$P_USER" ] && [ "$P_USER" != ubuntu ] && $CH "getent passwd ubuntu >/dev/null"; then
    $CH "pkill -u ubuntu 2>/dev/null; sleep 1; usermod -l '$P_USER' ubuntu && usermod -d '/home/$P_USER' -m '$P_USER' && (groupmod -n '$P_USER' ubuntu 2>/dev/null || true); for s in /etc/sudoers.d/*; do [ -f \"\$s\" ] && sed -i 's/\bubuntu\b/$P_USER/g' \"\$s\"; done" && echo "user: ubuntu -> $P_USER"
  fi
  [ -z "$P_USER" ] && P_USER=ubuntu
  # sudo ohne Passwort fuer den eingestellten Systemuser (Kevins Entwurf, 30.8.). Die alte
  # Laborregel 'knews-nopasswd' (steckt im v14-Image) fliegt raus — hier, damit auch das schon
  # ausgelieferte Image den Fix bekommt. Syntax per visudo pruefen: eine kaputte Datei in
  # sudoers.d sperrt sudo komplett.
  rm -f $U/etc/sudoers.d/knews-nopasswd
  printf '%s ALL=(ALL) NOPASSWD: ALL\n' "$P_USER" > $U/etc/sudoers.d/barra-user; chmod 0440 $U/etc/sudoers.d/barra-user
  if $CH "visudo -cf /etc/sudoers.d/barra-user >/dev/null 2>&1"; then echo "sudo: $P_USER ohne passwort"; else rm -f $U/etc/sudoers.d/barra-user; echo "WARNUNG: sudo-Regel fuer $P_USER ungueltig - nicht gesetzt"; fi
  if [ -n "$P_PASS" ]; then printf '%s:%s\n' "$P_USER" "$P_PASS" | $CH 'chpasswd' && echo "passwort gesetzt"; fi
  $CH "chage -d -1 '$P_USER' 2>/dev/null; chage -E -1 '$P_USER' 2>/dev/null" || true
  # SSH-Key(s): Werte per ENV in den chroot geben (NICHT in den bash -c-Text interpolieren) —
  # sonst bricht ein einfaches Anfuehrungszeichen im Key aus der Quotierung aus und laeuft als
  # root im chroot. Der bash -c-Rumpf ist single-quoted; BARRA_KEY/BARRA_USER kommen ueber env.
  # while-read verarbeitet mehrere Zeilen (mehrere Keys) korrekt.
  if [ -n "$P_KEY" ]; then
    chroot $U /usr/bin/env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      BARRA_KEY="$P_KEY" BARRA_USER="$P_USER" /bin/bash -c '
        H=$(getent passwd "$BARRA_USER" | cut -d: -f6)
        mkdir -p "$H/.ssh"; touch "$H/.ssh/authorized_keys"
        printf "%s\n" "$BARRA_KEY" | while IFS= read -r k; do
          [ -n "$k" ] || continue
          grep -qxF "$k" "$H/.ssh/authorized_keys" || printf "%s\n" "$k" >> "$H/.ssh/authorized_keys"
        done
        chmod 700 "$H/.ssh"; chmod 600 "$H/.ssh/authorized_keys"
        chown -R "$BARRA_USER:$BARRA_USER" "$H/.ssh"
      ' && echo "ssh-key hinterlegt"
  fi
  # Hostname
  if [ -n "$P_HOST" ]; then echo "$P_HOST" > $U/etc/hostname; sed -i "s/^127.0.1.1.*/127.0.1.1\t$P_HOST/" $U/etc/hosts; grep -q '127.0.1.1' $U/etc/hosts || printf '127.0.1.1\t%s\n' "$P_HOST" >> $U/etc/hosts; echo "hostname: $P_HOST"; fi
  # Zeitzone
  if [ -n "$P_TZ" ] && [ -f "$U/usr/share/zoneinfo/$P_TZ" ]; then ln -sf "/usr/share/zoneinfo/$P_TZ" $U/etc/localtime; echo "$P_TZ" > $U/etc/timezone; echo "zeitzone: $P_TZ"; fi
  # UI-Sprache (Container-Seite; Android-Store folgt unten)
  if [ -n "$P_LANG" ]; then echo "$P_LANG" > $U/etc/barra-lang; echo "sprache: $P_LANG"; fi
  # Verifikation VOR dem Marker: Benutzer muss existieren, sonst Abbruch (kein "preconfigured" bei Fehler)
  if ! $CH "getent passwd '$P_USER' >/dev/null"; then unmnt; echo "FEHLER: Benutzer $P_USER wurde nicht angelegt"; exit 1; fi
  echo "benutzer ok: $($CH "getent passwd '$P_USER' | cut -d: -f1,3,6")"
  # firstboot: nur noch Host-Keys+machine-id (Hostname/Passwortzwang sind erledigt) -> Marker "preconfigured"
  mkdir -p $U/var/lib/barra; touch $U/var/lib/barra/configured $U/var/lib/barra/preconfigured
  unmnt
  # Android-Store: WLAN + Ladegrenzen + Display
  S=/data/adb/baseos/config
  setkv(){ grep -v "^$1=" "$S" > "$S.tmp" 2>/dev/null; echo "$1=$2" >> "$S.tmp"; mv "$S.tmp" "$S"; chmod 600 "$S"; }
  [ -n "$P_SSID" ] && { setkv WIFI_SSID "$P_SSID"; setkv WIFI_PSK "$P_PSK"; setkv WLAN_ENABLED 1; echo "wlan: $P_SSID"; }
  [ -n "$P_CS" ] && setkv CHARGE_START "$P_CS"; [ -n "$P_CE" ] && setkv CHARGE_STOP "$P_CE"; [ -n "$P_DT" ] && setkv DISPLAY_TIMEOUT "$P_DT"
  [ -n "$P_LANG" ] && setkv LANG_UI "$P_LANG"
  chown 0:0 $S; chmod 600 $S
  rm -f "$PRE"
fi
echo "-- Kontrolle --"
echo "hooks: $(ls /data/adb/service.d/ | tr '\n' ' ')"
echo "bin:   $(ls /data/adb/baseos/bin/ | wc -l) Programme"
echo "rootfs: $(du -sh /data/local/ubuntu 2>/dev/null | cut -f1)"
echo "firstboot-Marker: $(ls /data/local/ubuntu/etc/barra-firstboot.pending 2>/dev/null || echo FEHLT)"
echo "OK - Base installiert. Neustart: adb reboot"
