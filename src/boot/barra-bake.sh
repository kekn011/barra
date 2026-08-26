#!/system/bin/sh
# barra-bake.sh — Base-Image-Payload (barra-base.tar.gz) aus einer STAGING-KOPIE bauen. Laeuft AUF dem Golden Node als root;
# der Node selbst bleibt unberuehrt (Keys, Creds, Benutzer). Ergebnis: /data/local/tmp/barra-base.tar.gz + .sha256
#   su -c 'sh /data/adb/baseos/bin/barra-bake.sh'
set -e
U=/data/local/ubuntu; B=/data/local/bake; OUT=/data/local/tmp/barra-base.tar.gz
UBUNTU_HASH='$6$barrabase$z78uePiq/N.XjnuCPe9/bkY0.00Ov6X9lS0wj4nOyCG/ERJm.3gXY8OcM0/osZ1D94s4X1BzHjartIfrN4uQP0'   # Passwort "ubuntu" (firstboot erzwingt Wechsel; Setup setzt eigenes)
echo "== barra-bake: Staging-Kopie =="
rm -rf $B; mkdir -p $B/ubuntu $B/adb
# 1) Rootfs kopieren — ohne Mountpoints/Laufzeit, Logs, Modelle, apt-Cache, Sockets.
# KIT-Artefakte (pya/tts/wake) IMMER auslassen: ALLE 5 Pakete sind SDK-Kits (24.8., v13) —
# der Bake darf sie nie einbacken, auch wenn sie auf dem Quell-Node installiert sind
# (llm/stt-Kits liegen unter /data/local/barra-* und damit eh ausserhalb von $U/$A).
cd $U && tar -cf -   --exclude=./proc --exclude=./sys --exclude=./dev --exclude=./run --exclude=./tmp --exclude=./oldroot --exclude=./core   --exclude=./var/log --exclude=./var/cache/apt --exclude=./var/lib/apt/lists --exclude=./var/lib/snapd   --exclude='./home/*/*.gguf' --exclude='./home/*/models' --exclude='./opt/hwbridge/*.sock' --exclude='./opt/hwbridge/*.log'   --exclude=./opt/barra-pya --exclude=./opt/barra/pya --exclude=./opt/barra-tts --exclude=./opt/barra-wake   . | (cd $B/ubuntu && tar -xpf -)
R=$B/ubuntu
# Nachputzen, falls ein Exclude nicht griff
rm -rf $R/proc/* $R/sys/* $R/dev/* $R/run/* $R/tmp/* $R/oldroot/* $R/core $R/var/log $R/var/cache/apt $R/var/lib/apt/lists $R/var/lib/snapd 2>/dev/null || true
find $R/home -name '*.gguf' -delete 2>/dev/null || true; rm -rf $R/home/*/models $R/opt/hwbridge/*.sock $R/opt/hwbridge/*.log 2>/dev/null || true
# Kit-Reste sicher entfernen (Container-Seite): Verzeichnisse, Werkzeuge, systemd-Units
rm -rf $R/opt/barra-pya $R/opt/barra/pya $R/opt/barra-tts $R/opt/barra-wake 2>/dev/null || true
rm -f $R/usr/local/bin/barra-diarize $R/usr/local/bin/barra-img $R/etc/systemd/system/barra-tts.service $R/etc/systemd/system/barra-wake.service 2>/dev/null || true
# Entwickler-Harnesses aus dem Bringup (Quellen in src/container) gehoeren nicht ins Produkt-Image:
# sie standen bis v14 in /usr/local/bin neben den echten Werkzeugen und waren fuer Nutzer nicht unterscheidbar.
rm -f $R/usr/local/bin/dmabuf-test $R/usr/local/bin/kms-test $R/usr/local/bin/touchtest $R/usr/local/bin/touchkms $R/usr/local/bin/wavsend $R/usr/local/bin/atone $R/usr/local/bin/colband 2>/dev/null || true
mkdir -p $R/proc $R/sys $R/dev $R/run $R/tmp $R/oldroot $R/var/log/journal $R/var/cache/apt $R/var/lib/apt/lists; chmod 1777 $R/tmp
: > $R/var/log/wtmp; : > $R/var/log/btmp; : > $R/var/log/lastlog
echo "Rootfs-Kopie: $(du -sm $R | cut -f1) MB"
echo "== Container-Seite generisch machen (wie barra-prepare-image) =="
rm -f $R/etc/ssh/ssh_host_*_key $R/etc/ssh/ssh_host_*_key.pub
: > $R/etc/machine-id; rm -f $R/var/lib/dbus/machine-id
rm -f $R/root/.bash_history $R/home/*/.bash_history $R/root/.ssh/known_hosts $R/home/*/.ssh/known_hosts $R/etc/barra-rename
rm -f $R/root/.ssh/authorized_keys   # kein Root-Key im Image (der Setup legt den Nutzer-Key an)
rm -f $R/var/lib/barra/configured $R/var/lib/barra/preconfigured
echo barra > $R/etc/hostname; sed -i 's/^127.0.1.1.*/127.0.1.1\tbarra/' $R/etc/hosts
touch $R/etc/barra-firstboot.pending
# Benutzer zurueck auf ubuntu (Golden Node heisst barra) — passwd/group/shadow/gshadow/sub*id/sudoers/Home
CUR=$(awk -F: '$3==1001{print $1}' $R/etc/passwd | head -1)
if [ -n "$CUR" ] && [ "$CUR" != ubuntu ]; then
  for f in passwd group shadow gshadow subuid subgid; do [ -f $R/etc/$f ] && sed -i "s/^$CUR:/ubuntu:/; s/:$CUR$/:ubuntu/; s/,$CUR,/,ubuntu,/g; s/,$CUR$/,ubuntu/; s/:$CUR,/:ubuntu,/" $R/etc/$f; done
  sed -i "s#:/home/$CUR:#:/home/ubuntu:#" $R/etc/passwd
  for s in $R/etc/sudoers.d/*; do [ -f "$s" ] && sed -i "s/\b$CUR\b/ubuntu/g" "$s"; done
  [ -d $R/home/$CUR ] && mv $R/home/$CUR $R/home/ubuntu
  echo "Benutzer $CUR -> ubuntu"
fi
# Passwort ubuntu setzen, Home aufraeumen
sed -i "s#^ubuntu:[^:]*:#ubuntu:$UBUNTU_HASH:#" $R/etc/shadow
rm -rf $R/home/ubuntu/.cache $R/home/ubuntu/.ssh/authorized_keys $R/home/ubuntu/*.log 2>/dev/null || true
# Versionsmarker (F5): ohne ihn kann niemand sagen, welcher Stand auf einem Node laeuft —
# weder der Nutzer im Bugreport noch wir beim Nachstellen. Format KEY=WERT (source-bar).
VER=${BARRA_VERSION:-unversioniert}
[ "$VER" = unversioniert ] && echo "WARNUNG: BARRA_VERSION nicht gesetzt — das Image bekommt keinen Versionsnamen"
{ echo "BARRA_VERSION=$VER"
  echo "BARRA_BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "BARRA_KERNEL=$(uname -r)"
} > $R/etc/barra-release
chmod 644 $R/etc/barra-release
echo "Versionsmarker: $(tr '
' ' ' < $R/etc/barra-release)"
echo "== Android-Seite kopieren + saeubern =="
A=$B/adb
cp -a /data/adb/baseos /data/adb/hwbridge /data/adb/service.d /data/adb/post-fs-data.d /data/adb/usbnet-modules $A/
cp /data/adb/nsprep $A/nsprep; cp /data/adb/wifi-rfkill.ko $A/wifi-rfkill.ko
# Magisk-Modul barra-bootanim (Boot-Branding) gezielt mitnehmen — NICHT ganz modules/ (fremde Module)
mkdir -p $A/modules; cp -a /data/adb/modules/barra-bootanim $A/modules/ 2>/dev/null || true
S=$A/baseos/config; [ -f $S ] && { grep -v '^WIFI_SSID=' $S | grep -v '^WIFI_PSK=' > $S.tmp; mv $S.tmp $S; chmod 600 $S; }
rm -rf $A/baseos/run; mkdir -p $A/baseos/run; chmod 700 $A/baseos/run
rm -f $A/baseos/state $A/baseos/disable $A/baseos/boot.log $A/baseos/bin/*.bak* 2>/dev/null; rm -rf $A/baseos/boot.lock 2>/dev/null
rm -f $A/hwbridge/*.log $A/hwbridge/*.pid $A/hwbridge/dash-run.sh 2>/dev/null
# Kit-Reste sicher entfernen (Android-Seite): Server-Skripte + Mic-Bridge kommen aus den Kits.
# WICHTIG: hier muss JEDES neue Kit eingetragen werden, das etwas unter /data/adb ablegt, sonst
# backt der naechste Bake es still mit ein. Am 25.8. genau so aufgefallen: Bild-Kit (imgserver.sh)
# und Dev-Kit (baseos/dev mit frida-server+glslang = 55 MB Fremdbinaries, dazu dev-mode/devdeploy
# und der service.d-Hook) waren nach dem v13-Bake dazugekommen und standen nicht in der Liste.
rm -f $A/baseos/bin/pyaserver.sh $A/baseos/bin/ttsserver.sh $A/baseos/bin/wakeserver.sh $A/baseos/bin/imgserver.sh $A/hwbridge/audiod-record 2>/dev/null
rm -rf $A/baseos/dev 2>/dev/null; rm -f $A/baseos/bin/barra-dev-mode.sh $A/baseos/bin/devdeploy.sh $A/service.d/55-barra-dev.sh 2>/dev/null
echo "adb-Teile: $(du -sm $A | cut -f1) MB (llm $(du -sm $A/baseos/llm 2>/dev/null | cut -f1) MB, tpu $(du -sm $A/baseos/tpu 2>/dev/null | cut -f1) MB)"
echo "== Kontrolle =="
grep -rl 'WIFI_PSK\|psk=' $A 2>/dev/null | head -3 || true
echo "Kit-Reste (muss ueberall 0 sein): dev=$(ls -d $A/baseos/dev 2>/dev/null | wc -l) frida=$(find $A -name 'frida*' 2>/dev/null | wc -l) kitserver=$(ls $A/baseos/bin/pyaserver.sh $A/baseos/bin/ttsserver.sh $A/baseos/bin/wakeserver.sh $A/baseos/bin/imgserver.sh 2>/dev/null | wc -l) devhook=$(ls $A/service.d/55-barra-dev.sh 2>/dev/null | wc -l) opt=$(ls -d $R/opt/barra-tts $R/opt/barra-wake $R/opt/barra-pya 2>/dev/null | wc -l)"
echo "Bringup-Harnesses (muss 0 sein): $(ls $R/usr/local/bin/dmabuf-test $R/usr/local/bin/kms-test $R/usr/local/bin/touchtest $R/usr/local/bin/touchkms $R/usr/local/bin/wavsend $R/usr/local/bin/atone 2>/dev/null | wc -l)  release: $(grep -c . $R/etc/barra-release)"
echo "host-keys: $(ls $R/etc/ssh/ssh_host_* 2>/dev/null | wc -l)  machine-id: $(wc -c < $R/etc/machine-id)  firstboot: $(ls $R/etc/barra-firstboot.pending)  user: $(awk -F: '$3==1001{print $1":"$6}' $R/etc/passwd)  gguf: $(find $R -name '*.gguf' | wc -l)"
echo "== packen =="
rm -f $OUT; cd $B && tar -czf $OUT ubuntu adb
sha256sum $OUT | tee $OUT.sha256; ls -la $OUT
rm -rf $B
echo "fertig: adb pull $OUT"
