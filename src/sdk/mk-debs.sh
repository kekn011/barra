#!/bin/bash
# ============================================================================
# mk-debs.sh — baut die barra-SDK-Debian-Pakete IM Container auf dem Node und
# installiert sie (ersetzt die /usr/local-Handinstallation durch echte Pakete).
# Erwartet: /tmp/barra-src (src/barra) + /tmp/barra-sdk (barra-smi, examples/).
#   libbarra0       Laufzeitbibliothek (/usr/lib/libbarra.so.0.2.0, SONAME .so.1)
#   libbarra-dev    Header, .so/.a, pkg-config (prefix /usr)
#   barra-tools     barra-smi + Beschleuniger-CLIs (/usr/bin)
#   barra-examples  Beispielquellen (/usr/share/barra/examples)
# Ausgabe: /tmp/barra-debs/out/*.deb ; danach dpkg -i + Aufraeumen /usr/local.
# ============================================================================
set -e
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TMPDIR=/tmp   # Android-su vererbt sonst /data/local/tmp -> existiert im chroot nicht (dpkg-deb temp file error)
VER=0.2.0; REV=1; ARCH=$(dpkg --print-architecture)
SRC=/tmp/barra-src; SDK=/tmp/barra-sdk
# Als root fuehren wir gleich build.sh AUS $SRC aus und dpkg-installieren daraus. Vorher
# sicherstellen, dass die Eingaben root-eigen und nicht gruppen/welt-schreibbar sind, sonst
# koennte ein unprivilegierter Prozess sie unterschieben/racen (root-Code-Ausfuehrung).
chk_safe(){
  [ -d "$1" ] || { echo "FEHLER: $1 fehlt"; exit 1; }
  [ "$(stat -c %u "$1" 2>/dev/null)" = 0 ] || { echo "FEHLER: $1 nicht root-eigen - Abbruch"; exit 1; }
  [ -z "$(find "$1" -maxdepth 0 -perm /022 2>/dev/null)" ] || { echo "FEHLER: $1 gruppen/welt-schreibbar - Abbruch"; exit 1; }
}
chk_safe "$SRC"; chk_safe "$SDK"
ST=$(mktemp -d); OUT=$ST/out       # Ausgabe in ein frisches, privates Verzeichnis
mkdir -p "$OUT"

echo "== libbarra bauen (gcc, im Container) =="
( cd "$SRC" && sh build.sh build >/dev/null )

ctl(){ # $1=pkgdir $2=name $3=depends $4=desc
  mkdir -p "$1/DEBIAN"
  cat > "$1/DEBIAN/control" <<EOF
Package: $2
Version: $VER-$REV
Architecture: $ARCH
Maintainer: barra project
Section: libs
Priority: optional
Depends: $3
Homepage: https://github.com/kekn011/barra
Description: $4
EOF
}

echo "== stage: libbarra0 =="
P0=$ST/libbarra0
mkdir -p "$P0/usr/lib"
install -m755 "$SRC/libbarra.so.$VER" "$P0/usr/lib/"
ln -sf "libbarra.so.$VER" "$P0/usr/lib/libbarra.so.1"
ctl "$P0" libbarra0 "libc6" "barra runtime library - unified dispatch over the Tensor G3 accelerator bridges (TPU/GPU/DSP) with zero-copy dmabuf sharing"
printf '#!/bin/sh\nldconfig\n' > "$P0/DEBIAN/postinst"; chmod 755 "$P0/DEBIAN/postinst"
printf '#!/bin/sh\nldconfig\n' > "$P0/DEBIAN/postrm";  chmod 755 "$P0/DEBIAN/postrm"

echo "== stage: libbarra-dev =="
PD=$ST/libbarra-dev
mkdir -p "$PD/usr/lib/pkgconfig" "$PD/usr/include"
install -m644 "$SRC/libbarra.a" "$PD/usr/lib/"
ln -sf "libbarra.so.$VER" "$PD/usr/lib/libbarra.so"
install -m644 "$SRC/barra.h" "$PD/usr/include/"
printf 'prefix=/usr\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n\nName: barra\nDescription: unified dispatch layer over the Pixel accelerator bridges (TPU/GPU/DSP/CPU)\nVersion: %s\nLibs: -L${libdir} -lbarra\nCflags: -I${includedir}\n' "$VER" > "$PD/usr/lib/pkgconfig/barra.pc"
ctl "$PD" libbarra-dev "libbarra0 (= $VER-$REV)" "barra development files - barra.h header, static/shared library links and pkg-config metadata"

echo "== stage: barra-tools =="
PT=$ST/barra-tools
mkdir -p "$PT/usr/bin"
install -m755 "$SDK/barra-smi" "$PT/usr/bin/barra-smi"
install -m755 "$SDK/barrac" "$PT/usr/bin/barrac"
for t in gpzc-cli gxpcli gxpvcli tpucli; do
  # idempotent: beim ERSTEN Lauf liegen die CLIs in /usr/local/bin, bei Wiederholungen
  # schon als Paket in /usr/bin — beide Quellen akzeptieren, sonst fliegen sie beim
  # "over"-Install aus dem Paket (und damit vom System).
  src=""
  [ -x "/usr/local/bin/$t" ] && src="/usr/local/bin/$t"
  [ -z "$src" ] && [ -x "/usr/bin/$t" ] && src="/usr/bin/$t"
  [ -n "$src" ] && install -m755 "$src" "$PT/usr/bin/$t" && echo "  + $t ($src)"
done
ctl "$PT" barra-tools "libbarra0 (= $VER-$REV)" "barra command line tools - barra-smi (accelerator status) and bridge CLIs"

echo "== stage: barra-examples =="
PE=$ST/barra-examples
mkdir -p "$PE/usr/share/barra/examples"
for f in barra_demo.c barra_zc_demo.c barra_tpu_zc_test.c barra_dsp_zc_test.c; do
  install -m644 "$SRC/$f" "$PE/usr/share/barra/examples/"
done
install -m644 "$SDK/examples/README.md" "$PE/usr/share/barra/examples/"
install -m755 "$SDK/examples/build.sh" "$PE/usr/share/barra/examples/"
mkdir -p "$PE/usr/share/barra/docs"
install -m644 "$SDK/docs/your-model-on-the-tpu.md" "$PE/usr/share/barra/docs/"
ctl "$PE" barra-examples "libbarra-dev" "barra SDK examples - small C programs driving TPU, GPU and DSP through libbarra"

echo "== dpkg-deb bauen =="
for p in libbarra0 libbarra-dev barra-tools barra-examples; do
  dpkg-deb -Zgzip -b "$ST/$p" "$OUT/${p}_$VER-${REV}_$ARCH.deb"
done
ls -la "$OUT"

echo "== installieren =="
dpkg -i "$OUT"/libbarra0_*.deb "$OUT"/libbarra-dev_*.deb "$OUT"/barra-tools_*.deb "$OUT"/barra-examples_*.deb

echo "== /usr/local-Duplikate raeumen =="
rm -f /usr/local/lib/libbarra.so* /usr/local/lib/libbarra.a /usr/local/lib/pkgconfig/barra.pc /usr/local/include/barra.h
for t in gpzc-cli gxpcli gxpvcli tpucli barra_demo barra_zc_demo barra_zc2_test barra_tpu_zc_test barra_dsp_zc_test; do
  rm -f "/usr/local/bin/$t"
done
ldconfig

echo "== Verifikation =="
pkg-config --modversion barra
pkg-config --cflags --libs barra
VERIFY=$(mktemp -d) && cp /usr/share/barra/examples/* "$VERIFY/" && ( cd "$VERIFY" && sh build.sh ); rm -rf "$VERIFY"
echo MKDEBSDONE
