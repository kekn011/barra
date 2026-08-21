#!/bin/sh
# libbarra — heterogene Dispatch-Schicht ueber die Pixel-HW-Bruecken (glibc, im Container bauen).
#   ./build.sh                    -> libbarra.a, libbarra.so, barra_demo, barra_zc_demo (im Quellverzeichnis)
#   sudo ./build.sh install       -> nach PREFIX (Default /usr/local); DESTDIR=<stage> fuer Payload-Bau
# Base hat gcc, aber kein make -> bewusst reines sh.
set -e
cd "$(dirname "$0")"
CC=${CC:-gcc}; PREFIX=${PREFIX:-/usr/local}; CFLAGS=${CFLAGS:--O2 -Wall -fPIC}; VERSION=0.2.0
build(){
  $CC $CFLAGS -c barra.c -o barra.o
  ar rcs libbarra.a barra.o
  $CC -shared -Wl,-soname,libbarra.so.1 -o libbarra.so.$VERSION barra.o
  ln -sf libbarra.so.$VERSION libbarra.so.1; ln -sf libbarra.so.$VERSION libbarra.so
  $CC $CFLAGS barra_demo.c libbarra.a -o barra_demo
  $CC $CFLAGS barra_zc_demo.c libbarra.a -o barra_zc_demo
  $CC $CFLAGS barra_zc2_test.c libbarra.a -o barra_zc2_test
  $CC $CFLAGS barra_tpu_zc_test.c libbarra.a -o barra_tpu_zc_test
  $CC $CFLAGS barra_tpu_raw.c libbarra.a -o barra_tpu_raw
  $CC $CFLAGS barra_tpu_zcbench.c libbarra.a -o barra_tpu_zcbench
  $CC $CFLAGS barra_dsp_zc_test.c libbarra.a -o barra_dsp_zc_test
  printf 'prefix=%s\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n\nName: barra\nDescription: heterogene Dispatch-Schicht ueber die Pixel-HW-Bruecken (TPU/GPU/DSP/CPU)\nVersion: %s\nLibs: -L${libdir} -lbarra\nCflags: -I${includedir}\n' "$PREFIX" "$VERSION" > barra.pc
  echo "gebaut: libbarra.a libbarra.so.$VERSION barra_demo barra_zc_demo barra_zc2_test barra_tpu_zc_test barra_dsp_zc_test barra.pc"
}
install_(){
  D="$DESTDIR$PREFIX"
  install -d "$D/lib/pkgconfig" "$D/include" "$D/bin" "$D/src/barra"
  install -m644 libbarra.a "$D/lib/"
  install -m755 libbarra.so.$VERSION "$D/lib/"
  ln -sf libbarra.so.$VERSION "$D/lib/libbarra.so.1"; ln -sf libbarra.so.$VERSION "$D/lib/libbarra.so"
  install -m644 barra.pc "$D/lib/pkgconfig/"
  install -m644 barra.h "$D/include/"
  install -m755 barra_demo barra_zc_demo barra_zc2_test barra_tpu_zc_test barra_dsp_zc_test "$D/bin/"
  install -m644 barra.c barra.h barra_demo.c barra_zc_demo.c barra_zc2_test.c barra_tpu_zc_test.c barra_dsp_zc_test.c build.sh "$D/src/barra/"; chmod 755 "$D/src/barra/build.sh"
  [ -z "$DESTDIR" ] && ldconfig 2>/dev/null || true
  echo "installiert nach $D (lib/include/pkgconfig/bin/src)"
}
case "${1:-build}" in
  build) build;;
  install) [ -f libbarra.a ] || build; install_;;
  clean) rm -f *.o libbarra.a libbarra.so* barra_demo barra_zc_demo barra_zc2_test barra_tpu_zc_test barra.pc;;
  *) echo "usage: $0 [build|install|clean]"; exit 1;;
esac
