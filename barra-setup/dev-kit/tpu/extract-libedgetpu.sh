#!/system/bin/sh
# libcomp_std.so ON-DEVICE erzeugen (nie mitgeliefert: gepatchter Vendor-Blob; die Vendor-Lib
# verlaesst das Geraet nicht). Patcht eine KOPIE von libedgetpu_tflite_compiler.so zum
# frida-freien Standalone-Compiler. tpuc1 nutzt das via COMPILER_SO=.../libcomp_std.so.
#   su -c 'sh /data/adb/baseos/dev/tpu/extract-libedgetpu.sh'
#
# PYTHON-FREI (der Container bringt kein python3 mit): die 4 Fixes per dd, mit Verifikation der
# Altwerte (falsche libedgetpu-Version -> Abbruch statt Blindpatch). Offsets/Werte = die von
# patch-compiler-standalone.py (in .text ist file-offset == vaddr):
#  1. Config-Name v1->v6 (adrp+add) @0x16b0978/097c  -> baut "v6_config.binarypb" (registriert)
#  2. chip_type-Mapper @0xb7549c/54a0: w0:=0x1a (mappt auf id 6 = rio/G3) + nop
V=${COMPILER_SO_SRC:-/vendor/lib64/libedgetpu_tflite_compiler.so}
OUT=${1:-/data/adb/baseos/dev/tpu/libcomp_std.so}
[ -f "$V" ] || { echo "Vendor-Compiler fehlt: $V"; exit 1; }

echo "kopiere Vendor-Compiler ($(wc -c < "$V") B) -> $OUT ..."
cp "$V" "$OUT" || { echo "Kopie fehlgeschlagen"; exit 2; }

# patch <offset-dez> <alt-hex-LE> <neu-oktal-bytes>
patch(){
  off=$1; oldhex=$2; newb=$3
  cur=$(dd if="$OUT" bs=1 skip="$off" count=4 2>/dev/null | od -An -tx1 | tr -d ' \n')
  [ "$cur" = "$oldhex" ] || { echo "Patch @$off: erwartet $oldhex, gefunden $cur -> ABBRUCH (andere libedgetpu-Version? patch-compiler-standalone.py-Offsets neu ermitteln)"; rm -f "$OUT"; exit 3; }
  printf "$newb" | dd of="$OUT" bs=1 seek="$off" count=4 conv=notrunc 2>/dev/null
  echo "  @$off: $oldhex -> gepatcht"
}

patch $((0x16b0978)) 4971fff0 '\251\164\377\320'   # config v1->v6 adrp  (d0ff74a9)
patch $((0x16b097c)) 29c13791 '\051\055\035\221'   # config v1->v6 add   (911d2d29)
patch $((0xb7549c))  1f3c0071 '\100\003\200\122'   # chip_type mov w0,#0x1a (52800340)
patch $((0xb754a0))  4d020054 '\037\040\003\325'   # chip_type nop        (d503201f)

chmod 644 "$OUT"
echo "OK -> $OUT (frida-freier Standalone-Compiler). Test: COMPILER_SO=$OUT tpuc1 <modell.tflite>"
