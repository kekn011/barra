#!/system/bin/sh
GX=/data/local/tmp/hb/gxpd3
SRC=/data/local/tmp
KD=/data/local/tmp/hb1
export GXPD_CACHE=1
export LD_LIBRARY_PATH=/data/adb/hwbridge:/system/lib64:/vendor/lib64
export GXPD_KDIR=$KD
mkdir -p $KD
for t in f2a fb8 fc0 fd3 fe3 f110 f190 f93 f202 efd e1079 e1185; do
  rm -f $KD/ker_*.elf
  cp $SRC/ker_$t.elf $KD/ker_dot.elf
  logcat -c 2>/dev/null
  DOT=$($GX x selftest 2>&1 | grep -E '^\[dot\]')
  EXC=$(logcat -d 2>/dev/null | grep -oE 'EXCCAUSE: [0-9]+, EXCCAUSE_ADDRESS: [0-9a-fx]+.*EPC1: [0-9a-f]+' | head -1)
  CR=$(logcat -d 2>/dev/null | grep -oE 'exccause: 0x[0-9a-f]+, exctype: 0x[0-9a-f]+, excvaddr: 0x[0-9a-f]+' | head -1)
  printf '%-7s | %s | %s | %s\n' "$t" "$DOT" "$EXC" "$CR"
done
