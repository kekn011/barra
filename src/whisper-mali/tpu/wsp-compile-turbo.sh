#!/system/bin/sh
T=/data/local/tmp; W=$T/wsptpu; P=$W/turbo/pkgs
LOG=$W/turbo/compile-all.log
: > $LOG
for L in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31; do
  for m in proj woc ffnresc; do
    if [ -f $P/l${L}_${m}.package ]; then echo "l${L}_${m}: SKIP" >> $LOG; continue; fi
    sh $W/wsptpu-compile.sh $P/l${L}_${m}.tflite $P/l${L}_${m}.package >/dev/null 2>&1
    if [ -f $P/l${L}_${m}.package ]; then echo "l${L}_${m}: OK" >> $LOG; else echo "l${L}_${m}: FAIL" >> $LOG; fi
  done
done
echo COMPILE_ALL_DONE >> $LOG
