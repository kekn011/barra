#!/system/bin/sh
# 72 Attention-Packages (qkv_L*/o_L*) on-device kompilieren. Nutzt tpuc1 + libcomp_std wie verify_proj.
setenforce 0 2>/dev/null
T=/data/local/tmp; A=$T/attn4b
cp /data/adb/baseos/tpu/tpuc1 $T/tpuc1 2>/dev/null; cp /data/adb/baseos/tpu/libcomp_std.so $T/libcomp_std.so 2>/dev/null
chmod 755 $T/tpuc1
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
NL=36; ok=0; fail=0
i=0
while [ $i -lt $NL ]; do
  for base in qkv_L$i o_L$i; do
    if [ -s $A/$base.package ]; then ok=$((ok+1)); i2=1; else
      rm -f $T/out.package /data/vendor/edgetpu/cache/_0 /data/vendor/edgetpu/cache/_0_checksum 2>/dev/null
      COMPILER_SO=$T/libcomp_std.so MODEL=$A/$base.tflite $T/tpuc1 >/dev/null 2>&1
      if [ -s $T/out.package ]; then mv $T/out.package $A/$base.package; ok=$((ok+1)); else echo "FEHLER: $base"; fail=$((fail+1)); fi
    fi
  done
  echo "L$i fertig ($ok ok, $fail Fehler)"
  i=$((i+1))
done
setenforce 1 2>/dev/null
ls $A/*.package | wc -l
echo "COMPILEDONE ok=$ok fail=$fail"
