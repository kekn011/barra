#!/system/bin/sh
KD=/data/local/tmp/hb2
rm -rf $KD; mkdir -p $KD
cp /data/local/tmp/obs_ker_dot.elf $KD/ker_dot.elf
cp /data/local/tmp/obs_ker_sumloop.elf $KD/ker_sumloop.elf
cp /data/local/tmp/obs_ker_vadd.elf $KD/ker_vadd.elf
export GXPD_KDIR=$KD
export GXPD_CACHE=1
export LD_LIBRARY_PATH=/data/adb/hwbridge:/system/lib64:/vendor/lib64
/data/local/tmp/hb/gxpd3 x selftest 2>&1 | grep -E 'vadd|dot|sum|mm2x2'
