#!/system/bin/sh
echo "=== prod gxpd alive? ==="; ps -A 2>/dev/null | grep -i gxpd | grep -v grep
echo "=== prod socket ==="; ls -la /data/local/ubuntu/opt/hwbridge/gxp.sock 2>/dev/null
echo "=== cleanup test dirs/files ==="
rm -rf /data/local/tmp/hb1 /data/local/tmp/hb2 /data/local/tmp/hbtest
rm -f /data/local/tmp/ker_*.elf /data/local/tmp/obs_ker_*.elf
rm -f /data/local/tmp/run_probe.sh /data/local/tmp/sweep*.sh /data/local/tmp/obsrun.sh /data/local/tmp/_dev_recon.sh
echo "removed. remaining ker in tmp:"; ls /data/local/tmp/ker_*.elf 2>/dev/null | wc -l
echo "=== KDIR prod untouched ==="; ls /data/adb/hwbridge/ | grep -cE 'ker_|gxpd'
