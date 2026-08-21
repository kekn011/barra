#!/system/bin/sh
echo "=== /data/local/tmp/hb ==="; ls -la /data/local/tmp/hb/ 2>/dev/null
echo "=== KDIR /data/adb/hwbridge ==="; ls -la /data/adb/hwbridge/ 2>/dev/null | grep -E 'gxpd|ker_|metrics|inbuilt|vscale'
echo "=== testdir ==="; ls -la /data/local/ubuntu/opt/hwbridge/test/ 2>/dev/null
echo "=== running gxpd ==="; ps -A 2>/dev/null | grep -i gxpd
echo "=== prod socket ==="; ls -la /data/local/ubuntu/opt/hwbridge/gxp.sock 2>/dev/null
