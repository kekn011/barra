echo "=== gxpd3 binary ==="
ls -la /data/local/tmp/hb/gxpd3 2>/dev/null || echo "no gxpd3 in hb"
echo "=== KDIR /data/adb/hwbridge ==="
ls -la /data/adb/hwbridge/ | grep -E 'ker_|gxpd|metrics'
echo "=== ubuntu opt/hwbridge ==="
ls -la /data/local/ubuntu/opt/hwbridge/ 2>/dev/null
echo "=== metrics stub ==="
ls -la /data/local/tmp/gxp_metrics_logger.so 2>/dev/null
