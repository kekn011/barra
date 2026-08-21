# stop only the private test daemon; leave production gxpd alone
pkill -f "gxpd3 /data/local/ubuntu/opt/hwbridge/test" 2>/dev/null
sleep 1
echo "=== test daemon procs (should be none) ==="
ps -A 2>/dev/null | grep -E "gxpd3 .*test" | grep -v grep || echo "none"
echo "=== production gxpd alive? ==="
ps -A 2>/dev/null | grep -E "hwbridge/gxpd" | grep -v grep || echo "PROD GXPD NOT FOUND"
echo "=== production socket ==="
ls -la /data/local/ubuntu/opt/hwbridge/gxp.sock
