# Test-gxpd3 mit Puffer-Option auf eigenem Socket starten (Produktions-gxpd bleibt)
H=/data/adb/hwbridge; T=/data/local/tmp/hb
pkill -f "gxpd3 /data/local/ubuntu/opt/hwbridge/test" 2>/dev/null; sleep 1
mkdir -p /data/local/ubuntu/opt/hwbridge/test; chmod 777 /data/local/ubuntu/opt/hwbridge/test
chmod 755 $T/gxpd3
export GXPD_KDIR=$H
export GXPD_CACHE=$1; [ -n "$2" ] && export GXPD_COH=$2
LD_LIBRARY_PATH=$H:/system/lib64:/vendor/lib64 setsid $T/gxpd3 /data/local/ubuntu/opt/hwbridge/test/gxp.sock </dev/null >$T/gxpexp.log 2>&1 &
sleep 4; grep -a "Set\|bereit\|hoert" $T/gxpexp.log | tail -4; ls -la /data/local/ubuntu/opt/hwbridge/test/
