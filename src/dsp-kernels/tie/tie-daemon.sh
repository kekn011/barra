#!/system/bin/sh
# Start a private gxpd3 for TIE tests. Own KDIR + own socket. Production gxpd untouched.
H=/data/adb/hwbridge
KDIR=/data/local/tmp/tie-kernels
SOCKDIR=/data/local/ubuntu/opt/hwbridge/test
GXPD=/data/local/tmp/hb/gxpd3
pkill -f "gxpd3 $SOCKDIR" 2>/dev/null; sleep 1
mkdir -p $SOCKDIR; chmod 755 $SOCKDIR   # nicht welt-beschreibbar (Daemon laeuft als root/shell)
mkdir -p $KDIR
chmod 755 $GXPD
export GXPD_KDIR=$KDIR
export GXPD_CACHE=1
LD_LIBRARY_PATH=$H:/system/lib64:/vendor/lib64 setsid $GXPD $SOCKDIR/gxp.sock </dev/null >/data/local/tmp/tie-gxpd.log 2>&1 &
sleep 5
echo "=== log tail ==="; tail -15 /data/local/tmp/tie-gxpd.log
echo "=== socket ==="; ls -la $SOCKDIR/gxp.sock
