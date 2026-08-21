#!/system/bin/sh
# Lauter/Leiser: Cirrus AMP PCM Gain (0..20, L+R). Aufruf: vol.sh up|down
H=/data/adb/hwbridge
cur=$($H/tinymix get "AMP PCM Gain" 2>/dev/null | awk '{print $1}')
case "$cur" in ''|*[!0-9]*) cur=10;; esac
STEP=2; MAX=20; MIN=0
if [ "$1" = "up" ]; then new=$((cur+STEP)); [ $new -gt $MAX ] && new=$MAX
else new=$((cur-STEP)); [ $new -lt $MIN ] && new=$MIN; fi
$H/tinymix set "AMP PCM Gain" $new >/dev/null 2>&1
$H/tinymix set "R AMP PCM Gain" $new >/dev/null 2>&1
echo "gain $cur -> $new"
