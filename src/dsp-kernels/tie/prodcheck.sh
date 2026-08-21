echo "=== all gxp processes ==="
ps -A -o PID,NAME,ARGS 2>/dev/null | grep -iE 'gxp' | grep -v grep || echo "no gxp procs via ARGS"
echo "=== ps -ef style ==="
ps -ef 2>/dev/null | grep -iE 'gxpd|hwbridge' | grep -v grep || echo "none"
echo "=== toybox ps fallback ==="
for p in /proc/[0-9]*; do c=$(cat $p/cmdline 2>/dev/null | tr '\0' ' '); case "$c" in *gxpd*) echo "$p: $c";; esac; done
echo "=== supervisor hw-bridges ==="
for p in /proc/[0-9]*; do c=$(cat $p/cmdline 2>/dev/null | tr '\0' ' '); case "$c" in *hw-bridges*|*hwbridge*) echo "$p: $c";; esac; done
