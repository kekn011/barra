#!/system/bin/sh
# Magisk service.d boot hook: set rio-TPU governor to performance at boot.
# Safe: TPU is idle-gated (0Hz when not computing); performance only raises clock under load.
DF=/sys/class/devfreq/1a000000.rio
[ -d "$DF" ] || exit 0
# wait for boot to settle so framework DVFS init doesn't override us
until [ "$(getprop sys.boot_completed)" = "1" ]; do sleep 2; done
sleep 5
echo performance > $DF/governor 2>/dev/null
echo 1119000000 > $DF/min_freq 2>/dev/null
log -t tpu-boost "rio governor=$(cat $DF/governor) min=$(cat $DF/min_freq)"
