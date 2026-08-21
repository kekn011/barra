#!/system/bin/sh
# tpu-boost: rio-TPU aus powersave (226MHz) auf vollen Takt (1.119GHz) heben.
# Die TPU ist idle-gated (0Hz idle) -> performance zieht Takt NUR unter Last. Gibt ~2.7x Compute.
DF=/sys/class/devfreq/1a000000.rio
case "$1" in
  on)  echo performance > $DF/governor; echo 1119000000 > $DF/min_freq
       echo "TPU boost ON: governor=$(cat $DF/governor) min=$(cat $DF/min_freq)" ;;
  off) echo 226000000 > $DF/min_freq 2>/dev/null; echo powersave > $DF/governor
       echo "TPU boost OFF: governor=$(cat $DF/governor)" ;;
  status) echo "governor=$(cat $DF/governor) cur=$(cat $DF/cur_freq) min=$(cat $DF/min_freq) max=$(cat $DF/max_freq)" ;;
  *) echo "usage: tpu-boost.sh {on|off|status}"; exit 1 ;;
esac
