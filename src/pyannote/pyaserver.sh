#!/system/bin/sh
# barra Diarization-Dienst — tpud mit dem ResNet34-Trunk-Package fuer barra-diarize.
#   pyaserver.sh start | stop | status | log
# Trennung: Binaries im Base (/data/adb/baseos/stt/tpud_pipe4 + Container /opt/barra/pya),
# Kit-Daten unter /data/local/ubuntu/opt/barra-pya (install-pya.ps1) — ein Verzeichnis
# fuer beide Welten: tpud (Android) liest das Package, sherpa (Container) Modelle+head.
# Pins: CPU/MIF/TPU performance solange der Dienst laeuft (PYA_NOPIN=1 laesst sie aus).
# Koexistenz: LLM ist tabu (RAM); STT parallel ist ok (TPU-Graphen 104+1 < ~157).
BIN=/data/adb/baseos/stt/tpud_pipe4
KIT=/data/local/ubuntu/opt/barra-pya
D=/data/local/ubuntu/opt/hwbridge/pya
R=/data/adb/baseos/run
F=/sys/class/devfreq

pins_on(){
  [ "$PYA_NOPIN" = "1" ] && return
  for CP in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > $CP/scaling_governor 2>/dev/null; done
  echo performance > $F/1a000000.rio/governor 2>/dev/null
  echo 1119000000 > $F/1a000000.rio/min_freq 2>/dev/null
  echo performance > $F/17000010.devfreq_mif/governor 2>/dev/null
}
pins_off(){
  for CP in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > $CP/scaling_governor 2>/dev/null; done
  echo interactive > $F/17000010.devfreq_mif/governor 2>/dev/null
  echo 0 > $F/1a000000.rio/min_freq 2>/dev/null
}

case "${1:-status}" in
  stop)
    pkill -f "tpud_pipe4 $D/tpu.sock" 2>/dev/null
    pins_off
    echo "gestoppt";;
  status)
    PT=$(pgrep -f "tpud_pipe4 $D/tpu.sock"|head -1)
    [ -n "$PT" ] && echo "pya-tpud laeuft (PID $PT)" || echo "pya-tpud aus"
    [ -S "$D/tpu.sock" ] && echo "Socket bereit: /opt/hwbridge/pya/tpu.sock (Container-Sicht)";;
  log) tail -30 $R/pya-tpud.log;;
  start)
    pgrep -f "tpud_pipe4 $D/tpu.sock" >/dev/null && { echo "laeuft schon (erst stop)"; exit 0; }
    # Koexistenz-Verbot (22.8.): LLM pinnt den Speicher — Diarization dazu riskiert OOM
    pgrep -f "llama-server" >/dev/null && { echo "KI-Chat (llmserver) laeuft - erst stoppen: llmserver.sh stop"; exit 1; }
    # Packages generisch: r34_trunk ODER eres_body ODER titanet-Segmentkette (alphabetisch = model_ids)
    PKG=$(ls $KIT/*.package 2>/dev/null | sort | tr '\n' ' ')
    [ -n "$PKG" ] || { echo "Diarization-Kit fehlt unter $KIT (install-pya.ps1 ausfuehren)"; exit 1; }
    mkdir -p $D $R
    rm -f $D/tpu.sock
    pins_on
    export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
    TPU_CPU=8 TPU_WARMUP=2 setsid $BIN $D/tpu.sock $PKG </dev/null >$R/pya-tpud.log 2>&1 &
    i=0; while [ $i -lt 60 ]; do grep -aq bereit $R/pya-tpud.log && break; sleep 1; i=$((i+1)); done
    grep -aq bereit $R/pya-tpud.log || { echo "tpud-Start FEHLGESCHLAGEN:"; tail -3 $R/pya-tpud.log; pins_off; exit 1; }
    chmod 666 $D/tpu.sock 2>/dev/null
    echo "gestartet: Diarization-TPU bereit - im Container: barra-diarize <wav> [sprecher]";;
  *) echo "pyaserver.sh start | stop | status | log";;
esac
