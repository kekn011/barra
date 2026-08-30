#!/system/bin/sh
# barra STT-Server — whisper-server (HTTP) mit TPU-Encoder (large-v3-turbo, q5_0-Decoder).
#   sttserver.sh start [modell.bin] [port]  |  stop  |  status  |  log
# Trennung (v10): Binaries im Base (/data/adb/baseos/stt: whisper-server/cli+Libs+tpud_pipe4),
# Kit-Daten (TPU-Packages + Modell) unter /data/local/barra-stt (install-stt.ps1).
# Defaults: greedy (-bs 1 -bo 1), Port 8090. Pins: CPU/MIF/GPU/TPU performance solange der
# Server laeuft (STT_NOPIN=1 laesst sie aus); stop stellt schedutil/interactive wieder her.
BIN=/data/adb/baseos/stt
KIT=/data/local/barra-stt
R=/data/adb/baseos/run
F=/sys/class/devfreq
DM=/data/local/ubuntu/opt/hwbridge/pf; DC=/data/local/ubuntu/opt/hwbridge/pfc
P=$KIT/turbo

. /data/adb/baseos/bin/barra-i18n.sh
# Koexistenz-Waechter (gemeinsam fuer alle Kit-Dienste, src/boot/barra-guard.sh). Fehlt er auf
# einem aelteren Base, laufen die Aufrufe ins Leere statt ins Messer.
G=/data/adb/baseos/bin/barra-guard.sh
if [ -f "$G" ]; then . "$G"; else guard_check(){ :; }; guard_need(){ echo 0; }; guard_expendable(){ :; }; fi
pins_on(){
  [ "$STT_NOPIN" = "1" ] && return
  for CP in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > $CP/scaling_governor 2>/dev/null; done
  echo performance > $F/1a000000.rio/governor 2>/dev/null
  echo 1119000000 > $F/1a000000.rio/min_freq 2>/dev/null
  echo performance > $F/17000010.devfreq_mif/governor 2>/dev/null
  echo 890000 > /sys/class/misc/mali0/device/scaling_min_freq 2>/dev/null
  echo 890000 > /sys/class/misc/mali0/device/hint_min_freq 2>/dev/null
}
pins_off(){
  for CP in /sys/devices/system/cpu/cpufreq/policy*; do echo schedutil > $CP/scaling_governor 2>/dev/null; done
  echo interactive > $F/17000010.devfreq_mif/governor 2>/dev/null
  echo 150000 > /sys/class/misc/mali0/device/scaling_min_freq 2>/dev/null   # 'echo 0' = No-op auf mali-devfreq
  echo 150000 > /sys/class/misc/mali0/device/hint_min_freq 2>/dev/null
}

start_tpuds(){
  # $P = Package-Ordner des Modells; Bestand wird gescannt (Layerzahl je Modell verschieden),
  # die REIHENFOLGE (Layer-Tripel, Cross, conv1, conv2) ist das model_id-Mapping von whisper-barra.
  mkdir -p $DM $DC
  # Nur die EIGENEN beiden tpuds treffen. Ein Kill nach Prozessnamen erschlaegt auch den der
  # Diarisierung (pyaserver haelt einen eigenen), deren Socket dann als Leiche liegen bleibt
  # -> "Connection refused" beim naechsten barra-diarize. Am 27.8. genau so aufgelaufen,
  # als Whisper nach pyaserver gestartet wurde. pyaserver macht es seit jeher ueber den Socket.
  pkill -f "tpud_pipe4 $DM/tpu.sock" 2>/dev/null; pkill -f "tpud_pipe4 $DC/tpu.sock" 2>/dev/null
  sleep 0.6; rm -f $DM/tpu.sock $DC/tpu.sock
  MODELS=""
  L=0; while [ -f $P/l${L}_proj.package ]; do MODELS="$MODELS $P/l${L}_proj.package $P/l${L}_woc.package $P/l${L}_ffnresc.package"; L=$((L+1)); done
  X=0; while [ -f $P/l${X}_cross.package ]; do MODELS="$MODELS $P/l${X}_cross.package"; X=$((X+1)); done
  [ -f $P/conv1.package ] && MODELS="$MODELS $P/conv1.package"
  if [ -f $P/conv2.package ]; then MODELS="$MODELS $P/conv2.package"
  elif [ -f $P/conv2w.package ]; then MODELS="$MODELS $P/conv2w.package"; fi
  CORE=$(ls $P/wsp_core*_b0.package 2>/dev/null | head -1)
  [ -n "$CORE" ] || { t stt.core_missing "$P"; return 1; }
  export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
  TPU_CPU=8 setsid $BIN/tpud_pipe4 $DM/tpu.sock $MODELS </dev/null >$R/stt-tpud-main.log 2>&1 &
  TPU_CPU=7 TPU_ZC_BOUNCE=2 setsid $BIN/tpud_pipe4 $DC/tpu.sock $CORE </dev/null >$R/stt-tpud-core.log 2>&1 &
  i=0; while [ $i -lt 240 ]; do grep -aq bereit $R/stt-tpud-main.log && grep -aq bereit $R/stt-tpud-core.log && return 0; sleep 1; i=$((i+1)); done
  t stt.tpud_fail; tail -3 $R/stt-tpud-main.log $R/stt-tpud-core.log; return 1
}

case "${1:-status}" in
  stop)
    pkill -f "$BIN/whisper-server" 2>/dev/null
    pkill -f "tpud_pipe4 $DM/tpu.sock" 2>/dev/null; pkill -f "tpud_pipe4 $DC/tpu.sock" 2>/dev/null
    pins_off
    t stt.stopped;;
  status)
    PW=$(pgrep -f "$BIN/whisper-server"|head -1)
    PT=$(pgrep -f "tpud_pipe4 $DM/tpu.sock"|head -1)
    [ -n "$PW" ] && t stt.running "$PW" || t stt.off
    [ -n "$PT" ] && t stt.tpud_on || t stt.tpud_off
    grep -a "wsp-barra. init ok" $R/sttserver.log 2>/dev/null | tail -1;;
  log) tail -50 $R/sttserver.log;;
  start)
    MDL="${2:-$KIT/models/ggml-large-v3-turbo-q5_0.bin}"
    PORT="${3:-8090}"
    pgrep -f "$BIN/whisper-server" >/dev/null && { t stt.already; exit 0; }
    [ -f "$MDL" ] || { t stt.model_missing "$MDL"; exit 1; }
    # Waechter (29.8.): Sachverbot stt<->llm + Speicherpruefung. Aufschlag 1100 MB ueber der
    # Modellgroesse (gemessen: 1636 MB bei 547 MB Modell).
    guard_check stt "$(guard_need "$MDL" 1100)" || exit 1
    mkdir -p $R
    pins_on
    # TPU-Encoder, wenn ein Package-Satz fuers Modell installiert ist; sonst CPU
    case "$(basename "$MDL")" in
      *large-v3-turbo*) P=$KIT/turbo;;
      *base*)           P=$KIT/base;;
      *tiny*)           P=$KIT/tiny;;
      *small*)          P=$KIT/small;;
      *medium*)         P=$KIT/medium;;
      *)                P=/nicht-vorhanden;;
    esac
    case "$([ -f "$P/enc_params.txt" ] && echo tpu || echo cpu)" in
      tpu)
        start_tpuds || exit 1
        WHISPER_BARRA=1 WSP_THREADS=4 WSP_PAIR=1 WSP_OVL=1 WSP_SYNC=1 \
        WSP_PKG_DIR=$P WSP_SOCK_MAIN=$DM WSP_SOCK_CORE=$DC LD_LIBRARY_PATH=$BIN \
          setsid $BIN/whisper-server -m "$MDL" --host 0.0.0.0 --port $PORT -t 8 -bs 1 -bo 1 \
          </dev/null >$R/sttserver.log 2>&1 &
        P=$!; guard_expendable "$P"   # sonst oom_score_adj -1000 = unkillbar
        t stt.started_tpu "$P" "$(basename $MDL)" "$PORT";;
      *)
        LD_LIBRARY_PATH=$BIN \
          setsid $BIN/whisper-server -m "$MDL" --host 0.0.0.0 --port $PORT -t 8 -bs 1 -bo 1 \
          </dev/null >$R/sttserver.log 2>&1 &
        P=$!; guard_expendable "$P"
        t stt.started_cpu "$P" "$(basename $MDL)" "$PORT";;
    esac
    t stt.hint "$PORT";;
  *) t stt.usage;;
esac
