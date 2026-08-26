#!/bin/bash
# barra-tts Kit-Launcher (selbst-enthaltend). Startet gpudecd (GPU-Vokoder, warm) + ttsd (HTTP-Worker).
# Vom systemd-Dienst barra-tts.service aufgerufen. Alle Pfade kit-intern (kein ~/tts noetig).
K=/opt/barra-tts
R=$K/runtime
export PATH=$R/usr/bin:$PATH
export LD_LIBRARY_PATH=$R/usr/lib/aarch64-linux-gnu:$K/sherpa/lib
export PYTHONHOME=$R/usr
export PYTHONPATH=$K/site
export ESPEAK_DATA_PATH=$R/usr/lib/aarch64-linux-gnu/espeak-ng-data
export TTS_ROOT=$K
export TTS_GPUDEC_SOCK=/run/barra-tts/gpudec.sock
export TTS_SHERPA=$K/sherpa/sherpa-onnx-offline-tts
export TTS_AUDIO_SOCK=/opt/hwbridge/audio.sock
export TTS_PORT=${TTS_PORT:-8095}
mkdir -p /run/barra-tts 2>/dev/null || true
# GPU-Vokoder-Daemons: EIN gpudecd je Stimme mit eigenem Socket.
# Wichtig: ein gpudecd traegt genau EIN Programm (Gewichte einer Stimme). Wuerden sich
# zwei Stimmen einen Daemon teilen, bekaeme die zweite still die Gewichte der ersten —
# es klaenge falsch, ohne dass irgendetwas fehlschlaegt.
for v in $K/voices/*/; do
  [ -f "$v/gpukit2/program2.json" ] || continue
  name=$(basename "${v%/}")
  sock=/run/barra-tts/gpudec-$name.sock
  pgrep -f "gpudecd $sock" >/dev/null && continue
  echo "[launch] GPU-Vokoder fuer $name -> $sock" >>/run/barra-tts/gpudecd.log
  ( while :; do
      $K/bin/gpudecd "$sock" "$v/gpukit2/program2.json" "$v/gpukit2"         >>/run/barra-tts/gpudecd.log 2>&1
      echo "[launch] gpudecd $name beendet (rc=$?), Neustart in 2s" >>/run/barra-tts/gpudecd.log
      sleep 2
    done ) &
done
exec python3.12 $K/bin/ttsd.py
