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
# GPU-Vokoder-Daemon (nur David braucht ihn; startet auch ohne, Piper geht dann trotzdem)
if ! pgrep -f "gpudecd /run/barra-tts/gpudec.sock" >/dev/null; then
  $K/bin/gpudecd /run/barra-tts/gpudec.sock \
    $K/voices/david/gpukit2/program2.json $K/voices/david/gpukit2 \
    >/run/barra-tts/gpudecd.log 2>&1 &
fi
exec python3.12 $K/bin/ttsd.py
