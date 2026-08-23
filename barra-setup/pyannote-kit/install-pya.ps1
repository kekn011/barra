# install-pya.ps1 — Diarization-Kit (Sprecher-Trennung) auf einen barra-Node schieben (USB/adb).
# Inhalt pyannote-kit.tar: kit/ (seg.onnx Segmentierung, resnet34.onnx wespeaker-Referenz,
# r34_trunk.package TPU-Embedding-Trunk, head.bin Pooling+FC-Kopf) -> /data/local/ubuntu/opt/barra-pya
# und base/ (sherpa-Binary, barra-diarize, pyaserver.sh) — ab Base v11 im Image, hier fuer v10 mitinstalliert.
# Danach: pyaserver.sh start (Android) + barra-diarize <wav> [sprecher] (im Container).
$ErrorActionPreference = "Stop"
$kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$adb = Join-Path (Split-Path -Parent $kit) "tools\adb.exe"
if (-not (Test-Path $adb)) { $adb = "adb" }

Write-Host "== barra Diarization-Kit: pyannote/wespeaker auf TPU =="
& $adb wait-for-device | Out-Null

Write-Host "1/2 Kit pushen (71 MB) ..."
& $adb push (Join-Path $kit "pyannote-kit.tar") /data/local/tmp/pya-kit.tar
& $adb shell "su -c 'cd /data/local/tmp && rm -rf pya-kit && mkdir pya-kit && cd pya-kit && tar -xf ../pya-kit.tar'"

Write-Host "2/2 Dateien verteilen ..."
& $adb shell "su -c 'U=/data/local/ubuntu; S=/data/local/tmp/pya-kit; mkdir -p `$U/opt/barra-pya `$U/opt/barra/pya && cp `$S/kit/* `$U/opt/barra-pya/ && chmod 644 `$U/opt/barra-pya/* && cp `$S/base/sherpa-onnx-offline-speaker-diarization `$U/opt/barra/pya/ && chmod 755 `$U/opt/barra/pya/sherpa-onnx-offline-speaker-diarization && cp `$S/base/barra-diarize `$U/usr/local/bin/barra-diarize && chmod 755 `$U/usr/local/bin/barra-diarize && cp `$S/base/pyaserver.sh /data/adb/baseos/bin/pyaserver.sh && chmod 755 /data/adb/baseos/bin/pyaserver.sh && rm -rf /data/local/tmp/pya-kit /data/local/tmp/pya-kit.tar && echo PYA_OK'"

Write-Host ""
Write-Host "Fertig. Start:   adb shell su -c 'sh /data/adb/baseos/bin/pyaserver.sh start'"
Write-Host "Nutzen (Node):   barra-diarize aufnahme.wav [sprecherzahl]"
Write-Host "Hinweis: nicht gleichzeitig mit dem KI-Chat (llmserver) betreiben."
