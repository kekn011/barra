# install-stt.ps1 — Whisper-STT-Kit auf einen barra-Node schieben (Base-Image v10+, per USB/adb).
# Inhalt: whisper-kit-turbo.tar (103 TPU-Packages fuer large-v3-turbo: Encoder+Cross-K/V+Conv-
# Frontend+Kern + Params -> /data/local/barra-stt/turbo) + ggml-large-v3-turbo-q5_0.bin
# (Decoder-Modell, q5_0 -> /data/local/barra-stt/models).
# Binaries (whisper-server, tpud_pipe4) und sttserver.sh sind ab v10 im Base-Image.
# Danach: sttserver.sh start -> HTTP-Transkription Port 8090 (greedy, TPU-Encoder, 29s-Audio ~7s).
$ErrorActionPreference = "Stop"
$kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$adb = Join-Path (Split-Path -Parent $kit) "tools\adb.exe"
if (-not (Test-Path $adb)) { $adb = "adb" }

Write-Host "== barra STT-Kit: whisper large-v3-turbo =="
& $adb wait-for-device | Out-Null

Write-Host "1/3 TPU-Packages pushen (670 MB) ..."
& $adb push (Join-Path $kit "whisper-kit-turbo.tar") /data/local/tmp/whisper-kit.tar
& $adb shell "su -c 'mkdir -p /data/local/barra-stt && cd /data/local/barra-stt && tar -xf /data/local/tmp/whisper-kit.tar && mkdir -p models && chmod -R 755 /data/local/barra-stt && rm /data/local/tmp/whisper-kit.tar'"

Write-Host "2/3 Decoder-Modell pushen (547 MB) ..."
& $adb push (Join-Path $kit "ggml-large-v3-turbo-q5_0.bin") /data/local/tmp/stt-model.bin
& $adb shell "su -c 'mv /data/local/tmp/stt-model.bin /data/local/barra-stt/models/ggml-large-v3-turbo-q5_0.bin'"

Write-Host "3/3 STT-Server starten ..."
& $adb shell "su -c 'sh /data/adb/baseos/bin/sttserver.sh start'"

Write-Host ""
Write-Host "Fertig. Transkription: curl -F file=@x.wav -F language=de http://<node-ip>:8090/inference"
Write-Host "Status:  adb shell su -c 'sh /data/adb/baseos/bin/sttserver.sh status'"
