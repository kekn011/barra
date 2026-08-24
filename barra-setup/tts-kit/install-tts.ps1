# install-tts.ps1 - TTS-Kit (Text-to-Speech: David GPU + Piper de/en) auf einen barra-Node schieben (USB/adb).
# Inhalt tts-kit.tar.gz: barra-tts/ (bin/gpudecd+ttsd.py+launch.sh, voices/{david,piper-de,piper-en},
# runtime/ eigene Python-Runtime, site/bajtts, sherpa/) -> /data/local/ubuntu/opt/barra-tts (chown 1001).
# barra-tts.service (Container-systemd, NICHT enabled) + ttsserver.sh (Steuerung). Manueller Start
# wie llm/stt/pya, KEIN Boot-Autostart. Danach: HTTP-Dienst Port 8095.
$ErrorActionPreference = "Stop"
$kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$adb = Join-Path (Split-Path -Parent $kit) "tools\adb.exe"
if (-not (Test-Path $adb)) { $adb = "adb" }

Write-Host "== barra TTS-Kit: David (GPU-Vokoder) + Piper Thorsten/Amy =="
& $adb wait-for-device | Out-Null

Write-Host "1/4 Kit + Steuerung pushen (~300 MB) ..."
& $adb push (Join-Path $kit "tts-kit.tar.gz") /data/local/tmp/tts-kit.tar.gz
& $adb push (Join-Path $kit "barra-tts.service") /data/local/tmp/barra-tts.service
& $adb push (Join-Path $kit "ttsserver.sh") /data/local/tmp/ttsserver.sh

Write-Host "2/4 Kit entpacken -> /opt/barra-tts (chown 1001) ..."
& $adb shell "su -c 'U=/data/local/ubuntu; cd `$U/opt && rm -rf barra-tts && tar -xzf /data/local/tmp/tts-kit.tar.gz && chown -R 1001:1001 barra-tts && echo EXTRACT_OK'"

Write-Host "3/4 Dienst + Steuerung installieren (kein Autostart) ..."
& $adb shell "su -c 'U=/data/local/ubuntu; cp /data/local/tmp/barra-tts.service `$U/etc/systemd/system/barra-tts.service && cp /data/local/tmp/ttsserver.sh /data/adb/baseos/bin/ttsserver.sh && chmod 755 /data/adb/baseos/bin/ttsserver.sh && rm -f /data/local/tmp/tts-kit.tar.gz /data/local/tmp/barra-tts.service /data/local/tmp/ttsserver.sh && echo SERVICE_OK'"

Write-Host "4/4 einmal starten (Sofort-Test; nach Reboot: ttsserver.sh start) ..."
$act = & $adb shell "su -M -c 'sh /data/adb/baseos/bin/ttsserver.sh start'"
Write-Host ("   " + ($act | Select-Object -Last 2))

Write-Host ""
Write-Host "Fertig. Steuerung:  adb shell su -M -c 'sh /data/adb/baseos/bin/ttsserver.sh start|stop|status'"
Write-Host "Test:   curl 'http://<node>:8095/say?voice=david&text=Hallo'"
Write-Host "Stimmen: david (Erzaehler, GPU), piper-de (Thorsten), piper-en (Amy)"
