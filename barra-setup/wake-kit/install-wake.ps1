# install-wake.ps1 - Weckwort-Kit ("Hey Barra") auf einen barra-Node schieben (USB/adb).
# Inhalt wake-kit.tar.gz: kit/ (wake-bridge + KWS-Modelle int8 + lib + tokens/keywords/on-wake.sh)
# -> /data/local/ubuntu/opt/barra-wake (chown 1001); base/ (audiod-record Mic-Bridge -> hwbridge,
# wakeserver.sh -> baseos/bin, barra-wake.service Container-systemd NICHT enabled).
# Manueller Start wie llm/stt/pya/tts, KEIN Boot-Autostart. Mic-Route setzt av-setup (Base).
$ErrorActionPreference = "Stop"
$kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$adb = Join-Path (Split-Path -Parent $kit) "tools\adb.exe"
if (-not (Test-Path $adb)) { $adb = "adb" }

Write-Host "== barra Weckwort-Kit: 'Hey Barra' (sherpa-onnx keyword spotter) =="
& $adb wait-for-device | Out-Null

Write-Host "1/3 Kit pushen (17 MB) ..."
& $adb push (Join-Path $kit "wake-kit.tar.gz") /data/local/tmp/wake-kit.tar.gz

Write-Host "2/3 verteilen (kit -> /opt/barra-wake, base -> hwbridge/baseos; kein Autostart) ..."
& $adb shell "su -c 'cd /data/local/tmp && rm -rf wake-kit && mkdir wake-kit && cd wake-kit && tar -xzf ../wake-kit.tar.gz && U=/data/local/ubuntu && mkdir -p `$U/opt/barra-wake && cp -a kit/. `$U/opt/barra-wake/ && chown -R 1001:1001 `$U/opt/barra-wake && cp base/audiod-record /data/adb/hwbridge/audiod-record && chmod 755 /data/adb/hwbridge/audiod-record && cp base/wakeserver.sh /data/adb/baseos/bin/wakeserver.sh && chmod 755 /data/adb/baseos/bin/wakeserver.sh && cp base/barra-wake.service `$U/etc/systemd/system/barra-wake.service && cd /data/local/tmp && rm -rf wake-kit wake-kit.tar.gz && echo WAKE_OK'"

Write-Host "3/3 einmal starten (Sofort-Test; nach Reboot: wakeserver.sh start) ..."
$act = & $adb shell "su -M -c 'sh /data/adb/baseos/bin/wakeserver.sh start'"
Write-Host ("   " + ($act | Select-Object -Last 2))

Write-Host ""
Write-Host "Fertig. Steuerung:  adb shell su -M -c 'sh /data/adb/baseos/bin/wakeserver.sh start|stop|status'"
Write-Host "Sag 'Hey Barra' - Treffer landet in /run/barra-wake/trigger (Hook: /opt/barra-wake/on-wake.sh)."
