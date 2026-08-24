# install-img.ps1 - Bildgenerator-Kit (Stable Diffusion 1.5, DreamShaper 8 LCM) auf einen barra-Node schieben (USB/adb).
# Inhalt img-kit.tar.gz: bin/ (sd-cli + sd-server, stable-diffusion.cpp mit Mali-GEMM- und Flash-Attention-Kernen)
# -> /data/local/barra-img/bin; base/imgserver.sh -> baseos/bin, base/barra-img -> Container /usr/local/bin.
# Modell DreamShaper8_LCM.safetensors (2,1 GB) + taesd.safetensors (10 MB) -> /data/local/barra-img/models.
# Manueller Start wie llm/stt/pya/tts/wake, KEIN Boot-Autostart. Dienst: Port 8096 (sdapi + OpenAI-Route).
$ErrorActionPreference = "Stop"
$kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$adb = Join-Path (Split-Path -Parent $kit) "tools\adb.exe"
if (-not (Test-Path $adb)) { $adb = "adb" }

Write-Host "== barra Bildgenerator-Kit: Stable Diffusion 1.5 DreamShaper 8 LCM =="
& $adb wait-for-device | Out-Null

Write-Host "1/4 Kit pushen (67 MB) ..."
& $adb push (Join-Path $kit "img-kit.tar.gz") /data/local/tmp/img-kit.tar.gz

Write-Host "2/4 verteilen (bin -> /data/local/barra-img, imgserver.sh -> baseos/bin, barra-img -> Container) ..."
& $adb shell "su -c 'D=/data/local/barra-img; mkdir -p `$D && cd `$D && rm -rf bin base && tar -xzf /data/local/tmp/img-kit.tar.gz && mkdir -p models && chmod -R 755 `$D/bin && cp base/imgserver.sh /data/adb/baseos/bin/imgserver.sh && chmod 755 /data/adb/baseos/bin/imgserver.sh && U=/data/local/ubuntu && cp base/barra-img `$U/usr/local/bin/barra-img && chmod 755 `$U/usr/local/bin/barra-img && rm -rf base /data/local/tmp/img-kit.tar.gz && echo IMG_OK'"

Write-Host "3/4 Modell pushen (2,1 GB + TAESD) ..."
& $adb push (Join-Path $kit "DreamShaper8_LCM.safetensors") /data/local/tmp/img-model.bin
& $adb push (Join-Path $kit "taesd.safetensors") /data/local/tmp/img-taesd.bin
& $adb shell "su -c 'D=/data/local/barra-img/models; mkdir -p `$D && mv /data/local/tmp/img-model.bin `$D/DreamShaper8_LCM.safetensors && mv /data/local/tmp/img-taesd.bin `$D/taesd.safetensors && chmod 644 `$D/* && ls -la `$D'"

Write-Host "4/4 fertig - kein Autostart."
Write-Host ""
Write-Host "Start:   adb shell su -c 'sh /data/adb/baseos/bin/imgserver.sh start'   (stop | status | log)"
Write-Host "Bild:    im Container  barra-img `"a red fox in the snow`" -o fuchs.png"
Write-Host "HTTP:    POST http://<node-ip>:8096/sdapi/v1/txt2img  {`"prompt`":`"...`"}  -> images[0] = base64-PNG"
