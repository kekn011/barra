# install-dev.ps1 - barra Dev-Kit (Kernel-Werkbank) auf einen Node schieben (USB/adb).
# Inhalt barra-dev-kit.tar.gz:
#   dev/   -> /data/adb/baseos/dev   (Android-Seite: GPU/TPU-Kernelarbeit lebt in bionic;
#            devbuild/devdoctor, devkit-env, gpu-kernels-Quellen, tpu/ (tpuc1+libcomp_std),
#            third-party/bin/ (glslang, frida-server via fetch-or-supply), harness/)
#   base/  -> /data/adb/baseos/bin   (barra-dev-mode.sh, devdeploy.sh)
#   service.d/ -> /data/adb/service.d (55-barra-dev.sh: Dev-Mode-Persistenz-Hook)
# Dev-Mode wird NICHT automatisch aktiviert (opt-in): 'barra-dev-mode.sh on'.
$ErrorActionPreference = "Stop"
$kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$adb = Join-Path (Split-Path -Parent $kit) "tools\adb.exe"
if (-not (Test-Path $adb)) { $adb = "adb" }
$tar = Join-Path $kit "barra-dev-kit.tar.gz"
if (-not (Test-Path $tar)) { Write-Host "FEHLT: $tar  (erst 'pack' bauen - siehe SPEC-P1.md)"; exit 1 }

Write-Host "== barra Dev-Kit: Kernel-Werkbank (GPU/TPU/DSP + Dev-Mode) =="
& $adb wait-for-device | Out-Null

Write-Host "1/3 Kit pushen ..."
& $adb push $tar /data/local/tmp/barra-dev-kit.tar.gz

Write-Host "2/3 verteilen (dev -> baseos/dev, base -> baseos/bin, Hook -> service.d) ..."
& $adb shell "su -c 'cd /data/local/tmp && rm -rf barra-dev && mkdir barra-dev && cd barra-dev && tar -xzf ../barra-dev-kit.tar.gz && B=/data/adb/baseos && mkdir -p `$B/dev && cp -a dev/. `$B/dev/ && chmod -R 755 `$B/dev && cp base/barra-dev-mode.sh `$B/bin/barra-dev-mode.sh && cp base/devdeploy.sh `$B/bin/devdeploy.sh && chmod 755 `$B/bin/barra-dev-mode.sh `$B/bin/devdeploy.sh && cp service.d/55-barra-dev.sh /data/adb/service.d/55-barra-dev.sh && chmod 755 /data/adb/service.d/55-barra-dev.sh && cd /data/local/tmp && rm -rf barra-dev barra-dev-kit.tar.gz && echo DEV_OK'"

Write-Host "3/3 Selbsttest ..."
# /data/adb ist Magisk-root-700 -> Selbsttest MUSS als root laufen (Dev-Kit ist root-only, wie die Daemons).
$doc = & $adb shell "su -c 'sh /data/adb/baseos/dev/bin/devdoctor.sh'"
$doc | ForEach-Object { Write-Host "   $_" }

Write-Host ""
Write-Host "Fertig. KEIN Dev-Mode aktiviert (opt-in, ueberdauert dann Reboot):"
Write-Host "  adb shell su -c 'sh /data/adb/baseos/bin/barra-dev-mode.sh on|off|status'"
Write-Host "Werkbank:  adb shell  ->  . /data/adb/baseos/dev/devkit-env.sh  ->  devdoctor / devbuild"
Write-Host "WARNUNG: Dev-Mode (permissive) nur fuer Dev-Geraete - nie auf einem Produktiv-Node."
