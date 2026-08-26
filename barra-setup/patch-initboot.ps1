# ============================================================================
# patch-initboot.ps1 — produce payload\init_boot-magisk.img locally, on the
# user's PC, from the stock init_boot.img + the official Magisk APK. barra does
# NOT ship a pre-patched init_boot (it would contain Google's stock ramdisk);
# both inputs are obtained from their sources at setup time.
#
# It runs Magisk's own boot_patch.sh (from the APK) on the connected phone as
# the plain adb shell user — NO root required (magiskboot is a self-contained
# binary that runs fine from /data/local/tmp). That matters because barra-core
# calls this right BEFORE root exists, on the freshly flashed stock system.
# The payload is pushed as a script file (never inlined through PowerShell —
# PS strips quotes before adb sees them).
#
# Inputs : stock\image\init_boot.img , payload\Magisk-*.apk
# Output : payload\init_boot-magisk.img
# ============================================================================
param(
  [string]$Stock   = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'stock'),
  [string]$Payload = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'payload')
)
$ErrorActionPreference = 'Stop'
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$adb = Join-Path $Kit 'tools\adb.exe'
$src = Join-Path $Stock 'image\init_boot.img'
$apk = Get-ChildItem (Join-Path $Payload 'Magisk-*.apk') | Select-Object -First 1
$out = Join-Path $Payload 'init_boot-magisk.img'

if (-not (Test-Path $src)) { throw "stock init_boot.img missing ($src) — run fetch-stock.ps1 first" }
if (-not $apk) {
  # Frueher brach das hier ab und schickte den Nutzer von Hand zu GitHub. Die APK ist in
  # models.psd1 gepinnt (URL + SHA-256), also holen wir sie selbst - barra verteilt sie nicht.
  $fetch = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'fetch-models.ps1'
  if (Test-Path $fetch) {
    Write-Host "Magisk APK fehlt - wird von der Originalquelle geladen (GPL-3.0, topjohnwu/Magisk) ..."
    & $fetch -Id magisk -Yes
    $apk = Get-ChildItem (Join-Path $Payload 'Magisk-*.apk') -ErrorAction SilentlyContinue | Select-Object -First 1
  }
}
if (-not $apk) { throw "Magisk APK missing in $Payload - fetch-models.ps1 -Id magisk holt sie; Quelle: https://github.com/topjohnwu/Magisk/releases" }

Write-Host "Patching init_boot.img with $($apk.Name) on the connected device (no root needed) …"
$sh = @'
set -e
cd /data/local/tmp/barra-ib
rm -rf mp && mkdir mp && cd mp
unzip -o ../magisk.apk 'lib/arm64-v8a/*' 'assets/*' >/dev/null
for f in lib/arm64-v8a/lib*.so; do cp "$f" "$(basename "$f" | sed 's/^lib//;s/\.so$//')"; done
cp assets/* . 2>/dev/null || true
chmod 755 magiskboot magiskinit ./*.sh 2>/dev/null || true
export KEEPVERITY=true KEEPFORCEENCRYPT=true
sh ./boot_patch.sh ../init_boot.img
mv new-boot.img ../init_boot-magisk.img
echo BARRA_PATCHED
'@ -replace "`r",""
$tmp = Join-Path $env:TEMP 'barra-patch-ib.sh'
[IO.File]::WriteAllText($tmp, $sh + "`n", [Text.Encoding]::ASCII)

& $adb wait-for-device | Out-Null
& $adb shell "mkdir -p /data/local/tmp/barra-ib" | Out-Null
& $adb push $src /data/local/tmp/barra-ib/init_boot.img | Out-Null
& $adb push $apk.FullName /data/local/tmp/barra-ib/magisk.apk | Out-Null
& $adb push $tmp /data/local/tmp/barra-ib/patch.sh | Out-Null
Remove-Item $tmp -Force -ErrorAction SilentlyContinue
# stderr mit einsammeln, aber ohne 2>&1-Falle (PS 5.1 + EAP=Stop macht aus nativen
# stderr-Zeilen sonst terminierende NativeCommandErrors)
$prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
$r = (& $adb shell "sh /data/local/tmp/barra-ib/patch.sh" 2>&1) | ForEach-Object { "$_" }
$ErrorActionPreference = $prevEap
if (($r -join "`n") -notmatch 'BARRA_PATCHED') { Write-Host ($r -join "`n"); throw 'boot_patch failed on device' }
& $adb pull /data/local/tmp/barra-ib/init_boot-magisk.img $out | Out-Null
& $adb shell "rm -rf /data/local/tmp/barra-ib" | Out-Null
if (-not (Test-Path $out) -or (Get-Item $out).Length -lt 1000000) { throw "pull failed or output too small: $out" }
Write-Host "OK: $out ($('{0:n0} bytes' -f (Get-Item $out).Length))"
