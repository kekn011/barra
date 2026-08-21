# ============================================================================
# fetch-stock.ps1 — download the Google factory image for the supported build
# and unpack it into .\stock, so barra-setup can flash from it. Because the
# factory image is Google's property, barra does NOT ship it; each user fetches
# it from Google's own server here, after accepting Google's terms.
#
# What it does:
#   1. shows Google's terms link and requires acknowledgement (unless -Yes)
#   2. downloads akita-<build>-factory-<hash>.zip from dl.google.com
#   3. verifies the SHA-256 (the hash is also Google's URL suffix — self-checking)
#   4. unpacks it into .\stock\ (bootloader-*.img, radio-*.img, image\ ...)
#   5. builds .\stock\super.img from the logical partitions with lpmake
#      (tools\lpmake.exe) — this keeps flashing on the driver-free bootloader path
#
# Pinned to the build barra v0.1 targets. Newer builds: bump $Build/$Sha/$Url
# (upgrades are anti-rollback-safe; the flasher never downgrades the bootloader).
# ============================================================================
param([switch]$Yes)
$ErrorActionPreference = 'Stop'
$Kit   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Stock = Join-Path $Kit 'stock'
$Tools = Join-Path $Kit 'tools'

# --- pinned target (verified 2026-08-21) comes from stock-target.ps1 (shared with barra-core) ---
. (Join-Path $Kit 'stock-target.ps1')
$Build = $StockBuild; $Url = $StockUrl; $Sha = $StockSha; $Zip = $StockZip
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# --- platform-tools (adb/fastboot) — fetched from Google, never redistributed ---
if (-not (Test-Path (Join-Path $Tools 'adb.exe')) -or -not (Test-Path (Join-Path $Tools 'fastboot.exe'))) {
  Write-Host "Fetching Google platform-tools (adb/fastboot) — subject to the Android SDK terms:"
  Write-Host "  https://developer.android.com/studio/terms"
  $ptZip = Join-Path $env:TEMP 'platform-tools-windows.zip'
  try { Start-BitsTransfer -Source $PlatformToolsUrl -Destination $ptZip -DisplayName 'barra: platform-tools' }
  catch { Invoke-WebRequest -Uri $PlatformToolsUrl -OutFile $ptZip }
  $ptDir = Join-Path $env:TEMP 'barra-platform-tools'
  Remove-Item $ptDir -Recurse -Force -ErrorAction SilentlyContinue
  Expand-Archive -Path $ptZip -DestinationPath $ptDir -Force
  New-Item -ItemType Directory -Force $Tools | Out-Null
  Get-ChildItem (Join-Path $ptDir 'platform-tools') | ForEach-Object { Move-Item $_.FullName $Tools -Force }
  Remove-Item $ptDir -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item $ptZip -Force -ErrorAction SilentlyContinue
  Write-Host "OK: adb/fastboot in $Tools"
}

Write-Host "== barra: fetch Google factory image ($Build) =="
Write-Host ""
Write-Host "The Google factory image is downloaded from Google's server and is"
Write-Host "subject to Google's terms:"
Write-Host "  https://developers.google.com/android/images#legal"
Write-Host "barra does not redistribute it."
Write-Host ""
if (-not $Yes) {
  $a = Read-Host "Type 'yes' to accept Google's terms and download"
  if ($a -ne 'yes') { Write-Host 'Aborted.'; exit 1 }
}

if ((Test-Path $Zip) -and ((Get-FileHash $Zip -Algorithm SHA256).Hash -eq $Sha.ToUpper())) {
  Write-Host "Already downloaded and verified: $Zip"
} else {
  Write-Host "Downloading (~3.5 GB) …"
  # BITS gives a progress bar and resumes; fall back to Invoke-WebRequest.
  try { Start-BitsTransfer -Source $Url -Destination $Zip -DisplayName 'barra: factory image' }
  catch { Invoke-WebRequest -Uri $Url -OutFile $Zip }
  Write-Host "Verifying SHA-256 …"
  $got = (Get-FileHash $Zip -Algorithm SHA256).Hash
  if ($got -ne $Sha.ToUpper()) { throw "SHA-256 mismatch! expected $Sha got $got — delete $Zip and retry." }
  Write-Host "OK ($got)"
}

Write-Host "Unpacking into $Stock …"
New-Item -ItemType Directory -Force $Stock | Out-Null
Expand-Archive -Path $Zip -DestinationPath $Stock -Force
# The zip unpacks as stock\akita-<build>\{bootloader,radio,image-<build>.zip,flash-all.*}
$inner = Get-ChildItem $Stock -Directory | Where-Object { $_.Name -like 'akita-*' } | Select-Object -First 1
if (-not $inner) { throw "unexpected factory zip layout under $Stock" }
Get-ChildItem $inner.FullName -File | ForEach-Object { Move-Item $_.FullName $Stock -Force }
$imgzip = Get-ChildItem $Stock -Filter 'image-akita-*.zip' | Select-Object -First 1
if (-not $imgzip) { throw "image-akita-*.zip not found in factory image" }
$imgdir = Join-Path $Stock 'image'
New-Item -ItemType Directory -Force $imgdir | Out-Null
Expand-Archive -Path $imgzip.FullName -DestinationPath $imgdir -Force
Remove-Item $imgzip.FullName -Force -ErrorAction SilentlyContinue   # unpacked — save the 3.6 GB
Remove-Item $inner.FullName -Recurse -Force -ErrorAction SilentlyContinue

# --- build super.img from the logical partitions (driver-free bootloader flash path) ---
& (Join-Path $Kit 'build-super.ps1') -Stock $Stock
Write-Host ""
Write-Host "Done. .\stock is ready — run barra-setup to flash."
