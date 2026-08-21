# ============================================================================
# build-super.ps1 — assemble stock\super.img from the unpacked factory logical
# partitions with lpmake (tools\lpmake.exe). This lets barra-setup flash the
# system on the driver-free bootloader path ("Weg B", no Windows fastbootd
# driver needed) instead of streaming logical partitions through fastbootd.
#
# Geometry is pinned for akita (from super_empty.img of the targeted factory
# image); partition sizes are read from the .img files at runtime, so a factory
# update only needs the pin in fetch-stock.ps1 bumped, not this file.
# ============================================================================
param([Parameter(Mandatory=$true)][string]$Stock)
$ErrorActionPreference = 'Stop'
$Kit    = Split-Path -Parent $MyInvocation.MyCommand.Path
$lpmake = Join-Path $Kit 'tools\lpmake.exe'
$img    = Join-Path $Stock 'image'
$out    = Join-Path $Stock 'super.img'

if (-not (Test-Path $lpmake)) {
  throw "tools\lpmake.exe is missing. barra builds it from AOSP for Windows; see docs/building-lpmake.md."
}

# --- pinned akita super geometry (super_empty.img, virtual A/B) ---
$SuperSize = 8531214336
$GroupSize = 8527020032
$MetaSize  = 65536
$MetaSlots = 3
$Parts = 'system','system_dlkm','system_ext','product','vendor','vendor_dlkm'

$a = @(
  '--metadata-size', $MetaSize, '--metadata-slots', $MetaSlots, '--super-name', 'super',
  '--virtual-ab', '--device', "super:$SuperSize",
  '--group', "google_dynamic_partitions_a:$GroupSize",
  '--group', "google_dynamic_partitions_b:$GroupSize"
)
foreach ($p in $Parts) {
  $f = Join-Path $img "$p.img"
  if (-not (Test-Path $f)) { throw "missing partition image: $f" }
  $sz = (Get-Item $f).Length
  # slot A carries the data; slot B is created empty (retrofit A/B, updated via OTA)
  $a += '--partition', "${p}_a:readonly:${sz}:google_dynamic_partitions_a"
  $a += '--image', "${p}_a=$f"
  $a += '--partition', "${p}_b:readonly:0:google_dynamic_partitions_b"
}
$a += '--sparse', '--output', $out

Write-Host "Building super.img with lpmake (~6.3 GB, a minute or two) …"
# lpmake logs progress on stderr. In PS 5.1 ANY stderr redirection of a native
# command wraps lines in ErrorRecords — with EAP=Stop the first info line would
# become a terminating NativeCommandError. Lower EAP around the call; only the
# exit code matters.
$errLog = Join-Path $env:TEMP 'barra-lpmake.err'
$prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
& $lpmake @a 2> $errLog
$code = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($code -ne 0) {
  Remove-Item $out -Force -ErrorAction SilentlyContinue   # no truncated super.img left behind
  Write-Host (Get-Content $errLog -Tail 5 -ErrorAction SilentlyContinue)
  throw "lpmake failed (exit $code) — see $errLog"
}
Write-Host "OK: $out ($('{0:n1} GB' -f ((Get-Item $out).Length/1e9)))"
