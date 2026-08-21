# ============================================================================
# stock-target.ps1 — the pinned Google factory build barra v0.1 targets.
# Single source of truth, dot-sourced by fetch-stock.ps1 AND barra-core.ps1.
# The SHA-256 equals the hash suffix in Google's own URL (self-checking).
# Newer builds: bump all three values (upgrades are anti-rollback-safe; the
# flasher never downgrades the bootloader).
# ============================================================================
$StockBuild = 'bp4a.260205.001'
$StockUrl   = 'https://dl.google.com/dl/android/aosp/akita-bp4a.260205.001-factory-661cb49b.zip'
$StockSha   = '661cb49bab85398126ef0bf44e4d81282c4c87e19f4dffa0e3f90017c32d08b0'
$StockZip   = Join-Path $env:TEMP ("akita-$StockBuild-factory.zip")

# Google platform-tools (adb.exe/fastboot.exe) — also fetched, never redistributed.
# "latest" is Google's stable alias; no pinned SHA (HTTPS direct from Google).
$PlatformToolsUrl = 'https://dl.google.com/android/repository/platform-tools-latest-windows.zip'
