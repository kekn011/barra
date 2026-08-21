﻿# Read-only-Test des Kerns gegen das angeschlossene Geraet (flasht NICHTS)
. "$PSScriptRoot\barra-core.ps1"
$script:Emit = { param($k,$d) if ($k -eq 'log') { return }; if ($d -is [hashtable]) { "[$k] " + (($d.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ') } else { "[$k] $d" } }
"tools: adb=$(Test-Path $script:ADB) fastboot=$(Test-Path $script:FB)"
"AdbState = '$(AdbState)'"
"FbState  = '$(FbState)'"
"DevState = '$(DevState)'"
if ((FbState) -in 'bootloader','fastbootd') {
  "unlocked           = $(FbVar 'unlocked')"
  "version-bootloader = $(FbVar 'version-bootloader')"
  "current-slot       = $(FbVar 'current-slot')"
  "is-userspace       = $(FbVar 'is-userspace')"
  $ai = Get-Content (Join-Path $script:Stock 'image\android-info.txt')
  $req = ($ai | Where-Object { $_ -match '^require version-bootloader=(.+)' } | ForEach-Object { $Matches[1] }) | Select-Object -First 1
  "req-bootloader     = $req  -> cur $(BlNum (FbVar 'version-bootloader')) vs req $(BlNum $req) -> upgrade noetig: $((BlNum (FbVar 'version-bootloader')) -lt (BlNum $req))"
}
"stock image dir ok: $(Test-Path (Join-Path $script:Stock 'image\super_empty.img'))"
"payload ok: $(Test-Path (Join-Path $script:Payload 'barra-base.tar.gz'))"
