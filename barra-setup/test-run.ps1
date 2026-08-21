﻿. "$PSScriptRoot\barra-core.ps1"
$script:Emit = { param($k,$d) }
$r = Run $script:FB 'devices' $null 15
"code=$($r.code) lines=$($r.lines.Count)"
$r.lines | ForEach-Object { "  [$_]" }
$r2 = Run $script:FB "getvar is-userspace" $null 10
"getvar code=$($r2.code): " + ($r2.lines -join ' | ')
"FbVar unlocked = '$(FbVar 'unlocked')'"
