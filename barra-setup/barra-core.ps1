# ============================================================================
# barra-core.ps1 — Windows-nativer Flash-Kern (adb.exe + fastboot.exe aus .\tools).
# Kein WSL, kein usbipd, kein Treiber, keine Admin-Rechte. Wird von der GUI (barra-setup.ps1)
# geladen; laeuft in einem Hintergrund-Runspace und meldet Ereignisse ueber $Emit:
#   Emit 'step'   @{ n=<0..5>; title=...; state='run|wait_user|wait_dev|ok|fail' }
#   Emit 'log'    'text'                       (rohe Zeile fuers Detail-Log)
#   Emit 'info'   'text'    | 'ok' | 'warn' | 'fail'
#   Emit 'tel'    'text'                       (📱 am Telefon)
#   Emit 'ask'    @{ text=...; id=... }        -> Antwort ueber $Answer-Queue ('j'/'n'/'ok')
#   Emit 'progress' @{ pct=..; text=.. }       (Flash-Fortschritt)
#   Emit 'done'   @{ host=..; ip=..; user=.. }
# UI-Texte kommen aus den i18n-Katalogen (T 'core.*' — die GUI initialisiert den Loader im
# Runspace VOR dem Laden dieser Datei); ohne Loader (Standalone-Test) faellt T auf den Key zurueck.
# Die Logik ist 1:1 aus dem bewaehrten Wizard (barra-flash) uebernommen: Zustandserkennung,
# Bootloader-Upgrade nur bei aelterem Stand, fastbootd-Wiederaufnahme, Magisk-Warten,
# Base-Install per device-install.sh, Erst-Boot-Wartung.
# ============================================================================
Set-StrictMode -Off
if (-not (Get-Command T -ErrorAction SilentlyContinue)) { function T([string]$Key){ $Key } }
$script:Kit    = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:Tools  = Join-Path $script:Kit 'tools'
$script:ADB    = Join-Path $script:Tools 'adb.exe'
$script:FB     = Join-Path $script:Tools 'fastboot.exe'
$script:Payload= Join-Path $script:Kit 'payload'
$script:Stock  = Join-Path $script:Kit 'stock'          # image\ + bootloader-*.img + radio-*.img
$script:Emit   = { param($kind,$data) }                 # GUI setzt das
$script:Answer = [System.Collections.Concurrent.ConcurrentQueue[string]]::new()
$script:Cancel = $false
$script:PreCfg = $null                                  # Pre-Einrichtung (Hashtable) oder $null

function Emit($kind,$data){ & $script:Emit $kind $data }
function Log($t){ Emit 'log' $t }
function Info($t){ Emit 'info' $t }
function Ok($t){ Emit 'ok' $t }
function Warn($t){ Emit 'warn' $t }
function Fail($t){ Emit 'fail' $t; throw "FAIL: $t" }
function Tel($t){ Emit 'tel' $t }
function Step($n,$title,$state){ Emit 'step' @{ n=$n; title=$title; state=$state } }
function Progress($pct,$text){ Emit 'progress' @{ pct=$pct; text=$text } }
function Ask($text,$id='ask'){
  Emit 'ask' @{ text=$text; id=$id }
  $a=$null; while (-not $script:Answer.TryDequeue([ref]$a)) { if ($script:Cancel) { throw 'CANCEL' }; Start-Sleep -Milliseconds 200 }
  return $a
}
function WaitUser($text){ [void](Ask $text 'ok') }
function Chk(){ if ($script:Cancel) { throw 'CANCEL' } }
function MMSS($el){ '{0:d2}:{1:d2}' -f [int]($el/60), ($el%60) }

# ---- Prozess-Aufrufe (ohne Fenster, mit Zeilen-Callback) ---------------------
function Run($exe,$argstr,[scriptblock]$onLine,$timeoutSec=600,$live='err'){
  # Robust + einfach: fastboot schreibt Fortschritt/Ergebnis auf STDERR, adb push auf STDOUT (mit CR statt Zeilenumbruch).
  # Den "live"-Strom lesen wir zeilenweise (ReadLine trennt auch an CR -> Fortschritt sofort),
  # den anderen asynchron komplett; alles ins Log.
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName=$exe; $psi.Arguments=$argstr; $psi.UseShellExecute=$false; $psi.CreateNoWindow=$true
  $psi.RedirectStandardOutput=$true; $psi.RedirectStandardError=$true
  $p = New-Object System.Diagnostics.Process; $p.StartInfo=$psi
  [void]$p.Start()
  if ($live -eq 'out') { $liveRd=$p.StandardOutput; $outTask=$p.StandardError.ReadToEndAsync() }
  else                 { $liveRd=$p.StandardError;  $outTask=$p.StandardOutput.ReadToEndAsync() }
  $lines = New-Object System.Collections.ArrayList
  $t0=Get-Date
  while ($true) {
    $l = $null
    $rt = $liveRd.ReadLineAsync()
    while (-not $rt.Wait(200)) {
      if (((Get-Date)-$t0).TotalSeconds -gt $timeoutSec) { try{$p.Kill()}catch{}; break }
      if ($script:Cancel) { try{$p.Kill()}catch{}; throw 'CANCEL' }
    }
    if (-not $rt.IsCompleted) { break }
    $l = $rt.Result
    if ($l -eq $null) { break }
    [void]$lines.Add($l); Log $l; if ($onLine) { & $onLine $l }
  }
  $p.WaitForExit()
  $so = $outTask.Result
  if ($so) { foreach ($l in ($so -split "`r?`n")) { if ($l -ne '') { [void]$lines.Add($l); Log $l; if ($onLine) { & $onLine $l } } } }
  return @{ code=$p.ExitCode; lines=@($lines) }
}
function AdbOut($a,$to=20){ (Run $script:ADB $a $null $to).lines -join "`n" }
function FbOut($a,$to=20){ (Run $script:FB  $a $null $to).lines -join "`n" }
function AdbSh($cmd,$to=30){ (AdbOut "shell $cmd" $to).Trim() }
function AdbSu($cmd,$to=60){ (AdbOut "shell su -c `"$cmd`"" $to).Trim() }

# ---- Zustandserkennung ---------------------------------------------------------
function AdbState(){ $s=(AdbOut 'get-state' 8).Trim(); if ($s -eq 'device') {'device'} elseif ($s -match '^unauthorized') {'unauthorized'} else {''} }
function FbState(){ $d=FbOut 'devices' 8; if ($d -match 'fastboot') { $u=FbOut 'getvar is-userspace' 8; if ($u -match 'is-userspace:\s*yes') {'fastbootd'} else {'bootloader'} } else {''} }
function FbVar($v){ $o=FbOut "getvar $v" 8; if ($o -match "$([regex]::Escape($v)):\s*(.+)") { $Matches[1].Trim() } else { '' } }
function DevState(){ $a=AdbState; if ($a) { return $a }; $f=FbState; if ($f) { return $f }; '' }
function UsbHint(){ Tel (T 'core.usb_hint') }

function WaitFor($what,$sec=180,$label=$null){  # what: device|bootloader|fastbootd|any-fb
  $t0=Get-Date; $lastHint=0
  while (((Get-Date)-$t0).TotalSeconds -lt $sec) {
    Chk
    $st = DevState
    $hit = switch ($what) { 'device' { $st -eq 'device' } 'bootloader' { $st -eq 'bootloader' } 'fastbootd' { $st -eq 'fastbootd' } 'any-fb' { $st -in 'bootloader','fastbootd' } }
    if ($hit) { return $true }
    if ($st -eq 'unauthorized') { Tel (T 'core.allow_debug') }
    $el=[int]((Get-Date)-$t0).TotalSeconds
    $lb = if ($label) { $label } else { (T 'core.wait_for' $what) }
    Progress -1 ("{0} · {1}" -f $lb, (MMSS $el))
    if ($el -ge 40 -and $lastHint -eq 0) { UsbHint; $lastHint=$el }
    Start-Sleep 3
  }
  return $false
}

# ---- Schritte -------------------------------------------------------------------
function Step0_Verbindung(){
  Step 0 (T 'setup.steps.connect') 'wait_dev'
  Info (T 'core.s0.intro')
  $t0=Get-Date; $hinted=$false
  while (((Get-Date)-$t0).TotalSeconds -lt 180) {
    Chk; $st=DevState
    if ($st -eq 'device') { Ok (T 'core.s0.connected'); Step 0 (T 'setup.steps.connect') 'ok'; return }
    if ($st -in 'bootloader','fastbootd') { Ok (T 'core.s0.mode' $st); Step 0 (T 'setup.steps.connect') 'ok'; return }
    if ($st -eq 'unauthorized') { Step 0 (T 'setup.steps.connect') 'wait_user'; Tel (T 'core.allow_debug') }
    $el=[int]((Get-Date)-$t0).TotalSeconds; Progress -1 ("{0} · {1}" -f (T 'core.s0.searching'), (MMSS $el))
    if ($el -ge 30 -and -not $hinted) { UsbHint; $hinted=$true }
    Start-Sleep 3
  }
  Step 0 (T 'setup.steps.connect') 'fail'; Fail (T 'core.s0.notfound')
}

function Step1_Unlock(){
  Step 1 (T 'setup.steps.unlock') 'run'
  if ((AdbState) -eq 'device') {
    $L=AdbSh 'getprop ro.boot.flash.locked'; $OEM=AdbSh 'getprop sys.oem_unlock_allowed'
    if ($L -eq '0') { Ok (T 'core.s1.already'); Step 1 (T 'setup.steps.unlock') 'ok'; return }
    if ($OEM -ne '1') {
      Step 1 (T 'setup.steps.unlock') 'wait_user'
      Tel (T 'core.s1.dev_opts')
      Tel (T 'core.s1.oem_on')
      WaitUser (T 'core.s1.when_done')
    }
    Info (T 'core.s1.to_bl'); [void](AdbOut 'reboot bootloader' 10); Start-Sleep 3
    if (-not (WaitFor 'bootloader' 120 (T 'core.fs.wait_bl'))) { Fail (T 'core.s1.bl_unreach') }
  }
  $st=FbState
  if ($st -eq 'fastbootd') { Ok (T 'core.s1.fastbootd_ok'); Step 1 (T 'setup.steps.unlock') 'ok'; return }
  if ($st -ne 'bootloader') { Fail (T 'core.s1.neither') }
  if ((FbVar 'unlocked') -eq 'yes') { Ok (T 'core.s1.unlocked'); Step 1 (T 'setup.steps.unlock') 'ok'; return }
  Step 1 (T 'setup.steps.unlock') 'wait_user'
  Warn (T 'core.s1.wipe_warn')
  if ((Ask (T 'core.s1.unlock_q') 'yn') -ne 'j') { throw 'CANCEL' }
  [void](FbOut 'flashing unlock' 10)
  Tel (T 'core.s1.tel_unlock')
  $t0=Get-Date
  while (((Get-Date)-$t0).TotalSeconds -lt 240) { Chk; if ((FbVar 'unlocked') -eq 'yes') { Ok (T 'core.s1.unlocked'); Step 1 (T 'setup.steps.unlock') 'ok'; return }; $el=[int]((Get-Date)-$t0).TotalSeconds; Progress -1 ("{0} · {1}" -f (T 'core.s1.wait_confirm'), (MMSS $el)); Start-Sleep 3 }
  Fail (T 'core.s1.not_confirmed')
}

# Bootloader-Version vergleichen: 'akita-14.5-...' -> 14.5
function BlNum($s){ if ($s -match 'akita-(\d+\.\d+)') { [double]$Matches[1] } else { 0 } }
# --- schritt-weiter Fortschritt (EIN Balken pro Schritt, byte-basiert) -----------------------
# PbBegin(bytes) legt das Budget eines Schritts fest; jeder FbFlash rechnet seinen Anteil ein
# (Sparse-Teile a/b werden innerhalb der Datei interpoliert). Ohne Budget: Balken pro Datei.
$script:PbTotal = 0; $script:PbDone = 0
function PbBegin($bytes){ $script:PbTotal=[double]$bytes; $script:PbDone=[double]0 }
function PbEnd(){ $script:PbTotal=0; $script:PbDone=0 }
function GB($b){ '{0:n1} GB' -f ($b/1e9) }
function PbShow($doneBytes,$label){
  if ($script:PbTotal -le 0) { return }
  # beide als double: mit Int32-0 als 1. Argument waehlt PS 5.1 Min(int,int) -> Overflow bei >2 GB (super.img)
  $d=[Math]::Min([double]$doneBytes,[double]$script:PbTotal)
  Progress ([int](100*$d/$script:PbTotal)) ("{0} · {1} / {2}" -f $label,(GB $d),(GB $script:PbTotal))
}

function FbFlash($part,$file,$label,$chunk=''){  # mit Fortschritts-Parsing (Sending sparse 'x' 3/16)
  # Windows-WinUSB bricht bei grossen Bulk-Transfers (Standard-Chunk ~256 MB) mit Fehler 31
  # ("AdbWriteEndpointSync failed") ab -> grosse Images mit kleineren Sparse-Chunks senden (-S 64M).
  # Bei Fehler 31 automatisch einmal mit noch kleineren Chunks (32M) wiederholen.
  $sizeArg = if ($chunk) { "-S $chunk " } else { '' }
  $size = [double](Get-Item $file).Length
  $base = $script:PbDone
  if ($script:PbTotal -gt 0) { PbShow $base $label } else { Progress 0 (T 'core.ff.sending' $label) }
  # Sparse-Zaehler (Teil a von b) fuer diese Datei: "Writing" nach einem Sparse-Teil = Teil a fertig
  # (a/b), NICHT pauschal 95 % — sonst pendelt der Balken pro Teil zwischen a/b und 95 %.
  $script:SpA = 0; $script:SpB = 0
  $r = Run $script:FB "${sizeArg}flash $part `"$file`"" { param($l)
    if ($l -match "Sending sparse '[^']+' (\d+)/(\d+)") { $a=[int]$Matches[1]; $b=[int]$Matches[2]; $script:SpA=$a; $script:SpB=$b
      if ($script:PbTotal -gt 0) { PbShow ($base + $size*($a-1)/$b) $label } else { Progress ([int](100*($a-1)/$b)) (T 'core.ff.part' $label $a $b) } }
    elseif ($l -match '^Writing ') {
      if ($script:SpB -gt 0) { $f = $script:SpA/$script:SpB } else { $f = 0.95 }
      if ($script:PbTotal -gt 0) { PbShow ($base + $size*$f) $label } else { Progress ([int](100*$f)) (T 'core.ff.writing' $label) } }
  } 1800
  $txt = $r.lines -join "`n"
  if ($r.code -ne 0 -or $txt -match 'FAILED|error:') {
    if ($txt -match 'AdbWriteEndpointSync|\(31\)|Write to device failed' -and $chunk -ne '32M') {
      Warn (T 'core.ff.usb_retry' $part)
      Start-Sleep 3
      if (-not (WaitFor 'bootloader' 60 (T 'core.fs.wait_bl'))) { throw (T 'core.ff.gone' $part) }
      $script:PbDone = $base
      return (FbFlash $part $file $label '32M')
    }
    throw (T 'core.ff.failed' $part)
  }
  $script:PbDone = $base + $size
  if ($script:PbTotal -gt 0) { PbShow $script:PbDone $label }
}

# ---- Stock-Beschaffung (Rechts-Umbau: das Google-Factory-Image wird NICHT mitgeliefert) ------
# Fehlen stock\image bzw. stock\super.img, laedt der Flow das Factory-Zip von Googles Server
# (Nutzer bestaetigt Googles Bedingungen ueber den Ask-Mechanismus), prueft die SHA-256 und
# ruft dann fetch-stock.ps1 -Yes als Kindprozess fuers Entpacken + super.img-Bau (lpmake) —
# fetch-stock findet das verifizierte Zip, ueberspringt den Download: EINE Logik-Quelle.
function Download-Url($url,$dst,$label){
  # BITS zuerst (Dienst, resuemiert selbst; Foreground = ungedrosselt), Fallback direkter
  # HTTPS-Stream. Beide melden Fortschritt ueber Progress und brechen auf Cancel ab.
  $job=$null
  try {
    Import-Module BitsTransfer -ErrorAction Stop
    $job = Start-BitsTransfer -Source $url -Destination $dst -Asynchronous -Priority Foreground -DisplayName 'barra-stock'
    while ($true) {
      if ($script:Cancel) { throw 'CANCEL' }
      $st = "$($job.JobState)"
      if ($st -eq 'Transferred') { Complete-BitsTransfer -BitsJob $job; $job=$null; return }
      if ($st -in 'Error','Fatal','Cancelled') { throw "BITS: $($job.ErrorDescription)" }
      # BytesTotal ist UInt64.MaxValue solange BITS die Groesse noch nicht kennt
      if ($job.BytesTotal -gt 0 -and $job.BytesTotal -lt [uint64]::MaxValue) { Progress ([int](100*$job.BytesTransferred/$job.BytesTotal)) ("{0} · {1} / {2}" -f $label,(GB $job.BytesTransferred),(GB $job.BytesTotal)) }
      Start-Sleep 1
    }
  } catch {
    if ($job) { try { Remove-BitsTransfer -BitsJob $job } catch {} }
    if ("$_" -like '*CANCEL*') { throw 'CANCEL' }
    Log "BITS failed ($_) - falling back to direct download"
  }
  [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
  $req=[System.Net.HttpWebRequest]::Create($url); $req.UserAgent='barra-setup'
  $resp=$req.GetResponse(); $total=[double]$resp.ContentLength
  $in=$resp.GetResponseStream(); $out=[IO.File]::Create($dst)
  try {
    $buf=New-Object byte[] 1048576; $done=[double]0; $lastPct=-1
    while (($n=$in.Read($buf,0,$buf.Length)) -gt 0) {
      if ($script:Cancel) { throw 'CANCEL' }
      $out.Write($buf,0,$n); $done+=$n
      if ($total -gt 0) { $pct=[int](100*$done/$total); if ($pct -ne $lastPct) { $lastPct=$pct; Progress $pct ("{0} · {1} / {2}" -f $label,(GB $done),(GB $total)) } }
    }
  } finally { $out.Close(); $in.Close(); try { $resp.Close() } catch {} }
}

# adb/fastboot beschaffen (Google platform-tools, nie redistribuiert) — MUSS vor Schritt 0
# laufen, alles Weitere braucht adb. lpmake.exe liegt dagegen im Repo (eigener Apache-2.0-Build).
function Ensure-Tools(){
  if ((Test-Path $script:ADB) -and (Test-Path $script:FB)) { return }
  . (Join-Path $script:Kit 'stock-target.ps1')
  Info (T 'core.tl.needed')
  if ((Ask (T 'core.tl.terms_q') 'yn') -ne 'j') { throw 'CANCEL' }
  $ptZip = Join-Path $env:TEMP 'platform-tools-windows.zip'
  Download-Url $PlatformToolsUrl $ptZip (T 'core.tl.dl_lbl')
  Progress -1 (T 'core.tl.unpack')
  $ptDir = Join-Path $env:TEMP 'barra-platform-tools'
  Remove-Item $ptDir -Recurse -Force -ErrorAction SilentlyContinue
  Expand-Archive -Path $ptZip -DestinationPath $ptDir -Force
  New-Item -ItemType Directory -Force $script:Tools | Out-Null
  Get-ChildItem (Join-Path $ptDir 'platform-tools') | ForEach-Object { Move-Item $_.FullName $script:Tools -Force }
  Remove-Item $ptDir -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item $ptZip -Force -ErrorAction SilentlyContinue
  if (-not (Test-Path $script:ADB) -or -not (Test-Path $script:FB)) { Fail (T 'core.tl.fail') }
  Ok (T 'core.tl.ok')
}

function Ensure-Stock(){
  $IMG = Join-Path $script:Stock 'image'
  $superimg = Join-Path $script:Stock 'super.img'
  if ((Test-Path (Join-Path $IMG 'super_empty.img')) -and (Test-Path $superimg)) { return }
  . (Join-Path $script:Kit 'stock-target.ps1')
  Info (T 'core.st.needed')
  if ((Ask (T 'core.st.terms_q') 'yn') -ne 'j') { throw 'CANCEL' }
  if ((Test-Path $StockZip) -and ((Get-FileHash $StockZip -Algorithm SHA256).Hash -eq $StockSha.ToUpper())) {
    Info (T 'core.st.cached')
  } else {
    Download-Url $StockUrl $StockZip (T 'core.st.dl_lbl')
    Progress -1 (T 'core.st.verifying')
    if ((Get-FileHash $StockZip -Algorithm SHA256).Hash -ne $StockSha.ToUpper()) {
      Remove-Item $StockZip -Force -ErrorAction SilentlyContinue
      Fail (T 'core.st.sha_bad')
    }
    Ok (T 'core.st.sha_ok')
  }
  Info (T 'core.st.unpack'); Progress -1 (T 'core.st.unpack')
  $r = Run 'powershell.exe' "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $script:Kit 'fetch-stock.ps1')`" -Yes" $null 3600 'out'
  if ($r.code -ne 0 -or -not (Test-Path $superimg) -or -not (Test-Path (Join-Path $IMG 'super_empty.img'))) { Fail (T 'core.st.fail') }
  Ok (T 'core.st.ok')
}

# ---- init_boot lokal patchen (Rechts-Umbau: kein vorgepatchtes init_boot im Kit) --------------
# Quelle ist das Stock-init_boot aus dem Factory-Image; Magisks boot_patch.sh laeuft als
# shell-User in /data/local/tmp auf dem verbundenen Telefon — braucht adb, KEIN root
# (zu diesem Zeitpunkt gibt es noch keins). patch-initboot.ps1 macht die Arbeit (Kindprozess).
function Ensure-InitBoot(){
  $out = Join-Path $script:Payload 'init_boot-magisk.img'
  if (Test-Path $out) { return }
  Info (T 'core.ib.needed')
  Ensure-Stock
  Progress -1 (T 'core.ib.patching'); Info (T 'core.ib.patching')
  $r = Run 'powershell.exe' "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $script:Kit 'patch-initboot.ps1')`"" $null 600 'out'
  if ($r.code -ne 0 -or -not (Test-Path $out) -or (Get-Item $out).Length -lt 1000000) { Fail (T 'core.ib.fail') }
  Ok (T 'core.ib.ok')
}

# Stock-Flash (Weg B): Bootloader-Upgrade nur bei aelter, Boot-Kette, EIN super.img, Wipe — alles im BOOTLOADER.
# Wird von Schritt 2 (Setup) UND von Werkszustand (Unflash) benutzt. Erwartet Geraet in bootloader/fastbootd.
function Flash-Stock(){
  Ensure-Stock
  $IMG = Join-Path $script:Stock 'image'
  if (-not (Test-Path (Join-Path $IMG 'super_empty.img'))) { Fail (T 'core.fs.missing' $IMG) }
  $st=FbState
  if ($st -eq 'fastbootd') {
    # Weg B braucht KEIN fastbootd. Steht das Geraet doch dort (Altlast), zurueck in den Bootloader.
    Info (T 'core.fs.to_bl')
    [void](FbOut 'reboot bootloader' 10)
    if (-not (WaitFor 'bootloader' 120 (T 'core.fs.wait_bl'))) { Fail (T 'core.fs.bl_unreach2') }
    $st='bootloader'
  }
  if ($st -ne 'bootloader') { Fail (T 'core.fs.not_fb') }
  # Fortschritts-Budget fuer den ganzen Schritt (Boot-Kette + super, ggf. Bootloader/Radio) in Bytes
  $bootparts = 'boot','init_boot','dtbo','vendor_kernel_boot','pvmfw','vendor_boot','vbmeta','vbmeta_system','vbmeta_vendor'
  $superimg = Join-Path $script:Stock 'super.img'
  if (-not (Test-Path $superimg)) { Fail (T 'core.fs.super_missing') }
  $budget = [double](Get-Item $superimg).Length
  foreach ($p in $bootparts) { $budget += (Get-Item (Join-Path $IMG "$p.img")).Length }
  PbBegin $budget
  # Phase 0: Bootloader-Upgrade nur wenn AELTER als verlangt (ARB-sicher); niemals Downgrade
  $ai = Get-Content (Join-Path $IMG 'android-info.txt') -ErrorAction SilentlyContinue
  $reqBl = ($ai | Where-Object { $_ -match '^require version-bootloader=(.+)' } | ForEach-Object { $Matches[1] }) | Select-Object -First 1
  $curBl = FbVar 'version-bootloader'
  if ($reqBl -and (BlNum $curBl) -lt (BlNum $reqBl)) {
    Info (T 'core.fs.bl_older' $curBl $reqBl)
    $bl = Get-ChildItem $script:Stock -Filter 'bootloader-akita-*.img' | Select-Object -First 1
    $rd = Get-ChildItem $script:Stock -Filter 'radio-akita-*.img' | Select-Object -First 1
    if (-not $bl -or -not $rd) { Fail (T 'core.fs.bl_files_missing') }
    $script:PbTotal += $bl.Length + $rd.Length
    FbFlash 'bootloader' $bl.FullName (T 'core.fs.bl_lbl'); [void](FbOut 'reboot-bootloader' 10); Start-Sleep 5
    if (-not (WaitFor 'bootloader' 120 (T 'core.fs.bl_restart'))) { Fail (T 'core.fs.bl_notback') }
    FbFlash 'radio' $rd.FullName (T 'core.fs.radio_lbl'); [void](FbOut 'reboot-bootloader' 10); Start-Sleep 5
    if (-not (WaitFor 'bootloader' 120 (T 'core.fs.bl_restart'))) { Fail (T 'core.fs.rd_notback') }
    Ok (T 'core.fs.bl_now' (FbVar 'version-bootloader'))
  } else { Info (T 'core.fs.bl_ok' $curBl $reqBl) }
  # Phase 1: Boot-Kette
  Info (T 'core.fs.p1')
  [void](FbOut 'erase avb_custom_key' 15)
  foreach ($p in $bootparts) { FbFlash $p (Join-Path $IMG "$p.img") (T 'core.fs.bootchain_lbl' $p) }
  Ok (T 'core.fs.bootchain_done')
  # Phase 2 (Weg B): EIN komplettes super.img im BOOTLOADER flashen — kein fastbootd noetig
  # (fastbootd braucht unter Windows einen extra Treiber; der Bootloader nicht). super.img wird
  # einmalig aus super_empty + den logischen Partitionen gebaut (lpmake) und liegt im stock-Ordner.
  Info (T 'core.fs.p2')
  FbFlash 'super' $superimg (T 'core.fs.system_lbl') '64M'
  # Phase 3: Wipe
  Info (T 'core.fs.p3')
  Progress 100 (T 'core.fs.p3')
  [void](FbOut 'erase userdata' 120); [void](FbOut 'erase metadata' 60)
  PbEnd
}

function Step2_Stock(){
  Step 2 (T 'setup.steps.stock') 'run'
  if ((AdbState) -eq 'device') {
    $bid=AdbSh 'getprop ro.build.id'
    if ($bid -eq 'BP4A.260205.001') { Ok (T 'core.s2.already'); Step 2 (T 'setup.steps.stock') 'ok'; return }
    Info (T 'core.s2.replace' $bid); [void](AdbOut 'reboot bootloader' 10); Start-Sleep 3
    if (-not (WaitFor 'bootloader' 120 (T 'core.fs.wait_bl'))) { Fail (T 'core.s1.bl_unreach') }
  }
  Flash-Stock
  Info (T 'core.s2.reboot_stock')
  [void](FbOut 'reboot' 10)
  Step 2 (T 'setup.steps.stock') 'wait_user'
  Tel (T 'core.s2.tel_setup')
  Tel (T 'core.s2.tel_debug')
  Step 2 (T 'setup.steps.stock') 'wait_dev'
  if (-not (WaitFor 'device' 900 (T 'core.s2.wait_android'))) { Fail (T 'core.s2.no_adb') }
  $bid=AdbSh 'getprop ro.build.id'; $kv=AdbSh 'uname -r'
  if ($bid -ne 'BP4A.260205.001') { Fail (T 'core.s2.unexpected' $bid) }
  Ok (T 'core.s2.ok' $kv); Step 2 (T 'setup.steps.stock') 'ok'
}

function Step3_KernelRoot(){
  Step 3 (T 'setup.steps.kernel') 'run'
  if ((AdbState) -eq 'device') {
    $kv=AdbSh 'uname -r'; $root=(AdbSu 'id' 15) -match 'uid=0'
    if ($kv -like '6.1.157-android14-11-g*' -and $root) { Ok (T 'core.s3.already' $kv); Step 3 (T 'setup.steps.kernel') 'ok'; return }
    if ($kv -notlike '6.1.157-android14-11-g*') {
      Ensure-InitBoot
      Info (T 'core.s3.magisk_install')
      $r = Run $script:ADB "install -r `"$(Join-Path $script:Payload 'Magisk-30.7.apk')`"" $null 120
      if (($r.lines -join ' ') -match 'Success') { Ok (T 'core.s3.magisk_ok') } else { Warn (T 'core.s3.magisk_fail') }
      Info (T 'core.s1.to_bl'); [void](AdbOut 'reboot bootloader' 10); Start-Sleep 3
      if (-not (WaitFor 'bootloader' 120 (T 'core.fs.wait_bl'))) { Fail (T 'core.s1.bl_unreach') }
    }
  }
  if ((FbState) -eq 'bootloader') {
    # Backstop: ohne gepatchtes init_boot nicht flashen — patchen geht nur mit Geraet in Android.
    if (-not (Test-Path (Join-Path $script:Payload 'init_boot-magisk.img'))) { Fail (T 'core.ib.missing_bl') }
    Info (T 'core.s3.flash')
    FbFlash 'boot' (Join-Path $script:Payload 'boot-lz4.img') (T 'core.s3.kernel_lbl')
    FbFlash 'init_boot' (Join-Path $script:Payload 'init_boot-magisk.img') (T 'core.s3.root_lbl')
    [void](FbOut 'reboot' 10)
    Step 3 (T 'setup.steps.kernel') 'wait_dev'
    if (-not (WaitFor 'device' 300 (T 'core.s3.restarting'))) { Fail (T 'core.s3.no_adb') }
    $kv=AdbSh 'uname -r'
    if ($kv -notlike '6.1.157-android14-11-g*') { Fail (T 'core.s3.wrong_kernel' $kv) }
    Ok (T 'core.s3.kernel_ok' $kv)
  }
  if ((AdbSu 'id' 15) -match 'uid=0') { Ok (T 'core.s3.root_ok'); Step 3 (T 'setup.steps.kernel') 'ok'; return }
  Step 3 (T 'setup.steps.kernel') 'wait_user'
  Info (T 'core.s3.magisk_note')
  Tel (T 'core.s3.tel1')
  Tel (T 'core.s3.tel2')
  Tel (T 'core.s3.tel3')
  # Magisk-App direkt oeffnen — der "Zusaetzliche Einrichtung"-Dialog steht dann sofort auf dem Schirm
  Info (T 'core.s3.open_app')
  [void](AdbSh 'monkey -p com.topjohnwu.magisk -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1' 15)
  $t0=Get-Date; $down=$false; $reminded=$false
  while (((Get-Date)-$t0).TotalSeconds -lt 480) {
    Chk
    if ((AdbState) -ne 'device') { $down=$true; Step 3 (T 'setup.steps.kernel') 'wait_dev' }
    else { if ((AdbSu 'id' 10) -match 'uid=0') { Ok (T 'core.s3.root_ok2'); Step 3 (T 'setup.steps.kernel') 'ok'; return }; if ($down) { Step 3 (T 'setup.steps.kernel') 'wait_user' } }
    $el=[int]((Get-Date)-$t0).TotalSeconds; Progress -1 ("{0} · {1}" -f (T 'core.s3.wait_magisk'), (MMSS $el))
    if ($el -ge 90 -and -not $down -and -not $reminded) { Tel (T 'core.s3.remind'); $reminded=$true }
    Start-Sleep 3
  }
  Fail (T 'core.s3.no_root')
}

function Step4_Base(){
  Step 4 (T 'setup.steps.payload') 'run'
  $installed = (AdbSu 'test -f /data/adb/baseos/bin/base-boot.sh && echo ja' 15) -eq 'ja'
  $preDone   = (AdbSu 'test -f /data/local/ubuntu/var/lib/barra/preconfigured && echo ja' 15) -eq 'ja'
  if ($installed -and ($preDone -or -not $script:PreCfg)) { Ok (T 'core.s4.already'); Step 4 (T 'setup.steps.payload') 'ok'; return }
  if ($installed) {
    # Base ist da, aber die Pre-Einrichtung wurde nie angewendet -> nur die nachholen (Container wird dafuer angehalten)
    Info (T 'core.s4.pre_missing')
    $pre = Join-Path $env:TEMP 'barra-preconfig.env'
    ($script:PreCfg.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "`n" | Set-Content $pre -Encoding ASCII -NoNewline
    [void](AdbSh 'mkdir -p /data/local/tmp/barra-kit' 10)
    [void](Run $script:ADB "push `"$(Join-Path $script:Kit 'device-install.sh')`" /data/local/tmp/barra-kit/" $null 60)
    [void](Run $script:ADB "push `"$pre`" /data/local/tmp/barra-kit/preconfig.env" $null 60)
    Remove-Item $pre -Force -ErrorAction SilentlyContinue
    Progress -1 (T 'core.s4.pre_apply')
    $r = Run $script:ADB "shell su -c 'sh /data/local/tmp/barra-kit/device-install.sh preconfig'" $null 300
    if (($r.lines -join ' ') -notmatch 'OK - Base installiert') { Fail (T 'core.s4.pre_fail') }
    Ok (T 'core.s4.pre_ok'); Step 4 (T 'setup.steps.payload') 'ok'; return
  }
  # Payload = tar.gz; wird UNVERAENDERT aufs Telefon geschoben und dort mit toybox tar -xz entpackt.
  # (Auf dem PC wird nichts entpackt: Ubuntu-Rootfs mit Symlinks/Rechten hat auf NTFS nichts verloren.)
  $tar = Join-Path $script:Payload 'barra-base.tar.gz'
  if (-not (Test-Path $tar)) { Fail (T 'core.s4.tar_missing') }
  # Pre-Einrichtung als Datei mitgeben
  $pre = Join-Path $env:TEMP 'barra-preconfig.env'
  if ($script:PreCfg) { ($script:PreCfg.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "`n" | Set-Content $pre -Encoding ASCII -NoNewline } elseif (Test-Path $pre) { Remove-Item $pre }
  Info (T 'core.s4.push')
  [void](AdbSh 'mkdir -p /data/local/tmp/barra-kit' 10)
  $pushed=$false
  for ($try=1; $try -le 3 -and -not $pushed; $try++) {
    Chk
    if ($try -gt 1) { Warn (T 'core.s4.retry' $try); Start-Sleep 4; if (-not (WaitFor 'device' 120 (T 'core.s4.wait_dev'))) { break } }
    $r = Run $script:ADB "push `"$tar`" /data/local/tmp/barra-kit/barra-base.tar.gz" { param($l) if ($l -match '\[\s*(\d+)%\]') { Progress ([int]$Matches[1]) (T 'core.s4.push_pct' $Matches[1]) } } 900 'out'
    if (($r.lines -join ' ') -match 'file pushed') { $pushed=$true }
  }
  if (-not $pushed) { Fail (T 'core.s4.push_fail') }
  [void](Run $script:ADB "push `"$(Join-Path $script:Kit 'device-install.sh')`" /data/local/tmp/barra-kit/" $null 60)
  if (Test-Path $pre) { [void](Run $script:ADB "push `"$pre`" /data/local/tmp/barra-kit/preconfig.env" $null 60); Remove-Item $pre -Force -ErrorAction SilentlyContinue }
  Info (T 'core.s4.install'); Progress -1 (T 'core.s4.install_prog')
  $r = Run $script:ADB "shell su -c 'sh /data/local/tmp/barra-kit/device-install.sh'" $null 900
  if (($r.lines -join ' ') -notmatch 'OK - Base installiert') { Fail (T 'core.s4.incomplete') }
  Ok (T 'core.s4.ok'); Step 4 (T 'setup.steps.payload') 'ok'
}

function Step5_Boot(){
  Step 5 (T 'setup.steps.firstboot') 'wait_dev'
  Info (T 'core.s5.intro')
  [void](AdbOut 'reboot' 10); Start-Sleep 5
  if (-not (WaitFor 'device' 300 (T 'core.s5.restarting'))) { Fail (T 'core.s5.no_adb') }
  $t0=Get-Date; $hn=''; $ip=''
  while (((Get-Date)-$t0).TotalSeconds -lt 600) {
    Chk
    # Erst-Einrichtung startet das Telefon mehrfach neu — waehrend adb weg ist, das ehrlich anzeigen
    if ((AdbState) -ne 'device') { $el=[int]((Get-Date)-$t0).TotalSeconds; Progress -1 ("{0} · {1}" -f (T 'core.s5.reboots'), (MMSS $el)); Start-Sleep 5; continue }
    $st = AdbSu 'cat /data/adb/baseos/state 2>/dev/null' 10
    $h  = AdbSu 'grep -c Uebergabe /data/adb/baseos/boot.log 2>/dev/null' 10
    $el=[int]((Get-Date)-$t0).TotalSeconds; Progress -1 ("{0} · {1}" -f (T 'core.s5.stage' ($st -replace '\s','')), (MMSS $el))
    if ([int]($h -replace '\D','0') -ge 1) { break }
    Start-Sleep 5
  }
  $hn = AdbSu 'cat /data/local/ubuntu/etc/hostname' 10
  $ip   = AdbSu 'ip -4 -o addr show wlan0 2>/dev/null | grep -oE [0-9.]+/ | head -1 | cut -d/ -f1' 10
  if (-not $ip) { $t1=Get-Date; while (((Get-Date)-$t1).TotalSeconds -lt 120 -and -not $ip) { Progress -1 (T 'core.s5.wait_ip'); Start-Sleep 5; $ip = AdbSu 'ip -4 -o addr show wlan0 2>/dev/null | grep -oE [0-9.]+/ | head -1 | cut -d/ -f1' 10 } }
  Ok (T 'core.s5.ok' "$hn$(if($ip){" · $ip"})"); Step 5 (T 'setup.steps.firstboot') 'ok'
  return @{ host=$hn; ip=$ip }
}

# ---- Werkszustand (Unflash): barra/Root/Kernel weg, Google-Stock A16 + Wipe, optional Bootloader sperren ----
function Run-Unflash(){
  try {
    Ensure-Tools
    Step0_Verbindung
    # 1: in den Bootloader
    Step 1 (T 'setup.steps.u_bl') 'run'
    if ((AdbState) -eq 'device') {
      Info (T 'core.uf.to_bl'); [void](AdbOut 'reboot bootloader' 10); Start-Sleep 3
      Step 1 (T 'setup.steps.u_bl') 'wait_dev'
      if (-not (WaitFor 'bootloader' 120 (T 'core.fs.wait_bl'))) { Fail (T 'core.s1.bl_unreach') }
      Step 1 (T 'setup.steps.u_bl') 'run'
    } elseif ((AdbState) -eq 'unauthorized') {
      Step 1 (T 'setup.steps.u_bl') 'wait_user'; Tel (T 'core.uf.allow')
      if (-not (WaitFor 'device' 120 (T 'core.uf.wait_grant'))) { Fail (T 'core.uf.no_grant') }
      [void](AdbOut 'reboot bootloader' 10); Start-Sleep 3
      Step 1 (T 'setup.steps.u_bl') 'wait_dev'
      if (-not (WaitFor 'bootloader' 120 (T 'core.fs.wait_bl'))) { Fail (T 'core.s1.bl_unreach') }
      Step 1 (T 'setup.steps.u_bl') 'run'
    }
    if ((FbState) -eq 'fastbootd') { [void](FbOut 'reboot bootloader' 10); Step 1 (T 'setup.steps.u_bl') 'wait_dev'; if (-not (WaitFor 'bootloader' 120 (T 'core.fs.wait_bl'))) { Fail (T 'core.s1.bl_unreach') }; Step 1 (T 'setup.steps.u_bl') 'run' }
    if ((FbState) -ne 'bootloader') { Fail (T 'core.uf.neither') }
    if ((FbVar 'unlocked') -ne 'yes') { Fail (T 'core.uf.locked_already') }
    Ok (T 'core.uf.bl_ok'); Step 1 (T 'setup.steps.u_bl') 'ok'
    # 2: Werks-Image + Wipe
    Step 2 (T 'setup.steps.u_stock') 'run'
    Warn (T 'core.uf.wipe_warn')
    Flash-Stock
    Ok (T 'core.uf.flashed'); Step 2 (T 'setup.steps.u_stock') 'ok'
    # 3: Bootloader sperren (optional)
    Step 3 (T 'setup.steps.u_lock') 'wait_user'
    Info (T 'core.uf.lock_info')
    if ((Ask (T 'core.uf.lock_q') 'yn') -eq 'j') {
      Step 3 (T 'setup.steps.u_lock') 'run'
      [void](FbOut 'flashing lock' 20)
      Step 3 (T 'setup.steps.u_lock') 'wait_user'
      Tel (T 'core.uf.tel_lock')
      $t0=Get-Date; $locked=$false
      while (((Get-Date)-$t0).TotalSeconds -lt 240) {
        Chk; $st=FbState
        if ($st -eq 'bootloader') { if ((FbVar 'unlocked') -eq 'no') { $locked=$true; break } }
        elseif (-not $st) { $locked=$true; break }   # Geraet weg = hat gesperrt und startet neu
        $el=[int]((Get-Date)-$t0).TotalSeconds; Progress -1 ("{0} · {1}" -f (T 'core.s1.wait_confirm'), (MMSS $el)); Start-Sleep 3
      }
      if ($locked) { Ok (T 'core.uf.locked'); Step 3 (T 'setup.steps.u_lock') 'ok'; if ((FbState) -eq 'bootloader') { [void](FbOut 'reboot' 10) } }
      else { Warn (T 'core.uf.not_locked'); Step 3 (T 'setup.steps.u_lock') 'ok'; [void](FbOut 'reboot' 10) }
    } else {
      Info (T 'core.uf.stays'); Step 3 (T 'setup.steps.u_lock') 'ok'; [void](FbOut 'reboot' 10)
    }
    Ok (T 'core.uf.done')
    Emit 'done' @{ kind='unflash' }
  }
  catch { if ("$_" -eq 'CANCEL' -or "$_" -like '*CANCEL*') { Info (T 'core.cancelled') } elseif ("$_" -notlike 'FAIL:*') { Emit 'fail' (T 'core.unexpected' "$_") } }
}

function Run-Flash(){
  try {
    Ensure-Tools; Step0_Verbindung; Step1_Unlock; Step2_Stock; Step3_KernelRoot; Step4_Base
    $bootInfo = Step5_Boot
    if ($script:PreCfg -and $script:PreCfg.KITS -and @($script:PreCfg.KITS).Count) {
      # Pakete als 7. Schritt im Flow; ein Paket-Fehler bricht den Flash NICHT ab
      # (der Node laeuft ja) — Warnung im Feed, Retry ueber das Panel der Fertig-Seite.
      Step 6 (T 'setup.steps.packages') 'run'
      try { Run-KitsInner; Step 6 (T 'setup.steps.packages') 'ok' }
      catch {
        if ("$_" -eq 'CANCEL' -or "$_" -like '*CANCEL*') { throw 'CANCEL' }
        Warn (T 'core.kit.fail' "$_"); Emit 'pkg' @{ text=(T 'core.kit.fail' "$_"); state='fail' }
        Step 6 (T 'setup.steps.packages') 'ok'
      }
    }
    Emit 'done' @{ host=$bootInfo.host; ip=$bootInfo.ip; user=$(if($script:PreCfg -and $script:PreCfg.USER){$script:PreCfg.USER}else{'ubuntu'}) }
  }
  catch { if ("$_" -eq 'CANCEL' -or "$_" -like '*CANCEL*') { Info (T 'core.cancelled') } elseif ("$_" -notlike 'FAIL:*') { Emit 'fail' (T 'core.unexpected' "$_") } }
}

# ---- Pakete (Fertig-Seite): LLM-Kit / STT-Kit auf den frisch geflashten Node ----
# Events gehen ueber 'pkg' (eigener Status auf Seite 3); Log-Zeilen wie gehabt ins Detail-Log.
function Pkg($text,$state='run'){
  Emit 'pkg' @{ text=$text; state=$state }
  Emit 'progress' @{ pct=-1; text=$text }   # zeigt den Text auch in der Statuskachel (In-Flow)
}
function KitPush($src,$dst,$label){
  $r = Run $script:ADB "push `"$src`" $dst" { param($l)
        if ($l -match '\[\s*(\d+)%\]') {
          Emit 'pkg' @{ text=("$label " + $matches[1] + '%'); state='run' }
          Emit 'progress' @{ pct=[int]$matches[1]; text=("$label " + $matches[1] + '%') }
        } } 1800 'out'
  if ($r.code -ne 0) { throw "push failed: $src" }
}
function Run-Kits(){
  try { Run-KitsInner }
  catch {
    if ("$_" -eq 'CANCEL' -or "$_" -like '*CANCEL*') { Pkg (T 'core.cancelled') 'fail' }
    else { Pkg (T 'core.kit.fail' "$_") 'fail' }
  }
}
function Run-KitsInner(){
    $kits = @(); if ($script:PreCfg -and $script:PreCfg.KITS) { $kits = @($script:PreCfg.KITS) }
    if (-not $kits.Count) { Pkg (T 'core.kit.all_ok') 'ok'; return }
    # Koexistenz-Verbot (22.8., am Geraet bewiesen): llm gleichzeitig mit stt/pya = OOM-Panic +
    # TPU-Graph-Limit. Bei Konflikt-Auswahl wird alles INSTALLIERT, aber KEIN Dienst gestartet.
    # stt+pya zusammen ist erlaubt (TPU-Graphen 104+1 unterm Limit).
    $bothKits = ($kits -contains 'llm') -and (($kits -contains 'stt') -or ($kits -contains 'pya'))
    Pkg (T 'core.kit.wait_dev')
    if ((AdbState) -ne 'device') { [void](Run $script:ADB 'wait-for-device' $null 90) }
    if ((AdbState) -ne 'device') { throw (T 'core.kit.no_dev') }
    # Modellwahl aus dem Setup (precfg.KITMODELS); ohne Angabe das Standard-Modell je Kit
    $km = @{}
    if ($script:PreCfg -and $script:PreCfg.KITMODELS) { $km = $script:PreCfg.KITMODELS }
    foreach ($k in $kits) {
      Chk
      if ($k -eq 'llm') {
        $name = T 'core.kit.llm_name'
        $mdl = if ($km['llm']) { $km['llm'] } else { 'qwen3-4b' }
        $mf = @{ 'qwen3-4b'='qwen3-4b.gguf'; 'qwen2.5-1.5b'='qwen2.5-1.5b.gguf'; 'glm-edge-4b'='glm-edge-4b-chat.gguf'; 'qwen38-distill'='qwen38-4b-distill.gguf'; 'gemma-e2b'='gemma-4-e2b-q4_0.gguf'; 'gemma-e4b'='gemma-4-e4b-q3_k_s.gguf' }[$mdl]
        if (-not $mf) { throw (T 'core.kit.fail' "$name (model id $mdl)") }
        # TPU-Attention-Kit je Modell (v7/v8-Pipeline; llmserver erkennt es am GGUF-Basenamen).
        # distill (nur 8/32 Full-Attention-Layer) und gemma-e4b (Q3, nicht im Base-Stack) haben keins.
        $attnTar = @{ 'qwen3-4b'='llm-attn-qwen3-4b.tar'; 'qwen2.5-1.5b'='llm-attn-qwen2.5-1.5b.tar'; 'glm-edge-4b'='llm-attn-glm-edge-4b-chat.tar'; 'gemma-e2b'='llm-attn-gemma-4-e2b-q4_0.tar' }[$mdl]
        if ($attnTar) {
          Pkg (T 'core.kit.push' $name)
          KitPush (Join-Path $script:Kit "llm-kit\$attnTar") '/data/local/tmp/llm-kit.tar' $name
          Pkg (T 'core.kit.extract' $name)
          $o = AdbSu 'mkdir -p /data/local/barra-attn && cd /data/local/barra-attn && tar -xf /data/local/tmp/llm-kit.tar && chmod -R 755 /data/local/barra-attn && rm /data/local/tmp/llm-kit.tar && echo KIT_OK' 600
          if ($o -notmatch 'KIT_OK') { throw (T 'core.kit.fail' "$name (tar)") }
        }
        Pkg (T 'core.kit.push_model' $name)
        KitPush (Join-Path $script:Kit "llm-kit\$mf") '/data/local/tmp/llm-model.gguf' $name
        $o = AdbSu ('H=$(ls -d /data/local/ubuntu/home/* | head -1); mkdir -p $H/models && mv /data/local/tmp/llm-model.gguf $H/models/' + $mf + ' && chown -R 1001:1001 $H/models && echo MDL_OK') 180
        if ($o -notmatch 'MDL_OK') { throw (T 'core.kit.fail' "$name (model)") }
        if (-not $bothKits) {
          Pkg (T 'core.kit.start' $name)
          [void](AdbSu ('H=$(ls -d /data/local/ubuntu/home/* | head -1); sh /data/adb/baseos/bin/llmserver.sh start $H/models/' + $mf) 300)
        }
        Ok (T 'core.kit.ok' $name)
      }
      elseif ($k -eq 'stt') {
        $name = T 'core.kit.stt_name'
        $mdl = if ($km['stt']) { $km['stt'] } else { 'turbo' }
        $mf = @{ 'turbo'='ggml-large-v3-turbo-q5_0.bin'; 'tiny'='ggml-tiny.bin'; 'base'='ggml-base.bin'; 'small'='ggml-small.bin'; 'medium'='ggml-medium-q5_0.bin' }[$mdl]
        if (-not $mf) { throw (T 'core.kit.fail' "$name (model id $mdl)") }
        # TPU-Encoder-Packages, wo ein Satz existiert (turbo, base); andere Modelle laufen auf der CPU
        $ptar = @{ 'turbo'='whisper-kit-turbo.tar'; 'base'='whisper-kit-base.tar'; 'tiny'='whisper-kit-tiny.tar'; 'small'='whisper-kit-small.tar'; 'medium'='whisper-kit-medium.tar' }[$mdl]
        if ($ptar -and -not (Test-Path (Join-Path $script:Kit "whisper-kit\$ptar"))) { $ptar = $null }
        if ($ptar) {
          Pkg (T 'core.kit.push' $name)
          KitPush (Join-Path $script:Kit "whisper-kit\$ptar") '/data/local/tmp/whisper-kit.tar' $name
          Pkg (T 'core.kit.extract' $name)
          $o = AdbSu 'mkdir -p /data/local/barra-stt && cd /data/local/barra-stt && tar -xf /data/local/tmp/whisper-kit.tar && mkdir -p models && chmod -R 755 /data/local/barra-stt && rm /data/local/tmp/whisper-kit.tar && echo KIT_OK' 600
          if ($o -notmatch 'KIT_OK') { throw (T 'core.kit.fail' "$name (tar)") }
        }
        Pkg (T 'core.kit.push_model' $name)
        KitPush (Join-Path $script:Kit "whisper-kit\$mf") '/data/local/tmp/stt-model.bin' $name
        $o = AdbSu ('mkdir -p /data/local/barra-stt/models && mv /data/local/tmp/stt-model.bin /data/local/barra-stt/models/' + $mf + ' && echo MDL_OK') 120
        if ($o -notmatch 'MDL_OK') { throw (T 'core.kit.fail' "$name (model)") }
        if (-not $bothKits) {
          Pkg (T 'core.kit.start' $name)
          [void](AdbSu ('sh /data/adb/baseos/bin/sttserver.sh start /data/local/barra-stt/models/' + $mf) 400)
        }
        Ok (T 'core.kit.ok' $name)
      }
      elseif ($k -eq 'pya') {
        $name = T 'core.kit.pya_name'
        $mdl = if ($km['pya']) { $km['pya'] } else { 'resnet34-en' }
        Pkg (T 'core.kit.push' $name)
        KitPush (Join-Path $script:Kit 'pyannote-kit\pyannote-kit.tar') '/data/local/tmp/pya-kit.tar' $name
        Pkg (T 'core.kit.extract' $name)
        $o = AdbSu 'cd /data/local/tmp && rm -rf pya-kit && mkdir pya-kit && cd pya-kit && tar -xf ../pya-kit.tar && U=/data/local/ubuntu && mkdir -p $U/opt/barra-pya $U/opt/barra/pya && cp kit/* $U/opt/barra-pya/ && chmod 644 $U/opt/barra-pya/* && cp base/sherpa-onnx-offline-speaker-diarization $U/opt/barra/pya/ && chmod 755 $U/opt/barra/pya/sherpa-onnx-offline-speaker-diarization && cp base/barra-diarize $U/usr/local/bin/barra-diarize && chmod 755 $U/usr/local/bin/barra-diarize && cp base/pyaserver.sh /data/adb/baseos/bin/pyaserver.sh && chmod 755 /data/adb/baseos/bin/pyaserver.sh && cd /data/local/tmp && rm -rf pya-kit pya-kit.tar && echo KIT_OK' 600
        if ($o -notmatch 'KIT_OK') { throw (T 'core.kit.fail' "$name (tar)") }
        if ($mdl -eq 'resnet34-zh') {
          # zh-Variante (TPU): gleiche Architektur — Trunk-Package/Kopf/ONNX im Kit ersetzen
          Pkg (T 'core.kit.push_model' $name)
          KitPush (Join-Path $script:Kit 'pyannote-kit\r34zh_trunk.package') '/data/local/tmp/pya-zh.pkg' $name
          KitPush (Join-Path $script:Kit 'pyannote-kit\head_zh.bin') '/data/local/tmp/pya-zh.head' $name
          KitPush (Join-Path $script:Kit 'pyannote-kit\resnet34_zh.onnx') '/data/local/tmp/pya-zh.onnx' $name
          $o = AdbSu 'K=/data/local/ubuntu/opt/barra-pya; mv /data/local/tmp/pya-zh.pkg $K/r34_trunk.package && mv /data/local/tmp/pya-zh.head $K/head.bin && mv /data/local/tmp/pya-zh.onnx $K/resnet34.onnx && chmod 644 $K/* && echo EMB_OK' 120
          if ($o -notmatch 'EMB_OK') { throw (T 'core.kit.fail' "$name (model)") }
          if (-not $bothKits) {
            Pkg (T 'core.kit.start' $name)
            [void](AdbSu 'sh /data/adb/baseos/bin/pyaserver.sh start' 180)
          }
        } elseif ($mdl -eq 'eres2net-zh') {
          # eres2net auf TPU: Multi-Output-Rumpf + Float-Tail; ersetzt die r34-Kit-Dateien
          Pkg (T 'core.kit.push_model' $name)
          KitPush (Join-Path $script:Kit 'pyannote-kit\eres_body.package') '/data/local/tmp/pya-e1' $name
          KitPush (Join-Path $script:Kit 'pyannote-kit\eres_tail.onnx') '/data/local/tmp/pya-e2' $name
          KitPush (Join-Path $script:Kit 'pyannote-kit\head_eres.bin') '/data/local/tmp/pya-e3' $name
          KitPush (Join-Path $script:Kit 'pyannote-kit\eres_params.txt') '/data/local/tmp/pya-e4' $name
          $o = AdbSu 'K=/data/local/ubuntu/opt/barra-pya; rm -f $K/r34_trunk.package $K/emb_cpu.onnx; mv /data/local/tmp/pya-e1 $K/eres_body.package && mv /data/local/tmp/pya-e2 $K/eres_tail.onnx && mv /data/local/tmp/pya-e3 $K/head.bin && mv /data/local/tmp/pya-e4 $K/eres_params.txt && chmod 644 $K/* && echo EMB_OK' 120
          if ($o -notmatch 'EMB_OK') { throw (T 'core.kit.fail' "$name (model)") }
          if (-not $bothKits) {
            Pkg (T 'core.kit.start' $name)
            [void](AdbSu 'sh /data/adb/baseos/bin/pyaserver.sh start' 180)
          }
        } elseif ($mdl -eq 'titanet-en') {
          # TitaNet auf TPU: 5-Segment-Kette + Float-Tail; ersetzt die r34-Kit-Dateien
          Pkg (T 'core.kit.push_model' $name)
          foreach ($tf in @('tita_seg0.package','tita_seg1.package','tita_seg2.package','tita_seg3.package','tita_seg4.package','tita_tail.onnx','tita_glue.bin','tita_params.txt')) {
            KitPush (Join-Path $script:Kit "pyannote-kit\$tf") "/data/local/tmp/$tf" $name
          }
          $o = AdbSu 'K=/data/local/ubuntu/opt/barra-pya; rm -f $K/r34_trunk.package $K/eres_body.package $K/eres_tail.onnx $K/eres_params.txt $K/emb_cpu.onnx $K/head.bin; mv /data/local/tmp/tita_seg0.package /data/local/tmp/tita_seg1.package /data/local/tmp/tita_seg2.package /data/local/tmp/tita_seg3.package /data/local/tmp/tita_seg4.package /data/local/tmp/tita_tail.onnx /data/local/tmp/tita_glue.bin /data/local/tmp/tita_params.txt $K/ && chmod 644 $K/* && echo EMB_OK' 180
          if ($o -notmatch 'EMB_OK') { throw (T 'core.kit.fail' "$name (model)") }
          if (-not $bothKits) {
            Pkg (T 'core.kit.start' $name)
            [void](AdbSu 'sh /data/adb/baseos/bin/pyaserver.sh start' 180)
          }
        } elseif (-not $bothKits) {
          Pkg (T 'core.kit.start' $name)
          [void](AdbSu 'sh /data/adb/baseos/bin/pyaserver.sh start' 180)
        }
        Ok (T 'core.kit.ok' $name)
      }
    }
    if ($bothKits) { Pkg (T 'core.kit.both_note') 'ok' } else { Pkg (T 'core.kit.all_ok') 'ok' }
}
