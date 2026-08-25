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
# Optionaler Hook, mit dem die GUI einen kooperativen Abbruch durchreicht (setzt $script:Cancel).
# Standalone (ohne GUI) bleibt er $null und wird uebersprungen.
$script:CancelCheck = $null
function RefreshCancel(){ if ($script:CancelCheck) { & $script:CancelCheck } }
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
  $a=$null; while (-not $script:Answer.TryDequeue([ref]$a)) { RefreshCancel; if ($script:Cancel) { throw 'CANCEL' }; Start-Sleep -Milliseconds 200 }
  return $a
}
function WaitUser($text){ [void](Ask $text 'ok') }
function Chk(){ RefreshCancel; if ($script:Cancel) { throw 'CANCEL' } }
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
      RefreshCancel; if ($script:Cancel) { try{$p.Kill()}catch{}; throw 'CANCEL' }
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
# su-Aufrufe: das Kommando muss als EIN Argument ueber die Windows→adb-Grenze (24.8., Kit-Install-
# Bug): CommandLineToArgvW konsumiert "..."-Quotes und adb joint argv OHNE Re-Quoting — die
# Geraete-Shell splittet Ketten dann an ;/&& und fuehrt NUR das 1. Kommando unter su aus, der
# Rest lief als shell (uid 2000) → mkdir unter /data/local = EACCES. Deshalb: shell "su -c '<cmd>'"
# — die Geraete-Shell parst die Single-Quotes, su bekommt die ganze Kette; $-Expansion erst in der
# root-Shell. BEDINGUNG: keine Anfuehrungszeichen im Kommando (Guard wirft sofort, statt still
# als shell zu laufen).
function SuQ($cmd){ if ($cmd -match '["'']') { throw "su-Kommando enthaelt Anfuehrungszeichen (nicht erlaubt): $cmd" }; "'" + $cmd + "'" }
function AdbSu($cmd,$to=60){ (AdbOut ("shell `"su -c " + (SuQ $cmd) + "`"") $to).Trim() }
# su -M (Mount-Namespace erhalten) — noetig fuer enter-systemd.sh (nsenter in den Container-systemd)
function AdbSuM($cmd,$to=60){ (AdbOut ("shell `"su -M -c " + (SuQ $cmd) + "`"") $to).Trim() }
# Langlaeufer ROBUST: schreibt $cmd als Skript aufs Geraet, startet es DETACHED (setsid -> ueberlebt
# adb-Abbruch, kein SIGHUP-Tod) und pollt eine Marker-Datei mit kurzen Reads (reconnected bei Bedarf).
# Volles set-x-Trace-Log fuer Diagnose. Rueckgabe: @{ ok; rc; log }. Loest das "Entpacken bricht ab,
# wenn USB waehrend des stillen Extract-Waits zuckt"-Problem (Kit-Install).
function AdbSuBg($cmd,$timeoutSec=900,$label=$null){
  $sh='/data/local/tmp/kitrun.sh'; $mk='/data/local/tmp/kitrun.done'; $lg='/data/local/tmp/kitrun.log'
  # Marker+Log world-readable: der Extract laeuft als root, der Poll liest als shell (adb shell cat).
  $wrap = "exec > $lg 2>&1`nset -x`n$cmd`nrc=`$?`nset +x`nchmod 666 $lg 2>/dev/null`necho R:`$rc > $mk`nchmod 666 $mk 2>/dev/null`n"
  $tmp = [IO.Path]::GetTempFileName()
  [IO.File]::WriteAllText($tmp, ($wrap -replace "`r`n","`n"))   # LF fuers Geraet
  $pr = Run $script:ADB "push `"$tmp`" $sh" $null 60 'out'
  Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  if ($pr.code -ne 0) { return @{ ok=$false; rc=-1; log='run-script push failed' } }
  # su -M (Mount-Master/globaler Namespace) NOETIG: direkt nach dem Boot sieht der Default-su-Namespace
  # eine stale FBE-Sicht -> /data/local wirkt unbeschreibbar, mkdir/tar scheitern minutenlang. -M sieht
  # den korrekten Mount-Zustand. (Memory-Falle: "adb-su nach Boot minutenlang krank, su -mm/-M".)
  [void](AdbSuM "rm -f $mk; setsid sh $sh </dev/null >/dev/null 2>&1 &" 30)
  $t0=Get-Date
  while (((Get-Date)-$t0).TotalSeconds -lt $timeoutSec) {
    Chk
    if ((AdbState) -ne 'device') { [void](WaitFor 'device' 180 (T 'core.wait_for' 'device')) }
    $m = AdbSh "cat $mk 2>/dev/null" 15
    if ($m -match 'R:(-?\d+)') {
      $rc=[int]$matches[1]; $log = if ($rc -eq 0) { AdbSh "tail -25 $lg 2>/dev/null" 20 } else { AdbSh "tail -60 $lg 2>/dev/null" 20 }
      return @{ ok=($rc -eq 0); rc=$rc; log=$log }
    }
    if ($label) { $el=[int]((Get-Date)-$t0).TotalSeconds; Pkg ("$label · " + (MMSS $el)) }
    Start-Sleep 3
  }
  $log = AdbSh "tail -60 $lg 2>/dev/null" 20
  return @{ ok=$false; rc=-2; log=("timeout; " + $log) }
}

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
  # F41: falls in stock-target.ps1 ein SHA-256 gepinnt ist, verifizieren (wie beim Factory-Image);
  # sonst warnen, dass adb/fastboot nur ueber TLS abgesichert sind.
  if ($PlatformToolsSha) {
    if ((Get-FileHash $ptZip -Algorithm SHA256).Hash -ne $PlatformToolsSha.ToUpper()) {
      Remove-Item $ptZip -Force -ErrorAction SilentlyContinue
      Fail 'platform-tools SHA-256 stimmt nicht - Abbruch'
    }
    Ok 'platform-tools Integritaet ok'
  } else {
    Warn 'platform-tools ungepinnt (nur TLS) - zum Pinnen PlatformToolsSha in stock-target.ps1 setzen'
  }
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
  # Exit-Status pruefen: ein stiller erase-Fehler (z.B. Geraet faellt nach dem super-Flash vom USB)
  # laesst frisch geflashtes System mit alten FBE-Keys zurueck -> Decrypt-Bootloop, der erst spaeter
  # als unklarer "kein adb"-Timeout auffaellt.
  $eu = Run $script:FB 'erase userdata' $null 120
  if ($eu.code -ne 0 -or (($eu.lines -join ' ') -match 'FAILED|error:')) { Fail 'fastboot erase userdata fehlgeschlagen - Abbruch (sonst Decrypt-Bootloop mit alten FBE-Keys)' }
  $em = Run $script:FB 'erase metadata' $null 60
  if ($em.code -ne 0 -or (($em.lines -join ' ') -match 'FAILED|error:')) { Fail 'fastboot erase metadata fehlgeschlagen - Abbruch (sonst Decrypt-Bootloop mit alten FBE-Keys)' }
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

# F5: Payload-Artefakt gegen payload/SHA256SUMS pruefen, BEVOR es geflasht/installiert wird.
# -NonFatal: nur warnen (fuer init_boot-magisk.img, das Ensure-InitBoot lokal neu patchen kann).
function Verify-PayloadFile($name, [switch]$NonFatal){
  $sums = Join-Path $script:Payload 'SHA256SUMS'
  $file = Join-Path $script:Payload $name
  $bail = { param($m) if ($NonFatal) { Warn $m } else { Fail $m } }
  if (-not (Test-Path $sums)) { & $bail "SHA256SUMS fehlt - $name nicht verifizierbar"; return }
  if (-not (Test-Path $file)) { & $bail "$name fehlt im payload-Ordner"; return }
  $want = $null
  foreach ($ln in (Get-Content $sums)) {
    $parts = $ln -split '\s+', 2
    if ($parts.Count -eq 2 -and $parts[1].Trim() -eq $name) { $want = $parts[0].Trim().ToLower(); break }
  }
  if (-not $want) { & $bail "$name nicht in SHA256SUMS gelistet"; return }
  $have = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
  if ($have -ne $want) { & $bail "$name SHA256 stimmt nicht (beschaedigt/getauscht?)"; return }
  Info "Integritaet ok: $name"
}

function Step3_KernelRoot(){
  Step 3 (T 'setup.steps.kernel') 'run'
  if ((AdbState) -eq 'device') {
    $kv=AdbSh 'uname -r'; $root=(AdbSu 'id' 15) -match 'uid=0'
    if ($kv -like '6.1.157-android14-11-g*' -and $root) { Ok (T 'core.s3.already' $kv); Step 3 (T 'setup.steps.kernel') 'ok'; return }
    if ($kv -notlike '6.1.157-android14-11-g*') {
      Ensure-InitBoot
      Verify-PayloadFile 'Magisk-30.7.apk'
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
    Verify-PayloadFile 'boot-lz4.img'                     # geshipptes Kernel-Image: fatal bei Mismatch
    Verify-PayloadFile 'init_boot-magisk.img' -NonFatal   # kann von Ensure-InitBoot lokal neu gepatcht sein -> nur warnen
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
    # F9: Push-Ergebnisse pruefen — schlaegt der preconfig.env-Push fehl, wendet das
    # Geraeteskript nichts an, meldet aber trotzdem Erfolg (Passwort/WLAN nie gesetzt).
    $rd = Run $script:ADB "push `"$(Join-Path $script:Kit 'device-install.sh')`" /data/local/tmp/barra-kit/" $null 60
    if (($rd.lines -join ' ') -notmatch 'file pushed') { Remove-Item $pre -Force -ErrorAction SilentlyContinue; Fail (T 'core.s4.pre_fail') }
    $rp = Run $script:ADB "push `"$pre`" /data/local/tmp/barra-kit/preconfig.env" $null 60
    if (($rp.lines -join ' ') -notmatch 'file pushed') { Remove-Item $pre -Force -ErrorAction SilentlyContinue; Fail (T 'core.s4.pre_fail') }
    [void](AdbSh 'chmod 600 /data/local/tmp/barra-kit/preconfig.env 2>/dev/null' 10)   # Klartext-Credentials sofort abschirmen
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
  Verify-PayloadFile 'barra-base.tar.gz'   # F5: vor dem Push gegen SHA256SUMS pruefen
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
  if (-not $pushed) { Remove-Item $pre -Force -ErrorAction SilentlyContinue; Fail (T 'core.s4.push_fail') }   # keine Klartext-env in %TEMP% zuruecklassen
  [void](Run $script:ADB "push `"$(Join-Path $script:Kit 'device-install.sh')`" /data/local/tmp/barra-kit/" $null 60)
  if (Test-Path $pre) { [void](Run $script:ADB "push `"$pre`" /data/local/tmp/barra-kit/preconfig.env" $null 60); [void](AdbSh 'chmod 600 /data/local/tmp/barra-kit/preconfig.env 2>/dev/null' 10); Remove-Item $pre -Force -ErrorAction SilentlyContinue }
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
        Warn (Kit-FailText "$_"); Emit 'pkg' @{ text=(Kit-FailText "$_"); state='fail' }
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
  $localSize = (Get-Item $src).Length
  for ($try=1; $try -le 3; $try++) {
    Chk
    if ((AdbState) -ne 'device') { [void](WaitFor 'device' 180 (T 'core.wait_for' 'device')) }
    $r = Run $script:ADB "push `"$src`" $dst" { param($l)
          if ($l -match '\[\s*(\d+)%\]') {
            Emit 'pkg' @{ text=("$label " + $matches[1] + '%'); state='run' }
            Emit 'progress' @{ pct=[int]$matches[1]; text=("$label " + $matches[1] + '%') }
          } } 1800 'out'
    # Groesse verifizieren: ein bei USB-Zucken abgerissener Push zeigt oft 100%, ist aber abgeschnitten
    # -> tar scheitert spaeter beim Entpacken. Deshalb Bytes auf dem Geraet gegen die lokale Groesse.
    if ($r.code -eq 0) {
      $rem = (AdbSh "wc -c < $dst 2>/dev/null" 30).Trim()
      if ($rem -eq "$localSize") { return }
      Log "KitPush: Groesse $rem != $localSize (Versuch $try) -> erneut"
    } else {
      Log "KitPush: adb push code=$($r.code) (Versuch $try) -> erneut"
    }
    Emit 'pkg' @{ text=("$label — Übertragung unvollständig, erneut ($try/3)"); state='run' }
    Start-Sleep 2
  }
  throw "push failed/truncated: $src"
}
function Kit-FailText($err){
  # Innere throws nutzen teils schon core.kit.fail — nicht doppelt wickeln (23.8.: "fehlgeschlagen: fehlgeschlagen:")
  $pfx = (T 'core.kit.fail' ([string][char]1)).Split([char]1)[0]
  if ($pfx -and "$err".StartsWith($pfx)) { return "$err" }
  T 'core.kit.fail' "$err"
}
function Run-Kits(){
  try { Run-KitsInner }
  catch {
    if ("$_" -eq 'CANCEL' -or "$_" -like '*CANCEL*') { Pkg (T 'core.cancelled') 'fail' }
    else { Pkg (Kit-FailText "$_") 'fail' }
  }
}
function Run-KitsInner(){
    # kdiag: Diagnose-Praeambel fuer die Extract-Skripte (Frisch-Flash-Bug 23.8.: mkdir unter
    # /data/local scheitert nur beim echten Flash-Lauf; errno war durch 2>/dev/null unsichtbar).
    # Loggt in das kitrun-Trace-Log: wer sind wir (id inkl. SELinux-Kontext), Mount-NS des
    # detached Kinds vs. global (proc/1), Mount-Sicht auf /data, Pfad-Zustand, AVC/fscrypt-Meldungen.
    # $D setzt jeder Aufrufer auf sein Zielverzeichnis.
    $KitDiag = 'kdiag(){ echo ==KDIAG $1 i=$i D=$D; id; getenforce; ls -ldZ /data /data/local /data/local/tmp $D; ls -l /proc/1/ns/mnt /proc/self/ns/mnt; grep -E " /data | /data/local/ubuntu " /proc/self/mounts; dmesg | grep -iE "avc|fscrypt|f2fs" | tail -5; echo ==KDIAG-END; }; '
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
          $r = AdbSuBg ($KitDiag + 'D=/data/local/barra-attn; i=0; while [ $i -lt 60 ]; do mkdir -p $D && touch $D/.wt && rm -f $D/.wt && break; [ $i = 0 ] && kdiag erst; [ $((i%12)) = 11 ] && kdiag lauf; sleep 5; i=$((i+1)); done; mkdir -p $D && touch $D/.wt && rm -f $D/.wt || kdiag final; cd $D && tar -xf /data/local/tmp/llm-kit.tar && chmod -R 755 $D && rm /data/local/tmp/llm-kit.tar') 900 (T 'core.kit.extract' $name)
          if (-not $r.ok) { throw (T 'core.kit.fail' "$name (tar rc=$($r.rc)): $($r.log)") }
        }
        Pkg (T 'core.kit.push_model' $name)
        KitPush (Join-Path $script:Kit "llm-kit\$mf") '/data/local/tmp/llm-model.gguf' $name
        $o = AdbSuM ('H=$(ls -d /data/local/ubuntu/home/* | head -1); mkdir -p $H/models && mv /data/local/tmp/llm-model.gguf $H/models/' + $mf + ' && chown -R 1001:1001 $H/models && echo MDL_OK') 180
        if ($o -notmatch 'MDL_OK') { throw (T 'core.kit.fail' "$name (model)") }
        # KEIN Auto-Start (Kevin): nur installieren; Start manuell via llmserver.sh start
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
          $r = AdbSuBg ($KitDiag + 'D=/data/local/barra-stt; i=0; while [ $i -lt 60 ]; do mkdir -p $D && touch $D/.wt && rm -f $D/.wt && break; [ $i = 0 ] && kdiag erst; [ $((i%12)) = 11 ] && kdiag lauf; sleep 5; i=$((i+1)); done; mkdir -p $D && touch $D/.wt && rm -f $D/.wt || kdiag final; cd $D && tar -xf /data/local/tmp/whisper-kit.tar && mkdir -p models && chmod -R 755 $D && rm /data/local/tmp/whisper-kit.tar') 900 (T 'core.kit.extract' $name)
          if (-not $r.ok) { throw (T 'core.kit.fail' "$name (tar rc=$($r.rc)): $($r.log)") }
        }
        Pkg (T 'core.kit.push_model' $name)
        KitPush (Join-Path $script:Kit "whisper-kit\$mf") '/data/local/tmp/stt-model.bin' $name
        $o = AdbSuM ('mkdir -p /data/local/barra-stt/models && mv /data/local/tmp/stt-model.bin /data/local/barra-stt/models/' + $mf + ' && echo MDL_OK') 120
        if ($o -notmatch 'MDL_OK') { throw (T 'core.kit.fail' "$name (model)") }
        # KEIN Auto-Start (Kevin): nur installieren; Start manuell via sttserver.sh start
        Ok (T 'core.kit.ok' $name)
      }
      elseif ($k -eq 'pya') {
        $name = T 'core.kit.pya_name'
        $mdl = if ($km['pya']) { $km['pya'] } else { 'resnet34-en' }
        Pkg (T 'core.kit.push' $name)
        KitPush (Join-Path $script:Kit 'pyannote-kit\pyannote-kit.tar') '/data/local/tmp/pya-kit.tar' $name
        Pkg (T 'core.kit.extract' $name)
        $r = AdbSuBg ($KitDiag + 'D=/data/local/ubuntu/opt; i=0; while [ $i -lt 60 ]; do touch $D/.wt && rm -f $D/.wt && break; [ $i = 0 ] && kdiag erst; [ $((i%12)) = 11 ] && kdiag lauf; sleep 5; i=$((i+1)); done; touch $D/.wt && rm -f $D/.wt || kdiag final; cd /data/local/tmp && rm -rf pya-kit && mkdir pya-kit && cd pya-kit && tar -xf ../pya-kit.tar && U=/data/local/ubuntu && mkdir -p $U/opt/barra-pya $U/opt/barra/pya && cp kit/* $U/opt/barra-pya/ && chmod 644 $U/opt/barra-pya/* && cp base/sherpa-onnx-offline-speaker-diarization $U/opt/barra/pya/ && chmod 755 $U/opt/barra/pya/sherpa-onnx-offline-speaker-diarization && cp base/barra-diarize $U/usr/local/bin/barra-diarize && chmod 755 $U/usr/local/bin/barra-diarize && cp base/pyaserver.sh /data/adb/baseos/bin/pyaserver.sh && chmod 755 /data/adb/baseos/bin/pyaserver.sh && cd /data/local/tmp && rm -rf pya-kit pya-kit.tar') 900 (T 'core.kit.extract' $name)
        if (-not $r.ok) { throw (T 'core.kit.fail' "$name (tar rc=$($r.rc)): $($r.log)") }
        if ($mdl -eq 'resnet34-zh') {
          # zh-Variante (TPU): gleiche Architektur — Trunk-Package/Kopf/ONNX im Kit ersetzen
          Pkg (T 'core.kit.push_model' $name)
          KitPush (Join-Path $script:Kit 'pyannote-kit\r34zh_trunk.package') '/data/local/tmp/pya-zh.pkg' $name
          KitPush (Join-Path $script:Kit 'pyannote-kit\head_zh.bin') '/data/local/tmp/pya-zh.head' $name
          KitPush (Join-Path $script:Kit 'pyannote-kit\resnet34_zh.onnx') '/data/local/tmp/pya-zh.onnx' $name
          $o = AdbSuM 'K=/data/local/ubuntu/opt/barra-pya; mv /data/local/tmp/pya-zh.pkg $K/r34_trunk.package && mv /data/local/tmp/pya-zh.head $K/head.bin && mv /data/local/tmp/pya-zh.onnx $K/resnet34.onnx && chmod 644 $K/* && echo EMB_OK' 120
          if ($o -notmatch 'EMB_OK') { throw (T 'core.kit.fail' "$name (model)") }
          # KEIN Auto-Start (Kevin)
        } elseif ($mdl -eq 'eres2net-zh') {
          # eres2net auf TPU: Multi-Output-Rumpf + Float-Tail; ersetzt die r34-Kit-Dateien
          Pkg (T 'core.kit.push_model' $name)
          KitPush (Join-Path $script:Kit 'pyannote-kit\eres_body.package') '/data/local/tmp/pya-e1' $name
          KitPush (Join-Path $script:Kit 'pyannote-kit\eres_tail.onnx') '/data/local/tmp/pya-e2' $name
          KitPush (Join-Path $script:Kit 'pyannote-kit\head_eres.bin') '/data/local/tmp/pya-e3' $name
          KitPush (Join-Path $script:Kit 'pyannote-kit\eres_params.txt') '/data/local/tmp/pya-e4' $name
          $o = AdbSuM 'K=/data/local/ubuntu/opt/barra-pya; rm -f $K/r34_trunk.package $K/emb_cpu.onnx; mv /data/local/tmp/pya-e1 $K/eres_body.package && mv /data/local/tmp/pya-e2 $K/eres_tail.onnx && mv /data/local/tmp/pya-e3 $K/head.bin && mv /data/local/tmp/pya-e4 $K/eres_params.txt && chmod 644 $K/* && echo EMB_OK' 120
          if ($o -notmatch 'EMB_OK') { throw (T 'core.kit.fail' "$name (model)") }
          # KEIN Auto-Start (Kevin)
        } elseif ($mdl -eq 'titanet-en') {
          # TitaNet auf TPU: 5-Segment-Kette + Float-Tail; ersetzt die r34-Kit-Dateien
          Pkg (T 'core.kit.push_model' $name)
          foreach ($tf in @('tita_seg0.package','tita_seg1.package','tita_seg2.package','tita_seg3.package','tita_seg4.package','tita_tail.onnx','tita_glue.bin','tita_params.txt')) {
            KitPush (Join-Path $script:Kit "pyannote-kit\$tf") "/data/local/tmp/$tf" $name
          }
          $o = AdbSuM 'K=/data/local/ubuntu/opt/barra-pya; rm -f $K/r34_trunk.package $K/eres_body.package $K/eres_tail.onnx $K/eres_params.txt $K/emb_cpu.onnx $K/head.bin; mv /data/local/tmp/tita_seg0.package /data/local/tmp/tita_seg1.package /data/local/tmp/tita_seg2.package /data/local/tmp/tita_seg3.package /data/local/tmp/tita_seg4.package /data/local/tmp/tita_tail.onnx /data/local/tmp/tita_glue.bin /data/local/tmp/tita_params.txt $K/ && chmod 644 $K/* && echo EMB_OK' 180
          if ($o -notmatch 'EMB_OK') { throw (T 'core.kit.fail' "$name (model)") }
          # KEIN Auto-Start (Kevin)
        }
        # KEIN Auto-Start (Kevin): Start manuell via pyaserver.sh start
        Ok (T 'core.kit.ok' $name)
      }
      elseif ($k -eq 'tts') {
        # TTS (Container-seitig, wie pya): Kit -> /opt/barra-tts, Container-systemd-Dienst barra-tts
        # (NICHT enabled -> kein Boot-Autostart) + ttsserver.sh Steuerung. Manueller Start wie llm/stt/pya.
        $name = T 'core.kit.tts_name'
        Pkg (T 'core.kit.push' $name)
        KitPush (Join-Path $script:Kit 'tts-kit\tts-kit.tar.gz') '/data/local/tmp/tts-kit.tar.gz' $name
        KitPush (Join-Path $script:Kit 'tts-kit\barra-tts.service') '/data/local/tmp/barra-tts.service' $name
        KitPush (Join-Path $script:Kit 'tts-kit\ttsserver.sh') '/data/local/tmp/ttsserver.sh' $name
        Pkg (T 'core.kit.extract' $name)
        $r = AdbSuBg ($KitDiag + 'D=/data/local/ubuntu/opt; i=0; while [ $i -lt 60 ]; do touch $D/.wt && rm -f $D/.wt && break; [ $i = 0 ] && kdiag erst; [ $((i%12)) = 11 ] && kdiag lauf; sleep 5; i=$((i+1)); done; touch $D/.wt && rm -f $D/.wt || kdiag final; U=/data/local/ubuntu; cd $U/opt && rm -rf barra-tts && tar -xzf /data/local/tmp/tts-kit.tar.gz && chown -R 1001:1001 barra-tts && cp /data/local/tmp/barra-tts.service $U/etc/systemd/system/barra-tts.service && cp /data/local/tmp/ttsserver.sh /data/adb/baseos/bin/ttsserver.sh && chmod 755 /data/adb/baseos/bin/ttsserver.sh && rm -f /data/local/tmp/tts-kit.tar.gz /data/local/tmp/barra-tts.service /data/local/tmp/ttsserver.sh') 900 (T 'core.kit.extract' $name)
        if (-not $r.ok) { throw (T 'core.kit.fail' "$name (tar rc=$($r.rc)): $($r.log)") }
        # KEIN Auto-Start (Kevin): Start manuell via 'ttsserver.sh start' (su -M, enter-systemd)
        Ok (T 'core.kit.ok' $name)
      }
      elseif ($k -eq 'wake') {
        # Weckwort (Container-Erkennung + Android-Mic-Bridge): kit -> /opt/barra-wake, base -> hwbridge/
        # baseos; Dienst barra-wake NICHT enabled, Steuerung wakeserver.sh. Manueller Start.
        $name = T 'core.kit.wake_name'
        Pkg (T 'core.kit.push' $name)
        KitPush (Join-Path $script:Kit 'wake-kit\wake-kit.tar.gz') '/data/local/tmp/wake-kit.tar.gz' $name
        Pkg (T 'core.kit.extract' $name)
        $r = AdbSuBg ($KitDiag + 'D=/data/local/ubuntu/opt; i=0; while [ $i -lt 60 ]; do touch $D/.wt && rm -f $D/.wt && break; [ $i = 0 ] && kdiag erst; [ $((i%12)) = 11 ] && kdiag lauf; sleep 5; i=$((i+1)); done; touch $D/.wt && rm -f $D/.wt || kdiag final; cd /data/local/tmp && rm -rf wake-kit && mkdir wake-kit && cd wake-kit && tar -xzf ../wake-kit.tar.gz && U=/data/local/ubuntu && mkdir -p $U/opt/barra-wake && cp -a kit/. $U/opt/barra-wake/ && chown -R 1001:1001 $U/opt/barra-wake && cp base/audiod-record /data/adb/hwbridge/audiod-record && chmod 755 /data/adb/hwbridge/audiod-record && cp base/wakeserver.sh /data/adb/baseos/bin/wakeserver.sh && chmod 755 /data/adb/baseos/bin/wakeserver.sh && cp base/barra-wake.service $U/etc/systemd/system/barra-wake.service && cd /data/local/tmp && rm -rf wake-kit wake-kit.tar.gz') 900 (T 'core.kit.extract' $name)
        if (-not $r.ok) { throw (T 'core.kit.fail' "$name (tar rc=$($r.rc)): $($r.log)") }
        # KEIN Auto-Start (Kevin): Start manuell via 'wakeserver.sh start' (su -M, enter-systemd)
        Ok (T 'core.kit.ok' $name)
      }
      elseif ($k -eq 'img') {
        # Bildgenerator (Android-seitig wie llm/stt): Kit -> /data/local/barra-img/{bin,models}, imgserver.sh -> baseos/bin,
        # Container-CLI barra-img -> /usr/local/bin. Manueller Start (imgserver.sh start, Port 8096), KEIN Boot-Autostart.
        $name = T 'core.kit.img_name'
        $mdl = if ($km['img']) { $km['img'] } else { 'dreamshaper-lcm' }
        $mf = @{ 'dreamshaper-lcm'='DreamShaper8_LCM.safetensors' }[$mdl]
        if (-not $mf) { throw (T 'core.kit.fail' "$name (model id $mdl)") }
        Pkg (T 'core.kit.push' $name)
        KitPush (Join-Path $script:Kit 'img-kit\img-kit.tar.gz') '/data/local/tmp/img-kit.tar.gz' $name
        Pkg (T 'core.kit.extract' $name)
        $r = AdbSuBg ($KitDiag + 'D=/data/local/barra-img; i=0; while [ $i -lt 60 ]; do mkdir -p $D && touch $D/.wt && rm -f $D/.wt && break; [ $i = 0 ] && kdiag erst; [ $((i%12)) = 11 ] && kdiag lauf; sleep 5; i=$((i+1)); done; mkdir -p $D && touch $D/.wt && rm -f $D/.wt || kdiag final; cd $D && rm -rf bin base && tar -xzf /data/local/tmp/img-kit.tar.gz && mkdir -p models && chmod -R 755 $D/bin && cp base/imgserver.sh /data/adb/baseos/bin/imgserver.sh && chmod 755 /data/adb/baseos/bin/imgserver.sh && U=/data/local/ubuntu && cp base/barra-img $U/usr/local/bin/barra-img && chmod 755 $U/usr/local/bin/barra-img && rm -rf base /data/local/tmp/img-kit.tar.gz') 900 (T 'core.kit.extract' $name)
        if (-not $r.ok) { throw (T 'core.kit.fail' "$name (tar rc=$($r.rc)): $($r.log)") }
        Pkg (T 'core.kit.push_model' $name)
        KitPush (Join-Path $script:Kit "img-kit\$mf") '/data/local/tmp/img-model.bin' $name
        KitPush (Join-Path $script:Kit 'img-kit\taesd.safetensors') '/data/local/tmp/img-taesd.bin' $name
        $o = AdbSuM ('D=/data/local/barra-img/models; mkdir -p $D && mv /data/local/tmp/img-model.bin $D/' + $mf + ' && mv /data/local/tmp/img-taesd.bin $D/taesd.safetensors && chmod 644 $D/* && echo MDL_OK') 180
        if ($o -notmatch 'MDL_OK') { throw (T 'core.kit.fail' "$name (model)") }
        # KEIN Auto-Start (Kevin): Start manuell via 'imgserver.sh start'
        Ok (T 'core.kit.ok' $name)
      }
    }
    if ($bothKits) { Pkg (T 'core.kit.both_note') 'ok' } else { Pkg (T 'core.kit.all_ok') 'ok' }
}
