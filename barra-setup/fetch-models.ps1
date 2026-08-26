# fetch-models.ps1 - Modelldateien beschaffen (fetch-or-supply), gesteuert von models.psd1.
#
# Prinzip: barra liefert FREMDE Modelle nicht mit, sondern laedt sie von der Originalquelle -
# genau wie fetch-stock.ps1 das Google-Factory-Image holt. Jede Datei ist per SHA-256 gepinnt;
# der Hash ist zugleich der Beweis, dass die Quelle die richtige Datei geliefert hat.
#
#   .\fetch-models.ps1 -List                     Statustabelle: was fehlt, was ist gepinnt
#   .\fetch-models.ps1                           alle upstream-Modelle mit gesetzter URL laden
#   .\fetch-models.ps1 -Id whisper-turbo         nur dieses eine
#   .\fetch-models.ps1 -Kit stt                  alle Modelle eines Kits
#   .\fetch-models.ps1 -Verify whisper-turbo -Url <kandidat>
#                                                Kandidatenquelle pruefen: laedt und vergleicht
#                                                gegen den gepinnten SHA, ohne die URL einzutragen
#   .\fetch-models.ps1 -Supply gemma-e2b -Path C:\pfad\datei.gguf
#                                                eigene Datei uebernehmen (fuer gated Quellen)
param(
  [switch]$List,
  [string]$Id,
  [string]$Kit,
  [string]$Verify,
  [string]$Url,
  [string]$Supply,
  [string]$Path,
  [switch]$Yes
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $root "models.psd1"
if (-not (Test-Path $manifestPath)) { throw "models.psd1 nicht gefunden neben diesem Skript." }
$M = Import-PowerShellDataFile -Path $manifestPath
# '_release' ist kein Modell, sondern sagt, WOHER unsere eigenen Artefakte kommen.
$Rel = $M['_release']
$M.Remove('_release') | Out-Null
$RelBase = ''
if ($Rel -and $Rel.base) { $RelBase = $Rel.base.TrimEnd('/') }

# Kit-ID -> Ordnername (historisch gewachsen: stt liegt in whisper-kit, pya in pyannote-kit)
$KitDir = @{ llm="llm-kit"; stt="whisper-kit"; pya="pyannote-kit"; tts="tts-kit"; wake="wake-kit"; img="img-kit"; dev="dev-kit"; payload="payload" }
function Target($e) {
  $d = $KitDir[$e.kit]
  if (-not $d) { throw "Unbekanntes Kit '$($e.kit)' im Manifest - Ordner in $KitDir ergaenzen." }
  Join-Path $root ("{0}\{1}" -f $d, $e.file)
}

function ShaOf($p) { (Get-FileHash $p -Algorithm SHA256).Hash.ToLower() }

# Woher kommt die Datei? Fremdmodelle tragen ihre URL selbst; unsere eigenen Artefakte
# liegen im Release und heissen dort wie die Datei.
function SourceUrl($e) {
  if ($e.origin -eq 'barra') {
    if (-not $RelBase) { return '' }
    return ($RelBase + '/' + $e.file)
  }
  if ($e.url) { return $e.url }
  return ''
}

function Status($e) {
  $t = Target $e
  if (-not (Test-Path $t)) { return "fehlt" }
  if ($e.sha256 -and (ShaOf $t) -ne $e.sha256.ToLower()) { return "SHA WEICHT AB" }
  return "da"
}

# ---------------------------------------------------------------- Liste
if ($List) {
  "{0,-20} {1,-5} {2,-34} {3,-14} {4}" -f "ID","KIT","DATEI","ZUSTAND","HERKUNFT"
  "{0,-20} {1,-5} {2,-34} {3,-14} {4}" -f ("-"*20),("-"*5),("-"*34),("-"*14),("-"*30)
  foreach ($k in ($M.Keys | Sort-Object)) {
    $e = $M[$k]
    $u = SourceUrl $e
    $herk = if ($e.gated)              { "gated -> -Supply" }
            elseif ($u)                { if ($e.origin -eq 'barra') { "aus dem Release" } else { "gepinnt" } }
            elseif ($e.origin -eq 'barra') { "RELEASE FEHLT" }
            else                       { "URL FEHLT" }
    "{0,-20} {1,-5} {2,-34} {3,-14} {4}" -f $k, $e.kit, $e.file, (Status $e), $herk
  }
  ""
  if (-not $RelBase) {
    Write-Host "HINWEIS: in models.psd1 ist keine Release-Adresse eingetragen ('_release'.base)." -ForegroundColor Yellow
    Write-Host "         Solange sie fehlt, laesst sich nichts holen, was barra selbst baut - inklusive"
    Write-Host "         payload/barra-base.tar.gz, ohne das kein Flash moeglich ist."
    Write-Host ""
  }
  $up   = @($M.Values | Where-Object { $_.origin -eq 'upstream' })
  $open = @($up | Where-Object { -not $_.url })
  $nolic= @($up | Where-Object { -not $_.license })
  "Fremdmodelle: $($up.Count) - davon ohne URL: $($open.Count), ohne Lizenzangabe: $($nolic.Count)"
  if ($open.Count)  { "  -> ohne URL laesst sich nichts laden; -Verify <id> -Url <kandidat> beweist eine Quelle ueber den SHA." }
  if ($nolic.Count) { "  -> ohne Lizenzangabe darf nichts veroeffentlicht werden (docs/models.md wird daraus erzeugt)." }
  exit 0
}

# ---------------------------------------------------------------- Supply
if ($Supply) {
  if (-not $M.ContainsKey($Supply)) { throw "Unbekannte ID '$Supply'. Bekannte: $($M.Keys -join ', ')" }
  if (-not $Path -or -not (Test-Path $Path)) { throw "-Supply braucht -Path <existierende Datei>" }
  $e = $M[$Supply]; $t = Target $e
  New-Item -ItemType Directory -Force -Path (Split-Path $t) | Out-Null
  Copy-Item $Path $t -Force
  $got = ShaOf $t
  if ($e.sha256 -and $got -ne $e.sha256.ToLower()) {
    Write-Host "[WARN]  $Supply : SHA weicht vom Pin ab." -ForegroundColor Yellow
    Write-Host "        erwartet $($e.sha256)"
    Write-Host "        bekommen $got"
    Write-Host "        Das ist eine ANDERE Datei als die, gegen die barra getestet wurde."
  } else {
    Write-Host "[ok]    $Supply uebernommen und gegen den Pin geprueft." -ForegroundColor Green
  }
  exit 0
}

# ---------------------------------------------------------------- Download-Kern
function Download($url, $dest) {
  New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
  # BITS zeigt Fortschritt und nimmt abgebrochene Downloads wieder auf; sonst Invoke-WebRequest.
  try   { Start-BitsTransfer -Source $url -Destination $dest -DisplayName "barra: $(Split-Path $dest -Leaf)" }
  catch { Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing }
}

# ---------------------------------------------------------------- Verify (Quelle beweisen)
if ($Verify) {
  if (-not $M.ContainsKey($Verify)) { throw "Unbekannte ID '$Verify'." }
  $e = $M[$Verify]
  # Reihenfolge: ausdrueckliche -Url, dann die gepinnte url, dann der Kandidat aus dem Manifest
  $cand = if ($Url) { $Url } elseif ($e.url) { $e.url } elseif ($e.urlCandidate -and $e.urlCandidate -like "http*") { $e.urlCandidate } else { throw "Keine URL und kein Kandidat fuer '$Verify' - mit -Url <kandidat> angeben." }
  if (-not $e.sha256) { throw "$Verify hat keinen SHA-Pin - ohne den ist nichts zu beweisen." }
  $tmp = Join-Path $env:TEMP ("barra-verify-" + $e.file)
  Write-Host "[pruefe] $Verify <- $cand"
  Download $cand $tmp
  $got = ShaOf $tmp
  Remove-Item $tmp -Force
  if ($got -eq $e.sha256.ToLower()) {
    Write-Host "[BEWIESEN] $Verify : SHA stimmt. Diese URL ist die Quelle - in models.psd1 als url eintragen." -ForegroundColor Green
  } else {
    Write-Host "[NEIN]     $Verify : SHA stimmt nicht - andere Datei." -ForegroundColor Red
    Write-Host "           gepinnt  $($e.sha256)"
    Write-Host "           geladen  $got"
  }
  exit 0
}

# ---------------------------------------------------------------- Laden
$sel = $M.Keys | Sort-Object
if ($Id)  { if (-not $M.ContainsKey($Id)) { throw "Unbekannte ID '$Id'." }; $sel = @($Id) }
if ($Kit) { $sel = @($sel | Where-Object { $M[$_].kit -eq $Kit }); if (-not $sel.Count) { throw "Kein Modell im Kit '$Kit'." } }

$todo = @()
foreach ($k in $sel) {
  $e = $M[$k]
  if ((Status $e) -eq 'da') { Write-Host "[da]     $k"; continue }
  if ($e.gated) { Write-Host "[gated]  $k : Quelle verlangt Zustimmung/Login - bitte -Supply $k -Path <datei>" -ForegroundColor Yellow; continue }
  $u = SourceUrl $e
  if (-not $u) {
    if ($e.origin -eq 'barra') {
      Write-Host "[offen]  $k : kommt aus dem barra-Release, aber in models.psd1 ist keine Release-Adresse eingetragen ('_release'.base)." -ForegroundColor Yellow
    } else {
      Write-Host "[offen]  $k : keine URL im Manifest. $($e.urlCandidate)" -ForegroundColor Yellow
    }
    continue
  }
  $todo += $k
}
if (-not $todo.Count) { Write-Host ""; Write-Host "Nichts zu laden."; exit 0 }

# Lizenzhinweis + Zustimmung, wie fetch-stock.ps1 es fuer Googles Bedingungen macht
# Zustimmung braucht es nur fuer FREMDE Lizenzen. Was barra selbst gebaut hat, steht unter
# Apache-2.0 wie der Rest des Projekts - dafuer den Nutzer zu fragen waere Theater.
$foreign = @($todo | Where-Object { $M[$_].origin -eq 'upstream' })
if ($foreign.Count) {
  Write-Host ""
  Write-Host "Diese Dateien werden von ihren Originalquellen geladen. barra verteilt sie nicht selbst;"
  Write-Host "es gelten die Lizenzen der jeweiligen Anbieter:"
  foreach ($k in $foreign) {
    $e = $M[$k]
    $lic = "LIZENZ NICHT ANGEGEBEN"
    if ($e.license) { $lic = $e.license }
    "  {0,-20} {1}" -f $k, $lic
    if ($e.licenseUrl) { "  {0,-20} {1}" -f "", $e.licenseUrl }
  }
  Write-Host ""
  if (-not $Yes) {
    $a = Read-Host "Mit 'ja' bestaetigen und laden"
    if ($a -ne 'ja' -and $a -ne 'yes') { Write-Host "Abgebrochen."; exit 1 }
  }
}
$own = @($todo | Where-Object { $M[$_].origin -eq 'barra' })
if ($own.Count) { Write-Host "$($own.Count) Datei(en) aus dem barra-Release (Apache-2.0)." }

# Zu grosse Dateien liegen geteilt im Release (GitHub: max 2 GB je Asset). Jeder Teil wird
# einzeln geprueft, dann zusammengesetzt und das Ergebnis gegen den Gesamt-SHA geprueft.
function DownloadParts($e, $dest) {
  $tmp = @()
  foreach ($pt in $e.parts) {
    $pu = $RelBase + '/' + $pt.file
    $pd = Join-Path (Split-Path $dest) $pt.file
    Write-Host "  Teil $($pt.file) <- $pu"
    Download $pu $pd
    $got = ShaOf $pd
    if ($pt.sha256 -and $got -ne $pt.sha256.ToLower()) {
      Remove-Item $pd -Force
      foreach ($x in $tmp) { Remove-Item $x -Force -ErrorAction SilentlyContinue }
      Write-Host "  Teil $($pt.file): SHA stimmt nicht - abgebrochen" -ForegroundColor Red
      return $false
    }
    $tmp += $pd
  }
  # zusammensetzen (streamend, damit nichts komplett in den Speicher muss)
  $out = [System.IO.File]::Create($dest)
  try { foreach ($x in $tmp) { $in = [System.IO.File]::OpenRead($x); try { $in.CopyTo($out) } finally { $in.Dispose() } } }
  finally { $out.Dispose() }
  foreach ($x in $tmp) { Remove-Item $x -Force -ErrorAction SilentlyContinue }
  return $true
}

$fail = 0
foreach ($k in $todo) {
  $e = $M[$k]; $t = Target $e
  if ($e.parts) {
    New-Item -ItemType Directory -Force -Path (Split-Path $t) | Out-Null
    Write-Host "[lade]   $k (in $($e.parts.Count) Teilen)"
    if (-not (DownloadParts $e $t)) { $fail++; continue }
    $got = ShaOf $t
    if ($got -ne $e.sha256.ToLower()) {
      Remove-Item $t -Force
      Write-Host "[FEHLER] $k : zusammengesetzte Datei hat den falschen SHA." -ForegroundColor Red
      $fail++
    } else { Write-Host "[ok]     $k geladen, zusammengesetzt und geprueft." -ForegroundColor Green }
    continue
  }
  $u = SourceUrl $e
  Write-Host "[lade]   $k <- $u"
  Download $u $t
  # Nicht jeder Eintrag hat einen Pin: die Pruefsummenliste selbst ist die Referenz und
  # traegt bewusst keinen eigenen Hash. Ohne diese Pruefung lief .ToLower() auf $null.
  if (-not $e.sha256) { Write-Host "[ok]     $k geladen (kein Pin - Referenzdatei)." -ForegroundColor Green; continue }
  $got = ShaOf $t
  if ($got -ne $e.sha256.ToLower()) {
    Remove-Item $t -Force
    Write-Host "[FEHLER] $k : SHA-Mismatch, Datei geloescht." -ForegroundColor Red
    Write-Host "         erwartet $($e.sha256)"
    Write-Host "         bekommen $got"
    $fail++
  } else {
    Write-Host "[ok]     $k geladen und geprueft." -ForegroundColor Green
  }
}
Write-Host ""
if ($fail) { Write-Host "$fail Datei(en) fehlgeschlagen." -ForegroundColor Red; exit 1 }
Write-Host "Fertig. Der Setup-Assistent bietet jetzt an, was hier liegt."
