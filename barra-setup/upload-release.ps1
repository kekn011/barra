# upload-release.ps1 - haengt die Release-Dateien an ein GitHub-Release und prueft sie vorher.
#
# Die Assets dieses Projekts entstehen NICHT in CI: Base-Image, Kernel, Kit-Archive und
# TPU-Packages sind lokale Build-Artefakte. release-please legt das Release deshalb als
# Entwurf an (docs/releasing.md); dieses Skript fuellt ihn.
#
#   .\upload-release.ps1                 Tag aus ..\version.txt, prueft und laedt hoch
#   .\upload-release.ps1 -Tag v0.2.0     anderes Ziel
#   .\upload-release.ps1 -VerifyOnly     nur pruefen, nichts hochladen
#
# Geprueft wird gegen models.psd1: Vorhandensein, Groesse und SHA-256 jeder Datei. Eine
# Abweichung bricht ab, BEVOR irgendetwas hochgeladen wird - ein halb falsches Release ist
# schlimmer als keins.
[CmdletBinding()]
param(
  [string]$Tag,
  [string]$Repo = 'kekn011/barra',
  [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Tag) {
  $vf = Join-Path (Split-Path $root) 'version.txt'
  if (-not (Test-Path $vf)) { throw "version.txt nicht gefunden - -Tag angeben." }
  $Tag = 'v' + (Get-Content $vf -Raw).Trim()
}

# Kit-ID -> Ordnername, wie in fetch-models.ps1 (historisch gewachsen).
$KitDir = @{ llm='llm-kit'; stt='whisper-kit'; pya='pyannote-kit'; tts='tts-kit'
             wake='wake-kit'; img='img-kit'; dev='dev-kit'; payload='payload' }

$M = Import-PowerShellDataFile (Join-Path $root 'models.psd1')
$rel = $M['_release']; $M.Remove('_release') | Out-Null
if ($rel.tag -ne $Tag) {
  throw "models.psd1 nennt $($rel.tag), hochgeladen werden soll $Tag. Erst _release nachziehen (docs/releasing.md)."
}

# Assets sammeln: nur unsere eigenen Dateien, geteilte Dateien als ihre Teile.
$assets = New-Object System.Collections.Generic.List[object]
foreach ($k in $M.Keys) {
  $e = $M[$k]
  if ($e.origin -ne 'barra') { continue }
  $dir = Join-Path $root $KitDir[$e.kit]
  if (-not $dir) { throw "Unbekanntes Kit '$($e.kit)' bei '$k'." }
  if ($e.parts) {
    foreach ($p in $e.parts) {
      $assets.Add([pscustomobject]@{ id=$k; name=$p.file; path=(Join-Path $dir $p.file); bytes=$p.bytes; sha=$p.sha256 })
    }
  } else {
    $assets.Add([pscustomobject]@{ id=$k; name=$e.file; path=(Join-Path $dir $e.file); bytes=$e.bytes; sha=$e.sha256 })
  }
}
Write-Host ("{0} Assets, {1:N2} GB, Ziel {2}" -f $assets.Count, (($assets | Measure-Object bytes -Sum).Sum/1GB), $Tag)

Write-Host "== pruefen =="
$fehler = @()
foreach ($a in $assets) {
  if (-not (Test-Path $a.path))              { $fehler += "$($a.name): fehlt lokal"; continue }
  $ist = (Get-Item $a.path).Length
  if ($ist -ne $a.bytes)                     { $fehler += "$($a.name): $ist Bytes statt $($a.bytes)"; continue }
  if (-not $a.sha) { Write-Host "  $($a.name): ohne Pin (Referenzdatei) - nur Groesse geprueft"; continue }
  if ((Get-FileHash $a.path -Algorithm SHA256).Hash.ToLower() -ne $a.sha.ToLower()) {
    $fehler += "$($a.name): SHA-256 weicht ab"
  }
}
if ($fehler) { $fehler | ForEach-Object { Write-Host "  FEHLER $_" -ForegroundColor Red }; throw "$($fehler.Count) Abweichung(en) - nichts hochgeladen." }
Write-Host "  alle $($assets.Count) Dateien stimmen mit dem Manifest ueberein." -ForegroundColor Green
if ($VerifyOnly) { return }

Write-Host "== hochladen (kleinste zuerst, damit Fehler frueh auffallen) =="
$n = 0
foreach ($a in ($assets | Sort-Object bytes)) {
  $n++
  $t0 = Get-Date
  & gh release upload $Tag $a.path -R $Repo --clobber
  if ($LASTEXITCODE -ne 0) { throw "Upload von $($a.name) fehlgeschlagen." }
  Write-Host ("  [{0,2}/{1}] {2,-42} {3,6} MB  {4,6:N1}s" -f $n, $assets.Count, $a.name, [math]::Round($a.bytes/1MB), ((Get-Date)-$t0).TotalSeconds)
}
Write-Host "fertig. Veroeffentlichen mit: gh release edit $Tag --draft=false -R $Repo" -ForegroundColor Green