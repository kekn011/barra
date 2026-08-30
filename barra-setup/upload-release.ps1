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

# Zweite Pruefung: die Kopien IN den Archiven. Die Kits und das Base-Image tragen eigene
# Kopien der Geraete-Skripte und der i18n-Kataloge - eine Korrektur in src/ erreicht das
# Produkt also nicht von selbst. Genau so sind ausgeliefert worden: die Vor-i18n-Fassung
# von sttserver.sh (27.8.2026) und vier weitere Skripte samt einem Katalog ohne die
# stt.- und pya.-Schluessel (28.8.2026) - sichtbar erst als rohe Schluessel auf dem Geraet.
$Kopien = @(
  @{ a='pyannote-kit\pyannote-kit.tar'; m='base/pyaserver.sh';                        q='src\pyannote\pyaserver.sh' }
  @{ a='pyannote-kit\pyannote-kit.tar'; m='base/barra-diarize';                       q='src\pyannote\barra-diarize' }
  @{ a='img-kit\img-kit.tar.gz';        m='base/imgserver.sh';                        q='src\img-kit\imgserver.sh' }
  @{ a='img-kit\img-kit.tar.gz';        m='base/barra-img';                           q='src\img-kit\barra-img' }
  @{ a='wake-kit\wake-kit.tar.gz';      m='./base/wakeserver.sh';                     q='src\wake\wakeserver.sh' }
  @{ a='payload\barra-base.tar.gz';     m='adb/baseos/bin/sttserver.sh';              q='src\whisper-mali\sttserver.sh' }
  @{ a='payload\barra-base.tar.gz';     m='adb/baseos/bin/llmserver.sh';              q='src\boot\llmserver.sh' }
  @{ a='payload\barra-base.tar.gz';     m='adb/baseos/bin/wifi-join.sh';              q='src\boot\wifi-join.sh' }
  # baseos-cfgd.sh liegt im Image ZWEIMAL - und NUR die hwbridge-Kopie wird gestartet.
  # Beide pruefen, sonst laeuft wieder die alte (26.8. genau so passiert).
  @{ a='payload\barra-base.tar.gz';     m='adb/hwbridge/baseos-cfgd.sh';              q='src\boot\baseos-cfgd.sh' }
  @{ a='payload\barra-base.tar.gz';     m='adb/baseos/bin/baseos-cfgd.sh';            q='src\boot\baseos-cfgd.sh' }
  @{ a='payload\barra-base.tar.gz';     m='adb/baseos/i18n/de.properties';            q='src\i18n\de.properties' }
  @{ a='payload\barra-base.tar.gz';     m='adb/baseos/i18n/en.properties';            q='src\i18n\en.properties' }
  @{ a='payload\barra-base.tar.gz';     m='ubuntu/usr/share/barra/i18n/de.properties'; q='src\i18n\de.properties' }
  @{ a='payload\barra-base.tar.gz';     m='ubuntu/usr/share/barra/i18n/en.properties'; q='src\i18n\en.properties' }
)
# Freistehende Skript-Assets: dieselbe Frage, ohne Archiv drumherum.
$Frei = @(
  @{ f='tts-kit\ttsserver.sh';     q='src\tts\ttsserver.sh' }
  # Der Waechter reist im Setup mit (kein Asset) - hier trotzdem gegen src/ pruefen, damit die
  # Kopie nicht auseinanderlaeuft. Genau diese Klasse Fehler ist dreimal ausgeliefert worden.
  @{ f='base\barra-guard.sh';      q='src\boot\barra-guard.sh' }
  @{ f='whisper-kit\sttserver.sh'; q='src\whisper-mali\sttserver.sh' }
)
Write-Host "== Kopien in den Archiven gegen src/ =="
$src = Split-Path $root
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("barra-relcheck-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
  foreach ($e in $Frei) {
    $f = Join-Path $root $e.f; $q = Join-Path $src $e.q
    if (-not (Test-Path $f)) { $fehler += "$($e.f): fehlt lokal"; continue }
    if ((Get-FileHash $f -Algorithm SHA256).Hash -ne (Get-FileHash $q -Algorithm SHA256).Hash) {
      $fehler += "$($e.f) weicht von $($e.q) ab"
    }
  }
  foreach ($g in ($Kopien | Group-Object { $_.a })) {
    $arch = Join-Path $root $g.Name
    if (-not (Test-Path $arch)) { $fehler += "$($g.Name): fehlt lokal"; continue }
    $glieder = @($g.Group | ForEach-Object { $_.m })
    & tar.exe -xf $arch -C $tmp @glieder
    if ($LASTEXITCODE -ne 0) { $fehler += "$($g.Name): Glieder nicht auslesbar ($($glieder -join ', '))"; continue }
    foreach ($e in $g.Group) {
      $ist  = Join-Path $tmp ($e.m -replace '/', '\')
      $soll = Join-Path $src $e.q
      if (-not (Test-Path $ist)) { $fehler += "$($g.Name): $($e.m) fehlt im Archiv"; continue }
      if ((Get-FileHash $ist -Algorithm SHA256).Hash -ne (Get-FileHash $soll -Algorithm SHA256).Hash) {
        $fehler += "$($g.Name): $($e.m) weicht von $($e.q) ab (auch Zeilenenden pruefen - CRLF im Arbeitsbaum?)"
      }
    }
  }
} finally { Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue }
if ($fehler) { $fehler | ForEach-Object { Write-Host "  FEHLER $_" -ForegroundColor Red }; throw "$($fehler.Count) Abweichung(en) - nichts hochgeladen." }
Write-Host "  alle $($Kopien.Count + $Frei.Count) Kopien stimmen mit src/ ueberein." -ForegroundColor Green

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