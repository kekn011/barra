# mk-model-docs.ps1 - erzeugt docs/models.md aus models.psd1.
# Nicht von Hand pflegen: das Manifest ist die Quelle, dieses Skript nur die Ausgabe.
#   .\mk-model-docs.ps1            schreibt ..\docs\models.md
#   .\mk-model-docs.ps1 -Check     prueft nur, ob die Datei aktuell waere (Exitcode 1 wenn nicht)
param([switch]$Check)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$M    = Import-PowerShellDataFile -Path (Join-Path $root "models.psd1")
$M.Remove('_release') | Out-Null
# Abgeleitete Artefakte erben die Lizenz ihres Ausgangsmodells - das steht in licenses.psd1.
$L = @{}
$lp = Join-Path $root "licenses.psd1"
if (Test-Path $lp) { $L = Import-PowerShellDataFile -Path $lp }
$out  = Join-Path (Split-Path $root) "docs\models.md"

$KitName = @{ llm='LLM (KI-Chat)'; stt='Spracherkennung'; pya='Sprecher-Trennung'
              tts='Sprachausgabe'; wake='Weckwort'; img='Bildgenerator'; dev='Dev-Kit' }

$Lines = New-Object System.Collections.Generic.List[string]
$Lines.Add('# Models and their licenses')
$Lines.Add('')
$Lines.Add('barra does **not** redistribute third-party models. The setup fetches each one from its')
$Lines.Add('original source, pinned to a specific revision and verified against a SHA-256, after you')
$Lines.Add('accept the terms — the same way it fetches Google''s factory image.')
$Lines.Add('')
$Lines.Add('Fetch them with `barra-setup\fetch-models.ps1`; `-List` shows what is present.')
$Lines.Add('')
$Lines.Add('> This file is generated from `barra-setup/models.psd1` by `mk-model-docs.ps1`.')
$Lines.Add('> Edit the manifest, not this file.')
$Lines.Add('')

# ---- Fremdmodelle nach Kit ----
$Lines.Add('## Third-party models (downloaded)')
$Lines.Add('')
foreach ($kit in @('llm','stt','pya','img','tts','wake')) {
  $rows = @($M.GetEnumerator() | Where-Object { $_.Value.kit -eq $kit -and $_.Value.origin -eq 'upstream' } | Sort-Object Key)
  if (-not $rows.Count) { continue }
  $Lines.Add("### $($KitName[$kit])")
  $Lines.Add('')
  $Lines.Add('| File | Source | License |')
  $Lines.Add('|---|---|---|')
  foreach ($r in $rows) {
    $e = $r.Value
    $src = if ($e.url) { "[link]($($e.url))" } elseif ($e.gated) { 'gated — supply it yourself' } else { '**not yet pinned**' }
    $lic = if ($e.license) { if ($e.licenseUrl) { "[$($e.license)]($($e.licenseUrl))" } else { $e.license } } else { '**unknown**' }
    $Lines.Add("| ``$($e.file)`` | $src | $lic |")
  }
  $Lines.Add('')
  foreach ($r in $rows) {
    if ($r.Value.licenseNote) { $Lines.Add("- **$($r.Value.file)** — $($r.Value.licenseNote)") }
  }
  if ($rows | Where-Object { $_.Value.licenseNote }) { $Lines.Add('') }
}

# ---- eigene Artefakte ----
$own = @($M.GetEnumerator() | Where-Object { $_.Value.origin -eq 'barra' } | Sort-Object Key)
$Lines.Add('## barra''s own artifacts')
$Lines.Add('')
$Lines.Add("These $($own.Count) files are our build output — TPU packages, kit archives and derived")
$Lines.Add('float tails — and they ship as GitHub release assets, not as downloads from third parties.')
$Lines.Add('')
$Lines.Add('**They are not automatically Apache-2.0.** A TPU package is not an original work: it is a')
$Lines.Add('third-party model''s weights, re-quantized and re-laid-out for the Tensor G3. It carries the')
$Lines.Add('licence of the model it came from. The table below names that model and that licence for')
$Lines.Add('every file; only our own code and shaders are Apache-2.0.')
$Lines.Add('')
$L2 = $L
$Lines.Add('| File | Kit | Derived from | License |')
$Lines.Add('|---|---|---|---|')
foreach ($r in $own) {
  $e = $L2[$r.Key]
  $df = 'barra (eigener Code)'; $lic = 'Apache-2.0'
  if ($e) {
    if ($e.derivedFrom) { $df = $e.derivedFrom }
    if ($e.license) { $lic = $e.license }
    if ($e.licenseUrl) { $lic = "[$lic]($($e.licenseUrl))" }
  }
  $Lines.Add("| ``$($r.Value.file)`` | $($KitName[$r.Value.kit]) | $df | $lic |")
}
$Lines.Add('')
$obl = @($own | Where-Object { $L2[$_.Key] -and $L2[$_.Key].obligations })
if ($obl.Count) {
  $Lines.Add('**Auflagen, die mit der Weitergabe uebernommen werden:**')
  $Lines.Add('')
  foreach ($r in $obl) { $Lines.Add("- ``$($r.Value.file)`` — $($L2[$r.Key].obligations)") }
  $Lines.Add('')
}
$Lines.Add('')

# ---- offene Punkte sichtbar machen statt verschweigen ----
$notes = @($M.GetEnumerator() | Where-Object { $_.Value.note } | Sort-Object Key)
if ($notes.Count) {
  $Lines.Add('## Known gaps')
  $Lines.Add('')
  $Lines.Add('Some of our kit archives still bundle third-party models. Those are **not** covered by the')
  $Lines.Add('table above and have to be split out before a release, so they can be fetched and licensed')
  $Lines.Add('like everything else:')
  $Lines.Add('')
  foreach ($n in $notes) { $Lines.Add("- ``$($n.Value.file)`` — $($n.Value.note)") }
  $Lines.Add('')
}

$text = ($Lines -join "`n") + "`n"

if ($Check) {
  # UTF-8 AUSDRUECKLICH lesen: PS 5.1 nimmt fuer BOM-lose Dateien sonst ANSI an,
  # dann verunglueckt jeder Gedankenstrich und der Vergleich schlaegt grundlos fehl.
  $enc = New-Object System.Text.UTF8Encoding($false)
  if ((Test-Path $out) -and ([System.IO.File]::ReadAllText($out, $enc) -eq $text)) { "docs/models.md ist aktuell."; exit 0 }
  "docs/models.md ist NICHT aktuell - mk-model-docs.ps1 ausfuehren."; exit 1
}
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
[System.IO.File]::WriteAllText($out, $text, (New-Object System.Text.UTF8Encoding($false)))
"geschrieben: $out"
$open = @($M.Values | Where-Object { $_.origin -eq 'upstream' -and -not $_.license })
if ($open.Count) { Write-Host "WARNUNG: $($open.Count) Modell(e) ohne Lizenzangabe - stehen als 'unknown' in der Tabelle." -ForegroundColor Yellow }
