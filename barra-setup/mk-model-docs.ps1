# mk-model-docs.ps1 - erzeugt docs/models.md aus models.psd1.
# Nicht von Hand pflegen: das Manifest ist die Quelle, dieses Skript nur die Ausgabe.
#   .\mk-model-docs.ps1            schreibt ..\docs\models.md
#   .\mk-model-docs.ps1 -Check     prueft nur, ob die Datei aktuell waere (Exitcode 1 wenn nicht)
param([switch]$Check)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$M    = Import-PowerShellDataFile -Path (Join-Path $root "models.psd1")
$out  = Join-Path (Split-Path $root) "docs\models.md"

$KitName = @{ llm='LLM (KI-Chat)'; stt='Spracherkennung'; pya='Sprecher-Trennung'
              tts='Sprachausgabe'; wake='Weckwort'; img='Bildgenerator'; dev='Dev-Kit' }

$L = New-Object System.Collections.Generic.List[string]
$L.Add('# Models and their licenses')
$L.Add('')
$L.Add('barra does **not** redistribute third-party models. The setup fetches each one from its')
$L.Add('original source, pinned to a specific revision and verified against a SHA-256, after you')
$L.Add('accept the terms — the same way it fetches Google''s factory image.')
$L.Add('')
$L.Add('Fetch them with `barra-setup\fetch-models.ps1`; `-List` shows what is present.')
$L.Add('')
$L.Add('> This file is generated from `barra-setup/models.psd1` by `mk-model-docs.ps1`.')
$L.Add('> Edit the manifest, not this file.')
$L.Add('')

# ---- Fremdmodelle nach Kit ----
$L.Add('## Third-party models (downloaded)')
$L.Add('')
foreach ($kit in @('llm','stt','pya','img','tts','wake')) {
  $rows = @($M.GetEnumerator() | Where-Object { $_.Value.kit -eq $kit -and $_.Value.origin -eq 'upstream' } | Sort-Object Key)
  if (-not $rows.Count) { continue }
  $L.Add("### $($KitName[$kit])")
  $L.Add('')
  $L.Add('| File | Source | License |')
  $L.Add('|---|---|---|')
  foreach ($r in $rows) {
    $e = $r.Value
    $src = if ($e.url) { "[link]($($e.url))" } elseif ($e.gated) { 'gated — supply it yourself' } else { '**not yet pinned**' }
    $lic = if ($e.license) { if ($e.licenseUrl) { "[$($e.license)]($($e.licenseUrl))" } else { $e.license } } else { '**unknown**' }
    $L.Add("| ``$($e.file)`` | $src | $lic |")
  }
  $L.Add('')
  foreach ($r in $rows) {
    if ($r.Value.licenseNote) { $L.Add("- **$($r.Value.file)** — $($r.Value.licenseNote)") }
  }
  if ($rows | Where-Object { $_.Value.licenseNote }) { $L.Add('') }
}

# ---- eigene Artefakte ----
$own = @($M.GetEnumerator() | Where-Object { $_.Value.origin -eq 'barra' } | Sort-Object Key)
$L.Add('## barra''s own artifacts')
$L.Add('')
$L.Add("These $($own.Count) files are our build output — TPU packages, kit archives and derived")
$L.Add('float tails. They are Apache-2.0 like the rest of the project and ship as GitHub release')
$L.Add('assets, not as downloads from third parties.')
$L.Add('')
$L.Add('| File | Kit |')
$L.Add('|---|---|')
foreach ($r in $own) { $L.Add("| ``$($r.Value.file)`` | $($KitName[$r.Value.kit]) |") }
$L.Add('')

# ---- offene Punkte sichtbar machen statt verschweigen ----
$notes = @($M.GetEnumerator() | Where-Object { $_.Value.note } | Sort-Object Key)
if ($notes.Count) {
  $L.Add('## Known gaps')
  $L.Add('')
  $L.Add('Some of our kit archives still bundle third-party models. Those are **not** covered by the')
  $L.Add('table above and have to be split out before a release, so they can be fetched and licensed')
  $L.Add('like everything else:')
  $L.Add('')
  foreach ($n in $notes) { $L.Add("- ``$($n.Value.file)`` — $($n.Value.note)") }
  $L.Add('')
}

$text = ($L -join "`n") + "`n"

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
