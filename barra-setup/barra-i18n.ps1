# ============================================================================
# barra-i18n.ps1 — Lokalisierungs-Loader fuer barra-setup (gleiche .properties wie auf dem Geraet).
# Kataloge: <kit>\i18n\{en,de,...}.properties  (key=wert, UTF-8, %s-Platzhalter, %% = Prozent, \n = Zeile)
# Nutzung:  . .\barra-i18n.ps1;  Initialize-BarraI18n -Dir "$Kit\i18n" -Lang de
#           T 'setup.p1.title'      T 'setup.err.user' $name
# Fallback-Kette: gewaehlte Sprache -> en -> der Key selbst.
# ============================================================================
$script:I18nTab = @{}
$script:I18nFallback = @{}
$script:I18nLang = 'en'

function Read-I18nFile([string]$Path) {
  $tab = @{}
  if (-not (Test-Path $Path)) { return $tab }
  foreach ($line in [IO.File]::ReadAllLines($Path, [Text.Encoding]::UTF8)) {
    if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
    $i = $line.IndexOf('=')
    $tab[$line.Substring(0, $i)] = $line.Substring($i + 1)
  }
  $tab
}

function Get-BarraI18nLanguages([string]$Dir) {
  Get-ChildItem (Join-Path $Dir '*.properties') -ErrorAction SilentlyContinue |
    ForEach-Object { $_.BaseName } | Sort-Object
}

function Initialize-BarraI18n([string]$Dir, [string]$Lang) {
  $script:I18nLang = $Lang
  $script:I18nFallback = Read-I18nFile (Join-Path $Dir 'en.properties')
  $script:I18nTab = if ($Lang -eq 'en') { $script:I18nFallback } else { Read-I18nFile (Join-Path $Dir "$Lang.properties") }
}

function T([string]$Key) {
  $v = $script:I18nTab[$Key]
  if ($null -eq $v) { $v = $script:I18nFallback[$Key] }
  if ($null -eq $v) { return $Key }
  # %% -> Literal-Prozent, %s -> Argumente der Reihe nach, \n -> Zeilenumbruch
  $v = $v.Replace('%%', [string][char]1)
  if ($args.Count -gt 0) {
    foreach ($a in $args) {
      $i = $v.IndexOf('%s')
      if ($i -lt 0) { break }
      $v = $v.Substring(0, $i) + [string]$a + $v.Substring($i + 2)
    }
  }
  $v.Replace([string][char]1, '%').Replace('\n', "`n")
}
