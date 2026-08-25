# fetch-devtools.ps1 - externe Toolchains fuers Dev-Kit beschaffen (fetch-or-supply).
# Prinzip (Kevin 25.8.): das Kit buendelt NUR unseren Code. Alles Externe wird ENTWEDER
# hier gepinnt geladen (wie fetch-stock.ps1 das Google-Image holt: URL + SHA256) ODER der
# User legt die Datei selbst ab. Ergebnis landet in dev\third-party\bin\ vor dem Packen.
#   .\fetch-devtools.ps1                 alle mit gesetztem Pin laden+pruefen
#   .\fetch-devtools.ps1 -Supply glslang -Path C:\pfad\glslang   eigene Datei uebernehmen
param([string]$Supply, [string]$Path)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dst  = Join-Path $root "third-party\bin"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

# Pin-Manifest: URL + SHA256 je Werkzeug. LEER lassen (''), solange nicht festgezurrt -> das
# Werkzeug wird dann uebersprungen mit Hinweis auf -Supply. (Offener Punkt aus dem Konzept:
# offizielle Quellen + Pins festzurren. aarch64: glslang=android/bionic, frida-server=android-arm64.)
$Pins = @(
  @{ name='glslang';      url=''; sha='1741431bded4f84a392b57ab7d0f0d75607b2ea532a94db888b17ce19bb32bac'; xz=$false; note='aarch64-android: kein Prebuilt -> host/build-glslang-android.sh (glslang @23076b37, NDK r27c) baut es, dann -Supply glslang. SHA = 25.8. am Node verifiziertes Binary (3,36 MB, c++_static).' }
  @{ name='frida-server'; url='https://github.com/frida/frida/releases/download/17.17.0/frida-server-17.17.0-android-arm64.xz'; sha='09d1fad867b27d69562a79289f4c412e85867f5d38ab72877036ed35e4223021'; xz=$true; note='frida-server 17.17.0-android-arm64 (SHA = der .xz; 25.8. verifiziert).' }
)

function Fetch($p){
  $out = Join-Path $dst $p.name
  if (-not $p.url) { Write-Host ("[offen]  {0}: kein Pin gesetzt - via -Supply {0} -Path <datei> ablegen. {1}" -f $p.name,$p.note); return }
  $dl = if ($p.xz) { "$out.xz" } else { $out }
  Write-Host "[lade]   $($p.name) <- $($p.url)"
  try { Start-BitsTransfer -Source $p.url -Destination $dl -DisplayName "barra dev: $($p.name)" }
  catch { Invoke-WebRequest -Uri $p.url -OutFile $dl -UseBasicParsing }
  if ($p.sha) {
    $got = (Get-FileHash $dl -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $p.sha.ToLower()) { Remove-Item $dl; throw "SHA-Mismatch $($p.name): $got != $($p.sha)" }
    Write-Host "[ok]     $($p.name): Download-SHA verifiziert"
  } else { Write-Host "[WARN]   $($p.name): geladen, aber kein SHA-Pin - vor Release festzurren" }
  if ($p.xz) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) { throw "python fehlt zum Entpacken von $($p.name).xz (sonst 'xz -d' manuell)" }
    & $py -c "import lzma,sys,shutil; shutil.copyfileobj(lzma.open(sys.argv[1]),open(sys.argv[2],'wb'))" $dl $out
    Remove-Item $dl
    Write-Host "[ok]     $($p.name): entpackt -> $out"
  }
}

if ($Supply) {
  if (-not $Path -or -not (Test-Path $Path)) { throw "-Supply braucht -Path <existierende datei>" }
  Copy-Item $Path (Join-Path $dst $Supply) -Force
  Write-Host "[ok]     $Supply aus $Path uebernommen (user-supplied)"
  exit 0
}
foreach ($p in $Pins) { Fetch $p }
Write-Host ""
Write-Host "Ziel: $dst"
Write-Host "Xtensa-Toolchain (DSP) ist Vendor-lizenziert -> nicht ladbar: user-supplied (Pfad im Host-Setup, P2)."
