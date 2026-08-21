# Baut bootanimation.zip (1080x2400@15fps) aus logo.png:
#   part0 = 10 Frames Fade-in (Logo + "barra")
#   part1 = 12 Frames Loop (3 pulsierende Ladepunkte)
# Zip MUSS store-only sein (bootanim mmapt die Eintraege).
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$W=1080; $H=2400; $FPS=15
$logoPath=Join-Path $PSScriptRoot '..\..\logo.png'
$work=Join-Path $env:TEMP 'barra-bootanim'
$out=Join-Path $PSScriptRoot 'bootanimation.zip'
Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory "$work\part0","$work\part1" | Out-Null

$logo=[System.Drawing.Bitmap]::new($logoPath)
$lw=700; $lh=[int]($logo.Height*$lw/$logo.Width)
$lx=[int](($W-$lw)/2); $ly=900-[int]($lh/2)          # Logo-Mitte bei y=900
$textY=1380.0
$dotsY=1560; $dotR=14; $dotGap=70

function New-Frame([double]$alpha,[double[]]$dots){
  $bmp=[System.Drawing.Bitmap]::new($W,$H)
  $g=[System.Drawing.Graphics]::FromImage($bmp)
  $g.Clear([System.Drawing.Color]::Black)
  $g.InterpolationMode='HighQualityBicubic'; $g.SmoothingMode='AntiAlias'; $g.TextRenderingHint='AntiAlias'
  # Logo mit Alpha
  $cm=[System.Drawing.Imaging.ColorMatrix]::new(); $cm.Matrix33=[single]$alpha
  $ia=[System.Drawing.Imaging.ImageAttributes]::new(); $ia.SetColorMatrix($cm)
  $dst=[System.Drawing.Rectangle]::new($lx,$ly,$lw,$lh)
  $g.DrawImage($logo,$dst,0,0,$logo.Width,$logo.Height,[System.Drawing.GraphicsUnit]::Pixel,$ia)
  # Wortmarke
  $f=[System.Drawing.Font]::new('Segoe UI Semibold',96,[System.Drawing.GraphicsUnit]::Pixel)
  $b=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb([int](255*$alpha),240,242,246))
  $sz=$g.MeasureString('barra',$f)
  $g.DrawString('barra',$f,$b,[single](($W-$sz.Width)/2),[single]$textY)
  # Ladepunkte
  if ($dots) {
    $x0=[int]($W/2 - $dotGap)
    for ($d=0; $d -lt 3; $d++) {
      $v=[int](255*$dots[$d])
      $db=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb($v,61,111,224))
      $g.FillEllipse($db,$x0+$d*$dotGap-$dotR,$dotsY-$dotR,2*$dotR,2*$dotR)
      $db.Dispose()
    }
  }
  $f.Dispose(); $b.Dispose(); $ia.Dispose(); $g.Dispose()
  return $bmp
}

for ($i=0; $i -lt 10; $i++) {
  $a=($i+1)/10.0
  $bmp=New-Frame $a $null
  $bmp.Save(("{0}\part0\{1:d3}.png" -f $work,$i),[System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
}
for ($i=0; $i -lt 12; $i++) {
  $p=$i/12.0
  $dots=@(0,1,2 | ForEach-Object { 0.25 + 0.75*(([math]::Sin(2*[math]::PI*($p - $_/3.0))+1)/2) })
  $bmp=New-Frame 1.0 $dots
  $bmp.Save(("{0}\part1\{1:d3}.png" -f $work,$i),[System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
}
$logo.Dispose()
[IO.File]::WriteAllText("$work\desc.txt","$W $H $FPS`np 1 0 part0`np 0 0 part1`n")

New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
Remove-Item $out -Force -ErrorAction SilentlyContinue
$fs=[IO.File]::Create($out)
$zip=[IO.Compression.ZipArchive]::new($fs,[IO.Compression.ZipArchiveMode]::Create)
foreach ($rel in @('desc.txt') + (0..9 | ForEach-Object {'part0/{0:d3}.png' -f $_}) + (0..11 | ForEach-Object {'part1/{0:d3}.png' -f $_})) {
  $e=$zip.CreateEntry($rel,[IO.Compression.CompressionLevel]::NoCompression)
  $es=$e.Open(); $bytes=[IO.File]::ReadAllBytes("$work\$($rel -replace '/','\')"); $es.Write($bytes,0,$bytes.Length); $es.Close()
}
$zip.Dispose(); $fs.Close()
Write-Output ("OK: {0} ({1:n1} MB)" -f $out,((Get-Item $out).Length/1e6))
