# Baut bootanimation.zip (1080x2400@15fps) aus logo.png:
#   part0 = 10 Frames Fade-in (Logo + "barra")
#   part1 = 12 Frames Loop (3 pulsierende Ladepunkte)
# Zip MUSS store-only sein (bootanim mmapt die Eintraege).
# WICHTIG (Deploy): das Magisk-Modul barra-bootanim braucht die Datei unter BEIDEN
# Namen — bootanimation.zip UND bootanimation-dark.zip. Pixel spielt im Dark-Theme
# die -dark-Variante; ueberlagert man nur die helle, laeuft weiter die Google-Anim.
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

# --- Zip STORE-ONLY selbst schreiben. ACHTUNG: .NET ZipArchive mit
# CompressionLevel.NoCompression erzeugt trotzdem method=8 (Deflate mit
# stored-Bloecken) — bootanimation/libziparchive kann Frames aber nur bei
# ECHTEM method=0 mmapen und faellt sonst STILL auf die eingebaute
# Google-Animation zurueck (so am 21.8. auf dem Geraet diagnostiziert).
Add-Type -TypeDefinition @'
public static class BarraCrc {
  public static uint Compute(byte[] d){ uint[] t=new uint[256];
    for(uint i=0;i<256;i++){ uint c=i; for(int k=0;k<8;k++) c=(c&1)!=0?0xEDB88320u^(c>>1):c>>1; t[i]=c; }
    uint crc=0xFFFFFFFFu; foreach(byte b in d) crc=t[(crc^b)&0xFF]^(crc>>8); return crc^0xFFFFFFFFu; }
}
'@
function Crc32([byte[]]$d){ [BarraCrc]::Compute($d) }

New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
Remove-Item $out -Force -ErrorAction SilentlyContinue
$ms=[IO.File]::Create($out); $bw=[IO.BinaryWriter]::new($ms)
$central=@()
foreach ($rel in @('desc.txt') + (0..9 | ForEach-Object {'part0/{0:d3}.png' -f $_}) + (0..11 | ForEach-Object {'part1/{0:d3}.png' -f $_})) {
  $data=[IO.File]::ReadAllBytes("$work\$($rel -replace '/','\')")
  $nb=[Text.Encoding]::ASCII.GetBytes($rel); $crc=Crc32 $data; $off=[uint32]$ms.Position
  $bw.Write([uint32]0x04034b50); $bw.Write([uint16]20); $bw.Write([uint16]0); $bw.Write([uint16]0)   # LFH, v2.0, keine Flags, STORED
  $bw.Write([uint16]0); $bw.Write([uint16]0)                                                        # Zeit/Datum 0 (deterministisch)
  $bw.Write([uint32]$crc); $bw.Write([uint32]$data.Length); $bw.Write([uint32]$data.Length)
  $bw.Write([uint16]$nb.Length); $bw.Write([uint16]0); $bw.Write($nb); $bw.Write($data)
  $central += ,@{n=$nb; crc=$crc; len=[uint32]$data.Length; off=$off}
}
$cdStart=[uint32]$ms.Position
foreach ($e in $central) {
  $bw.Write([uint32]0x02014b50); $bw.Write([uint16]20); $bw.Write([uint16]20); $bw.Write([uint16]0); $bw.Write([uint16]0)
  $bw.Write([uint16]0); $bw.Write([uint16]0); $bw.Write([uint32]$e.crc); $bw.Write([uint32]$e.len); $bw.Write([uint32]$e.len)
  $bw.Write([uint16]$e.n.Length); $bw.Write([uint16]0); $bw.Write([uint16]0); $bw.Write([uint16]0); $bw.Write([uint16]0)
  $bw.Write([uint32]0); $bw.Write([uint32]$e.off); $bw.Write($e.n)
}
$cdLen=[uint32]($ms.Position-$cdStart)
$bw.Write([uint32]0x06054b50); $bw.Write([uint16]0); $bw.Write([uint16]0)
$bw.Write([uint16]$central.Count); $bw.Write([uint16]$central.Count); $bw.Write([uint32]$cdLen); $bw.Write([uint32]$cdStart); $bw.Write([uint16]0)
$bw.Close(); $ms.Close()
Write-Output ("OK: {0} ({1:n1} MB, store-only)" -f $out,((Get-Item $out).Length/1e6))
