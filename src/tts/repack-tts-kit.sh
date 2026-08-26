#!/bin/bash
# tts-kit.tar.gz umbauen: Stimme "David" entfernen, die Piper-Stimmen auf den
# GPU-Vokoder umstellen. Laeuft auf dem PC (kein Node noetig), MUSS aber in WSL
# laufen — git-bash setzt auf NTFS keine Exec-Bits, die Binaries im Kit waeren tot.
#
#   sh repack-tts-kit.sh <alt.tar.gz> <arbeitsverzeichnis-mit-gpukits> <neu.tar.gz>
#
# Erwartet im Arbeitsverzeichnis (Ausgabe von dump-dec-onnx + gen-gpudec2 + export-front-onnx):
#   gpukit-de/{program2.json,weights16.bin,bias16.bin}   piper-de-front.onnx
#   gpukit-en/{program2.json,weights16.bin,bias16.bin}   piper-en-front.onnx
set -e
OLD=$1; WORK=$2; NEW=$3
[ -f "$OLD" ] && [ -d "$WORK" ] && [ -n "$NEW" ] || { sed -n '2,9p' "$0"; exit 1; }
SRC=$(cd "$(dirname "$0")" && pwd)

case "$(uname -r)" in *icrosoft*|*WSL*) ;; *) echo "FEHLER: in WSL ausfuehren (Exec-Bits)"; exit 1;; esac

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
echo "== entpacke =="
tar -xzf "$OLD" -C "$TMP"
K=$TMP/barra-tts
[ -d "$K" ] || { echo "unerwartete Kit-Struktur"; exit 1; }

echo "== David entfernen =="
rm -rf "$K/voices/david"

echo "== Piper-Stimmen auf den GPU-Vokoder umstellen =="
add_voice() {   # $1 = Verzeichnisname, $2 = espeak-Stimme, $3/$4 = Label de/en
  d=$K/voices/$1
  [ -d "$d" ] || { echo "  $1: fehlt im Kit, uebersprungen"; return; }
  mkdir -p "$d/gpukit2"
  cp "$WORK/gpukit-${1#piper-}/program2.json" "$WORK/gpukit-${1#piper-}/weights16.bin" \
     "$WORK/gpukit-${1#piper-}/bias16.bin" "$d/gpukit2/"
  cp "$WORK/$1-front.onnx" "$d/front.onnx"
  # model.onnx bleibt liegen: sherpa ist der Rueckfallweg, wenn gpudecd nicht laeuft
  cat > "$d/voice.json" <<JSON
{"engine":"piper-gpu","front_onnx":"front.onnx","tokens":"tokens.txt","gpukit":"gpukit2",
 "espeak_voice":"$2","model":"model.onnx","data_dir":"espeak-ng-data",
 "gpu_sock":"/run/barra-tts/gpudec-$1.sock",
 "scales":[0.667,1.0,0.8],"sample_rate":22050,"sid":0,
 "label_de":"$3","label_en":"$4"}
JSON
  echo "  $1: gpukit2 + front.onnx + voice.json(piper-gpu)"
}
add_voice piper-de de "Thorsten (Deutsch, GPU)" "Thorsten (German, GPU)"
add_voice piper-en en-us "Amy (Englisch, GPU)" "Amy (English, GPU)"

echo "== Dienst-Dateien aus src/ nachziehen =="
cp "$SRC/ttsd.py"    "$K/bin/ttsd.py"
cp "$SRC/launch.sh"  "$K/bin/launch.sh"
cp "$SRC/tools/piper_ids.py" "$K/bin/piper_ids.py"
chmod 755 "$K/bin/launch.sh" "$K/bin/gpudecd" 2>/dev/null || true

echo "== Kontrolle =="
# NUR unsere Stimme suchen - espeak-ng bringt eine eigene Variante 'voices/!v/david' mit,
# die nichts mit uns zu tun hat und einen naiven Zaehler falsch alarmieren laesst.
echo -n "  Stimme david (muss 0 sein): "; find "$K/voices" -maxdepth 1 -iname 'david' | wc -l
echo -n "  Verweise auf voices/david (muss 0 sein): "; grep -rl "voices/david" "$K/bin" 2>/dev/null | wc -l
echo -n "  Stimmen: "; ls "$K/voices" | tr '\n' ' '; echo
for v in "$K"/voices/*/; do
  echo "    $(basename "$v"): engine=$(sed -n 's/.*"engine":"\([^"]*\)".*/\1/p' "$v/voice.json")"
done

echo "== packen =="
tar -czf "$NEW" -C "$TMP" barra-tts
echo "fertig: $NEW  ($(du -m "$NEW" | cut -f1) MB, vorher $(du -m "$OLD" | cut -f1) MB)"
sha256sum "$NEW"
