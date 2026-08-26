#!/bin/bash
# barra-stt-port — ein Whisper-Modell (ggml-Format) fuer den TPU-Encoder portieren:
# ein Werkzeuglauf ueber die generische Whisper-Familien-Pipeline.
#   ./barra-stt-port.sh <name> <ggml-modell.bin> <NL> <H>
#   (tiny: NL=4 H=6 | base: 6/8 | small: 12/12 | medium: 24/16 | turbo: 32/20)
# Ablauf (Dev-PC mit WSL-TF + Node per USB):
#   1. PC: gen-model.sh (Referenz, Kern v5, Layer-/Cross-/Conv-Pakete als 16x8-tflites)
#      + Kopfgruppen-Regroup wenn H > 10 (Kern-Limit ~10-12 Koepfe; HG = H/2)
#   2. Node: bewachter Compile ALLER tflites (Speicher-Waechter — Compiler-Panic-Lektion!)
#   3. Node: Packages + enc_params.txt nach /data/local/barra-stt/<name>, Kit-Tar zurueck
# Danach: sttserver.sh start <name> (bzw. Kit via install-stt/Wizard verteilen).
# Hinweis projw: whisper_pkgs7 setzt die Params-Keys (dim/heads/headgrp/projw) selbst;
# medium/turbo (H>=16) liefen damit — bei neuen Grossmodellen enc_params.txt gegenpruefen.
set -e
NAME="$1"; GGML="$2"; NL="$3"; H="$4"
[ -f "$GGML" ] && [ -n "$H" ] || { echo "Nutzung: barra-stt-port.sh <name> <ggml.bin> <NL> <H>"; exit 1; }
SRC="$(cd "$(dirname "$0")" && pwd)"
WD=${BARRA_TOOLCHAIN:?BARRA_TOOLCHAIN setzen (TPU-Toolchain-Datenverzeichnis)}/whisper
OUT=$WD/${NAME}_pkgs
BARRA_REPO=${BARRA_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
# adb: PATH zuerst, sonst das mitgelieferte Windows-adb aus dem Repo
WINADB="${BARRA_ADB:-$BARRA_REPO/barra-setup/tools/adb.exe}"
ADB=${ADB:-$(command -v adb.exe >/dev/null 2>&1 && echo adb.exe || { [ -x "$WINADB" ] && echo "$WINADB" || echo adb; })}
# Windows-adb braucht Windows-Pfade fuer lokale Dateien (WSL-Bruecke)
apath(){ case "$ADB" in *.exe) wslpath -w "$1";; *) printf '%s' "$1";; esac; }

echo "== 1/3 PC-Generierung =="
bash "$SRC/gen-model.sh" "$NAME" "$GGML" "$NL" "$H"
if [ "$H" -gt 10 ]; then
  HG=$((H / 2))
  echo "== 1b/3 Kopfgruppen-Regroup (H=$H > Kern-Limit -> HG=$HG) =="
  bash "$SRC/core-regroup.sh" "$NAME" "$HG" "$H"
fi

echo "== 2/3 Compile auf dem Node ($(ls "$OUT"/*.tflite | wc -l) tflites) =="
$ADB shell "mkdir -p /data/local/tmp/sttport-$NAME"
$ADB push "$(apath "$OUT")/." /data/local/tmp/sttport-$NAME/
cat > "$WD/_sttcomp.sh" <<EOF
#!/system/bin/sh
T=/data/local/tmp; A=\$T/sttport-$NAME; K=/data/local/barra-stt/$NAME
setenforce 0 2>/dev/null
cp /data/adb/baseos/tpu/tpuc1 \$T/tpuc1 2>/dev/null; cp /data/adb/baseos/tpu/libcomp_std.so \$T/libcomp_std.so 2>/dev/null
chmod 755 \$T/tpuc1
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
( while :; do
    a=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
    [ "\$a" -lt 1200000 ] && { echo "GUARD: \${a}kB -> pkill tpuc1"; pkill -x tpuc1; sleep 2; }
    sleep 0.5
  done ) &
G=\$!
ok=0; fail=0
for f in \$A/*.tflite \$A/cross/*.tflite \$A/conv/*.tflite; do
  [ -f "\$f" ] || continue
  base=\$(basename \$f .tflite)
  [ -s \$A/\$base.package ] && { ok=\$((ok+1)); continue; }
  AV=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
  [ "\$AV" -lt 1500000 ] && { echo "ABBRUCH: \${AV}kB frei"; kill \$G; exit 1; }
  rm -f \$T/out.package /data/vendor/edgetpu/cache/_0 /data/vendor/edgetpu/cache/_0_checksum 2>/dev/null
  COMPILER_SO=\$T/libcomp_std.so MODEL=\$f timeout 300 \$T/tpuc1 >/dev/null 2>&1
  if [ -s \$T/out.package ]; then mv \$T/out.package \$A/\$base.package; ok=\$((ok+1)); else echo "FEHLER: \$base"; fail=\$((fail+1)); fi
done
kill \$G 2>/dev/null
setenforce 1 2>/dev/null
echo "STTCOMPILE ok=\$ok fail=\$fail"
[ \$fail -gt 0 ] && exit 1
mkdir -p \$K; rm -f \$K/*
for p in \$A/*.package; do ln \$p \$K/ || cp \$p \$K/; done
# Params + Float-Glue-Dateien (flach, wie die bestehenden Kits)
for f in \$A/*.txt \$A/*.f32 \$A/cross/*.txt \$A/conv/*.txt; do [ -f "\$f" ] && cp "\$f" \$K/; done
chmod -R 755 /data/local/barra-stt/$NAME
cd /data/local/barra-stt
tar -cf \$T/whisper-kit-$NAME.tar $NAME
chmod 644 \$T/whisper-kit-$NAME.tar
echo "KIT_OK \$(ls \$K | wc -l) Dateien, \$(wc -c < \$T/whisper-kit-$NAME.tar) B"
EOF
sed -i 's/\r$//' "$WD/_sttcomp.sh"
$ADB push "$(apath "$WD/_sttcomp.sh")" /data/local/tmp/sttcomp.sh
$ADB shell "su -c 'sh /data/local/tmp/sttcomp.sh'" | tail -4

echo "== 3/3 Kit-Tar zurueckholen =="
$ADB pull /data/local/tmp/whisper-kit-$NAME.tar "$(apath "$WD/../whisper-kit-$NAME.tar")"
echo ""
echo "Fertig: $WD/../whisper-kit-$NAME.tar + installiert unter /data/local/barra-stt/$NAME"
echo "Test:   adb shell su -c 'sh /data/adb/baseos/bin/sttserver.sh start $NAME'"
