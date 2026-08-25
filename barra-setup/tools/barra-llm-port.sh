#!/bin/bash
# barra-llm-port — ein GGUF-Standardmodell (Qwen/Llama/GLM-artig, GQA, HD=128) fuer den
# TPU-Attention-Offload portieren: KEIN Code, ein Werkzeuglauf.
#   ./barra-llm-port.sh <modell.gguf>
# Ablauf (Dev-PC mit WSL-TF + Node per USB):
#   1. GGUF auf den Node, Kalibrier-Dump aus dem echten llama.cpp-Lauf (BARRA_ATTN_CALDUMP)
#   2. gen_attn_gen.py: GGUF + Dumps -> TPU-Packages (16x8) + attn.meta (v7) + aux_attn.bin
#   3. Compile auf dem Node (tpuc1, mit Speicher-Waechter)
#   4. Kit nach /data/local/barra-attn/<name> + Kit-Tar (llm-attn-<name>.tar) zurueck
# Danach: llmserver.sh start <modell> schaltet den Offload automatisch an.
# Grenzen: Standard-Decoder (RMSNorm, NEOX/NORM-RoPE, optional Bias/q/k-Norm, HD=128).
# Exoten (Gemma-4-Klasse: HD!=128/KV-Sharing) sind eigene Etappen — siehe README.
set -e
GGUF="$1"
[ -f "$GGUF" ] || { echo "Nutzung: barra-llm-port.sh <modell.gguf>"; exit 1; }
NAME=$(basename "$GGUF" .gguf)
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK=${BARRA_PORT_WORK:-$HOME/barra-port/$NAME}   # nicht mehr maschinenspezifisch; via $BARRA_PORT_WORK ueberschreibbar
mkdir -p "$WORK/cal" "$WORK/pkg"
WINADB="/mnt/c/Users/kevin/AppData/Local/Microsoft/WinGet/Packages/Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe/platform-tools/adb.exe"
ADB=${ADB:-$(command -v adb.exe >/dev/null 2>&1 && echo adb.exe || { [ -x "$WINADB" ] && echo "$WINADB" || echo adb; })}
# Windows-adb braucht Windows-Pfade fuer lokale Dateien (WSL-Bruecke)
apath(){ case "$ADB" in *.exe) wslpath -w "$1";; *) printf '%s' "$1";; esac; }

echo "== 1/4 Kalibrier-Dump auf dem Node =="
$ADB push "$(apath "$GGUF")" /data/local/tmp/$NAME.gguf
cat > "$WORK/caldump.sh" <<EOF
#!/system/bin/sh
T=/data/local/tmp; L=/data/adb/baseos/llm
pgrep -f "\$L/llama-server" >/dev/null && { echo "ABBRUCH: llmserver laeuft (erst stop)"; exit 1; }
pgrep -x tpud_pipe4 >/dev/null && { echo "ABBRUCH: tpud laeuft"; exit 1; }
AV=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
[ "\$AV" -lt 3500000 ] && { echo "ABBRUCH: nur \${AV}kB frei"; exit 1; }
mkdir -p \$T/cal-$NAME; rm -f \$T/cal-$NAME/cal_*.bin
head -c 12000 \$T/wiki.test.raw > \$T/calprompt.txt 2>/dev/null || head -c 12000 \$L/../bin/llmserver.sh > \$T/calprompt.txt
. \$L/env.sh
export BARRA_ATTN_CALDUMP=\$T/cal-$NAME
timeout 300 \$L/llama-cli -m \$T/$NAME.gguf -ngl 99 -f \$T/calprompt.txt -n 1 -c 4096 -b 512 -ub 512 -no-cnv </dev/null >\$T/cal.out 2>\$T/cal.err &
PP=\$!
while kill -0 \$PP 2>/dev/null; do
  AV=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
  [ "\$AV" -lt 1300000 ] && { echo "WATCHDOG: kill"; kill -9 \$PP; break; }
  sleep 1
done
wait \$PP 2>/dev/null
echo "Dateien: \$(ls \$T/cal-$NAME | wc -l)"
EOF
sed -i 's/\r$//' "$WORK/caldump.sh"
$ADB push "$(apath "$WORK/caldump.sh")" /data/local/tmp/caldump-port.sh
$ADB shell "su -c 'sh /data/local/tmp/caldump-port.sh'"
$ADB pull /data/local/tmp/cal-$NAME/. "$(apath "$WORK/cal")/"

echo "== 2/4 Packages generieren (TFLite 16x8) =="
python3 "$SRC/gen_attn_gen.py" "$GGUF" "$WORK/cal" "$WORK/pkg" 512

echo "== 3/4 Compile auf dem Node =="
$ADB shell "mkdir -p /data/local/tmp/port-$NAME"
$ADB push "$(apath "$WORK/pkg")/." /data/local/tmp/port-$NAME/
cat > "$WORK/comp.sh" <<EOF
#!/system/bin/sh
setenforce 0 2>/dev/null
trap 'setenforce 1 2>/dev/null' EXIT   # SELinux auch bei Fehlerabbruch wieder enforcing setzen
T=/data/local/tmp; A=\$T/port-$NAME
cp /data/adb/baseos/tpu/tpuc1 \$T/tpuc1 2>/dev/null; cp /data/adb/baseos/tpu/libcomp_std.so \$T/libcomp_std.so 2>/dev/null
chmod 755 \$T/tpuc1
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
ok=0; fail=0
for f in \$A/*.tflite; do
  base=\$(basename \$f .tflite)
  [ -s \$A/\$base.package ] && { ok=\$((ok+1)); continue; }
  AV=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
  [ "\$AV" -lt 1500000 ] && { echo "ABBRUCH: \${AV}kB frei"; exit 1; }
  rm -f \$T/out.package /data/vendor/edgetpu/cache/_0 /data/vendor/edgetpu/cache/_0_checksum 2>/dev/null
  COMPILER_SO=\$T/libcomp_std.so MODEL=\$f timeout 300 \$T/tpuc1 >/dev/null 2>&1
  if [ -s \$T/out.package ]; then mv \$T/out.package \$A/\$base.package; ok=\$((ok+1)); else echo "FEHLER: \$base"; fail=\$((fail+1)); fi
done
setenforce 1 2>/dev/null
echo "COMPILE ok=\$ok fail=\$fail"
[ \$fail -gt 0 ] && exit 1 || true
EOF
sed -i 's/\r$//' "$WORK/comp.sh"
$ADB push "$(apath "$WORK/comp.sh")" /data/local/tmp/comp-port.sh
$ADB shell "su -c 'sh /data/local/tmp/comp-port.sh'" | tee "$WORK/compile.log"
grep -q "fail=0" "$WORK/compile.log" || { echo "Compile-Fehler"; exit 1; }

echo "== 4/4 Kit installieren + Tar sichern =="
cat > "$WORK/kit.sh" <<EOF
#!/system/bin/sh
T=/data/local/tmp; A=\$T/port-$NAME; K=/data/local/barra-attn/$NAME
mkdir -p \$K; rm -f \$K/*
for p in \$A/*.package; do ln \$p \$K/ || cp \$p \$K/; done
cp \$A/attn_gen.meta \$K/attn.meta
cp \$A/aux_attn.bin \$K/
ls \$A/*.package | sed 's#.*/##; s/\.package//' | sort -t_ -k2.2n > \$K/pkglist.tmp
# Reihenfolge: je Layer q(1)/q2/kv/o — aus dem Namensschema rekonstruieren
NL=\$(head -1 \$K/attn.meta | { read _ _ _ _ _ nl _ _; echo \$nl; })
: > \$K/pkglist.txt
i=0
while [ \$i -lt \$NL ]; do
  for ty in q q1 q2 kv o; do [ -f \$K/\${ty}_L\$i.package ] && echo "\${ty}_L\$i" >> \$K/pkglist.txt; done
  i=\$((i+1))
done
rm \$K/pkglist.tmp
chmod -R 755 /data/local/barra-attn
tar -cf \$T/llm-attn-$NAME.tar -C /data/local/barra-attn $NAME
chmod 644 \$T/llm-attn-$NAME.tar
echo "KIT_OK \$(wc -l < \$K/pkglist.txt) Pakete"
EOF
sed -i 's/\r$//' "$WORK/kit.sh"
$ADB push "$(apath "$WORK/kit.sh")" /data/local/tmp/kit-port.sh
$ADB shell "su -c 'sh /data/local/tmp/kit-port.sh'"
$ADB pull /data/local/tmp/llm-attn-$NAME.tar "$(apath "$WORK/../llm-attn-$NAME.tar")"
echo ""
echo "Fertig: Kit installiert (/data/local/barra-attn/$NAME) + $WORK/../llm-attn-$NAME.tar"
echo "Start:  adb shell su -c 'sh /data/adb/baseos/bin/llmserver.sh start <pfad>/$NAME.gguf'"
