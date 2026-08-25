#!/bin/bash
# barra-emb-port — eine wespeaker-ResNet34-Variante (Speaker-Embedding, beliebige Sprache)
# fuer die TPU portieren: ein Werkzeuglauf, kein Code.
#   ./barra-emb-port.sh <wespeaker_r34.onnx> <name>        (z.B. r34fr)
# Ablauf (Dev-PC mit WSL + Node per USB):
#   1. PC: fixieren/Trunk/Kopf/16x8-Quant + PC-Verify (wespeaker_r34_build.py)
#   2. Node: bewachter TPU-Compile (compile-std + Speicher-Waechter)
#   3. Node: TPU-Verify gegen die PC-Sim-Referenz (ODT=2! Kaltstart-Hinweis beachten)
#   4. Kit-Dateien: <name>_trunk.package + head_<name>.bin + Original-ONNX (pyannote-Kit-Muster;
#      Installation ERSETZT r34_trunk.package/head.bin/resnet34.onnx im Kit — wie der zh-Zweig)
# Grenzen: NUR wespeaker-ResNet34. eres2net/TitaNet-artige Modelle (Sigmoid-Gate-Quantkiller)
# brauchen individuelle Splits — Rezepte in src/pyannote + Memory.
set -e
ONNX="$1"; NAME="$2"
[ -f "$ONNX" ] && [ -n "$NAME" ] || { echo "Nutzung: barra-emb-port.sh <r34.onnx> <name>"; exit 1; }
SRC="$(cd "$(dirname "$0")" && pwd)"
D=${BARRA_EMB_WORK:-/mnt/c/Users/kevin/projects/pixel-cluster-base/tpu-toolchain/pyannote}
WINADB="/mnt/c/Users/kevin/AppData/Local/Microsoft/WinGet/Packages/Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe/platform-tools/adb.exe"
ADB=${ADB:-$(command -v adb.exe >/dev/null 2>&1 && echo adb.exe || { [ -x "$WINADB" ] && echo "$WINADB" || echo adb; })}
# Windows-adb braucht Windows-Pfade fuer lokale Dateien (WSL-Bruecke)
apath(){ case "$ADB" in *.exe) wslpath -w "$1";; *) printf '%s' "$1";; esac; }

echo "== 1/4 PC-Build (Quant + Sim-Verify) =="
python3 "$SRC/wespeaker_r34_build.py" "$ONNX" "$NAME" "$D"

echo "== 2/4 TPU-Compile auf dem Node =="
$ADB push "$(apath "$D/${NAME}_trunk_16x8.tflite")" /data/local/tmp/${NAME}_trunk.tflite
cat > "$D/_embcomp.sh" <<EOF
#!/system/bin/sh
T=/data/local/tmp
AV=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
[ "\$AV" -lt 1500000 ] && { echo "ABBRUCH: \${AV}kB frei"; exit 1; }
( while :; do
    a=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
    [ "\$a" -lt 1200000 ] && pkill -x tpuc1
    sleep 0.5
  done ) &
G=\$!
setenforce 0 2>/dev/null
cp /data/adb/baseos/tpu/tpuc1 \$T/tpuc1 2>/dev/null; cp /data/adb/baseos/tpu/libcomp_std.so \$T/libcomp_std.so 2>/dev/null
chmod 755 \$T/tpuc1
rm -f \$T/out.package /data/vendor/edgetpu/cache/_0 /data/vendor/edgetpu/cache/_0_checksum 2>/dev/null
LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 COMPILER_SO=\$T/libcomp_std.so MODEL=\$T/${NAME}_trunk.tflite timeout 300 \$T/tpuc1 >/dev/null 2>&1
kill \$G 2>/dev/null
setenforce 1 2>/dev/null
[ -s \$T/out.package ] && { mv \$T/out.package \$T/${NAME}_trunk.package; chmod 644 \$T/${NAME}_trunk.package; echo "COMPILE_OK \$(wc -c < \$T/${NAME}_trunk.package) B"; } || { echo COMPILE_FAIL; exit 1; }
EOF
sed -i 's/\r$//' "$D/_embcomp.sh"
$ADB push "$(apath "$D/_embcomp.sh")" /data/local/tmp/embcomp.sh
$ADB shell "su -c 'sh /data/local/tmp/embcomp.sh'" | grep -E "COMPILE_OK|FAIL|ABBRUCH"

echo "== 3/4 TPU-Verify (TPU == PC-Sim; nach frischem Boot erst einen STT/Whisper-Lauf machen — Kaltstart-Falle) =="
$ADB push "$(apath "$D/${NAME}_verify_in.bin")" /data/local/tmp/${NAME}_vin.bin
cat > "$D/_embver.sh" <<EOF
#!/system/bin/sh
T=/data/local/tmp; E=\$T/e2ehw
/system/bin/toybox pkill -9 -x tpud_pipe4 2>/dev/null; sleep 1
mkdir -p \$E; chmod 777 \$E
for s in gpuzc.sock gxp.sock gpu.sock; do ln -sf /data/local/ubuntu/opt/hwbridge/\$s \$E/\$s; done
rm -f \$E/tpu.sock
cp /data/adb/baseos/stt/tpud_pipe4 \$T/tpud_pipe4 2>/dev/null; chmod 755 \$T/tpud_pipe4 2>/dev/null
TPU_PIPE=1 TPU_PIPE_N=1 TPU_PIPE_IN=\$T/${NAME}_vin.bin TPU_PIPE_SAVE=\$T/${NAME}_vout.bin \
  TPU_CPU=8 LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 timeout 120 \$T/tpud_pipe4 \$E/tpu.sock \$T/${NAME}_trunk.package >\$T/embver.log 2>&1
chmod 644 \$T/${NAME}_vout.bin 2>/dev/null
[ -s \$T/${NAME}_vout.bin ] && echo VERIFY_RAW_OK || { tail -3 \$T/embver.log; echo VERIFY_FAIL; }
EOF
sed -i 's/\r$//' "$D/_embver.sh"
$ADB push "$(apath "$D/_embver.sh")" /data/local/tmp/embver.sh
$ADB shell "su -c 'sh /data/local/tmp/embver.sh'" | tail -2
$ADB pull /data/local/tmp/${NAME}_vout.bin "$(apath "$D/${NAME}_vout.bin")"
python3 - "$D" "$NAME" <<'EOF'
import numpy as np, struct, sys
D, NAME = sys.argv[1], sys.argv[2]
h = open(f"{D}/head_{NAME}.bin","rb").read(28)
_,_,_ = struct.unpack("<III", h[:12]); ISC, OSC = struct.unpack("<dd", h[12:28])
ref = np.fromfile(f"{D}/{NAME}_verify_ref.f32", dtype=np.float32).astype(np.float64)
out = np.fromfile(f"{D}/{NAME}_vout.bin", dtype=np.int16).astype(np.float64)[:ref.size]*OSC
cos = ref@out/(np.linalg.norm(ref)*np.linalg.norm(out))
print("TPU-vs-Sim cos = %.6f %s" % (cos, "OK" if cos > 0.999 else "— PRUEFEN (Kaltstart? erst STT-Lauf)"))
EOF

echo "== 4/4 Kit-Dateien =="
$ADB pull /data/local/tmp/${NAME}_trunk.package "$(apath "$D/${NAME}_trunk.package")"
echo ""
echo "Kit-Dateien (pyannote-Kit-Muster, ersetzen r34_trunk.package/head.bin/resnet34.onnx):"
echo "  $D/${NAME}_trunk.package"
echo "  $D/head_${NAME}.bin"
echo "  $ONNX  (als resnet34.onnx ins Kit)"
