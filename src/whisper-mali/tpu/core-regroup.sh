#!/bin/bash
# Kern mit Kopfgruppen neu bauen + enc_params (core_in/core_out/headgrp) nachziehen.
#   core-regroup.sh <name> <HG> <HTOT>
set -e
NAME=$1; HG=$2; HTOT=$3
cd /mnt/c/Users/kevin/projects/pixel-cluster-base/src/whisper-mali/tpu
OUT=/mnt/c/Users/kevin/projects/pixel-cluster-base/tpu-toolchain/whisper/${NAME}_pkgs
rm -f "$OUT/wsp_core5_b0.tflite" "$OUT/wsp_core5_b0.qparams.json"
python3 whisper_core.py "$OUT/ref_base_l0.npz" "$OUT/wsp_core5g${HG}_b0" 16x8 375 375 $HG 64 0 v5 $HTOT 0 2>&1 | grep -E "WROTE|cos" | tail -2
python3 - "$OUT" $HG <<'EOF'
import json, re, sys
out, hg = sys.argv[1], sys.argv[2]
q = json.load(open(f"{out}/wsp_core5g{hg}_b0.qparams.json"))
p = open(f"{out}/enc_params.txt").read()
p = re.sub(r"core_in=.*", "core_in=%r" % q["x"]["isc"], p)
p = re.sub(r"core_out=.*", "core_out=%r" % q["out"]["osc"], p)
p = re.sub(r"headgrp=.*", "headgrp=%s" % hg, p)
open(f"{out}/enc_params.txt", "w").write(p)
print("enc_params: core_in=%r core_out=%r headgrp=%s" % (q["x"]["isc"], q["out"]["osc"], hg))
EOF
echo REGROUP_DONE
