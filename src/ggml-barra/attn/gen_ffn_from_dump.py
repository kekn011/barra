#!/usr/bin/env python3
"""gen_ffn_from_dump.py — FFN-Packages fuer den TPU-FFN-Hook, kalibriert aus ECHTEN Abzuegen.

  L0=<erste> L1=<hinter der letzten> python3 gen_ffn_from_dump.py

Je Layer i: ffn_L{i}.tflite = Wd(silu(Wg h) * (Wu h)), Batch B=32, 16x8 (int16-Aktivierungen,
int8-Gewichte). Kalibrierung kommt aus cal_ffn_L{i}.bin, die der Vulkan-Hook mit
BARRA_FFN_CALDUMP=<dir> abnimmt (ggml-vulkan-barra.inc, battn_ffn_caldump).

WARUM NICHT gen_ffn_pkgs.py (tpu-toolchain): das baut das ganze Modell in numpy nach, um die
Aktivierungen zu erzeugen. Bei hybriden Modellen (qwen3.5: 24 von 33 Schichten sind DeltaNet)
muesste es die Delta-Regel mit nachbauen. Der Abzug funktioniert dagegen fuer jedes Modell,
das durch unseren Vulkan-Pfad laeuft. (29.8.2026)

Meta: "D FF B bits NL split out_div out_div_gu Bd" + je Layer "isc izp osc ozp".
Die optionalen Kopffelder IMMER schreiben — der Parser des alten Backends liest den Zeilenrest
mit fgets und frisst sonst die erste Datenzeile ("meta Layer N ungueltig").

Grenzen am Geraet (29.8.2026 gemessen): 30 Pakete -> RegisterGraph FAIL; 29 geladen + Modell
-> OOM-Kernelpanik; ab ~8 Modellen gleichzeitig -> ZC_MAXH=32 in tpud.c bremst die Inferenz.
Praktisch getestet und gemessen: 16 Layer, 50/50-Aufteilung, 8 % schnellerer Prefill.
"""
import sys, os, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
import gguf, tensorflow as tf
B=os.path.dirname(os.path.abspath(__file__))
G="/mnt/c/Users/kevin/projects/barra-fremd/barra-setup/llm-kit/qwen38-4b-distill.gguf"
OUT=B+"/ffnkit"; CAL=OUT
D,FF,BATCH=2560,9216,32
import os as _os
L0=int(_os.environ.get("L0","0")); L1=int(_os.environ.get("L1","16")); NL=L1
r=gguf.GGUFReader(G); TENS={t.name:t for t in r.tensors}
def ten(n):
    t=TENS[n]; d=np.asarray(t.data)
    a=d.astype(np.float32) if t.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.BF16) else gguf.quants.dequantize(d,t.tensor_type).astype(np.float32)
    return a.reshape([int(x) for x in reversed(t.shape)])
meta=[]
for L in range(L0, L1):
    p=f"blk.{L}."
    Wg=tf.constant(ten(p+"ffn_gate.weight").T); Wu=tf.constant(ten(p+"ffn_up.weight").T); Wd=tf.constant(ten(p+"ffn_down.weight").T)
    h=np.fromfile(f"{CAL}/cal_ffn_L{L}.bin",dtype=np.float32).reshape(-1,D)
    cal=[h[i:i+BATCH] for i in range(0,min(len(h),BATCH*8),BATCH)]
    @tf.function(input_signature=[tf.TensorSpec([BATCH,D],tf.float32,name="h")])
    def fn(x): return {"y": tf.matmul(tf.nn.silu(tf.matmul(x,Wg))*tf.matmul(x,Wu), Wd)}
    c=tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()])
    c.optimizations=[tf.lite.Optimize.DEFAULT]; c.representative_dataset=lambda: ([x] for x in cal)
    c.target_spec.supported_ops=[tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    c.inference_input_type=tf.int16; c.inference_output_type=tf.int16
    pth=f"{OUT}/ffn_L{L}.tflite"; open(pth,"wb").write(c.convert())
    it=tf.lite.Interpreter(model_path=pth); it.allocate_tensors()
    di,do=it.get_input_details()[0],it.get_output_details()[0]
    meta.append((float(di["quantization"][0]),int(di["quantization"][1]),float(do["quantization"][0]),int(do["quantization"][1])))
    print(f"L{L}: isc={meta[-1][0]:.3e} osc={meta[-1][2]:.3e}", flush=True)
    del Wg,Wu,Wd
with open(f"{OUT}/ffn_L{L0}_{L1}.meta","w") as f:
    f.write(f"{D} {FF} {BATCH} 16 {NL}\n")
    for m in meta: f.write(f"{m[0]:.8e} {m[1]} {m[2]:.8e} {m[3]}\n")
print(f"fertig: {NL} Layer, meta geschrieben", flush=True)
