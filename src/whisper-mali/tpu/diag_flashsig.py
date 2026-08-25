import numpy as np, tensorflow as tf, json, sys
# flashsig-16x8 am PC sezieren: alle Zwischentensoren dumpen (preserve_all_tensors) und die
# Stufe finden, an der die Quantisierung kippt. Vergleich gegen float32-Interpreter desselben Modells.
#   python diag_flashsig.py <prefix-16x8> [topN]
pre = sys.argv[1]; topN = int(sys.argv[2]) if len(sys.argv) > 2 else 25
q = json.load(open(pre + ".qparams.json"))
dt = np.int16 if q["quant"] == "16x8" else np.int8

it = tf.lite.Interpreter(model_path=pre + ".tflite", experimental_preserve_all_tensors=True)
it.allocate_tensors()
di = it.get_input_details()[0]
xq = np.fromfile(pre + ".in.bin", dtype=dt).reshape(di["shape"])
it.set_tensor(di["index"], xq); it.invoke()

rows = []
for td in it.get_tensor_details():
    try:
        v = it.get_tensor(td["index"])
    except Exception:
        continue
    if v is None or v.size < 8: continue
    sc, zp = (td["quantization"] or (0, 0))[:2]
    if not sc: continue                      # nur quantisierte Aktivierungen
    f = (v.astype(np.float32) - zp)*sc
    # Saettigungsgrad + genutzter Anteil des Quantbereichs
    lim = 32767 if v.dtype == np.int16 else 127
    sat = float((np.abs(v) >= lim).mean())
    used = float(np.abs(v).max())/lim if v.size else 0
    rows.append((td["index"], td["name"][:80], tuple(v.shape), float(sc), sat, used,
                 float(f.min()), float(f.max())))
print("idx  sat%%  used%%  scale      range                shape  name")
for r in sorted(rows, key=lambda r: -r[4])[:topN]:
    print("%4d %5.1f %6.1f  %.3e [%9.3f %9.3f] %s %s" % (r[0], 100*r[4], 100*r[5], r[3], r[6], r[7], r[2], r[1]))
