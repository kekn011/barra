import numpy as np, tensorflow as tf, json, sys
# PC-Interpreter-Vergleich eines exportierten Parts gegen die numpy-Referenzstufe.
#   python cmp_l0.py <prefix> <ref.npz> <refkey> [mult]
pre, refnpz, key = sys.argv[1], sys.argv[2], sys.argv[3]
mult = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
q = json.load(open(pre + ".qparams.json"))
dt = np.int16 if q["quant"] == "16x8" else np.int8
it = tf.lite.Interpreter(model_path=pre + ".tflite")
it.allocate_tensors()
di, do = it.get_input_details()[0], it.get_output_details()[0]
xq = np.fromfile(pre + ".in.bin", dtype=dt).reshape(di["shape"])
it.set_tensor(di["index"], xq); it.invoke()
y = it.get_tensor(do["index"]).astype(np.float32)
yf = ((y - q["ozp"])*q["osc"]).reshape(-1)
ref = (np.load(refnpz)[key].astype(np.float32)*mult).reshape(-1)
cos = float((yf*ref).sum())/(np.linalg.norm(yf)*np.linalg.norm(ref)+1e-12)
rel = np.linalg.norm(yf-ref)/(np.linalg.norm(ref)+1e-12)
print("%s %s vs %s: cos=%.5f rel_l2=%.2f%%  (out range %.3f..%.3f, ref %.3f..%.3f)"
      % (pre.split("/")[-1], q["quant"], key, cos, 100*rel, yf.min(), yf.max(), ref.min(), ref.max()))
