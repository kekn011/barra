import numpy as np, tensorflow as tf, sys, os
import whisper_ref as W
# Whisper-Kuer: Cross-K/V-Packages fuer die TPU (ersetzt build_graph_cross, 1,34s Mali-GEMMs).
# Je Text-Layer EIN 16x8-Package: 1x1-CONV 1280->2560, W=[Wk^T*Kscale | Wv^T], Bias=[0|bv]
# (k hat in Whisper keinen Bias; Kscale=(n_audio_state/n_audio_head)^-0.25 wie build_graph_cross).
# Input = embd_enc (ln_post-Ausgang) aus der Referenz, Kalibrierung wie proj (Jitter-Rezept).
#   python gen_cross.py <ggml-bin> <ref_all.npz> <outdir>
# Output: l{L}_cross.tflite + cross_params.txt (ntext, X{L}_isc/izp/osc/ozp) + cross_ref-Check.

model, refnpz, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
hp, _f, t = W.parse_ggml(model)
D = hp["n_audio_state"]; NLE = hp["n_audio_layer"]; NX = hp["n_text_layer"]
S = 1500
Kscale = (D/hp["n_audio_head"])**-0.25
print("D=%d NLenc=%d NLtext=%d Kscale=%.6f" % (D, NLE, NX, Kscale), flush=True)

ref = np.load(refnpz)
xL = ref["L%d_out" % (NLE-1)].reshape(S, D).astype(np.float32)
enc = W.layernorm(xL, t["encoder.ln_post.weight"], t["encoder.ln_post.bias"]).astype(np.float32)
enc4 = enc.reshape(1, S, 1, D)
rng = np.random.default_rng(23)

def convert(fn, rep, name):
    c = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()])
    c.optimizations = [tf.lite.Optimize.DEFAULT]; c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    c.inference_input_type = tf.int16; c.inference_output_type = tf.int16
    buf = c.convert()
    open(os.path.join(outdir, name + ".tflite"), "wb").write(buf)
    print("WROTE", name, len(buf), flush=True)
    return buf

def qparams(buf):
    it = tf.lite.Interpreter(model_content=buf)
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    return (float(di["quantization"][0]), int(di["quantization"][1]),
            float(do["quantization"][0]), int(do["quantization"][1]))

def run_q(buf, x, isc, izp, osc, ozp, shape):
    it = tf.lite.Interpreter(model_content=buf); it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    xq = np.clip(np.round(x/isc + izp), -32768, 32767).astype(np.int16)
    it.set_tensor(di["index"], xq.reshape(di["shape"])); it.invoke()
    return ((it.get_tensor(do["index"]).astype(np.float32) - ozp)*osc).reshape(shape)

P = {}
for L in range(NX):
    p = "decoder.blocks.%d.cross_attn." % L
    Wk = t[p+"key.weight"]; Wv = t[p+"value.weight"]; bv = t[p+"value.bias"]
    Wkv = tf.constant(np.concatenate([Wk.T*Kscale, Wv.T], 1).reshape(1, 1, D, 2*D).astype(np.float32))
    bkv = tf.constant(np.concatenate([np.zeros(D, np.float32), bv], 0).astype(np.float32))

    @tf.function(input_signature=[tf.TensorSpec([1, S, 1, D], tf.float32)])
    def cross_fn(h4):
        return tf.nn.conv2d(h4, Wkv, 1, "VALID") + bkv

    def rep():
        yield [enc4]
        for sf in (0.85, 1.0, 1.15):
            for _ in range(3):
                yield [(enc4*sf + rng.standard_normal(enc4.shape).astype(np.float32)*0.05*enc4.std()).astype(np.float32)]

    buf = convert(cross_fn, rep, "l%d_cross" % L)
    P[L] = qparams(buf)
    isc, izp, osc, ozp = P[L]
    y = run_q(buf, enc4, isc, izp, osc, ozp, (S, 2*D))
    kref = (enc @ Wk.T.astype(np.float32))*Kscale
    vref = enc @ Wv.T.astype(np.float32) + bv
    for nm, a, b in (("k", y[:, :D], kref), ("v", y[:, D:], vref)):
        cos = float((a*b).sum())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
        rel = float(np.linalg.norm(a-b))/(np.linalg.norm(b)+1e-12)
        print("CHECK L%d %s: cos=%.6f rel=%.4f" % (L, nm, cos, rel), flush=True)

with open(os.path.join(outdir, "cross_params.txt"), "w") as f:
    f.write("ntext=%d\n" % NX)
    for L in range(NX):
        isc, izp, osc, ozp = P[L]
        f.write("X%d_isc=%r\nX%d_izp=%d\nX%d_osc=%r\nX%d_ozp=%d\n" % (L, isc, L, izp, L, osc, L, ozp))
print("DONE", flush=True)
