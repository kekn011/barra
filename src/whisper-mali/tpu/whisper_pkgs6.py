import numpy as np, tensorflow as tf, sys, json, os
import whisper_ref as W
# Whisper-Stufe-3 M3b: Packages fuer ALLE Encoder-Layer (proj 16x8 + woc/ffnresc int8-1x1-Conv;
# Kern bleibt das eine geteilte wsp_core5_b0). Kalibrierung je Layer mit echten Stufen aus
# ref_base_all.npz. Am Ende PC-KETTEN-CHECK ueber alle Layer inkl. Inter-Layer-Requant
# (int8-ffn-out -> int16-proj-in) exakt wie der Geraete-Treiber. Schreibt enc_params.txt.
#   python whisper_pkgs6.py <ggml-bin> <ref_all.npz> <outdir> [nlayers]

model, refnpz, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
NL = int(sys.argv[4]) if len(sys.argv) > 4 else 6
os.makedirs(outdir, exist_ok=True)
hp, _f, t = W.parse_ggml(model)
D = hp["n_audio_state"]; H = hp["n_audio_head"]; HD = D//H
S = 1500; BQ = 375; nb = S//BQ; sc = HD**-0.25
CORE_ISC = 0.0002111698704538867
CORE_OSC = 0.00851402897387743
ref = np.load(refnpz)
rng = np.random.default_rng(11)

def ln_tf(x, g, b):
    m = tf.reduce_mean(x, -1, keepdims=True); d = x - m
    v = tf.reduce_mean(d*d, -1, keepdims=True)
    return d*tf.math.rsqrt(v + 1e-5)*g + b

def convert(fn, rep, name, quant):
    c = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()])
    c.optimizations = [tf.lite.Optimize.DEFAULT]; c.representative_dataset = rep
    if quant == "16x8":
        c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        c.inference_input_type = tf.int16; c.inference_output_type = tf.int16
    else:
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.int8; c.inference_output_type = tf.int8
    buf = c.convert()
    open(os.path.join(outdir, name + ".tflite"), "wb").write(buf)
    print("WROTE", name, len(buf), flush=True)
    return buf

def qparams(buf):
    it = tf.lite.Interpreter(model_content=buf)
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    return (float(di["quantization"][0]), int(di["quantization"][1]),
            float(do["quantization"][0]), int(do["quantization"][1]))

def run_q(buf, x, dt, lim, isc, izp, osc, ozp, shape):
    it = tf.lite.Interpreter(model_content=buf); it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    xq = np.clip(np.round(x/isc + izp), -lim-1, lim).astype(dt)
    it.set_tensor(di["index"], xq.reshape(di["shape"])); it.invoke()
    return ((it.get_tensor(do["index"]).astype(np.float32) - ozp)*osc).reshape(shape)

P = {}
bufs = {}
for L in range(NL):
    p = "encoder.blocks.%d." % L
    C = lambda a: tf.constant(a.astype(np.float32))
    Wq = C(t[p+"attn.query.weight"].T * sc); bq = C(t[p+"attn.query.bias"] * sc)
    Wk = C(t[p+"attn.key.weight"].T * sc)
    Wv = C(t[p+"attn.value.weight"].T);  bv = C(t[p+"attn.value.bias"])
    g1 = C(t[p+"attn_ln.weight"]); c1 = C(t[p+"attn_ln.bias"])
    g2 = C(t[p+"mlp_ln.weight"]);  c2 = C(t[p+"mlp_ln.bias"])
    Wo4 = tf.constant(t[p+"attn.out.weight"].T.reshape(1, 1, D, D).astype(np.float32))
    bo_ = C(t[p+"attn.out.bias"])
    FF = t[p+"mlp.0.weight"].shape[0]
    W1c = tf.constant(t[p+"mlp.0.weight"].T.reshape(1, 1, D, FF).astype(np.float32))
    W2c = tf.constant(t[p+"mlp.2.weight"].T.reshape(1, 1, FF, D).astype(np.float32))
    b1_ = C(t[p+"mlp.0.bias"]); b2_ = C(t[p+"mlp.2.bias"])

    @tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
    def proj(x):
        h = ln_tf(x, g1, c1)
        hd = lambda z: tf.transpose(tf.reshape(z, [1, S, H, HD]), [0, 2, 1, 3])
        return tf.concat([hd(tf.matmul(h, Wq) + bq), hd(tf.matmul(h, Wk)), hd(tf.matmul(h, Wv) + bv)], 2)

    @tf.function(input_signature=[tf.TensorSpec([1, S, 1, D], tf.float32)])
    def woc(ctx):
        return tf.nn.conv2d(ctx, Wo4, 1, "VALID") + bo_

    @tf.function(input_signature=[tf.TensorSpec([1, S, 1, D], tf.float32)])
    def ffnresc(x1):
        h2 = ln_tf(x1, g2, c2)
        f = tf.nn.conv2d(tf.nn.gelu(tf.nn.conv2d(h2, W1c, 1, "VALID") + b1_, approximate=False), W2c, 1, "VALID") + b2_
        return x1 + f

    xin = ref["L%d_xin" % L].reshape(1, S, D).astype(np.float32)
    ctxr = ref["L%d_ctx" % L].reshape(1, S, 1, D).astype(np.float32)
    midr = ref["L%d_mid" % L].reshape(1, S, 1, D).astype(np.float32)
    def rep_x():
        yield [xin]
        for sf2 in (0.85, 1.0, 1.15):
            for _ in range(3):
                yield [(xin*sf2 + rng.standard_normal(xin.shape).astype(np.float32)*0.05*xin.std()).astype(np.float32)]
    def rep_ctx():
        yield [ctxr]
        for sf2 in (0.9, 1.1): yield [ctxr*sf2]
    def rep_mid():
        yield [midr]
        for sf2 in (0.9, 1.1): yield [midr*sf2]

    bufs["p%d" % L] = convert(proj, rep_x, "l%d_proj" % L, "16x8")
    bufs["w%d" % L] = convert(woc, rep_ctx, "l%d_woc" % L, "int8")
    bufs["f%d" % L] = convert(ffnresc, rep_mid, "l%d_ffnresc" % L, "16x8")
    P[L] = {"proj": qparams(bufs["p%d" % L]), "woc": qparams(bufs["w%d" % L]), "ffn": qparams(bufs["f%d" % L])}

# ---- PC-Ketten-Check ueber alle Layer (Interpreter + Glue wie der Treiber) ----
def core_np(Y):
    # exakte Attention auf dequantisiertem proj-Out (der Kern ist auf dem Geraet verifiziert;
    # hier simulieren wir seine Quantisierungskanten: in int16 CORE_ISC, out int16 CORE_OSC)
    Yq = np.clip(np.round(Y/CORE_ISC), -32768, 32767).astype(np.int16).astype(np.float32)*CORE_ISC
    q = Yq[:, :, 0:S, :]; k = Yq[:, :, S:2*S, :]; v = Yq[:, :, 2*S:3*S, :]
    s = q @ k.transpose(0, 1, 3, 2); s -= s.max(-1, keepdims=True)
    e = np.exp(s); l = e.sum(-1, keepdims=True)
    num = e @ v
    nq = np.clip(np.round(num/CORE_OSC), -32768, 32767).astype(np.int16).astype(np.float32)*CORE_OSC
    lq = np.clip(np.round(l/CORE_OSC), -32768, 32767).astype(np.int16).astype(np.float32)*CORE_OSC
    return nq/np.maximum(lq, 1e-6)

x = ref["x0"].reshape(1, S, D).astype(np.float32)
for L in range(NL):
    pi, pz, po, poz = P[L]["proj"]
    xq = np.clip(np.round(x/pi + pz), -32768, 32767).astype(np.int16).astype(np.float32)
    x_used = (xq - pz)*pi
    Y = run_q(bufs["p%d" % L], x_used, np.int16, 32767, pi, pz, po, poz, (1, H, 3*S, HD))
    ctx = core_np(Y).transpose(0, 2, 1, 3).reshape(1, S, D)
    wi, wz, wo_, woz = P[L]["woc"]
    att = run_q(bufs["w%d" % L], ctx.reshape(1, S, 1, D), np.int8, 127, wi, wz, wo_, woz, (1, S, D))
    x1 = x_used + att
    fi, fz, fo, foz = P[L]["ffn"]
    x = run_q(bufs["f%d" % L], x1.reshape(1, S, 1, D), np.int16, 32767, fi, fz, fo, foz, (1, S, D))
    refo = ref["L%d_out" % L].reshape(1, S, D)
    cos = float((x*refo).sum())/(np.linalg.norm(x)*np.linalg.norm(refo)+1e-12)
    print("KETTE L%d: cos=%.5f" % (L, cos), flush=True)

# Params + Treiber-Bins
with open(os.path.join(outdir, "enc_params.txt"), "w") as f:
    f.write("core_in=%r\ncore_out=%r\nnlayers=%d\n" % (CORE_ISC, CORE_OSC, NL))
    for L in range(NL):
        pi, pz, po, poz = P[L]["proj"]; wi, wz, wo_, woz = P[L]["woc"]; fi, fz, fo, foz = P[L]["ffn"]
        f.write("L%d_pin=%r\nL%d_pout=%r\nL%d_wisc=%r\nL%d_wizp=%d\nL%d_wosc=%r\nL%d_wozp=%d\nL%d_fisc=%r\nL%d_fizp=%d\nL%d_fosc=%r\nL%d_fozp=%d\n"
                % (L, pi, L, po, L, wi, L, wz, L, wo_, L, woz, L, fi, L, fz, L, fo, L, foz))
x0 = ref["x0"].reshape(1, S, D).astype(np.float32)
pi0 = P[0]["proj"][0]; pz0 = P[0]["proj"][1]
np.clip(np.round(x0/pi0 + pz0), -32768, 32767).astype(np.int16).tofile(os.path.join(outdir, "enc_x0.in.bin"))
ref["L%d_out" % (NL-1)].astype(np.float32).tofile(os.path.join(outdir, "enc_out_ref.f32"))
print("DONE", flush=True)
