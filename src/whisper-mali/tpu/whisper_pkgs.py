import numpy as np, tensorflow as tf, sys, json, os
import whisper_ref as W
# Whisper-Stufe-3 M3: Layer als Package-Satz. proj(16x8) -> 4x core-v5(16x8, Q-Offset gebacken)
# -> CPU-Glue (Division, Head-Transpose, Requant) -> tail(int8: Wo+Residual+LN+GELU-FFN+Residual).
# Erzeugt tflites + qparams + in/ref-Bins und macht den PC-KETTEN-CHECK (Interpreter + Glue vs numpy).
#   python whisper_pkgs.py <ggml-bin> <ref.npz> <outdir> [layer]

model, refnpz, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
LAYER = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.makedirs(outdir, exist_ok=True)
hp, _f, t = W.parse_ggml(model)
D = hp["n_audio_state"]; H = hp["n_audio_head"]; HD = D//H
S = 1500; BQ = 375; nb = S//BQ; sc = HD**-0.25
p = "encoder.blocks.%d." % LAYER
C = lambda a: tf.constant(a.astype(np.float32))
Wq = C(t[p+"attn.query.weight"].T * sc); bq = C(t[p+"attn.query.bias"] * sc)
Wk = C(t[p+"attn.key.weight"].T * sc)
Wv = C(t[p+"attn.value.weight"].T);  bv = C(t[p+"attn.value.bias"])
Wo = C(t[p+"attn.out.weight"].T);    bo = C(t[p+"attn.out.bias"])
g1 = C(t[p+"attn_ln.weight"]); c1 = C(t[p+"attn_ln.bias"])
g2 = C(t[p+"mlp_ln.weight"]);  c2 = C(t[p+"mlp_ln.bias"])
W1 = C(t[p+"mlp.0.weight"].T); b1 = C(t[p+"mlp.0.bias"])
W2 = C(t[p+"mlp.2.weight"].T); b2 = C(t[p+"mlp.2.bias"])

def ln(x, g, b):
    m = tf.reduce_mean(x, -1, keepdims=True)
    d = x - m
    v = tf.reduce_mean(d*d, -1, keepdims=True)
    return d*tf.math.rsqrt(v + 1e-5)*g + b
def recip(b): r = tf.math.rsqrt(b); return r*r
def rexp(z): s = tf.sigmoid(z*(-1.0)); return (1.0 - s)*recip(s)
ONES = tf.constant(np.ones((BQ, 1), np.float32))

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
def proj(x):
    h = ln(x, g1, c1)
    def hd(z): return tf.transpose(tf.reshape(z, [1, S, H, HD]), [0, 2, 1, 3])
    q = hd(tf.matmul(h, Wq) + bq)
    k = hd(tf.matmul(h, Wk))
    v = hd(tf.matmul(h, Wv) + bv)
    return tf.concat([q, k, v], 2)                 # [1,H,3S,HD]

def mk_core(OFF):
    @tf.function(input_signature=[tf.TensorSpec([1, H, 3*S, HD], tf.float32)])
    def core(y):
        qb = y[:, :, OFF:OFF+BQ, :]
        k  = y[:, :, S:2*S, :]
        v  = y[:, :, 2*S:3*S, :]
        m = None
        for j in range(nb):
            kj = k[:, :, j*BQ:(j+1)*BQ, :]
            mj = tf.reduce_max(tf.matmul(qb, kj, transpose_b=True), -1, keepdims=True)
            m = mj if m is None else tf.maximum(m, mj)
        num = None; den = None
        for j in range(nb):
            kj = k[:, :, j*BQ:(j+1)*BQ, :]; vj = v[:, :, j*BQ:(j+1)*BQ, :]
            s = tf.matmul(qb, kj, transpose_b=True)
            a = tf.nn.softmax(s, -1)
            c = tf.matmul(a, vj)
            beta = tf.matmul(rexp(s - m), ONES)
            nj = beta*c
            num = nj if num is None else num + nj
            den = beta if den is None else den + beta
        return tf.concat([num, den], -1)           # [1,H,BQ,HD+1]
    return core

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32),
                              tf.TensorSpec([1, S, D], tf.float32)])
def tail(ctx, x):
    att = tf.matmul(ctx, Wo) + bo
    x1 = x + att
    h2 = ln(x1, g2, c2)
    f = tf.matmul(tf.nn.gelu(tf.matmul(h2, W1) + b1, approximate=False), W2) + b2
    return x1 + f

@tf.function(input_signature=[tf.TensorSpec([1, 2*S, D], tf.float32)])
def tail1(z):
    # 1-Input-Variante (der 2-Input-tail haengt den On-Device-Compiler): z = [ctx | x] entlang S.
    ctx = z[:, 0:S, :]; x = z[:, S:2*S, :]
    att = tf.matmul(ctx, Wo) + bo
    x1 = x + att
    h2 = ln(x1, g2, c2)
    f = tf.matmul(tf.nn.gelu(tf.matmul(h2, W1) + b1, approximate=False), W2) + b2
    return x1 + f

ref = np.load(refnpz)
x0 = ref["x0"].reshape(1, S, D).astype(np.float32)
def headsnp(a): return a.reshape(1, S, H, HD).transpose(0, 2, 1, 3).astype(np.float32)
qh = headsnp(ref["st_q"])*sc; kh = headsnp(ref["st_k"])*sc; vh = headsnp(ref["st_v"])
Y = np.concatenate([qh, kh, vh], 2)                # proj-Referenzausgabe [1,H,3S,HD]
ctx_np = ref["st_ctx"].reshape(1, S, D).astype(np.float32)
rng = np.random.default_rng(9)

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

def rep_x():
    yield [x0]
    for sf in (0.85, 1.0, 1.15):
        for _ in range(3):
            yield [(x0*sf + rng.standard_normal(x0.shape).astype(np.float32)*0.05*x0.std()).astype(np.float32)]
def rep_y():
    yield [Y]
    for sf in (0.9, 1.1): yield [Y*sf]
def rep_tail():
    yield [ctx_np, x0]
    for sf in (0.9, 1.1): yield [ctx_np*sf, x0*sf]

bufs = {}
bufs["proj"] = convert(proj, rep_x, "l%d_proj" % LAYER, "16x8")
for i in range(nb):
    bufs["core%d" % i] = convert(mk_core(i*BQ), rep_y, "l%d_core%d" % (LAYER, i), "16x8")
bufs["tail"] = convert(tail, rep_tail, "l%d_tail" % LAYER, "int8")

# ---- PC-KETTEN-CHECK: Interpreter + CPU-Glue, exakt wie der Geraete-Treiber ----
def run1(buf, feeds):
    it = tf.lite.Interpreter(model_content=buf); it.allocate_tensors()
    dis = it.get_input_details(); do = it.get_output_details()[0]
    qps = []
    for d, f in zip(dis, feeds):
        isc, izp = d["quantization"]
        lim = 32767 if d["dtype"] == np.int16 else 127
        xq = np.clip(np.round(f/isc + izp), -lim-1, lim).astype(d["dtype"])
        it.set_tensor(d["index"], xq); qps.append((float(isc), int(izp)))
    it.invoke()
    osc, ozp = do["quantization"]
    y = (it.get_tensor(do["index"]).astype(np.float32) - ozp)*osc
    return y, qps, (float(osc), int(ozp))

py, pin_qp, pout_qp = run1(bufs["proj"], [x0])
cos_p = float((py*Y).sum())/(np.linalg.norm(py)*np.linalg.norm(Y)+1e-12)
print("proj: cos=%.5f" % cos_p, flush=True)

ctx_dev = np.zeros((1, H, S, HD), np.float32)
for i in range(nb):
    cy, _, _ = run1(bufs["core%d" % i], [py])
    num = cy[..., :HD]; den = np.maximum(cy[..., HD:], 1e-6)
    ctx_dev[:, :, i*BQ:(i+1)*BQ, :] = num/den
ctxT = ctx_dev.transpose(0, 2, 1, 3).reshape(1, S, D)
cos_c = float((ctxT*ctx_np).sum())/(np.linalg.norm(ctxT)*np.linalg.norm(ctx_np)+1e-12)
print("core-Kette: cos=%.5f" % cos_c, flush=True)

out_dev, _, _ = run1(bufs["tail"], [ctxT, x0])
out_ref = ref["out"].reshape(1, S, D)
cos_o = float((out_dev*out_ref).sum())/(np.linalg.norm(out_dev)*np.linalg.norm(out_ref)+1e-12)
rel_o = np.linalg.norm(out_dev-out_ref)/(np.linalg.norm(out_ref)+1e-12)
print("LAYER-KETTE PC: cos=%.5f rel_l2=%.2f%%" % (cos_o, 100*rel_o), flush=True)

# qparams + Bins fuer die Geraeteseite
meta = {"layer": LAYER, "S": S, "D": D, "H": H, "HD": HD, "BQ": BQ,
        "proj_in": pin_qp[0], "proj_out": pout_qp}
json.dump(meta, open(os.path.join(outdir, "l%d_meta.json" % LAYER), "w"))
x0q = np.clip(np.round(x0/pin_qp[0][0] + pin_qp[0][1]), -32768, 32767).astype(np.int16)
x0q.tofile(os.path.join(outdir, "l%d_x0.in.bin" % LAYER))
Y.astype(np.float32).tofile(os.path.join(outdir, "l%d_proj_ref.f32" % LAYER))
out_ref.astype(np.float32).tofile(os.path.join(outdir, "l%d_out_ref.f32" % LAYER))
print("DONE", flush=True)
