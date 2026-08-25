import numpy as np, tensorflow as tf, sys, json
import whisper_ref as W
# Whisper-Stufe-3 M2: base-Encoder-Block i als TFLite (int8 oder 16x8) mit ECHTEN Gewichten.
# Attention = flashsig (exakte Flash-Attention ohne exp/div, BLK-Bloecke); LN manuell; GELU exakt.
# Kalibrierung mit echtem x0 aus ref_base_l0.npz (+ Jitter). q/k-Scale HD^-0.25 in Gewichte gefaltet.
#   python whisper_l0.py <ggml-bin> <ref.npz> <outprefix> [int8|16x8] [layer] [BLK] [part]
# part: full (Default) | attn (out=att, Input x) | ffn (Input x1=mid, out=ffn) | ln1 | proj (q-Projektion)

model, refnpz, outpre = sys.argv[1], sys.argv[2], sys.argv[3]
quant = sys.argv[4] if len(sys.argv) > 4 else "int8"
LAYER = int(sys.argv[5]) if len(sys.argv) > 5 else 0
BLK = int(sys.argv[6]) if len(sys.argv) > 6 else 375
PART = sys.argv[7] if len(sys.argv) > 7 else "full"

hp, _f, t = W.parse_ggml(model)
D = hp["n_audio_state"]; H = hp["n_audio_head"]; HD = D//H; S = 1500
nb = S//BLK; sc = HD**-0.25
p = "encoder.blocks.%d." % LAYER
C = lambda a: tf.constant(a.astype(np.float32))
Wq = C(t[p+"attn.query.weight"].T * sc); bq = C(t[p+"attn.query.bias"] * sc)
Wk = C(t[p+"attn.key.weight"].T * sc)
Wv = C(t[p+"attn.value.weight"].T); bv = C(t[p+"attn.value.bias"])
Wo = C(t[p+"attn.out.weight"].T);   bo = C(t[p+"attn.out.bias"])
g1 = C(t[p+"attn_ln.weight"]); c1 = C(t[p+"attn_ln.bias"])
g2 = C(t[p+"mlp_ln.weight"]);  c2 = C(t[p+"mlp_ln.bias"])
W1 = C(t[p+"mlp.0.weight"].T); b1 = C(t[p+"mlp.0.bias"])
W2 = C(t[p+"mlp.2.weight"].T); b2 = C(t[p+"mlp.2.bias"])

def ln(x, g, b):
    m = tf.reduce_mean(x, -1, keepdims=True)
    d = x - m
    v = tf.reduce_mean(d*d, -1, keepdims=True)     # SQUARE ist in 16x8 nicht supported
    return d*tf.math.rsqrt(v + 1e-5)*g + b
def recip(b): r = tf.math.rsqrt(b); return r*r
def rexp(z): s = tf.sigmoid(z*(-1.0)); return (1.0 - s)*recip(s)   # NEG ist in 16x8 nicht supported

ONES = tf.constant(np.ones((BLK, 1), np.float32))
def rowsum(p):
    # 16x8-FALLE: reduce_sum erzwingt Eingangs-Skala am Ausgang -> l saturiert bei 1.0.
    # Matmul mit Einsen-Vektor bekommt eine EIGENE kalibrierte Ausgangs-Skala.
    return tf.matmul(p, ONES)

def attn_of(h):
    def proj(hb):
        q = tf.matmul(hb, Wq) + bq
        k = tf.matmul(hb, Wk)
        v = tf.matmul(hb, Wv) + bv
        f = lambda z: tf.transpose(tf.reshape(z, [1, BLK, H, HD]), [0, 2, 1, 3])
        return f(q), f(k), f(v)
    P = [proj(h[:, b*BLK:(b+1)*BLK, :]) for b in range(nb)]
    qB = [u[0] for u in P]; kB = [u[1] for u in P]; vB = [u[2] for u in P]
    outs = []
    for i in range(nb):
        qi = qB[i]; m = None; l = None; acc = None
        for j in range(nb):
            s = tf.matmul(qi, kB[j], transpose_b=True)
            mj = tf.reduce_max(s, -1, keepdims=True)
            if m is None:
                m = mj; pr = rexp(s - m); l = rowsum(pr); acc = tf.matmul(pr, vB[j])
            else:
                mn = tf.maximum(m, mj); corr = rexp(m - mn); pr = rexp(s - mn)
                l = l*corr + rowsum(pr); acc = acc*corr + tf.matmul(pr, vB[j]); m = mn
        ctx = acc*recip(l)
        outs.append(tf.reshape(tf.transpose(ctx, [0, 2, 1, 3]), [1, BLK, D]))
    return tf.matmul(tf.concat(outs, 1), Wo) + bo

def attn_qsplit(h):
    # Exakte Attention mit ECHTER Softmax-Op: Queries in BLK-Bloecke, volle K/V, Softmax-Breite S.
    q = tf.matmul(h, Wq) + bq
    k = tf.matmul(h, Wk)
    v = tf.matmul(h, Wv) + bv
    kh = tf.transpose(tf.reshape(k, [1, S, H, HD]), [0, 2, 1, 3])
    vh = tf.transpose(tf.reshape(v, [1, S, H, HD]), [0, 2, 1, 3])
    outs = []
    for i in range(nb):
        qb = q[:, i*BLK:(i+1)*BLK, :]
        qbh = tf.transpose(tf.reshape(qb, [1, BLK, H, HD]), [0, 2, 1, 3])
        a = tf.nn.softmax(tf.matmul(qbh, kh, transpose_b=True), -1)     # [1,H,BLK,S]
        outs.append(tf.reshape(tf.transpose(tf.matmul(a, vh), [0, 2, 1, 3]), [1, BLK, D]))
    return tf.matmul(tf.concat(outs, 1), Wo) + bo

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
def part_attn(x):
    return attn_of(ln(x, g1, c1))

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
def part_attnq(x):
    return attn_qsplit(ln(x, g1, c1))

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
def part_attnq1(x):
    # Bisektion: NUR Query-Block 0 (kein Loop/Concat/Wo) -> [1,BLK,D]
    h = ln(x, g1, c1)
    q = tf.matmul(h, Wq) + bq
    k = tf.matmul(h, Wk)
    v = tf.matmul(h, Wv) + bv
    kh = tf.transpose(tf.reshape(k, [1, S, H, HD]), [0, 2, 1, 3])
    vh = tf.transpose(tf.reshape(v, [1, S, H, HD]), [0, 2, 1, 3])
    qb = q[:, 0:BLK, :]
    qbh = tf.transpose(tf.reshape(qb, [1, BLK, H, HD]), [0, 2, 1, 3])
    a = tf.nn.softmax(tf.matmul(qbh, kh, transpose_b=True), -1)
    return tf.reshape(tf.transpose(tf.matmul(a, vh), [0, 2, 1, 3]), [1, BLK, D])

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
def part_kvh(x):
    # Bisektion: nur Projektionen + Head-Transposes in voller Laenge
    h = ln(x, g1, c1)
    k = tf.matmul(h, Wk)
    v = tf.matmul(h, Wv) + bv
    kh = tf.transpose(tf.reshape(k, [1, S, H, HD]), [0, 2, 1, 3])
    vh = tf.transpose(tf.reshape(v, [1, S, H, HD]), [0, 2, 1, 3])
    return kh + vh

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
def block_q(x):
    x1 = x + attn_qsplit(ln(x, g1, c1))
    h2 = ln(x1, g2, c2)
    f = tf.matmul(tf.nn.gelu(tf.matmul(h2, W1) + b1, approximate=False), W2) + b2
    return x1 + f

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
def part_ffn(x1):
    h2 = ln(x1, g2, c2)
    return tf.matmul(tf.nn.gelu(tf.matmul(h2, W1) + b1, approximate=False), W2) + b2

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
def part_ln1(x):
    return ln(x, g1, c1)

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
def part_proj(x):
    return tf.matmul(ln(x, g1, c1), Wq) + bq

@tf.function(input_signature=[tf.TensorSpec([1, S, D], tf.float32)])
def block(x):
    x1 = x + attn_of(ln(x, g1, c1))
    h2 = ln(x1, g2, c2)
    f = tf.matmul(tf.nn.gelu(tf.matmul(h2, W1) + b1, approximate=False), W2) + b2
    return x1 + f

FN = {"full": block, "fullq": block_q, "attn": part_attn, "attnq": part_attnq,
      "attnq1": part_attnq1, "kvh": part_kvh,
      "ffn": part_ffn, "ln1": part_ln1, "proj": part_proj}[PART]
ref = np.load(refnpz)
calkey = "st_mid" if PART == "ffn" else "x0"
x0 = ref[calkey].reshape(1, S, D).astype(np.float32)
rng = np.random.default_rng(3)
def repdata():
    def g():
        yield [x0]
        for s in (0.85, 1.0, 1.15):
            for _ in range(4):
                yield [(x0*s + rng.standard_normal(x0.shape).astype(np.float32)*0.05*x0.std()).astype(np.float32)]
    return g

c = tf.lite.TFLiteConverter.from_concrete_functions([FN.get_concrete_function()])
c.optimizations = [tf.lite.Optimize.DEFAULT]; c.representative_dataset = repdata()
if quant == "16x8":
    c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    c.inference_input_type = tf.int16; c.inference_output_type = tf.int16
else:
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8; c.inference_output_type = tf.int8
buf = c.convert()
open(outpre + ".tflite", "wb").write(buf)

it = tf.lite.Interpreter(model_content=buf)
di, do = it.get_input_details()[0], it.get_output_details()[0]
isc, izp = float(di["quantization"][0]), int(di["quantization"][1])
osc, ozp = float(do["quantization"][0]), int(do["quantization"][1])
json.dump({"isc": isc, "izp": izp, "osc": osc, "ozp": ozp, "S": S, "D": D, "BLK": BLK,
           "layer": LAYER, "quant": quant}, open(outpre + ".qparams.json", "w"))
dt = np.int16 if quant == "16x8" else np.int8
lim = 32767 if quant == "16x8" else 127
xq = np.clip(np.round(x0/isc + izp), -lim-1, lim).astype(dt)
xq.tofile(outpre + ".in.bin")
print("WROTE", outpre + ".tflite", len(buf), "bytes; in isc/izp=%.6g/%d out osc/ozp=%.6g/%d" % (isc, izp, osc, ozp))
