import numpy as np, tensorflow as tf, sys, json
# Whisper-Stufe-3 Option A: GEWICHTSLOSES Attention-Kern-Package (16x8).
# Ein Query-Block [1,H,BQ,HD] gegen volle K/V [1,H,S,HD], Flash ueber BLK-Key-Bloecke,
# rowsum-Fix (Matmul statt reduce_sum). Multi-Input-TFLite (qb,k,v), Output ctx [1,H,BQ,HD].
# Kalibrierung + PC-Vergleich aus echten Layer-0-Stufen (ref_base_l0.npz, q/k mit HD^-0.25 skaliert).
#   python whisper_core.py <ref.npz> <outprefix> [16x8|int8] [BQ] [BLK] [H] [HD]

refnpz, outpre = sys.argv[1], sys.argv[2]
quant = sys.argv[3] if len(sys.argv) > 3 else "16x8"
BQ  = int(sys.argv[4]) if len(sys.argv) > 4 else 375
BLK = int(sys.argv[5]) if len(sys.argv) > 5 else 375
H   = int(sys.argv[6]) if len(sys.argv) > 6 else 8
HD  = int(sys.argv[7]) if len(sys.argv) > 7 else 64
HTOT = int(sys.argv[10]) if len(sys.argv) > 10 else H   # Gesamtkoepfe im Modell (Head-Subset-Packages)
HOFF = int(sys.argv[11]) if len(sys.argv) > 11 else 0
ref = np.load(refnpz)
S = ref["st_q"].shape[0]; D = HTOT*HD; nb = S//BLK; sc = HD**-0.25

def heads(a):  # [S,D] -> [1,H,S,HD] (Subset HOFF..HOFF+H von HTOT Koepfen)
    return a.reshape(1, S, HTOT, HD).transpose(0, 2, 1, 3)[:, HOFF:HOFF+H].astype(np.float32)
qh = heads(ref["st_q"])*sc; kh = heads(ref["st_k"])*sc; vh = heads(ref["st_v"])
qb0 = qh[:, :, 0:BQ, :]

ONES = tf.constant(np.ones((BLK, 1), np.float32))
def recip(b): r = tf.math.rsqrt(b); return r*r
def rexp(z): s = tf.sigmoid(z*(-1.0)); return (1.0 - s)*recip(s)

@tf.function(input_signature=[tf.TensorSpec([1, H, BQ, HD], tf.float32),
                              tf.TensorSpec([1, H, S, HD], tf.float32),
                              tf.TensorSpec([1, H, S, HD], tf.float32)])
def core(qb, k, v):
    m = None; l = None; acc = None
    for j in range(nb):
        kj = k[:, :, j*BLK:(j+1)*BLK, :]; vj = v[:, :, j*BLK:(j+1)*BLK, :]
        s = tf.matmul(qb, kj, transpose_b=True)
        mj = tf.reduce_max(s, -1, keepdims=True)
        if m is None:
            m = mj; p = rexp(s - m); l = tf.matmul(p, ONES); acc = tf.matmul(p, vj)
        else:
            mn = tf.maximum(m, mj); corr = rexp(m - mn); p = rexp(s - mn)
            l = l*corr + tf.matmul(p, ONES); acc = acc*corr + tf.matmul(p, vj); m = mn
    return acc*recip(l)

QBLOCK = int(sys.argv[8]) if len(sys.argv) > 8 else -1   # >=0: concat-1-Input-Variante mit gebackenem Query-Block
X = np.concatenate([qh[:, :, QBLOCK*BQ:(QBLOCK+1)*BQ, :] if QBLOCK >= 0 else qh[:, :, 0:BQ, :], kh, vh], axis=2)

@tf.function(input_signature=[tf.TensorSpec([1, H, BQ + 2*S, HD], tf.float32)])
def core1(x):
    qb = x[:, :, 0:BQ, :]
    k  = x[:, :, BQ:BQ+S, :]
    v  = x[:, :, BQ+S:BQ+2*S, :]
    return core(qb, k, v)

@tf.function(input_signature=[tf.TensorSpec([1, H, BQ + 2*S, HD], tf.float32)])
def core3(x):
    # Variante 3: ZWEI-PASS-Flash — Pass 1 globales Zeilenmax (keine LUTs), Pass 2 p ohne corr-Kette.
    # Kostet die Score-Matmuls doppelt, eliminiert aber die fehlerverstaerkenden Korrektur-Multiplikationen.
    qb = x[:, :, 0:BQ, :]
    k  = x[:, :, BQ:BQ+S, :]
    v  = x[:, :, BQ+S:BQ+2*S, :]
    m = None
    for j in range(nb):
        kj = k[:, :, j*BLK:(j+1)*BLK, :]
        mj = tf.reduce_max(tf.matmul(qb, kj, transpose_b=True), -1, keepdims=True)
        m = mj if m is None else tf.maximum(m, mj)
    l = None; acc = None
    for j in range(nb):
        kj = k[:, :, j*BLK:(j+1)*BLK, :]; vj = v[:, :, j*BLK:(j+1)*BLK, :]
        p = rexp(tf.matmul(qb, kj, transpose_b=True) - m)
        lj = tf.matmul(p, ONES); aj = tf.matmul(p, vj)
        l = lj if l is None else l + lj
        acc = aj if acc is None else acc + aj
    return acc*recip(l)

@tf.function(input_signature=[tf.TensorSpec([1, H, BQ + 2*S, HD], tf.float32)])
def core4(x):
    # Variante 4: recip(l) RAUS aus der TPU (rsqrt-Spline-Verdacht) — Ausgabe [acc|l] unnormalisiert,
    # die Division acc/l macht die CPU-Glue. Zwei-Pass wie v3.
    qb = x[:, :, 0:BQ, :]
    k  = x[:, :, BQ:BQ+S, :]
    v  = x[:, :, BQ+S:BQ+2*S, :]
    m = None
    for j in range(nb):
        kj = k[:, :, j*BLK:(j+1)*BLK, :]
        mj = tf.reduce_max(tf.matmul(qb, kj, transpose_b=True), -1, keepdims=True)
        m = mj if m is None else tf.maximum(m, mj)
    l = None; acc = None
    for j in range(nb):
        kj = k[:, :, j*BLK:(j+1)*BLK, :]; vj = v[:, :, j*BLK:(j+1)*BLK, :]
        p = rexp(tf.matmul(qb, kj, transpose_b=True) - m)
        lj = tf.matmul(p, ONES); aj = tf.matmul(p, vj)
        l = lj if l is None else l + lj
        acc = aj if acc is None else acc + aj
    return tf.concat([acc, l], -1)          # [1,H,BQ,HD+1]

@tf.function(input_signature=[tf.TensorSpec([1, H, BQ + 2*S, HD], tf.float32)])
def core5(x):
    # Variante 5: block-lokale ECHTE SOFTMAX-Op (LUT-Qualitaet, 375 breit = kompilierbar) +
    # exakte Block-Kombination ueber Zeilen-Gewichte beta_j = rowsum(exp(s_j - m)) (sigmoid-Kette
    # nur noch fuer 4 Skalare/Zeile). Ausgabe [num|den] unnormalisiert -> CPU dividiert.
    qb = x[:, :, 0:BQ, :]
    k  = x[:, :, BQ:BQ+S, :]
    v  = x[:, :, BQ+S:BQ+2*S, :]
    m = None
    for j in range(nb):
        kj = k[:, :, j*BLK:(j+1)*BLK, :]
        mj = tf.reduce_max(tf.matmul(qb, kj, transpose_b=True), -1, keepdims=True)
        m = mj if m is None else tf.maximum(m, mj)
    num = None; den = None
    for j in range(nb):
        kj = k[:, :, j*BLK:(j+1)*BLK, :]; vj = v[:, :, j*BLK:(j+1)*BLK, :]
        s = tf.matmul(qb, kj, transpose_b=True)
        a = tf.nn.softmax(s, -1)                 # block-lokal, exakt normiert
        c = tf.matmul(a, vj)                     # [1,H,BQ,HD]
        beta = tf.matmul(rexp(s - m), ONES)      # [1,H,BQ,1]
        nj = beta*c
        num = nj if num is None else num + nj
        den = beta if den is None else den + beta
    return tf.concat([num, den], -1)             # [1,H,BQ,HD+1]

@tf.function(input_signature=[tf.TensorSpec([1, H, BQ + 2*S, HD], tf.float32)])
def core2(x):
    # Variante 2: ECHTE Softmax-Op ueber die volle 1500er-Zeile (keine Online-Kette).
    qb = x[:, :, 0:BQ, :]
    k  = x[:, :, BQ:BQ+S, :]
    v  = x[:, :, BQ+S:BQ+2*S, :]
    a = tf.nn.softmax(tf.matmul(qb, k, transpose_b=True), -1)   # [1,H,BQ,S]
    return tf.matmul(a, v)

rng = np.random.default_rng(5)
def repdata():
    nbq = S // BQ   # Anzahl BQ-grosser Query-Bloecke (nb = S//BLK zaehlt BLK-Bloecke)
    if QBLOCK >= 0:
        yield [X]
        for i in range(nbq):
            yield [np.concatenate([qh[:, :, i*BQ:(i+1)*BQ, :], kh, vh], axis=2)]
        for sf in (0.9, 1.1):
            yield [X*sf]
    else:
        yield [qb0, kh, vh]
        for i in range(1, nbq):
            yield [qh[:, :, i*BQ:(i+1)*BQ, :], kh, vh]
        for sf in (0.9, 1.1):
            yield [qb0*sf, kh*sf, vh*sf]

VAR = sys.argv[9] if len(sys.argv) > 9 else "v1"
FN = ({"v1": core1, "v2": core2, "v3": core3, "v4": core4, "v5": core5}[VAR]) if QBLOCK >= 0 else core
c = tf.lite.TFLiteConverter.from_concrete_functions([FN.get_concrete_function()])
c.optimizations = [tf.lite.Optimize.DEFAULT]; c.representative_dataset = repdata
if quant == "16x8":
    c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    c.inference_input_type = tf.int16; c.inference_output_type = tf.int16
else:
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8; c.inference_output_type = tf.int8
buf = c.convert()
open(outpre + ".tflite", "wb").write(buf)
print("WROTE", outpre + ".tflite", len(buf), "bytes")

# numpy-Referenz fuer den Query-Block (exakte Softmax)
if QBLOCK >= 0: qb0 = qh[:, :, QBLOCK*BQ:(QBLOCK+1)*BQ, :]
s = qb0 @ kh.transpose(0, 1, 3, 2)
s = s - s.max(-1, keepdims=True)
e = np.exp(s); l_np = e.sum(-1, keepdims=True); a = e/l_np
ctx_ref = (a @ vh).astype(np.float32)
if VAR in ("v4", "v5"):
    ctx_raw_ref = np.concatenate([(e @ vh), l_np], -1).astype(np.float32)  # RAW [acc|l] fuer Geraete-REF

# PC-Interpreter-Check
it = tf.lite.Interpreter(model_content=buf); it.allocate_tensors()
dis = it.get_input_details(); do = it.get_output_details()[0]
dt = np.int16 if quant == "16x8" else np.int8
lim = 32767 if quant == "16x8" else 127
qp = {}
if QBLOCK >= 0:
    d = dis[0]
    isc, izp = d["quantization"]
    xq = np.clip(np.round(X/isc + izp), -lim-1, lim).astype(dt)
    it.set_tensor(d["index"], xq)
    qp["x"] = {"isc": float(isc), "izp": int(izp), "shape": [int(t) for t in d["shape"]]}
    xq.tofile(outpre + ".in.bin")
else:
    feed = {"qb": qb0, "k": kh, "v": vh}
    def which(d):
        n = str(d["name"]).lower()
        for key in ("qb", "k", "v"):
            if n.endswith(key + ":0") or ("_%s:" % key) in n or n == key: return key
        return "qb" if tuple(d["shape"])[2] == BQ else None
    assert sorted(filter(None, (which(d) for d in dis))) == ["k", "qb", "v"], [d["name"] for d in dis]
    for d in dis:
        shp = tuple(d["shape"]); key = which(d)
        isc, izp = d["quantization"]
        xq = np.clip(np.round(feed[key]/isc + izp), -lim-1, lim).astype(dt)
        it.set_tensor(d["index"], xq)
        qp[key] = {"isc": float(isc), "izp": int(izp), "name": str(d["name"]), "shape": [int(t) for t in shp]}
        xq.tofile(outpre + ".in_%s.bin" % key)
it.invoke()
y = it.get_tensor(do["index"]).astype(np.float32)
osc, ozp = do["quantization"]
yf = (y - ozp)*osc
qp["out"] = {"osc": float(osc), "ozp": int(ozp)}; qp["quant"] = quant
json.dump(qp, open(outpre + ".qparams.json", "w"))
if VAR in ("v4", "v5"):
    ctx_raw_ref.tofile(outpre + ".ref_out.f32")
    acc_q = yf[..., :HD]; l_q = np.maximum(yf[..., HD:], 1e-6)
    yf = (acc_q/l_q).astype(np.float32)      # CPU-Glue-Division nachgestellt
else:
    ctx_ref.astype(np.float32).tofile(outpre + ".ref_out.f32")
cos = float((yf*ctx_ref).sum())/(np.linalg.norm(yf)*np.linalg.norm(ctx_ref)+1e-12)
rel = np.linalg.norm(yf-ctx_ref)/(np.linalg.norm(ctx_ref)+1e-12)
print("PC core %s %s: cos=%.5f rel_l2=%.2f%%" % (quant, VAR, cos, 100*rel))
