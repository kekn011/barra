import numpy as np, tensorflow as tf, sys
# Echte int8-Multi-Head-Attention, seq-parametrisch. Geometrie = Whisper (D=512,H=8,HD=64).
# full_{S}   : volle SxS-Attention  -> testet die seq-Decke
# chunk_{S}_{BLK}: block-lokale Attention (BLKxBLK je Block) -> testet Flash-Style-Workaround
rng = np.random.default_rng(7)
D, H, HD = 512, 8, 64
scale = 1.0/np.sqrt(HD)
Wqkv = tf.constant(rng.standard_normal((D,3*D)).astype(np.float32)*0.02)
Wo   = tf.constant(rng.standard_normal((D,D)).astype(np.float32)*0.02)

def build_full(S):
    @tf.function(input_signature=[tf.TensorSpec([1,S,D], tf.float32)])
    def f(x):
        qkv = tf.matmul(x, Wqkv)
        q,k,v = tf.split(qkv,3,axis=-1)
        q = tf.transpose(tf.reshape(q,[1,S,H,HD]),[0,2,1,3])
        k = tf.transpose(tf.reshape(k,[1,S,H,HD]),[0,2,1,3])
        v = tf.transpose(tf.reshape(v,[1,S,H,HD]),[0,2,1,3])
        a = tf.nn.softmax(tf.matmul(q,k,transpose_b=True)*scale,-1)   # [1,H,S,S]
        ctx = tf.matmul(a,v)                                          # [1,H,S,HD]
        ctx = tf.reshape(tf.transpose(ctx,[0,2,1,3]),[1,S,D])
        return tf.matmul(ctx, Wo)
    return f, [1,S,D]

def build_chunk(S, BLK):
    # Block-lokal, aber Bloecke als BATCH-Dim -> Attention bleibt 4D [nb,H,BLK,BLK]
    assert S % BLK == 0
    nb = S//BLK
    @tf.function(input_signature=[tf.TensorSpec([1,S,D], tf.float32)])
    def f(x):
        qkv = tf.matmul(x, Wqkv)                         # [1,S,3D]
        qkv = tf.reshape(qkv,[nb,BLK,3*D])               # Bloecke -> Batch
        q,k,v = tf.split(qkv,3,axis=-1)                  # [nb,BLK,D]
        q = tf.transpose(tf.reshape(q,[nb,BLK,H,HD]),[0,2,1,3])   # [nb,H,BLK,HD]
        k = tf.transpose(tf.reshape(k,[nb,BLK,H,HD]),[0,2,1,3])
        v = tf.transpose(tf.reshape(v,[nb,BLK,H,HD]),[0,2,1,3])
        a = tf.nn.softmax(tf.matmul(q,k,transpose_b=True)*scale,-1)   # [nb,H,BLK,BLK] (4D)
        ctx = tf.matmul(a,v)                                          # [nb,H,BLK,HD]
        ctx = tf.reshape(tf.transpose(ctx,[0,2,1,3]),[nb,BLK,D])
        ctx = tf.reshape(ctx,[1,S,D])
        return tf.matmul(ctx, Wo)
    return f, [1,S,D]

def build_cslice(S, BLK):
    # Chunking via slice+concat: jeder Block ist STANDARD-4D-Attention [1,BLK,D] (= attn_full_BLK, kompiliert)
    assert S % BLK == 0
    nb = S//BLK
    @tf.function(input_signature=[tf.TensorSpec([1,S,D], tf.float32)])
    def f(x):
        outs=[]
        for i in range(nb):
            xb = x[:, i*BLK:(i+1)*BLK, :]                 # [1,BLK,D]
            qkv = tf.matmul(xb, Wqkv)
            q,k,v = tf.split(qkv,3,axis=-1)
            q = tf.transpose(tf.reshape(q,[1,BLK,H,HD]),[0,2,1,3])
            k = tf.transpose(tf.reshape(k,[1,BLK,H,HD]),[0,2,1,3])
            v = tf.transpose(tf.reshape(v,[1,BLK,H,HD]),[0,2,1,3])
            a = tf.nn.softmax(tf.matmul(q,k,transpose_b=True)*scale,-1)
            ctx = tf.matmul(a,v)
            outs.append(tf.reshape(tf.transpose(ctx,[0,2,1,3]),[1,BLK,D]))
        ctx = tf.concat(outs, axis=1)                     # [1,S,D]
        return tf.matmul(ctx, Wo)
    return f, [1,S,D]

def build_qsplit(S, BLK):
    # EXAKTE Full-Attention: nur Queries in BLK-Bloecke, volle K/V. Score-Matrix je Block = BLK x S.
    assert S % BLK == 0
    nb = S//BLK
    @tf.function(input_signature=[tf.TensorSpec([1,S,D], tf.float32)])
    def f(x):
        qkv = tf.matmul(x, Wqkv)
        q,k,v = tf.split(qkv,3,axis=-1)
        kh = tf.transpose(tf.reshape(k,[1,S,H,HD]),[0,2,1,3])   # [1,H,S,HD]
        vh = tf.transpose(tf.reshape(v,[1,S,H,HD]),[0,2,1,3])
        outs=[]
        for i in range(nb):
            qb = q[:, i*BLK:(i+1)*BLK, :]
            qbh = tf.transpose(tf.reshape(qb,[1,BLK,H,HD]),[0,2,1,3])   # [1,H,BLK,HD]
            a = tf.nn.softmax(tf.matmul(qbh,kh,transpose_b=True)*scale,-1)  # [1,H,BLK,S]
            ctx = tf.matmul(a,vh)                                            # [1,H,BLK,HD]
            outs.append(tf.reshape(tf.transpose(ctx,[0,2,1,3]),[1,BLK,D]))
        return tf.matmul(tf.concat(outs,axis=1), Wo)
    return f, [1,S,D]

def build_flash(S, BLK):
    # EXAKTE Full-Attention via Online-Softmax: Queries UND Keys in BLK-Bloecke, kein SxS und kein S-breiter softmax.
    assert S % BLK == 0
    nb = S//BLK
    @tf.function(input_signature=[tf.TensorSpec([1,S,D], tf.float32)])
    def f(x):
        qkv = tf.matmul(x, Wqkv)
        q,k,v = tf.split(qkv,3,axis=-1)
        def heads(t): return tf.transpose(tf.reshape(t,[1,S,H,HD]),[0,2,1,3])  # [1,H,S,HD]
        qh,kh,vh = heads(q),heads(k),heads(v)
        outs=[]
        for i in range(nb):                       # ueber Query-Bloecke
            qi = qh[:,:, i*BLK:(i+1)*BLK, :]       # [1,H,BLK,HD]
            m = None; l = None; acc = None
            for j in range(nb):                   # online ueber Key-Bloecke
                kj = kh[:,:, j*BLK:(j+1)*BLK, :]; vj = vh[:,:, j*BLK:(j+1)*BLK, :]
                s = tf.matmul(qi,kj,transpose_b=True)*scale          # [1,H,BLK,BLK]
                mj = tf.reduce_max(s,axis=-1,keepdims=True)          # [1,H,BLK,1]
                if m is None:
                    m = mj; p = tf.exp(s-m); l = tf.reduce_sum(p,-1,keepdims=True); acc = tf.matmul(p,vj)
                else:
                    mnew = tf.maximum(m,mj); corr = tf.exp(m-mnew)
                    p = tf.exp(s-mnew)
                    l = l*corr + tf.reduce_sum(p,-1,keepdims=True)
                    acc = acc*corr + tf.matmul(p,vj)
                    m = mnew
            ctx = acc / l                          # [1,H,BLK,HD]
            outs.append(tf.reshape(tf.transpose(ctx,[0,2,1,3]),[1,BLK,D]))
        return tf.matmul(tf.concat(outs,axis=1), Wo)
    return f, [1,S,D]

def build_flashsig(S, BLK):
    # Exaktes Flash OHNE exp/reciprocal/div: Kehrwert via rsqrt^2 (alle Argumente >0).
    #   exp(z) = (1-sigmoid(-z)) * (1/sigmoid(-z)),  1/b = rsqrt(b)^2
    assert S % BLK == 0
    nb = S//BLK
    def recip(b): r=tf.math.rsqrt(b); return r*r
    def rexp(z): s=tf.sigmoid(-z); return (1.0 - s) * recip(s)
    @tf.function(input_signature=[tf.TensorSpec([1,S,D], tf.float32)])
    def f(x):
        def proj(xb):                                   # [1,BLK,D] -> q,k,v je [1,H,BLK,HD] (375-Zeilen-FC!)
            qkv = tf.matmul(xb, Wqkv)
            q,k,v = tf.split(qkv,3,axis=-1)
            h = lambda t: tf.transpose(tf.reshape(t,[1,BLK,H,HD]),[0,2,1,3])
            return h(q),h(k),h(v)
        P = [proj(x[:, b*BLK:(b+1)*BLK, :]) for b in range(nb)]   # erst schneiden, dann projizieren
        qB=[p[0] for p in P]; kB=[p[1] for p in P]; vB=[p[2] for p in P]
        outs=[]
        for i in range(nb):
            qi=qB[i]; m=None; l=None; acc=None
            for j in range(nb):
                s = tf.matmul(qi,kB[j],transpose_b=True)*scale
                mj = tf.reduce_max(s,axis=-1,keepdims=True)
                if m is None:
                    m=mj; p=rexp(s-m); l=tf.reduce_sum(p,-1,keepdims=True); acc=tf.matmul(p,vB[j])
                else:
                    mnew=tf.maximum(m,mj); corr=rexp(m-mnew); p=rexp(s-mnew)
                    l=l*corr+tf.reduce_sum(p,-1,keepdims=True); acc=acc*corr+tf.matmul(p,vB[j]); m=mnew
            ctx = acc*recip(l)
            outs.append(tf.reshape(tf.transpose(ctx,[0,2,1,3]),[1,BLK,D]))
        return tf.matmul(tf.concat(outs,axis=1), Wo)
    return f, [1,S,D]

def build_probe(kind):
    # Mini-Proben [1,8,64,64] zum Isolieren des LowerHlops-Crashers
    sh=[1,8,64,64]
    @tf.function(input_signature=[tf.TensorSpec(sh, tf.float32)])
    def f(x):
        if kind=="rexp":   return tf.math.reciprocal(tf.sigmoid(-x)) - 1.0
        if kind=="rmax":   return x - tf.reduce_max(x,axis=-1,keepdims=True)
        if kind=="emax":   return tf.maximum(x, 0.5*x)
        if kind=="recip":  return tf.math.reciprocal(x + 2.0)
        if kind=="sig":    return tf.sigmoid(x)
        if kind=="div":    return x / (tf.abs(x) + 1.0)                  # Tensor/Tensor-Division
        if kind=="rsqrt":  return tf.math.rsqrt(tf.abs(x) + 0.5)         # -> rescaling<RSQRT>?
        if kind=="recvia": r=tf.math.rsqrt(tf.abs(x)+0.5); return x*r*r  # 1/b = rsqrt(b)^2
        if kind=="expdiv":                                             # exp via Division statt reciprocal
            s = tf.sigmoid(-x); return (1.0 - s) / s
        if kind=="msoft":  # manueller softmax-Nenner
            p = tf.math.reciprocal(tf.sigmoid(-(x - tf.reduce_max(x,-1,keepdims=True)))) - 1.0
            return tf.reduce_sum(p,axis=-1,keepdims=True)
        return x
    return f, sh

def rep(shape):
    def g():
        for _ in range(16):
            yield [rng.standard_normal(shape).astype(np.float32)]
    return g
def to_int8(fn, shape, name):
    c = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()])
    c.optimizations=[tf.lite.Optimize.DEFAULT]; c.representative_dataset=rep(shape)
    c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type=tf.int8; c.inference_output_type=tf.int8
    open(name+".tflite","wb").write(c.convert()); print("WROTE",name+".tflite", flush=True)

for arg in sys.argv[1:]:
    if arg.startswith("full_"):
        S=int(arg.split("_")[1]); fn,sh=build_full(S); to_int8(fn,sh,"attn_"+arg)
    elif arg.startswith("chunk_"):
        _,S,BLK=arg.split("_"); fn,sh=build_chunk(int(S),int(BLK)); to_int8(fn,sh,"attn_"+arg)
    elif arg.startswith("cslice_"):
        _,S,BLK=arg.split("_"); fn,sh=build_cslice(int(S),int(BLK)); to_int8(fn,sh,"attn_"+arg)
    elif arg.startswith("qsplit_"):
        _,S,BLK=arg.split("_"); fn,sh=build_qsplit(int(S),int(BLK)); to_int8(fn,sh,"attn_"+arg)
    elif arg.startswith("flash_"):
        _,S,BLK=arg.split("_"); fn,sh=build_flash(int(S),int(BLK)); to_int8(fn,sh,"attn_"+arg)
    elif arg.startswith("flashsig_"):
        _,S,BLK=arg.split("_"); fn,sh=build_flashsig(int(S),int(BLK)); to_int8(fn,sh,"attn_"+arg)
    elif arg.startswith("probe_"):
        k=arg.split("_",1)[1]; fn,sh=build_probe(k); to_int8(fn,sh,"attn_"+arg)
print("DONE", flush=True)
