import numpy as np, struct, sys, math, wave
# Whisper-Stufe-3 M2: ggml-base.bin parsen (Hparams+Mel-Filter+Gewichte), numpy-Referenz-Forward
# des Encoders (Mel -> Conv-Frontend -> Block 0 [-> alle Bloecke]), Referenz-Dumps fuer die
# Geraete-Verifikation. Format = whisper.cpp convert-pt-to-ggml (Magic ggml, dims reversed, f16).
#   python whisper_ref.py <ggml-base.bin> <test.wav> <outdir> [nlayers]

def parse_ggml(path):
    f = open(path, "rb")
    magic = struct.unpack("<i", f.read(4))[0]
    assert magic == 0x67676d6c, hex(magic)
    keys = ("n_vocab n_audio_ctx n_audio_state n_audio_head n_audio_layer "
            "n_text_ctx n_text_state n_text_head n_text_layer n_mels ftype").split()
    hp = dict(zip(keys, struct.unpack("<11i", f.read(44))))
    n_mel, n_fft = struct.unpack("<2i", f.read(8))
    filters = np.frombuffer(f.read(n_mel*n_fft*4), dtype=np.float32).reshape(n_mel, n_fft).copy()
    n_vocab = struct.unpack("<i", f.read(4))[0]
    for _ in range(n_vocab):
        ln = struct.unpack("<i", f.read(4))[0]
        f.read(ln)
    tensors = {}
    while True:
        head = f.read(12)
        if len(head) < 12: break
        n_dims, name_len, ttype = struct.unpack("<3i", head)
        dims = struct.unpack("<%di" % n_dims, f.read(4*n_dims))
        name = f.read(name_len).decode()
        shape = tuple(reversed(dims))          # ne[0] = innerste Achse
        n = int(np.prod(shape))
        if ttype == 0:
            data = np.frombuffer(f.read(4*n), dtype=np.float32).copy()
        elif ttype == 1:
            data = np.frombuffer(f.read(2*n), dtype=np.float16).astype(np.float32)
        else:
            # Quantisierte ggml-Typen (>=2) haben ein anderes Byte-Layout; der else-Zweig
            # las sie frueher still als f32 (falsche Groesse/Interpretation).
            raise ValueError("whisper_ref: Tensor '%s' hat nicht unterstuetzten ggml-Typ %d (nur f32/f16)" % (name, ttype))
        a = data.reshape(shape)
        if 1 in shape: a = np.squeeze(a)   # Biases liegen mit 1er-Achse im File
        tensors[name] = a
    f.close()
    return hp, filters, tensors

def log_mel(wav_path, filters, n_samples=480000, n_fft=400, hop=160):
    w = wave.open(wav_path, "rb")
    assert w.getframerate() == 16000 and w.getnchannels() == 1
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    x = np.zeros(n_samples, np.float32); x[:min(len(pcm), n_samples)] = pcm[:n_samples]
    x = np.pad(x, (n_fft//2, n_fft//2), mode="reflect")
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)       # periodisches Hann wie torch
    n_frames = 1 + (len(x) - n_fft) // hop                    # 3001
    idx = np.arange(n_fft)[None, :] + hop*np.arange(n_frames)[:, None]
    frames = x[idx] * win
    spec = np.fft.rfft(frames, axis=1)                        # [3001, 201]
    mag = (np.abs(spec)**2).astype(np.float32)[:-1].T         # letzte Frame weg -> [201, 3000]
    mel = filters @ mag
    logspec = np.log10(np.maximum(mel, 1e-10))
    logspec = np.maximum(logspec, logspec.max() - 8.0)
    return ((logspec + 4.0) / 4.0).astype(np.float32)         # [80, 3000]

def _erf(x):
    # Abramowitz/Stegun 7.1.26, |err| < 1.5e-7 — vektorisiert (np hat kein erf, scipy fehlt im venv)
    s = np.sign(x); x = np.abs(x.astype(np.float64))
    t = 1.0/(1.0 + 0.3275911*x)
    y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*np.exp(-x*x)
    return s*y
def gelu(x): return (0.5*x*(1.0 + _erf(x/np.sqrt(2.0)))).astype(np.float32)

def conv1d(x, w, b, stride, pad):
    # x [Cin, T], w [Cout, Cin, K] -> [Cout, T_out]
    Cout, Cin, K = w.shape
    xp = np.pad(x, ((0,0),(pad,pad)))
    T_out = (xp.shape[1] - K)//stride + 1
    cols = np.stack([xp[:, i*stride:i*stride+K].ravel() for i in range(T_out)], axis=1)  # [Cin*K, T_out]
    return (w.reshape(Cout, Cin*K) @ cols + b[:, None]).astype(np.float32)

def layernorm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return ((x - m)/np.sqrt(v + eps)*g + b).astype(np.float32)

def block_forward(x, t, i, H):
    # x [S, D]; Pre-LN-Block; Rueckgabe (out, stages-dict)
    S, D = x.shape; HD = D // H; sc = HD**-0.25
    p = "encoder.blocks.%d." % i
    h = layernorm(x, t[p+"attn_ln.weight"], t[p+"attn_ln.bias"])
    q = h @ t[p+"attn.query.weight"].T + t[p+"attn.query.bias"]
    k = h @ t[p+"attn.key.weight"].T
    v = h @ t[p+"attn.value.weight"].T + t[p+"attn.value.bias"]
    qh = (q.reshape(S, H, HD).transpose(1,0,2)) * sc
    kh = (k.reshape(S, H, HD).transpose(1,0,2)) * sc
    vh = v.reshape(S, H, HD).transpose(1,0,2)
    s = qh @ kh.transpose(0,2,1)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s); a = e / e.sum(-1, keepdims=True)
    ctx = (a @ vh).transpose(1,0,2).reshape(S, D)
    att = ctx @ t[p+"attn.out.weight"].T + t[p+"attn.out.bias"]
    x1 = x + att
    h2 = layernorm(x1, t[p+"mlp_ln.weight"], t[p+"mlp_ln.bias"])
    f1 = gelu(h2 @ t[p+"mlp.0.weight"].T + t[p+"mlp.0.bias"])
    f2 = f1 @ t[p+"mlp.2.weight"].T + t[p+"mlp.2.bias"]
    out = (x1 + f2).astype(np.float32)
    return out, {"ln1": h, "q": q, "k": k, "v": v, "ctx": ctx, "att": att, "mid": x1, "ffn": f2}

if __name__ == "__main__":
    model, wavf, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    nlayers = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    hp, filters, t = parse_ggml(model)
    print("hparams", {k: hp[k] for k in ("n_audio_state","n_audio_head","n_audio_layer","n_mels")})
    mel = log_mel(wavf, filters)
    c1 = gelu(conv1d(mel, t["encoder.conv1.weight"], t["encoder.conv1.bias"], 1, 1))
    c2 = gelu(conv1d(c1,  t["encoder.conv2.weight"], t["encoder.conv2.bias"], 2, 1))
    x0 = (c2.T + t["encoder.positional_embedding"]).astype(np.float32)   # [1500, D]
    print("x0", x0.shape, "range", float(x0.min()), float(x0.max()))
    stages_all = {}
    per_layer = {}
    x = x0
    for i in range(nlayers):
        xin = x
        x, st = block_forward(x, t, i, hp["n_audio_head"])
        print("block %d out range %.3f %.3f  |mean| %.4f" % (i, x.min(), x.max(), np.abs(x).mean()))
        if i == 0: stages_all = st
        per_layer["L%d_xin" % i] = xin
        for k in ("q", "k", "v", "ctx", "att", "mid"): per_layer["L%d_%s" % (i, k)] = st[k]
        per_layer["L%d_out" % i] = x
    import os; os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "ref_base_l0.npz"), mel=mel, x0=x0, out=per_layer["L0_out"], **{"st_"+k: v for k, v in stages_all.items()})
    if nlayers > 1:
        # ln_post fuer den finalen Encoder-Ausgang
        enc = layernorm(x, t["encoder.ln_post.weight"], t["encoder.ln_post.bias"])
        np.savez(os.path.join(outdir, "ref_base_all.npz"), mel=mel, x0=x0, enc=enc, **per_layer)
        print("WROTE", os.path.join(outdir, "ref_base_all.npz"))
    print("WROTE", os.path.join(outdir, "ref_base_l0.npz"))
