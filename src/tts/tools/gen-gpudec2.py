# GPU-f16-Decoder-Generator: W_mat je Conv (gepaddet, f16) in einen Blob, program2.json mit
# conv_gemm+epilogue-Ops, Sub-Pixel-Upsample (wide) + shuffle, Residual/MRF-Glue.
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hifigan_np import load
from subpixel_mod import subpixel_weights

# gen-gpudec2.py <dec-dir> <out-dir>
#   <dec-dir>  Ausgabe von dump-dec.py (Coqui) oder dump-dec-onnx.py (Piper)
#   <out-dir>  bekommt program2.json + weights16.bin + bias16.bin
DEC = sys.argv[1] if len(sys.argv) > 1 else "../baj-out"
OUT = sys.argv[2] if len(sys.argv) > 2 else "gpukit2"

TM, TK = 128, 32
os.makedirs(OUT, exist_ok=True)
W, man, folded = load(DEC)
rates = man["upsample_rates"]; uks = man["upsample_kernel_sizes"]
rks = man["resblock_kernel_sizes"]; rds = man["resblock_dilation_sizes"]
ch0 = man["upsample_initial_channel"]
RB_TYPE = int(man.get("resblock_type", 1))   # 1 = convs1/convs2, 2 = convs.N
# leaky-Kodierung im Shader: 0 = keine, 1 = Koerpersteigung (0.1), 2 = 0.01 vor conv_post.
# HiFi-GAN ruft dort ein blankes F.leaky_relu() auf -> PyTorch-Vorgabe 0.01, nicht 0.1.
SLOPE = float(man.get("lrelu_slope", 0.1))
SLOPE_F = float(man.get("lrelu_slope_final", 0.01))
LEAKY_FINAL = 1 if abs(SLOPE_F - SLOPE) < 1e-9 else 2

def up(x, m): return (x + m - 1) // m * m

blob = []   # f16 arrays
def put_wmat(w, tm):   # w [Cout, Cin*K] -> [Mp, Kg] gepaddet; Mp auf Kachel tm
    Cout, CK = w.shape
    Mp, Kg = up(Cout, tm), up(CK, TK)
    Wm = np.zeros((Mp, Kg), np.float16)
    Wm[:Cout, :CK] = w.astype(np.float16)
    off = sum(a.size for a in blob)
    blob.append(Wm.ravel())
    return off, Mp, Kg

biasblob = []
def put_bias(b):
    off = sum(a.size for a in biasblob)
    biasblob.append(b.astype(np.float16))
    return off

pkgs = []   # conv-meta list
def conv_meta(name, w2d, b, Cin, Kk, dil, pad, leaky, Cout):
    tm = 32 if Cout <= 64 else 128   # kleine Cout: schlanke Kachel gegen Padding-Verschwendung
    woff, Mp, Kg = put_wmat(w2d, tm)
    boff = put_bias(b)
    m = dict(name=name, woff=woff, Mp=Mp, Kg=Kg, boff=boff, Cin=Cin, K=Kk, dil=dil, pad=pad,
             leaky=int(leaky), Cout=Cout, tm=tm)
    pkgs.append(m); return m

ops = []
def emit(o): ops.append(o)

# conv_pre (kein leaky)
w, b = folded("conv_pre"); Cout, Cin, K = w.shape
m = conv_meta("pre", w.reshape(Cout, Cin*K), b, Cin, K, 1, (K-1)//2, False, Cout)
emit(dict(op="conv", **{k: m[k] for k in ("woff","Mp","Kg","boff","Cin","K","dil","pad","leaky","Cout","tm")},
          src="z", dst="o", res="", tanh=0))

for i, (r, uk) in enumerate(zip(rates, uks)):
    w, b = folded(f"ups.{i}"); Cin, Cout, _ = w.shape
    Wreg, J = subpixel_weights(w, r, uk)   # [Cout*r, Cin, J]
    Ce = Cout*r
    # conv_gemm rechnet als normalen Conv out[t]=Σ_k W[k]·x[t-(J-1)+k]; Sub-Pixel braucht
    # out[q]=Σ_j Wreg[j]·x[q-j] -> k=J-1-j, also Tap-Achse umkehren.
    Wrev = Wreg[:, :, ::-1].copy()
    mu = conv_meta(f"up{i}", Wrev.reshape(Ce, Cin*J), np.repeat(b, r), Cin, J, 1, (J-1), True, Ce)
    # Sub-Pixel-Conv: pad beidseitig J-1 -> Ausgang T+(J-1); conv_gemm mit pad=J-1, dil=1
    emit(dict(op="conv", **{k: mu[k] for k in ("woff","Mp","Kg","boff","Cin","K","dil","pad","leaky","Cout","tm")},
              src="o", dst="u_conv", res="", tanh=0, outpad=J-1))   # outpad = zusaetzliche Ausgangslaenge
    emit(dict(op="shuffle", src="u_conv", dst="u", r=r, Cout=Cout, pad=(uk-r)//2, K=uk, J=J))
    rbouts = []
    for j, (k, dils) in enumerate(zip(rks, rds)):
        rb = f"resblocks.{i*len(rks)+j}"
        if RB_TYPE == 2:
            # ResBlock2: EIN Conv je Dilatation, jeder mit eigenem Residual.
            # Puffer abwechseln, damit kein Conv seine eigene Quelle ueberschreibt.
            cur = "u"
            for pi, d in enumerate(dils):
                dst = f"x{j}" if pi % 2 == 0 else f"y{j}"
                w1, b1 = folded(f"{rb}.convs.{pi}"); C1o, C1i, K1 = w1.shape
                m1 = conv_meta(f"s{i}_rb{j}_p{pi}", w1.reshape(C1o, C1i*K1), b1, C1i, K1,
                               d, d*(K1-1)//2, True, C1o)
                emit(dict(op="conv", **{kk: m1[kk] for kk in ("woff","Mp","Kg","boff","Cin","K","dil","pad","leaky","Cout","tm")},
                          src=cur, dst=dst, res=cur, tanh=0))
                cur = dst
            rbouts.append(cur)
            continue
        for pi, d in enumerate(dils):
            c1src = "u" if pi == 0 else f"x{j}"   # erstes Paar liest u (spart copy)
            c2res = "u" if pi == 0 else f"x{j}"   # Residual des ersten Paars = u
            xt = f"xt{j}"                          # eigenes xt je Resblock (Batch-Parallelitaet)
            w1, b1 = folded(f"{rb}.convs1.{pi}"); C1o, C1i, K1 = w1.shape
            m1 = conv_meta(f"s{i}_rb{j}_p{pi}_c1", w1.reshape(C1o, C1i*K1), b1, C1i, K1, d, d*(K1-1)//2, True, C1o)
            emit(dict(op="conv", **{kk: m1[kk] for kk in ("woff","Mp","Kg","boff","Cin","K","dil","pad","leaky","Cout","tm")}, src=c1src, dst=xt, res="", tanh=0))
            w2, b2 = folded(f"{rb}.convs2.{pi}"); C2o, C2i, K2 = w2.shape
            m2 = conv_meta(f"s{i}_rb{j}_p{pi}_c2", w2.reshape(C2o, C2i*K2), b2, C2i, K2, 1, (K2-1)//2, True, C2o)
            emit(dict(op="conv", **{kk: m2[kk] for kk in ("woff","Mp","Kg","boff","Cin","K","dil","pad","leaky","Cout","tm")}, src=xt, dst=f"x{j}", res=c2res, tanh=0))
        rbouts.append(f"x{j}")
    emit(dict(op="mrf", a=rbouts[0], b=rbouts[1], c=rbouts[2], dst="o"))

w, b = folded("conv_post"); Cout, Cin, K = w.shape
# vor conv_post gilt die andere LeakyReLU-Steigung (0.01 statt 0.1) - der Kernel
# kennt nur ein leaky-Flag, deshalb steht der Wert im Programm und wird geprueft.
mp = conv_meta("post", w.reshape(Cout, Cin*K), b, Cin, K, 1, (K-1)//2, LEAKY_FINAL, Cout)
emit(dict(op="conv", **{k: mp[k] for k in ("woff","Mp","Kg","boff","Cin","K","dil","pad","leaky","Cout","tm")}, src="o", dst="wav", res="", tanh=1))

full = np.concatenate(blob); full.tofile(f"{OUT}/weights16.bin")
biasf = np.concatenate(biasblob); biasf.tofile(f"{OUT}/bias16.bin")
json.dump(dict(nconv=len(pkgs), ops=ops, resblock_type=RB_TYPE,
               lrelu_slope=man.get("lrelu_slope", 0.1),
               lrelu_slope_final=man.get("lrelu_slope_final", 0.01)),
          open(f"{OUT}/program2.json", "w"))
print(f"{len(pkgs)} convs, {len(ops)} ops, weights {full.size*2/1e6:.1f}MB bias {biasf.size*2/1e6:.2f}MB")
