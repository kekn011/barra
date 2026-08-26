# Float-Simulator von program2.json (aus weights16.bin/bias16.bin), exakt wie conv_gemm+epilogue+
# shuffle+mrf rechnen. Vergleicht gegen HiFiGAN. Isoliert Verdrahtung/Gewichte von GPU/f16.
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hifigan_np import HiFiGAN
DUMP = os.environ.get("DUMPSLOT")

# gpu-sim.py <gpukit-dir> <dec-dir> [z.f32]
KIT = sys.argv[1] if len(sys.argv) > 1 else "gpukit2"
DEC = sys.argv[2] if len(sys.argv) > 2 else None
ZF  = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("ZFILE", "ztiny.f32")
prog = json.load(open(os.path.join(KIT, "program2.json")))
wblob = np.fromfile(os.path.join(KIT, "weights16.bin"), dtype=np.float16).astype(np.float32)
bblob = np.fromfile(os.path.join(KIT, "bias16.bin"), dtype=np.float16).astype(np.float32)
def up128(x): return (x+127)//128*128

# Kodierung wie im Shader: 1 = 0.1 (Koerper), 2 = 0.01 (vor conv_post).
SLOPES = {0: 0.0, 1: 0.1, 2: 0.01}
def leaky(x, mode=1): return np.where(x>=0, x, x*SLOPES.get(int(mode), 0.1))

slots = {}
z = np.fromfile(ZF, dtype=np.float32)
z = z.reshape(192, z.size // 192)
slots["z"] = z

for o in prog["ops"]:
    op = o["op"]
    if op == "conv":
        src = slots[o["src"]]; Cin=o["Cin"]; K=o["K"]; dil=o["dil"]; pad=o["pad"]; Cout=o["Cout"]
        Mp=o["Mp"]; Kg=o["Kg"]; woff=o["woff"]; boff=o["boff"]; outpad=o.get("outpad",0)
        T = src.shape[1]; Tout = T+outpad
        Wm = wblob[woff:woff+Mp*Kg].reshape(Mp,Kg)[:Cout,:Cin*K].reshape(Cout,Cin,K)
        b = bblob[boff:boff+Cout]
        x = leaky(src, o["leaky"]) if o["leaky"] else src
        # out[oc,t] = sum_{ci,k} Wm[oc,ci,k] * x[ci, t-pad+k*dil], t=0..Tout-1
        xp = np.pad(x, ((0,0),(pad, pad+outpad+K)))   # genug rechts
        out = np.zeros((Cout,Tout), np.float32)
        for k in range(K):
            out += Wm[:,:,k] @ xp[:, k*dil : k*dil+Tout]
        out += b[:,None]
        if o["res"]:
            out = out + slots[o["res"]]
        if o.get("tanh"): out = np.tanh(out)
        slots[o["dst"]] = out
    elif op == "shuffle":
        uc = slots[o["src"]]; r=o["r"]; Cout=o["Cout"]; pad=o["pad"]; K=o["K"]; J=o["J"]
        Tconv = uc.shape[1]; Tin = Tconv-(J-1); full=(Tin-1)*r+K; outT=full-2*pad
        u = np.zeros((Cout,outT), np.float32)
        for n in range(outT):
            m=n+pad; q=m//r; s=m%r
            u[:,n] = uc[np.arange(Cout)*r+s, q]
        slots[o["dst"]] = u
    elif op == "mrf":
        slots[o["dst"]] = (slots[o["a"]]+slots[o["b"]]+slots[o["c"]])/3.0
    if DUMP and o.get("dst")==DUMP:
        s = slots[DUMP]
        dv = np.fromfile("slot_dump.f32", dtype=np.float32)
        m = min(s.size, dv.size); a=s.ravel()[:m]; b=dv[:m]
        cos=float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
        print(f"SLOT {DUMP} sim{s.shape} gpu{dv.size} cos={cos:.6f} maxabs={np.abs(a-b).max():.4f}")
        sys.exit(0)

wav = slots["wav"][0]
ref = HiFiGAN(dec_dir=DEC).forward(z)
n=min(len(wav),len(ref))
cos=float((wav[:n]*ref[:n]).sum()/(np.linalg.norm(wav[:n])*np.linalg.norm(ref[:n])+1e-9))
print(f"GPU-SIM(float) vs numpy: len {len(wav)}/{len(ref)} cos={cos:.6f} maxabs={np.abs(wav[:n]-ref[:n]).max():.4f}")
