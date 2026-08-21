#!/usr/bin/env python3
"""cmp_ref.py — Vulkan-Referenz (dump_ref) gegen numpy-f32 aus demselben h (dump_href)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
import gguf
G=sys.argv[1]; DD=sys.argv[2]; L=int(sys.argv[3]) if len(sys.argv)>3 else 0
S=512; H=32; NKV=8; HD=128; GRP=4; D=2560; eps=1e-6; base=1e6
r=gguf.GGUFReader(G); TENS={t.name:t for t in r.tensors}
def ten(name):
    t=TENS[name]; d=np.asarray(t.data)
    a=d.astype(np.float32) if t.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.BF16) else gguf.quants.dequantize(d,t.tensor_type).astype(np.float32)
    return a.reshape([int(x) for x in reversed(t.shape)])
def cosd(name,a,b):
    a=np.asarray(a,np.float64).reshape(-1); b=np.asarray(b,np.float64).reshape(-1)
    cs=a@b/np.sqrt((a@a)*(b@b)+1e-30); rl=np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-30)
    print(f"{name:36s} cos={cs:.6f} rel_l2={rl*100:.2f}%")
h=np.fromfile(os.path.join(DD,f"dump_href_L{L}_f32.bin"),np.float32).reshape(S,D)
ref=np.fromfile(os.path.join(DD,f"dump_ref_L{L}_f32.bin"),np.float32).reshape(S,D)
hold=np.fromfile(os.path.join(DD,"dump_h_f32.bin"),np.float32).reshape(S,D)
ours=np.fromfile(os.path.join(DD,"dump_odst_f32.bin"),np.float32).reshape(S,D)
cosd("href vs dump_h (Laeufe identisch?)",h,hold)
p=f"blk.{L}."
Wq=ten(p+"attn_q.weight"); Wk=ten(p+"attn_k.weight"); Wv=ten(p+"attn_v.weight"); Wo=ten(p+"attn_output.weight")
gq=ten(p+"attn_q_norm.weight").reshape(HD); gk=ten(p+"attn_k_norm.weight").reshape(HD)
inv=base**(-np.arange(0,HD,2,dtype=np.float64)/HD)
t_=np.arange(S,dtype=np.float64)[:,None]
CO=np.cos(t_*inv).astype(np.float32); SI=np.sin(t_*inv).astype(np.float32)
def rope(v):
    lo=v[...,:64]; hi=v[...,64:]
    return np.concatenate([lo*CO[:,None,:]-hi*SI[:,None,:], hi*CO[:,None,:]+lo*SI[:,None,:]],-1)
def rmsn(x,g): return x/np.sqrt(np.mean(x*x,-1,keepdims=True)+eps)*g
qf=rope(rmsn((h@Wq.T).reshape(S,H,HD),gq)); kf=rope(rmsn((h@Wk.T).reshape(S,NKV,HD),gk)); vf=(h@Wv.T).reshape(S,NKV,HD)
MASK=np.tril(np.ones((S,S),bool)); sc=1.0/np.sqrt(HD)
ctx=np.zeros((S,H,HD),np.float32)
for hh in range(H):
    g=hh//GRP
    s=(qf[:,hh]@kf[:,g].T)*sc
    s=np.where(MASK,s,-np.inf)
    e=np.exp(s-s.max(-1,keepdims=True)); pw=e/e.sum(-1,keepdims=True)
    ctx[:,hh]=pw@vf[:,g]
onp=ctx.reshape(S,H*HD)@Wo.T
cosd("VULKAN-ref vs numpy-f32(href)",ref,onp)
cosd("UNSER odst vs numpy-f32(href)",ours,onp)
cosd("UNSER odst vs VULKAN-ref",ours,ref)
# Zeilenprofil der Abweichung Vulkan vs numpy
rl=np.linalg.norm(ref-onp,axis=1)/(np.linalg.norm(onp,axis=1)+1e-20)
print("rel_l2 je Zeile: mean %.3f  median %.3f  max %.3f (Zeile %d)  erste Zeilen: %s" % (rl.mean(),np.median(rl),rl.max(),rl.argmax(),np.round(rl[:8],3)))
