#!/usr/bin/env python3
"""cmp_stages4.py — Stufen-Bisektion der v2.3-Geraete-Dumps (16x8, q gechunkt) gegen numpy aus dump_h_f32.
Meta v4: 20 Werte/Layer (q1 isc izp osc ozp | q2 | k | v | o). Dumps int16, q in q_i8 (q1) + q2_i8."""
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
import gguf
G=sys.argv[1]; META=sys.argv[2]; DD=sys.argv[3]; L=int(sys.argv[4]) if len(sys.argv)>4 else 0
S=512; H=32; NKV=8; HD=128; GRP=4; D=2560
r=gguf.GGUFReader(G)
TENS={t.name:t for t in r.tensors}
def ten(name):
    t=TENS[name]; d=np.asarray(t.data)
    a=d.astype(np.float32) if t.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.BF16) else gguf.quants.dequantize(d,t.tensor_type).astype(np.float32)
    return a.reshape([int(x) for x in reversed(t.shape)])
eps=1e-6; base=1e6
meta=[]
with open(META) as f:
    f.readline()
    for line in f: meta.append([float(v) for v in line.split()])
m=meta[L]
q1_isc,q1_izp,q1_osc,q1_ozp = m[0:4]
q2_isc,q2_izp,q2_osc,q2_ozp = m[4:8]
k_isc,k_izp,k_osc,k_ozp     = m[8:12]
v_isc,v_izp,v_osc,v_ozp     = m[12:16]
o_isc,o_izp,o_osc,o_ozp     = m[16:20]
def rd(n,dt): return np.fromfile(os.path.join(DD,"dump_"+n),dt)
def cosd(name,a,b):
    a=np.asarray(a,np.float64).reshape(-1); b=np.asarray(b,np.float64).reshape(-1)
    cs=a@b/np.sqrt((a@a)*(b@b)+1e-30); rl=np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-30)
    print(f"{name:34s} cos={cs:.6f} rel_l2={rl*100:.2f}%")
h=rd("h_f32.bin",np.float32).reshape(S,D)
srow=rd("srow_f32.bin",np.float32)
srowc=rd("srowc_f32.bin",np.float32)
x16=rd("x_i8.bin",np.int16).reshape(S,D).astype(np.float32)
q1=rd("q_i8.bin",np.int16).reshape(S,H*HD//2).astype(np.float32)
q2=rd("q2_i8.bin",np.int16).reshape(S,H*HD//2).astype(np.float32)
k16=rd("k_i8.bin",np.int16).reshape(S,NKV*HD).astype(np.float32)
v16=rd("v_i8.bin",np.int16).reshape(S,NKV*HD).astype(np.float32)
qk=rd("qk_f16.bin",np.float16).astype(np.float32)
aof=rd("aof_f16.bin",np.float16).astype(np.float32).reshape(S,H*HD)
ao16=rd("ao_i8.bin",np.int16).reshape(S,H*HD).astype(np.float32)
o16=rd("o_i8.bin",np.int16).reshape(S,D).astype(np.float32)
odst=rd("odst_f32.bin",np.float32).reshape(S,D)
p=f"blk.{L}."
Wq=ten(p+"attn_q.weight"); Wk=ten(p+"attn_k.weight"); Wv=ten(p+"attn_v.weight"); Wo=ten(p+"attn_output.weight")
gq=ten(p+"attn_q_norm.weight").reshape(HD); gk=ten(p+"attn_k_norm.weight").reshape(HD)
# Stufe 0: srow korrekt? (geklemmt: sr=max(1, mx/(32767*isc)))
mx=np.abs(h).max(-1)
srow_exp=np.maximum(1.0, mx/(32767.0*q1_isc))
cosd("srow (geklemmte Zeilennorm)",srow[:S],srow_exp)
# Stufe 1: x_i16 Dequant vs h
hq=(x16-q1_izp)*q1_isc*srow[:S,None]
cosd("h nach int16 (dequant)",hq,h)
# Stufe 2: TPU q/k/v vs numpy (auf Basis hq)
qexp=hq@Wq.T; kexp=hq@Wk.T; vexp=hq@Wv.T
qd=np.concatenate([(q1-q1_ozp)*q1_osc,(q2-q2_ozp)*q2_osc],axis=1)*srow[:S,None]
kd=(k16-k_ozp)*k_osc*srow[:S,None]; vd=(v16-v_ozp)*v_osc*srow[:S,None]
cosd("TPU q1 (dequant vs numpy)",(q1-q1_ozp)*q1_osc*srow[:S,None],qexp[:,:H*HD//2])
cosd("TPU q2 (dequant vs numpy)",(q2-q2_ozp)*q2_osc*srow[:S,None],qexp[:,H*HD//2:])
cosd("TPU k",kd,kexp)
cosd("TPU v",vd,vexp)
# Stufe 3: rope/norm vs numpy (auf Basis qd/kd = was die GPU sah)
inv=base**(-np.arange(0,HD,2,dtype=np.float64)/HD)
t_=np.arange(S,dtype=np.float64)[:,None]
CO=np.cos(t_*inv).astype(np.float32); SI=np.sin(t_*inv).astype(np.float32)
def rope(v):
    lo=v[...,:64]; hi=v[...,64:]
    return np.concatenate([lo*CO[:,None,:]-hi*SI[:,None,:], hi*CO[:,None,:]+lo*SI[:,None,:]],-1)
def rmsn(x,g): return x/np.sqrt(np.mean(x*x,-1,keepdims=True)+eps)*g
qr_exp=rope(rmsn(qd.reshape(S,H,HD),gq)); kr_exp=rope(rmsn(kd.reshape(S,NKV,HD),gk))
qg=qk[:S*H*HD].reshape(S,H,HD); kg=qk[S*H*HD:].reshape(S,NKV,HD)
cosd("GPU q nach Norm+RoPE",qg,qr_exp)
cosd("GPU k nach Norm+RoPE",kg,kr_exp)
# Stufe 4: Attention (kausal, aus qg/kg/vd) vs GPU-ctx (aof)
MASK=np.tril(np.ones((S,S),bool)); sc=1.0/np.sqrt(HD)
ctx=np.zeros((S,H,HD),np.float32)
vdr=vd.reshape(S,NKV,HD)
for hh in range(H):
    g=hh//GRP
    s=(qg[:,hh]@kg[:,g].T)*sc
    s=np.where(MASK,s,-np.inf)
    e=np.exp(s-s.max(-1,keepdims=True)); pw=e/e.sum(-1,keepdims=True)
    ctx[:,hh]=pw@vdr[:,g]
cosd("GPU ctx (aof vs numpy-Attn)",aof,ctx.reshape(S,H*HD))
# Stufe 5: ctx-Quant
cm=np.abs(aof).max(-1)
srowc_exp=np.maximum(1.0, cm/(32767.0*o_isc))
cosd("srow_c",srowc[:S],srowc_exp)
aoq=(ao16-o_izp)*o_isc*srowc[:S,None]
cosd("ctx nach int16 (dequant)",aoq,aof)
# Stufe 6: TPU o
oexp=aoq@Wo.T
od=(o16-o_ozp)*o_osc*srowc[:S,None]
cosd("TPU o (dequant vs numpy)",od,oexp)
cosd("odst (final vs numpy-Kette)",odst,oexp)
# Gesamt: odst vs volle f32-Referenz aus h
qf=rope(rmsn((h@Wq.T).reshape(S,H,HD),gq)); kf=rope(rmsn((h@Wk.T).reshape(S,NKV,HD),gk)); vf=(h@Wv.T).reshape(S,NKV,HD)
ctxf=np.zeros((S,H,HD),np.float32)
for hh in range(H):
    g=hh//GRP
    s=(qf[:,hh]@kf[:,g].T)*sc
    s=np.where(MASK,s,-np.inf)
    e=np.exp(s-s.max(-1,keepdims=True)); pw=e/e.sum(-1,keepdims=True)
    ctxf[:,hh]=pw@vf[:,g]
cosd("odst vs f32-Referenz komplett",odst,ctxf.reshape(S,H*HD)@Wo.T)
