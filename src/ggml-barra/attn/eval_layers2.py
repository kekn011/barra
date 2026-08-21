#!/usr/bin/env python3
"""eval_layers2.py — per-Layer-Genauigkeit der v2.1-Pipeline (Zeilennorm + Split q/k/v/o) offline."""
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
import gguf
G=sys.argv[1]; META=sys.argv[2]; S=512
r=gguf.GGUFReader(G); arch=bytes(r.fields["general.architecture"].parts[-1]).decode()
def fld(n,d=None):
    f=r.fields.get(n)
    if f is None: return d
    v=f.parts[f.data[0]]; return v.tolist()[0] if hasattr(v,"tolist") and v.size==1 else v
NL=fld(f"{arch}.block_count"); D=fld(f"{arch}.embedding_length"); H=fld(f"{arch}.attention.head_count")
NKV=fld(f"{arch}.attention.head_count_kv"); HD=fld(f"{arch}.attention.key_length",D//H); GRP=H//NKV
eps=float(fld(f"{arch}.attention.layer_norm_rms_epsilon",1e-6)); base=float(fld(f"{arch}.rope.freq_base",1e6))
TENS={t.name:t for t in r.tensors}
def ten(name):
    t=TENS[name]; d=np.asarray(t.data)
    a=d.astype(np.float32) if t.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.BF16) else gguf.quants.dequantize(d,t.tensor_type).astype(np.float32)
    return a.reshape([int(x) for x in reversed(t.shape)])
def rms(x,g): return x/np.sqrt(np.mean(x*x,-1,keepdims=True)+eps)*g
def silu(x): return x/(1+np.exp(-x))
meta=[]
with open(META) as f:
    f.readline()
    for line in f: meta.append([float(v) for v in line.split()])
seed=[9707,11,358,1079,264,4128,1614,13,3555,374,697,829,30,358,2776,264,15235,18328,11,323,358,646,1492,498,448,264,6884,2088,315,9079,13,3555,11,1035,498,1075,311,1414,30,5209,3291,752,264,2699,911,6133,13,358]
toks=(seed*((S//len(seed))+1))[:S]; T=S
emb=ten("token_embd.weight"); x=emb[toks].astype(np.float32); del emb
inv=base**(-np.arange(0,HD,2,dtype=np.float64)/HD)
t_=np.arange(T,dtype=np.float64)[:,None]
COS=np.cos(t_*inv).astype(np.float32); SIN=np.sin(t_*inv).astype(np.float32)
def rope(v):
    lo=v[...,:HD//2]; hi=v[...,HD//2:]
    return np.concatenate([lo*COS[:,None,:]-hi*SIN[:,None,:], hi*COS[:,None,:]+lo*SIN[:,None,:]],-1)
MASK=np.tril(np.ones((T,T),bool)); sc=1.0/np.sqrt(HD)
f16=lambda a:a.astype(np.float16).astype(np.float32)
def rn_q8(x,isc,izp):
    """Zeilennorm-Quant: x -> (dequant, srow)"""
    m=np.abs(x).max(-1,keepdims=True); m[m==0]=1.0
    z=np.clip(np.round(x/(m*isc))+izp,-128,127)
    return ((z-izp)*isc*m).astype(np.float32), m
def attn(qh,kh,vh):
    ctx=np.zeros((T,H,HD),np.float32)
    for hh in range(H):
        g=hh//GRP
        s=f16((f16(qh[:,hh])/16)@(f16(kh[:,g])/16).T)*256*sc
        s=np.where(MASK,s,-np.inf)
        e=np.exp(s-s.max(-1,keepdims=True)); p=f16(e/e.sum(-1,keepdims=True))
        ctx[:,hh]=p@vh[:,g]
    return ctx.reshape(T,H*HD)
print("L   cos_o    rel_l2")
res=[]
for i in range(NL):
    p=f"blk.{i}."; m=meta[i]
    h=rms(x,ten(p+"attn_norm.weight"))
    Wq=ten(p+"attn_q.weight"); Wk=ten(p+"attn_k.weight"); Wv=ten(p+"attn_v.weight"); Wo=ten(p+"attn_output.weight")
    gq=ten(p+"attn_q_norm.weight").reshape(HD); gk=ten(p+"attn_k_norm.weight").reshape(HD)
    q=h@Wq.T; k=h@Wk.T; v=h@Wv.T
    qh=rope(rms(q.reshape(T,H,HD),gq)); kh=rope(rms(k.reshape(T,NKV,HD),gk))
    o_ref=attn(qh,kh,v.reshape(T,NKV,HD))@Wo.T
    # v2.1: h zeilennorm-int8 (gemeinsam), Ausgaenge je Paket int8 mit srow-Rueckskala
    hm=np.abs(h).max(-1,keepdims=True); hm[hm==0]=1.0
    hq=((np.clip(np.round(h/(hm*m[0]))+int(m[1]),-128,127)-int(m[1]))*m[0]*hm).astype(np.float32)
    def outq(y,osc,ozp): return ((np.clip(np.round(y/(hm*osc))+ozp,-128,127)-ozp)*osc*hm).astype(np.float32)
    q2=outq(hq@Wq.T,m[2],int(m[3])); k2=outq(hq@Wk.T,m[6],int(m[7])); v2=outq(hq@Wv.T,m[10],int(m[11]))
    qh2=rope(rms(q2.reshape(T,H,HD),gq)); kh2=rope(rms(k2.reshape(T,NKV,HD),gk))
    ctx2=f16(attn(qh2,kh2,v2.reshape(T,NKV,HD)))
    cm=np.abs(ctx2).max(-1,keepdims=True); cm[cm==0]=1.0
    cq=((np.clip(np.round(ctx2/(cm*m[12]))+int(m[13]),-128,127)-int(m[13]))*m[12]*cm).astype(np.float32)
    o2raw=cq@Wo.T
    o2=((np.clip(np.round(o2raw/(cm*m[14]))+int(m[15]),-128,127)-int(m[15]))*m[14]*cm).astype(np.float32)
    a=o2.reshape(-1).astype(np.float64); b=o_ref.reshape(-1).astype(np.float64)
    cos=a@b/np.sqrt((a@a)*(b@b)); rl=np.linalg.norm(a-b)/np.linalg.norm(b)
    res.append(cos)
    print(f"{i:2d}  {cos:.5f}  {rl*100:6.2f}%", flush=True)
    x=x+o_ref
    hf=rms(x,ten(p+"ffn_norm.weight"))
    x=x+(silu(hf@ten(p+"ffn_gate.weight").T)*(hf@ten(p+"ffn_up.weight").T))@ten(p+"ffn_down.weight").T
print("min/med:",min(res),sorted(res)[len(res)//2])
