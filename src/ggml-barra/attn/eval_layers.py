#!/usr/bin/env python3
"""eval_layers.py — pro Layer: wie genau ist der int8-Attention-Block (per-tensor, wie die Packages)
gegen f32? Simuliert exakt unsere Pipeline: h int8(isc/izp) -> qkv f32(W int8-quant? nein: TPU rechnet
W int8 — hier naehern wir nur die AKTIVIERUNGS-Quantisierung + f16-Attention + ctx-int8; W-Quant kommt
per Package-Verify spaeter dazu). Ausgabe: per-Layer cos des o-Ausgangs + Empfehlung Offload-Liste."""
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
def q8(a,isc,izp): return (np.clip(np.round(a/isc)+izp,-128,127)-izp)*isc
def attn(qh,kh,vh):
    ctx=np.zeros((T,H,HD),np.float32)
    for hh in range(H):
        g=hh//GRP
        s=f16((f16(qh[:,hh])/16)@(f16(kh[:,g])/16).T)*256*sc
        s=np.where(MASK,s,-np.inf)
        e=np.exp(s-s.max(-1,keepdims=True)); p=f16(e/e.sum(-1,keepdims=True))
        ctx[:,hh]=p@vh[:,g]
    return ctx.reshape(T,H*HD)
print("L   cos_o    rel_l2   h|max  isc     clip_in%")
res=[]
for i in range(NL):
    p=f"blk.{i}."; m=meta[i]
    h=rms(x,ten(p+"attn_norm.weight"))
    Wq=ten(p+"attn_q.weight"); Wk=ten(p+"attn_k.weight"); Wv=ten(p+"attn_v.weight"); Wo=ten(p+"attn_output.weight")
    gq=ten(p+"attn_q_norm.weight").reshape(HD); gk=ten(p+"attn_k_norm.weight").reshape(HD)
    # f32-Referenz
    q=h@Wq.T; k=h@Wk.T; v=h@Wv.T
    qh=rope(rms(q.reshape(T,H,HD),gq)); kh=rope(rms(k.reshape(T,NKV,HD),gk))
    o_ref=attn(qh,kh,v.reshape(T,NKV,HD))@Wo.T   # gleiche f16-Attention, isoliert NUR die int8-Stufen
    # int8-Pipeline: h int8 -> qkv (osc int8) -> attn -> ctx int8 -> o (osc int8)
    hq=q8(h,m[0],int(m[1]))
    clip=100.0*np.mean(np.abs(np.round(h/m[0])+int(m[1]))>127)
    qkv2=np.concatenate([hq@Wq.T,hq@Wk.T,hq@Wv.T],-1)
    qkv2=q8(qkv2,m[2],int(m[3]))
    q2=qkv2[:,:H*HD]; k2=qkv2[:,H*HD:H*HD+NKV*HD]; v2=qkv2[:,H*HD+NKV*HD:]
    qh2=rope(rms(q2.reshape(T,H,HD),gq)); kh2=rope(rms(k2.reshape(T,NKV,HD),gk))
    ctx2=q8(attn(qh2,kh2,v2.reshape(T,NKV,HD)),m[4],int(m[5]))
    o2=q8(ctx2@Wo.T,m[6],int(m[7]))
    a=o2.reshape(-1).astype(np.float64); b=o_ref.reshape(-1).astype(np.float64)
    cos=a@b/np.sqrt((a@a)*(b@b)); rl=np.linalg.norm(a-b)/np.linalg.norm(b)
    res.append(cos)
    print(f"{i:2d}  {cos:.5f}  {rl*100:6.2f}%  {np.abs(h).max():6.1f} {m[0]:.4f}  {clip:.2f}", flush=True)
    # echter Forward weiter (f32)
    x=x+ (attn(qh,kh,v.reshape(T,NKV,HD))@Wo.T)
    hf=rms(x,ten(p+"ffn_norm.weight"))
    x=x+(silu(hf@ten(p+"ffn_gate.weight").T)*(hf@ten(p+"ffn_up.weight").T))@ten(p+"ffn_down.weight").T
good=[i for i,c in enumerate(res) if c>0.99]
print("Layer mit cos>0.99:",len(good),good)
