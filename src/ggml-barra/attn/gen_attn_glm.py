#!/usr/bin/env python3
"""gen_attn_glm.py — Attention-Offload-Packages fuer GLM-Edge-4B (chatglm, 40 Layer, D=3072,
H=24, NKV=6, HD=128, voller RoPE 128, KEIN Bias, KEIN q/k-Norm, FFN fused-SWIGLU).

Unterschiede zu gen_attn_pkgs6.py (Qwen3-4B):
  - q als EIN Package (3072-out int16 — Compile-Frage, deshalb GLM_LIMIT-Smoke zuerst)
  - kein q/k-RMSNorm: RoPE direkt auf den rohen Koepfen; aux-GQ/GK = 1.0 (Layout kompatibel,
    Runtime-Kernel l6 ueberspringt die Norm)
  - FFN fused: up[3072,16384], swiglu = up[:, :FF] * silu(up[:, FF:]) (ggml: Gate in 2. Haelfte)
  - meta Version 6: 12 Werte/Layer (q kv o je isc izp osc ozp), 3 Pkg/Layer -> mids 3L/3L+1/3L+2

  python3 gen_attn_glm.py <gguf> <outdir> [S=512]      (env GLM_LIMIT=n: nur Layer 0..n-1)
"""
import sys, os, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
import gguf, tensorflow as tf

G=sys.argv[1]; OUT=sys.argv[2]; S=int(sys.argv[3]) if len(sys.argv)>3 else 512
os.makedirs(OUT, exist_ok=True)
r=gguf.GGUFReader(G); arch=bytes(r.fields["general.architecture"].parts[-1]).decode()
def fld(n,d=None):
    f=r.fields.get(n)
    if f is None: return d
    v=f.parts[f.data[0]]; return v.tolist()[0] if hasattr(v,"tolist") and v.size==1 else v
NL=fld(f"{arch}.block_count"); D=fld(f"{arch}.embedding_length"); H=fld(f"{arch}.attention.head_count")
NKV=fld(f"{arch}.attention.head_count_kv"); FF=fld(f"{arch}.feed_forward_length")
eps=float(fld(f"{arch}.attention.layer_norm_rms_epsilon",1e-5)); base=float(fld(f"{arch}.rope.freq_base",1e4))
HD=fld(f"{arch}.attention.key_length", D//H); GRP=H//NKV
LIM=int(os.environ.get("GLM_LIMIT", NL))
print(f"arch={arch} NL={NL} D={D} H={H} NKV={NKV} HD={HD} FF={FF} base={base} LIM={LIM}", flush=True)
TENS={t.name:t for t in r.tensors}
def ten(name):
    t=TENS[name]; d=np.asarray(t.data)
    a=d.astype(np.float32) if t.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.BF16) else gguf.quants.dequantize(d,t.tensor_type).astype(np.float32)
    return a.reshape([int(x) for x in reversed(t.shape)])
def rms(x,g): return x/np.sqrt(np.mean(x*x,-1,keepdims=True)+eps)*g
def silu(x): return x/(1+np.exp(-x))

def qparams(path):
    it=tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    di=it.get_input_details()[0]; do=it.get_output_details()[0]
    return di["quantization"], do["quantization"]

def conv_pkg(name,Wt,Kin,Nout,calib_rows):
    Wc=Wt.T.reshape(1,1,Kin,Nout).astype(np.float32); tW=tf.constant(Wc)
    @tf.function(input_signature=[tf.TensorSpec([1,S,1,Kin],tf.float32)])
    def conv(xx): return tf.nn.conv2d(xx,tW,strides=1,padding='VALID')
    cal=[calib_rows.reshape(1,S,1,Kin).astype(np.float32)]
    c=tf.lite.TFLiteConverter.from_concrete_functions([conv.get_concrete_function()])
    c.optimizations=[tf.lite.Optimize.DEFAULT]; c.representative_dataset=lambda:([v] for v in cal)
    c.target_spec.supported_ops=[tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    c.inference_input_type=tf.int16; c.inference_output_type=tf.int16
    path=os.path.join(OUT,f"{name}.tflite"); open(path,"wb").write(c.convert())
    return qparams(path)

seed=[9707,11,358,1079,264,4128,1614,13,3555,374,697,829,30,358,2776,264,15235,18328,11,323,358,646,1492,498,448,264,6884,2088,315,9079,13,3555,11,1035,498,1075,311,1414,30,5209,3291,752,264,2699,911,6133,13,358]
toks=(seed*((S//len(seed))+1))[:S]; T=S
emb=ten("token_embd.weight"); x=emb[toks].astype(np.float32); del emb
inv=base**(-np.arange(0,HD,2,dtype=np.float64)/HD)
t_=np.arange(T,dtype=np.float64)[:,None]
COS=np.cos(t_*inv).astype(np.float32); SIN=np.sin(t_*inv).astype(np.float32)
def rope(v):
    # chatglm = LLAMA_ROPE_TYPE_NORM: Rotation auf AUFEINANDERFOLGENDEN Paaren (x[2i],x[2i+1]),
    # Frequenz i — NICHT NEOX-half-split wie Qwen!
    p=v.reshape(v.shape[:-1]+(HD//2,2))
    lo=p[...,0]; hi=p[...,1]
    out=np.empty_like(p)
    out[...,0]=lo*COS[:,None,:]-hi*SIN[:,None,:]
    out[...,1]=hi*COS[:,None,:]+lo*SIN[:,None,:]
    return out.reshape(v.shape)
MASK=np.tril(np.ones((T,T),bool)); sc=1.0/np.sqrt(HD)
meta=[]
GQ=np.ones((NL,HD),np.float32); GK=np.ones((NL,HD),np.float32)
for i in range(LIM):
    p=f"blk.{i}."
    h=rms(x,ten(p+"attn_norm.weight"))
    Wq=ten(p+"attn_q.weight"); Wk=ten(p+"attn_k.weight"); Wv=ten(p+"attn_v.weight"); Wo=ten(p+"attn_output.weight")
    q_=conv_pkg(f"q_L{i}",Wq,D,H*HD,h)
    Wkv=np.concatenate([Wk,Wv],axis=0)
    kv=conv_pkg(f"kv_L{i}",Wkv,D,2*NKV*HD,h)
    m=[]
    # o braucht ctx als Kalibrierung -> erst Forward rechnen, dann o-Package
    q=h@Wq.T; k=h@Wk.T; v=h@Wv.T
    qh=rope(q.reshape(T,H,HD)); kh=rope(k.reshape(T,NKV,HD)); vh=v.reshape(T,NKV,HD)
    ctx=np.zeros((T,H,HD),np.float32)
    for hh in range(H):
        g=hh//GRP
        s=(qh[:,hh]@kh[:,g].T)*sc
        s=np.where(MASK,s,-np.inf)
        e=np.exp(s-s.max(-1,keepdims=True)); pw=e/e.sum(-1,keepdims=True)
        ctx[:,hh]=pw@vh[:,g]
    ctxf=ctx.reshape(T,H*HD)
    o_=conv_pkg(f"o_L{i}",Wo,H*HD,D,ctxf)
    for qz in (q_,kv,o_): m += [qz[0][0],int(qz[0][1]),qz[1][0],int(qz[1][1])]
    meta.append(m)
    print(f"L{i}: q isc={q_[0][0]:.3e} osc={q_[1][0]:.3e} | kv osc={kv[1][0]:.3e} | o osc={o_[1][0]:.3e}", flush=True)
    x=x+ctxf@Wo.T
    hf=rms(x,ten(p+"ffn_norm.weight"))
    hu=hf@ten(p+"ffn_up.weight").T
    x=x+(hu[:,:FF]*silu(hu[:,FF:]))@ten(p+"ffn_down.weight").T

with open(os.path.join(OUT,"attn_glm.meta"),"w") as f:
    f.write(f"{D} {H} {NKV} {HD} {FF} {NL} {S} 6\n")
    for m in meta:
        f.write(" ".join(f"{v:.8e}" if isinstance(v,float) else str(v) for v in m)+"\n")
aux=np.concatenate([GQ.reshape(-1),GK.reshape(-1),COS.reshape(-1),SIN.reshape(-1)]).astype(np.float32)
aux.tofile(os.path.join(OUT,"aux_attn.bin"))
print(f"fertig: {OUT} ({LIM}/{NL} Layer)", flush=True)
