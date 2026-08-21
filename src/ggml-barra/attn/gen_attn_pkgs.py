#!/usr/bin/env python3
"""gen_attn_pkgs.py — qkv+o Attention-Projektions-Packages fuer ALLE Layer (ggml-barra v2 / Attention-Offload).
Qwen3-4B, 1x1-Conv int8, M=S=512. Kalibrierung aus echtem numpy-Forward (Prompt-Seed wie gen_real_proj.py).

  python3 gen_attn_pkgs.py <gguf> <outdir> [S=512]

Schreibt nach <outdir>:
  qkv_L{i}.tflite / o_L{i}.tflite  (i=0..NL-1)
  attn.meta  : Zeile 1 "D H NKV HD FF NL S", dann je Layer "qkv_isc qkv_izp qkv_osc qkv_ozp o_isc o_izp o_osc o_ozp"
  aux_attn.bin : gamma_q[NL][HD] f32, gamma_k[NL][HD] f32, cos[S*HD/2] f32, sin[S*HD/2] f32
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
eps=float(fld(f"{arch}.attention.layer_norm_rms_epsilon",1e-6)); base=float(fld(f"{arch}.rope.freq_base",1e6))
HD=fld(f"{arch}.attention.key_length", D//H); GRP=H//NKV
print(f"arch={arch} NL={NL} D={D} H={H} NKV={NKV} HD={HD} FF={FF} S={S} eps={eps} base={base}", flush=True)
TENS={t.name:t for t in r.tensors}
def ten(name):
    t=TENS[name]; d=np.asarray(t.data)
    a=d.astype(np.float32) if t.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.BF16) else gguf.quants.dequantize(d,t.tensor_type).astype(np.float32)
    return a.reshape([int(x) for x in reversed(t.shape)])
def has(n): return n in TENS
def rms(x,g): return x/np.sqrt(np.mean(x*x,-1,keepdims=True)+eps)*g
def silu(x): return x/(1+np.exp(-x))

seed=[9707,11,358,1079,264,4128,1614,13,3555,374,697,829,30,358,2776,264,15235,18328,11,323,358,646,1492,498,448,264,6884,2088,315,9079,13,3555,11,1035,498,1075,311,1414,30,5209,3291,752,264,2699,911,6133,13,358]
toks=(seed*((S//len(seed))+1))[:S]; T=S
emb=ten("token_embd.weight"); x=emb[toks].astype(np.float32); del emb
inv=base**(-np.arange(0,HD,2,dtype=np.float64)/HD)
t_=np.arange(T,dtype=np.float64)[:,None]
COS=np.cos(t_*inv).astype(np.float32); SIN=np.sin(t_*inv).astype(np.float32)   # [T,HD/2]
def rope(v):  # [T,h,HD]
    lo=v[...,:HD//2]; hi=v[...,HD//2:]
    c=COS[:,None,:]; s=SIN[:,None,:]
    return np.concatenate([lo*c-hi*s, hi*c+lo*s],-1)
MASK=np.tril(np.ones((T,T),bool))

def conv_pkg(name,Wt,Kin,Nout,calib_rows):
    Wc=Wt.T.reshape(1,1,Kin,Nout).astype(np.float32); tW=tf.constant(Wc)
    @tf.function(input_signature=[tf.TensorSpec([1,S,1,Kin],tf.float32)])
    def conv(xx): return tf.nn.conv2d(xx,tW,strides=1,padding='VALID')
    cal=[calib_rows.reshape(1,S,1,Kin).astype(np.float32)]
    c=tf.lite.TFLiteConverter.from_concrete_functions([conv.get_concrete_function()])
    c.optimizations=[tf.lite.Optimize.DEFAULT]; c.representative_dataset=lambda:([v] for v in cal)
    c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type=tf.int8; c.inference_output_type=tf.int8
    blob=c.convert(); path=os.path.join(OUT,f"{name}.tflite"); open(path,"wb").write(blob)
    it=tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    di=it.get_input_details()[0]; do=it.get_output_details()[0]
    return di["quantization"], do["quantization"]

meta=[]; GQ=np.zeros((NL,HD),np.float32); GK=np.zeros((NL,HD),np.float32)
sc=1.0/np.sqrt(HD)
for i in range(NL):
    p=f"blk.{i}."
    h=rms(x,ten(p+"attn_norm.weight"))
    Wq=ten(p+"attn_q.weight"); Wk=ten(p+"attn_k.weight"); Wv=ten(p+"attn_v.weight")
    q=h@Wq.T; k=h@Wk.T; v=h@Wv.T
    gq=ten(p+"attn_q_norm.weight").reshape(HD); gk=ten(p+"attn_k_norm.weight").reshape(HD)
    GQ[i]=gq; GK[i]=gk
    qh=rope(rms(q.reshape(T,H,HD),gq)); kh=rope(rms(k.reshape(T,NKV,HD),gk)); vh=v.reshape(T,NKV,HD)
    ctx=np.zeros((T,H,HD),np.float32)
    for hh in range(H):
        g=hh//GRP
        s=(qh[:,hh]@kh[:,g].T)*sc
        s=np.where(MASK,s,-np.inf)
        e=np.exp(s-s.max(-1,keepdims=True)); pw=e/e.sum(-1,keepdims=True)
        ctx[:,hh]=pw@vh[:,g]
    ctx2=ctx.reshape(T,H*HD)
    o=ctx2@ten(p+"attn_output.weight").T
    # Packages mit ECHTEN Kalibrierdaten dieses Layers
    Wqkv=np.concatenate([Wq,Wk,Wv],0)
    qq=conv_pkg(f"qkv_L{i}",Wqkv,D,Wqkv.shape[0],h)
    oq=conv_pkg(f"o_L{i}",ten(p+"attn_output.weight"),H*HD,D,ctx2)
    meta.append((qq[0][0],qq[0][1],qq[1][0],qq[1][1],oq[0][0],oq[0][1],oq[1][0],oq[1][1]))
    print(f"L{i}: h|max={np.abs(h).max():.2f} ctx|max={np.abs(ctx2).max():.2f} qkv isc/osc={qq[0][0]:.4e}/{qq[1][0]:.4e} zp={qq[0][1]}/{qq[1][1]} o isc/osc={oq[0][0]:.4e}/{oq[1][0]:.4e} zp={oq[0][1]}/{oq[1][1]}", flush=True)
    x=x+o
    hf=rms(x,ten(p+"ffn_norm.weight"))
    x=x+(silu(hf@ten(p+"ffn_gate.weight").T)*(hf@ten(p+"ffn_up.weight").T))@ten(p+"ffn_down.weight").T

with open(os.path.join(OUT,"attn.meta"),"w") as f:
    f.write(f"{D} {H} {NKV} {HD} {FF} {NL} {S}\n")
    for m in meta: f.write(f"{m[0]:.8e} {int(m[1])} {m[2]:.8e} {int(m[3])} {m[4]:.8e} {int(m[5])} {m[6]:.8e} {int(m[7])}\n")
aux=np.concatenate([GQ.reshape(-1),GK.reshape(-1),COS.reshape(-1),SIN.reshape(-1)]).astype(np.float32)
aux.tofile(os.path.join(OUT,"aux_attn.bin"))
print(f"fertig: {OUT} aux={aux.nbytes}B meta={NL} Layer", flush=True)
