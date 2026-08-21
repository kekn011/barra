#!/usr/bin/env python3
"""gen_attn_pkgs7.py — L0-Rekalibrierung (v2.4): Die Seed-Kalibrierung unterschaetzt L0s Wertebereiche
(q2 Vokab-max 2,17 vs Rail 1,62; v 1,04 vs 0,47; ctx/o an der Rail) → q1/q2/kv/o fuer L0 mit
MARGIN-skalierter Kalibrierung neu konvertieren (cal = x*MARGIN → isc/osc wachsen um MARGIN,
int16 hat dafuer Headroom; Eingangs-Ausreisser faengt weiter die geklemmte Zeilennorm).

  python3 gen_attn_pkgs7.py <gguf> <outdir> <attn6.meta> [MARGIN=2.5] [S=512]

Schreibt nach <outdir>: q1_L0/q2_L0/kv_L0/o_L0.tflite + attn7.meta (= attn6.meta mit neuer L0-Zeile)
"""
import sys, os, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
import gguf, tensorflow as tf

G=sys.argv[1]; OUT=sys.argv[2]; META6=sys.argv[3]
MARGIN=float(sys.argv[4]) if len(sys.argv)>4 else 2.5
S=int(sys.argv[5]) if len(sys.argv)>5 else 512
os.makedirs(OUT, exist_ok=True)
r=gguf.GGUFReader(G); arch=bytes(r.fields["general.architecture"].parts[-1]).decode()
def fld(n,d=None):
    f=r.fields.get(n)
    if f is None: return d
    v=f.parts[f.data[0]]; return v.tolist()[0] if hasattr(v,"tolist") and v.size==1 else v
D=fld(f"{arch}.embedding_length"); H=fld(f"{arch}.attention.head_count")
NKV=fld(f"{arch}.attention.head_count_kv"); HD=fld(f"{arch}.attention.key_length", D//H); GRP=H//NKV
eps=float(fld(f"{arch}.attention.layer_norm_rms_epsilon",1e-6)); base=float(fld(f"{arch}.rope.freq_base",1e6))
TENS={t.name:t for t in r.tensors}
def ten(name):
    t=TENS[name]; d=np.asarray(t.data)
    a=d.astype(np.float32) if t.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.BF16) else gguf.quants.dequantize(d,t.tensor_type).astype(np.float32)
    return a.reshape([int(x) for x in reversed(t.shape)])
def rms(x,g): return x/np.sqrt(np.mean(x*x,-1,keepdims=True)+eps)*g

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
p="blk.0."
emb=ten("token_embd.weight")
h=rms(emb[toks].astype(np.float32),ten(p+"attn_norm.weight")); del emb
Wq=ten(p+"attn_q.weight"); Wk=ten(p+"attn_k.weight"); Wv=ten(p+"attn_v.weight"); Wo=ten(p+"attn_output.weight")
gq=ten(p+"attn_q_norm.weight").reshape(HD); gk=ten(p+"attn_k_norm.weight").reshape(HD)
NHALF=(H*HD)//2
hM=h*MARGIN
q1=conv_pkg("q1_L0",Wq[:NHALF],D,NHALF,hM)
q2=conv_pkg("q2_L0",Wq[NHALF:],D,NHALF,hM)
Wkv=np.concatenate([Wk,Wv],axis=0)
kv=conv_pkg("kv_L0",Wkv,D,2*NKV*HD,hM)
# ctx fuer die o-Kalibrierung (f32-Referenz-Attention wie gen5/gen6)
inv=base**(-np.arange(0,HD,2,dtype=np.float64)/HD)
t_=np.arange(T,dtype=np.float64)[:,None]
COS=np.cos(t_*inv).astype(np.float32); SIN=np.sin(t_*inv).astype(np.float32)
def rope(v):
    lo=v[...,:HD//2]; hi=v[...,HD//2:]
    return np.concatenate([lo*COS[:,None,:]-hi*SIN[:,None,:], hi*COS[:,None,:]+lo*SIN[:,None,:]],-1)
MASK=np.tril(np.ones((T,T),bool)); sc=1.0/np.sqrt(HD)
q=h@Wq.T; k=h@Wk.T; v=h@Wv.T
qh=rope(rms(q.reshape(T,H,HD),gq)); kh=rope(rms(k.reshape(T,NKV,HD),gk)); vh=v.reshape(T,NKV,HD)
ctx=np.zeros((T,H,HD),np.float32)
for hh in range(H):
    g=hh//GRP
    s=(qh[:,hh]@kh[:,g].T)*sc
    s=np.where(MASK,s,-np.inf)
    e=np.exp(s-s.max(-1,keepdims=True)); pw=e/e.sum(-1,keepdims=True)
    ctx[:,hh]=pw@vh[:,g]
o=conv_pkg("o_L0",Wo,H*HD,D,ctx.reshape(T,H*HD)*MARGIN)
m=[]
for qz in (q1,q2,kv,o): m += [qz[0][0],int(qz[0][1]),qz[1][0],int(qz[1][1])]
print("L0 neu: q1 isc=%.4e osc=%.4e | q2 osc=%.4e | kv osc=%.4e | o isc=%.4e osc=%.4e" % (
    m[0],m[2],m[6],m[10],m[12],m[14]))
print("Rails neu: q1=%.3f q2=%.3f kv=%.3f ctx=%.3f o=%.3f (Vokab-Soll: q 2.17, kv 1.04/1.20, ctx<=1.04)" % (
    32767*m[2],32767*m[6],32767*m[10],32767*m[12],32767*m[14]))
lines=open(META6).read().splitlines()
lines[1]=" ".join(f"{x:.8e}" if isinstance(x,float) else str(x) for x in m)
open(os.path.join(OUT,"attn7.meta"),"w").write("\n".join(lines)+"\n")
print(f"fertig: {OUT} (attn7.meta = attn6.meta mit neuer L0-Zeile)")
