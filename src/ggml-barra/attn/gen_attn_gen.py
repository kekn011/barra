#!/usr/bin/env python3
"""gen_attn_gen.py — GENERISCHER Attention-Offload-Generator (meta v7).
Kein Modell-Nachbau: Kalibrierdaten kommen aus dem Geraete-Dump (BARRA_ATTN_CALDUMP,
cal_h_L*.bin [S,D] + cal_ctx_L*.bin [S,H*HD] f32), Gewichte+Bias aus dem GGUF.
Packages: q (Bias eingebacken), kv ([Wk;Wv]+[bk;bv]), o — 3/Layer, 16x8, q ungechunkt.

  python3 gen_attn_gen.py <gguf> <caldir> <outdir> [S=512]
  env RMODE=0|2 ueberschreibt die Arch-Zuordnung (0=NORM/chatglm-llama, 2=NEOX/qwen).
"""
import sys, os, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
import gguf, tensorflow as tf

G=sys.argv[1]; CAL=sys.argv[2]; OUT=sys.argv[3]; S=int(sys.argv[4]) if len(sys.argv)>4 else 512
os.makedirs(OUT, exist_ok=True)
r=gguf.GGUFReader(G); arch=bytes(r.fields["general.architecture"].parts[-1]).decode()
def fld(n,d=None):
    f=r.fields.get(n)
    if f is None: return d
    v=f.parts[f.data[0]]; return v.tolist()[0] if hasattr(v,"tolist") and v.size==1 else v
NL=fld(f"{arch}.block_count"); D=fld(f"{arch}.embedding_length"); H=fld(f"{arch}.attention.head_count")
NKV=fld(f"{arch}.attention.head_count_kv"); FF=fld(f"{arch}.feed_forward_length")
base=float(fld(f"{arch}.rope.freq_base",1e4)); HD=fld(f"{arch}.attention.key_length", D//H)
RMAP={"qwen2":2,"qwen3":2,"chatglm":0,"llama":0}
rmode=int(os.environ.get("RMODE", RMAP.get(arch,2)))
TENS={t.name:t for t in r.tensors}
has_bias = "blk.0.attn_q.bias" in TENS
has_qknorm = "blk.0.attn_q_norm.weight" in TENS
# q-Ausgang >3072 int16 kompiliert nicht auf dem Geraet (Qwen3-4B-Falle) -> 2 Chunks (Kopf-Haelften)
qchunk = 1 if H*HD > 3072 else 0
print(f"arch={arch} NL={NL} D={D} H={H} NKV={NKV} HD={HD} base={base} rmode={rmode} bias={int(has_bias)} qknorm={int(has_qknorm)} qchunk={qchunk}", flush=True)
def ten(name):
    t=TENS[name]; d=np.asarray(t.data)
    a=d.astype(np.float32) if t.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.BF16) else gguf.quants.dequantize(d,t.tensor_type).astype(np.float32)
    return a.reshape([int(x) for x in reversed(t.shape)])

def qparams(path):
    it=tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    di=it.get_input_details()[0]; do=it.get_output_details()[0]
    return di["quantization"], do["quantization"]

def conv_pkg(name,Wt,Bv,Kin,Nout,calib_rows):
    Wc=Wt.T.reshape(1,1,Kin,Nout).astype(np.float32); tW=tf.constant(Wc)
    tB=tf.constant(Bv.astype(np.float32)) if Bv is not None else None
    @tf.function(input_signature=[tf.TensorSpec([1,S,1,Kin],tf.float32)])
    def conv(xx):
        y=tf.nn.conv2d(xx,tW,strides=1,padding='VALID')
        return y+tB if tB is not None else y
    cal=[calib_rows.reshape(1,S,1,Kin).astype(np.float32)]
    c=tf.lite.TFLiteConverter.from_concrete_functions([conv.get_concrete_function()])
    c.optimizations=[tf.lite.Optimize.DEFAULT]; c.representative_dataset=lambda:([v] for v in cal)
    c.target_spec.supported_ops=[tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    c.inference_input_type=tf.int16; c.inference_output_type=tf.int16
    path=os.path.join(OUT,f"{name}.tflite"); open(path,"wb").write(c.convert())
    return qparams(path)

# ---------- Gemma-4-Zweig (meta v8): 2 Layer-Typen (swa HD_swa / full HD), KV-Sharing,
# V-Norm auf der GPU (Packages unveraendert reine Projektionen), full-q gechunkt (>3072).
if arch=="gemma4":
    HDS=fld(f"{arch}.attention.key_length_swa")
    NKVS=1
    base_swa=float(fld(f"{arch}.rope.freq_base_swa",1e4))
    # Layer-Typ aus Tensor-Shapes: wq-out==H*HD -> full, sonst swa; kv-Layer an attn_k.weight
    types=[]
    for i in range(NL):
        full = int(TENS[f"blk.{i}.attn_q.weight"].shape[1])==H*HD
        haskv = f"blk.{i}.attn_k.weight" in TENS
        types.append((1 if full else 0) if haskv else (3 if full else 2))
    print("Typen:", "".join(str(t) for t in types), flush=True)
    HDMAX=HD
    GQ=np.ones((NL,HDMAX),np.float32); GK=np.ones((NL,HDMAX),np.float32)
    meta=[]; pkglist=[]
    for i in range(NL):
        p=f"blk.{i}."; ty=types[i]
        hd = HD if ty in (1,3) else HDS
        h=np.fromfile(os.path.join(CAL,f"cal_h_L{i}.bin"),dtype=np.float32).reshape(S,D)
        ctx=np.fromfile(os.path.join(CAL,f"cal_ctx_L{i}.bin"),dtype=np.float32).reshape(S,H*hd)
        GQ[i,:hd]=ten(p+"attn_q_norm.weight").reshape(hd)
        if ty in (0,1): GK[i,:hd]=ten(p+"attn_k_norm.weight").reshape(hd)
        Wq=ten(p+"attn_q.weight"); Wo=ten(p+"attn_output.weight")
        if H*hd>3072:
            HH=H*hd//2
            q1=conv_pkg(f"q1_L{i}",Wq[:HH],None,D,HH,h); q2=conv_pkg(f"q2_L{i}",Wq[HH:],None,D,HH,h)
            pkglist += [f"q1_L{i}",f"q2_L{i}"]
        else:
            q1=conv_pkg(f"q_L{i}",Wq,None,D,H*hd,h); q2=None
            pkglist.append(f"q_L{i}")
        kv=None
        if ty in (0,1):
            Wk=ten(p+"attn_k.weight"); Wv=ten(p+"attn_v.weight")
            kv=conv_pkg(f"kv_L{i}",np.concatenate([Wk,Wv],axis=0),None,D,2*hd,h)
            pkglist.append(f"kv_L{i}")
        o_=conv_pkg(f"o_L{i}",Wo,None,H*hd,D,ctx)
        pkglist.append(f"o_L{i}")
        m=[]
        for qz in (q1, q2 if q2 is not None else q1, kv if kv is not None else q1, o_):
            m += [qz[0][0],int(qz[0][1]),qz[1][0],int(qz[1][1])]
        if kv is None: m[8]=0.0; m[10]=0.0   # shared: kv-Skalen ungueltig markieren
        meta.append(m)
        print(f"L{i} ty={ty} hd={hd}: q isc={q1[0][0]:.3e} osc={q1[1][0]:.3e} | o osc={o_[1][0]:.3e}", flush=True)
    with open(os.path.join(OUT,"attn_gen.meta"),"w") as f:
        f.write(f"{D} {H} {NKVS} {HD} {FF} {NL} {S} 8\n")
        f.write(f"{rmode} 0 1 1\n")
        f.write(f"{HDS} 1 2\n")
        f.write(" ".join(str(t) for t in types)+"\n")
        for m in meta:
            f.write(" ".join(f"{v:.8e}" if isinstance(v,float) else str(v) for v in m)+"\n")
    # aux v8: GQ/GK NL x HDMAX + zwei Tabellen (swa: S x HDS/2; full: S x HD/2 MIT freq_factors)
    t_=np.arange(S,dtype=np.float64)[:,None]
    inv_s=base_swa**(-np.arange(0,HDS,2,dtype=np.float64)/HDS)
    # rope_freqs kann global ("rope_freqs.weight") ODER je Layer heissen; Faktoren bis 1e30 =
    # "diese Dimension nicht rotieren" (Gemma-proportional-RoPE) — ohne sie ist Full-RoPE falsch!
    ff_name=None
    for t in TENS:
        if t.endswith("rope_freqs.weight"): ff_name=t; break
    factors=ten(ff_name).reshape(HD//2).astype(np.float64) if ff_name else np.ones(HD//2)
    inv_f=base**(-np.arange(0,HD,2,dtype=np.float64)/HD)/factors
    CS=np.cos(t_*inv_s).astype(np.float32); SS=np.sin(t_*inv_s).astype(np.float32)
    CF=np.cos(t_*inv_f).astype(np.float32); SF=np.sin(t_*inv_f).astype(np.float32)
    aux=np.concatenate([GQ.reshape(-1),GK.reshape(-1),CS.reshape(-1),SS.reshape(-1),CF.reshape(-1),SF.reshape(-1)]).astype(np.float32)
    aux.tofile(os.path.join(OUT,"aux_attn.bin"))
    open(os.path.join(OUT,"pkglist.txt"),"w").write("\n".join(pkglist)+"\n")
    print(f"fertig: {OUT} ({NL} Layer, {len(pkglist)} Packages, meta v8 gemma, freq_factors={'ja' if ff_name else 'nein'})", flush=True)
    sys.exit(0)

meta=[]
GQ=np.ones((NL,HD),np.float32); GK=np.ones((NL,HD),np.float32)
for i in range(NL):
    p=f"blk.{i}."
    h=np.fromfile(os.path.join(CAL,f"cal_h_L{i}.bin"),dtype=np.float32).reshape(S,D)
    ctx=np.fromfile(os.path.join(CAL,f"cal_ctx_L{i}.bin"),dtype=np.float32).reshape(S,H*HD)
    Wq=ten(p+"attn_q.weight"); Wk=ten(p+"attn_k.weight"); Wv=ten(p+"attn_v.weight"); Wo=ten(p+"attn_output.weight")
    bq=ten(p+"attn_q.bias") if has_bias else None
    bkv=np.concatenate([ten(p+"attn_k.bias"),ten(p+"attn_v.bias")]) if has_bias else None
    bo=ten(p+"attn_output.bias") if (p+"attn_output.bias") in TENS else None
    if has_qknorm:
        GQ[i]=ten(p+"attn_q_norm.weight").reshape(HD); GK[i]=ten(p+"attn_k_norm.weight").reshape(HD)
    if qchunk:
        HH=H*HD//2
        q1=conv_pkg(f"q1_L{i}",Wq[:HH],bq[:HH] if bq is not None else None,D,HH,h)
        q2=conv_pkg(f"q2_L{i}",Wq[HH:],bq[HH:] if bq is not None else None,D,HH,h)
    else:
        q1=conv_pkg(f"q_L{i}",Wq,bq,D,H*HD,h); q2=None
    kv=conv_pkg(f"kv_L{i}",np.concatenate([Wk,Wv],axis=0),bkv,D,2*NKV*HD,h)
    o_=conv_pkg(f"o_L{i}",Wo,bo,H*HD,D,ctx)
    m=[]
    grp=(q1,q2,kv,o_) if qchunk else (q1,kv,o_)
    for qz in grp: m += [qz[0][0],int(qz[0][1]),qz[1][0],int(qz[1][1])]
    meta.append(m)
    print(f"L{i}: q isc={q1[0][0]:.3e} osc={q1[1][0]:.3e} | kv osc={kv[1][0]:.3e} | o osc={o_[1][0]:.3e}", flush=True)

with open(os.path.join(OUT,"attn_gen.meta"),"w") as f:
    f.write(f"{D} {H} {NKV} {HD} {FF} {NL} {S} 7\n")
    f.write(f"{rmode} {int(has_bias)} {int(has_qknorm)} {qchunk}\n")
    for m in meta:
        f.write(" ".join(f"{v:.8e}" if isinstance(v,float) else str(v) for v in m)+"\n")
inv=base**(-np.arange(0,HD,2,dtype=np.float64)/HD)
t_=np.arange(S,dtype=np.float64)[:,None]
COS=np.cos(t_*inv).astype(np.float32); SIN=np.sin(t_*inv).astype(np.float32)
aux=np.concatenate([GQ.reshape(-1),GK.reshape(-1),COS.reshape(-1),SIN.reshape(-1)]).astype(np.float32)
aux.tofile(os.path.join(OUT,"aux_attn.bin"))
print(f"fertig: {OUT} ({NL} Layer, meta v7, qknorm={int(has_qknorm)} qchunk={qchunk})", flush=True)
