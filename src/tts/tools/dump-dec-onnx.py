#!/usr/bin/env python3
"""Decoder-Gewichte + Manifest aus einem VITS-ONNX herausloesen (Piper-Weg).

Gegenstueck zu dump-dec.py, das einen Coqui-Torch-Checkpoint braucht. Piper-Stimmen
kommen als EIN model.onnx; hier wird der HiFi-GAN-Decoder daraus extrahiert.

Alles Architektonische wird AUS DEM GRAPHEN GELESEN, nicht angenommen:
  upsample_rates        <- stride der ConvTranspose-Knoten
  upsample_kernel_sizes <- Kernelmass der ups-Gewichte
  resblock_dilations    <- dilations-Attribut der Conv-Knoten
  resblock_typ          <- 'convs1/convs2' (ResBlock1) vs 'convs.N' (ResBlock2)

  python dump-dec-onnx.py <model.onnx> <out-dir>

Ergebnis: <out-dir>/dec_weights.npz + dec_manifest.json + z_tensor.txt
(z_tensor.txt nennt den Graph-Tensor, der in conv_pre laeuft = Schnittstelle
 zwischen Front und Decoder; export-front-onnx.py schneidet dort.)
"""
import json
import os
import re
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def main(model_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = onnx.load(model_path)
    g = m.graph

    init = {i.name: i for i in g.initializer}
    dec = {n: numpy_helper.to_array(t) for n, t in init.items() if n.startswith("dec.")}
    if not dec:
        sys.exit("Keine 'dec.*'-Initializer gefunden — ist das ein VITS-ONNX mit erhaltenen Modulnamen?")

    # --- Schnittstelle Front -> Decoder ---
    pre = [n for n in g.node if "dec.conv_pre.weight" in n.input]
    if not pre:
        sys.exit("dec.conv_pre nicht gefunden.")
    z_tensor = pre[0].input[0]

    # --- Attribute der dec-Convs einsammeln ---
    def attr(n, name):
        for a in n.attribute:
            if a.name == name:
                return list(a.ints) if len(a.ints) else a.i
        return None

    dil_of, stride_of = {}, {}
    for n in g.node:
        w = [i for i in n.input if i.startswith("dec.") and i.endswith(".weight")]
        if not w or n.op_type not in ("Conv", "ConvTranspose"):
            continue
        key = w[0][:-len(".weight")]
        d = attr(n, "dilations") or [1]
        s = attr(n, "strides") or [1]
        dil_of[key] = int(d[0])
        stride_of[key] = int(s[0])

    # --- Upsample-Stufen ---
    ups = sorted([k for k in dec if re.fullmatch(r"dec\.ups\.\d+\.weight", k)],
                 key=lambda s: int(s.split(".")[2]))
    rates = [stride_of[k[:-len(".weight")]] for k in ups]
    uks = [int(dec[k].shape[2]) for k in ups]
    up_init_ch = int(dec["dec.conv_pre.weight"].shape[0])
    inter_ch = int(dec["dec.conv_pre.weight"].shape[1])

    # --- Resblock-Typ bestimmen ---
    rb_keys = [k for k in dec if k.startswith("dec.resblocks.")]
    if any(".convs1." in k for k in rb_keys):
        rb_type = 1          # ResBlock1: convs1.{i} + convs2.{i}
    elif any(re.search(r"\.convs\.\d+\.", k) for k in rb_keys):
        rb_type = 2          # ResBlock2: convs.{i}, jeder ein eigener Residual-Schritt
    else:
        sys.exit("Unbekannte Resblock-Struktur: %s" % sorted(rb_keys)[:4])

    n_rb = 1 + max(int(k.split(".")[2]) for k in rb_keys)
    n_stage = len(ups)
    per_stage = n_rb // n_stage
    if n_rb % n_stage:
        sys.exit("Resblock-Zahl %d nicht durch Stufen %d teilbar" % (n_rb, n_stage))

    # Kernelgroessen der ersten Stufe = resblock_kernel_sizes
    def rb_first_conv(idx):
        for cand in ("dec.resblocks.%d.convs1.0.weight" % idx, "dec.resblocks.%d.convs.0.weight" % idx):
            if cand in dec:
                return cand
        sys.exit("kein erster Conv in resblock %d" % idx)

    rks = [int(dec[rb_first_conv(j)].shape[2]) for j in range(per_stage)]

    # Dilatationen je Resblock der ersten Stufe (gilt fuer alle Stufen gleich)
    rds = []
    for j in range(per_stage):
        if rb_type == 1:
            ds = []
            i = 0
            while "dec.resblocks.%d.convs1.%d.weight" % (j, i) in dec:
                ds.append(dil_of["dec.resblocks.%d.convs1.%d" % (j, i)])
                i += 1
        else:
            ds = []
            i = 0
            while "dec.resblocks.%d.convs.%d.weight" % (j, i) in dec:
                ds.append(dil_of["dec.resblocks.%d.convs.%d" % (j, i)])
                i += 1
        rds.append(ds)

    # --- LeakyReLU-Steigungen aus dem Graphen lesen, NICHT annehmen ---
    # HiFi-GAN benutzt LRELU_SLOPE=0.1 in ups/resblocks, aber vor conv_post ein blankes
    # F.leaky_relu() -> PyTorch-Vorgabe 0.01. Genau diese eine Stelle kostet sonst
    # Kosinus 0.998 statt 1.000 (am 26.8. genau so aufgelaufen).
    conv_post_node = [n for n in g.node if "dec.conv_post.weight" in n.input]
    slopes = {}
    for n in g.node:
        if n.op_type != "LeakyRelu" or not n.name.startswith("/dec"):
            continue
        a = [x.f for x in n.attribute if x.name == "alpha"]
        if not a:
            continue
        feeds_post = bool(conv_post_node) and n.output[0] in conv_post_node[0].input
        slopes.setdefault("final" if feeds_post else "body", round(float(a[0]), 6))

    man = dict(
        source=os.path.basename(model_path),
        lrelu_slope=slopes.get("body", 0.1),
        lrelu_slope_final=slopes.get("final", 0.01),
        resblock_type=rb_type,
        inter_channels=inter_ch,
        upsample_initial_channel=up_init_ch,
        upsample_rates=rates,
        upsample_kernel_sizes=uks,
        resblock_kernel_sizes=rks,
        resblock_dilation_sizes=rds,
        z_tensor=z_tensor,
    )

    # Praefix 'dec.' abstreifen — hifigan_np/gen-gpudec2 erwarten 'conv_pre', 'ups.0', ...
    flat = {k[len("dec."):]: v.astype(np.float32) for k, v in dec.items()}
    np.savez(os.path.join(out_dir, "dec_weights.npz"), **flat)
    with open(os.path.join(out_dir, "dec_manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    with open(os.path.join(out_dir, "z_tensor.txt"), "w") as f:
        f.write(z_tensor + "\n")

    print("Decoder extrahiert nach %s" % out_dir)
    print("  ResBlock-Typ         : %d  (%s)" % (rb_type, "convs1/convs2" if rb_type == 1 else "convs.N"))
    print("  inter_channels (z)   : %d" % inter_ch)
    print("  upsample_initial_ch  : %d" % up_init_ch)
    print("  upsample_rates       : %s   (Produkt = %d)" % (rates, int(np.prod(rates))))
    print("  upsample_kernels     : %s" % uks)
    print("  resblock_kernels     : %s" % rks)
    print("  resblock_dilations   : %s" % rds)
    print("  Tensoren             : %d" % len(flat))
    print("  LeakyReLU-Steigung   : %s (Koerper) / %s (vor conv_post)"
          % (man["lrelu_slope"], man["lrelu_slope_final"]))
    print("  z-Tensor im Graphen  : %s" % z_tensor)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
