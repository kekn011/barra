#!/usr/bin/env python3
"""Numerische Abnahme des extrahierten Decoders — ohne Ton.

Prueft, ob die numpy-Referenz (hifigan_np) aus den extrahierten Gewichten dieselbe
Wellenform erzeugt wie der Decoder-Teil des Original-ONNX. Das ist der Beweis, dass
dump-dec-onnx.py richtig extrahiert und hifigan_np den Modelltyp richtig rechnet —
bevor irgendetwas auf die GPU geht.

  python verify-dec.py <model.onnx> <dec-dir> [phoneme-anzahl]

Ablauf: das Original-ONNX wird um einen zusaetzlichen Ausgang erweitert (der
z-Tensor aus dem Manifest). Ein Lauf liefert damit z UND die Referenz-Wellenform;
die numpy-Referenz bekommt dasselbe z. Verglichen wird per Kosinus und max. Abweichung.
Es wird nichts abgespielt — nur gerechnet.
"""
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort


def main(model_path, dec_dir, nph=40):
    man = json.load(open(os.path.join(dec_dir, "dec_manifest.json")))
    z_name = man["z_tensor"]

    # --- ONNX um den z-Tensor als Ausgang erweitern ---
    m = onnx.load(model_path)
    if not any(o.name == z_name for o in m.graph.output):
        vi = onnx.helper.ValueInfoProto()
        vi.name = z_name
        m.graph.output.append(vi)
    tmp = os.path.join(dec_dir, "_with_z.onnx")
    onnx.save(m, tmp)

    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    outs = [o.name for o in sess.get_outputs()]

    rng = np.random.default_rng(0)
    ids = rng.integers(low=1, high=60, size=(1, nph)).astype(np.int64)
    feed = {
        "input": ids,
        "input_lengths": np.array([nph], dtype=np.int64),
        "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # deterministisch
    }
    res = sess.run(outs, feed)
    got = dict(zip(outs, res))
    wav_ref = np.asarray(got[outs[0]]).reshape(-1).astype(np.float32)
    z = np.asarray(got[z_name]).astype(np.float32)
    while z.ndim > 2:
        z = z[0]
    print("z: %s   Referenz-Wellenform: %d Abtastwerte" % (z.shape, wav_ref.size))

    # --- numpy-Referenz auf demselben z ---
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hifigan_np import HiFiGAN
    hg = HiFiGAN(dec_dir=dec_dir)
    wav_np = hg.forward(z).astype(np.float32).reshape(-1)
    print("numpy-Wellenform    : %d Abtastwerte" % wav_np.size)

    n = min(wav_ref.size, wav_np.size)
    if wav_ref.size != wav_np.size:
        print("HINWEIS: Laengen weichen ab (%d vs %d) - verglichen wird der gemeinsame Teil."
              % (wav_ref.size, wav_np.size))
    a, b = wav_ref[:n], wav_np[:n]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mx = float(np.max(np.abs(a - b)))
    rms = float(np.sqrt(np.mean((a - b) ** 2)))
    print()
    print("  Kosinus            : %.8f" % cos)
    print("  max. Abweichung    : %.3e" % mx)
    print("  RMS-Abweichung     : %.3e" % rms)
    os.remove(tmp)
    ok = cos > 0.9999 and mx < 1e-3
    print()
    print("  ERGEBNIS: %s" % ("GRUEN — Extraktion und Modelltyp stimmen" if ok
                              else "ROT — Abweichung zu gross, Extraktion pruefen"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40))
