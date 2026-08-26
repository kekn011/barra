#!/usr/bin/env python3
"""Ende-zu-Ende-Abnahme der gesamten Kette — ohne Ton.

Vergleicht:
    Original-model.onnx (ein Stueck)
  gegen
    Front-ONNX -> z -> program2.json im Float-Simulator -> Wellenform

Damit ist alles abgedeckt, was zwischen Original und Geraet liegt: Schnitt, Extraktion,
Sub-Pixel-Umschreibung, Programm-Verdrahtung, f16-Gewichte, leaky-Kodierung. Was danach
noch abweichen kann, ist allein die GPU selbst.

  python verify-e2e.py <model.onnx> <front.onnx> <gpukit-dir> <dec-dir> [phoneme]
"""
import json
import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))


def run_onnx(path, ids, want=None):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {"input": ids,
            "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
            "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32)}
    outs = [o.name for o in s.get_outputs()]
    res = dict(zip(outs, s.run(outs, feed)))
    return res[want] if want else res[outs[0]]


def main(model, front, kit, dec, nph=40):
    ids = np.random.default_rng(7).integers(1, 60, size=(1, nph)).astype(np.int64)

    wav_ref = np.asarray(run_onnx(model, ids), dtype=np.float32).reshape(-1)
    z = np.asarray(run_onnx(front, ids), dtype=np.float32)
    while z.ndim > 2:
        z = z[0]
    print("Front -> z %s ;  Original-Wellenform %d Abtastwerte" % (z.shape, wav_ref.size))

    zf = os.path.join(kit, "_z_e2e.f32")
    z.tofile(zf)
    out = subprocess.run([sys.executable, os.path.join(HERE, "gpu-sim.py"), kit, dec, zf],
                         capture_output=True, text=True)
    os.remove(zf)
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr.strip()
    print("Simulator: %s" % line)
    if "cos=" not in line:
        print("Simulator lieferte kein Ergebnis."); return 1
    cos_sim = float(line.split("cos=")[1].split()[0])

    # Der Simulator vergleicht gegen die numpy-Referenz; hier zusaetzlich gegen das ORIGINAL,
    # damit auch der Schnitt mitgeprueft wird.
    sys.path.insert(0, HERE)
    from hifigan_np import HiFiGAN
    wav_chain = HiFiGAN(dec_dir=dec).forward(z).astype(np.float32).reshape(-1)
    n = min(wav_ref.size, wav_chain.size)
    a, b = wav_ref[:n], wav_chain[:n]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mx = float(np.max(np.abs(a - b)))
    print()
    print("  Kette gegen Original : cos %.8f   max %.3e" % (cos, mx))
    print("  Programm gegen numpy : cos %.6f" % cos_sim)
    ok = cos > 0.9999 and cos_sim > 0.9999
    print()
    print("  ERGEBNIS: %s" % ("GRUEN — die ganze Kette bis zur GPU stimmt"
                              if ok else "ROT — Abweichung, Kette pruefen"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:5], int(sys.argv[5]) if len(sys.argv) > 5 else 40))
