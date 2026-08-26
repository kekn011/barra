#!/usr/bin/env python3
"""Front-ONNX aus einem Piper-model.onnx schneiden (alles bis zum Decoder-Eingang).

Gegenstueck zu export-front.py, das einen Coqui-Torch-Checkpoint braucht. Hier wird der
Graph an der Stelle geschnitten, die dump-dec-onnx.py im Manifest als z_tensor notiert hat:
Ausgang des Fronts ist z[192,T], genau das, was gpudecd als Eingabe erwartet.

  python export-front-onnx.py <model.onnx> <dec-dir> <out-front.onnx>
"""
import json
import os
import sys

import numpy as np
import onnx


def main(model_path, dec_dir, out_path):
    man = json.load(open(os.path.join(dec_dir, "dec_manifest.json")))
    z = man["z_tensor"]

    m = onnx.load(model_path)
    ins = [i.name for i in m.graph.input]
    print("Schneide %s  ->  %s" % (ins, z))

    onnx.utils.extract_model(model_path, out_path, ins, [z], check_model=False)

    f = onnx.load(out_path)
    onnx.checker.check_model(f)
    outs = [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])
            for o in f.graph.output]
    print("Front geschrieben: %s" % out_path)
    print("  Knoten        : %d  (Original %d)" % (len(f.graph.node), len(m.graph.node)))
    print("  Initializer   : %d  (Original %d)" % (len(f.graph.initializer), len(m.graph.initializer)))
    print("  Ausgang       : %s" % outs)
    print("  Groesse       : %.1f MB  (Original %.1f MB)"
          % (os.path.getsize(out_path) / 1e6, os.path.getsize(model_path) / 1e6))

    # Der Decoder darf NICHT mehr drinstecken - sonst haetten wir nichts gespart.
    left = [i.name for i in f.graph.initializer if i.name.startswith("dec.")]
    if left:
        print("  WARNUNG: %d dec.*-Gewichte noch im Front: %s" % (len(left), left[:3]))
        return 1
    print("  dec.*-Gewichte im Front: 0 — Schnitt sauber")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
