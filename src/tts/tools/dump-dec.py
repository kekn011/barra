# Gemeinsame Basis fuer TPU+GPU-Port des HiFi-GAN-Decoders:
# 1) Gewichte als npz (aus dem Torch-Checkpoint, klare Namen)
# 2) Struktur-Manifest (JSON)
# 3) Referenz-Dumps: z-Eingaenge + fp32-Wave-Ausgaenge fuer 10 Saetze (Verify + int8-Kalibrierung)
import json

import numpy as np
import torch

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits

config = VitsConfig()
config.load_json("baj-config.json")
model = Vits.init_from_config(config)
model.load_checkpoint(config, "david.pth", eval=True)
model.eval()

dec = model.waveform_decoder
sd = dec.state_dict()
np.savez("baj-out/dec_weights.npz", **{k: v.numpy() for k, v in sd.items()})
print("gewichte:", len(sd), "tensoren")

manifest = {
    "upsample_rates": list(config.model_args.upsample_rates_decoder),
    "upsample_kernel_sizes": list(config.model_args.upsample_kernel_sizes_decoder),
    "upsample_initial_channel": config.model_args.upsample_initial_channel_decoder,
    "resblock_kernel_sizes": list(config.model_args.resblock_kernel_sizes_decoder),
    "resblock_dilation_sizes": [list(d) for d in config.model_args.resblock_dilation_sizes_decoder],
    "resblock_type": config.model_args.resblock_type_decoder,
    "in_channels": 192,
    "keys": sorted(sd.keys()),
}
with open("baj-out/dec_manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)
print(json.dumps({k: v for k, v in manifest.items() if k != "keys"}, indent=1))

# Referenz-Dumps ueber den ONNX-Front (deterministisch, scales 0/1/0)
import onnxruntime as ort
front = ort.InferenceSession("baj-out/david-front.onnx", providers=["CPUExecutionProvider"])
decs = ort.InferenceSession("baj-out/david-dec.onnx", providers=["CPUExecutionProvider"])
import sys
sys.path.insert(0, ".")
from bajtts.frontend import BajFrontend
fe = BajFrontend("baj-out/tokens.txt")

SENTS = [
    "The natural world is the greatest source of excitement, the greatest source of visual beauty.",
    "Here, on this remarkable little phone, an entire voice is brought to life.",
    "In the heart of the forest, life finds a way to flourish against all odds.",
    "Every evening the tide returns, carrying with it the memory of distant storms.",
    "Numbers like 42, 1999 and 3.14 must also sound completely natural.",
    "A short one.",
    "What we observe here is nothing less than extraordinary; a moment of pure wonder.",
    "The engine hummed quietly while the sensors recorded every subtle change.",
    "Deep beneath the waves, creatures of impossible beauty drift through the dark.",
    "And so, at the end of the day, the little machine simply kept on speaking.",
]
zs, ws = [], []
for s in SENTS:
    ids = np.asarray(fe.text_to_ids(s), dtype=np.int64)[None]
    feed = {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
            "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32)}
    z = front.run(None, feed)[0]
    w = decs.run(None, {decs.get_inputs()[0].name: z})[0]
    zs.append(z[0])
    ws.append(w[0, 0])
    print(f"dump: T={z.shape[2]} samples={w.shape[2]}")
np.savez("baj-out/dec_calib.npz", **{f"z{i}": z for i, z in enumerate(zs)},
         **{f"w{i}": w for i, w in enumerate(ws)})
print("DUMPS_OK")
