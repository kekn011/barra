#!/usr/bin/env python3
# david-front.onnx exportieren: gibt den waveform_decoder-INPUT (z*y_mask) aus statt der WAV.
# Trick: waveform_decoder -> Identitaet. Danach front-onnx (noise=0) auf ids0 laufen und z[192,T]
# als channel-major f32 dumpen (fuer gpudec2). Aufruf:
#   python export-front.py <config.json> <david.pth> <ids0.npy> <out_onnx> <out_z.f32>
import sys, numpy as np, torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits

cfg_path, ckpt_path, ids_path, out_onnx, out_z = sys.argv[1:6]

config = VitsConfig(); config.load_json(cfg_path)
model = Vits.init_from_config(config); model.load_checkpoint(config, ckpt_path, eval=True); model.eval()
print("inter_channels:", getattr(model, "inter_channels", "?"))

class IdDecoder(torch.nn.Module):
    # waveform_decoder(z_slice, g=g) -> gibt z_slice unveraendert zurueck
    def forward(self, x, g=None):
        return x
model.waveform_decoder = IdDecoder()

_orig = torch.onnx.export
def _ts(*a, **k):
    k["dynamo"] = False
    return _orig(*a, **k)
torch.onnx.export = _ts

def onnx_inference(text, text_lengths, scales):
    model.inference_noise_scale = scales[0]
    model.length_scale = scales[1]
    model.inference_noise_scale_dp = scales[2]
    return model.inference(text, aux_input={"x_lengths": text_lengths, "d_vectors": None,
                           "speaker_ids": None, "language_ids": None, "durations": None})["model_outputs"]
model.forward = onnx_inference

seq = torch.randint(low=0, high=131, size=(1, 100), dtype=torch.long)
seq_len = torch.LongTensor([seq.size(1)])
scales_t = torch.FloatTensor([0.667, 1.0, 0.8])
torch.onnx.export(model, (seq, seq_len, scales_t), out_onnx, opset_version=15,
    input_names=["input", "input_lengths", "scales"], output_names=["z"],
    dynamic_axes={"input": {0: "b", 1: "ph"}, "input_lengths": {0: "b"}, "z": {0: "b", 2: "t"}})
print("front exportiert:", out_onnx)

import onnx
m = onnx.load(out_onnx)
print("front outputs:", [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])

# --- front deterministisch (noise=0) auf ids0 laufen -> z[1,192,T] ---
import onnxruntime as ort
ids = np.load(ids_path)
sess = ort.InferenceSession(out_onnx, providers=["CPUExecutionProvider"])
inames = [i.name for i in sess.get_inputs()]
feed = {inames[0]: ids, inames[1]: np.array([ids.shape[1]], dtype=np.int64),
        inames[2]: np.array([0.0, 1.0, 0.0], dtype=np.float32)}
z = sess.run(None, feed)[0]           # [1,192,T]
z = np.asarray(z, dtype=np.float32)[0]  # [192,T]
print("z shape:", z.shape, "-> T =", z.shape[1])
z.tofile(out_z)                        # channel-major [192,T] = c*T+t
print("z gedumpt:", out_z, z.nbytes, "bytes")
