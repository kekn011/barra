#!/usr/bin/env python3
"""wespeaker_r34_build.py — GENERISCH: eine wespeaker-ResNet34-Variante (beliebige Sprache)
fuer die TPU vorbereiten. PC-Teil von barra-emb-port: fixieren, Trunk extrahieren ('343'),
Kopf dumpen (seg_1 + mean_vec), onnx2tf, 16x8-Quant, PC-Verify, Node-Artefakte.

  python3 wespeaker_r34_build.py <wespeaker_r34.onnx> <name> [outdir]

Schreibt nach <outdir> (Default tpu-toolchain/pyannote):
  <name>_trunk_16x8.tflite  head_<name>.bin  <name>_verify_in.bin  <name>_verify_ref.f32
Grenzen: NUR die wespeaker-ResNet34-Familie (plain-Conv-Trunk, Tensor '343', model.seg_1-Kopf).
Andere Architekturen (eres2net/TitaNet/...) sind eigene Rezepte — siehe pyannote-Memory/README.
"""
import numpy as np, onnx, wave, struct, os, sys, subprocess
from onnx import utils as outils
import kaldi_native_fbank as knf
import onnxruntime as ort

SRC = sys.argv[1]; NAME = sys.argv[2]
# Ausgabeverzeichnis via argv[3] oder $BARRA_OUT, Default cwd-relativ (nicht mehr maschinenspezifisch)
D = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("BARRA_OUT", os.path.abspath("tpu-toolchain/pyannote"))

m = onnx.load(SRC)
tens = {v.name for v in m.graph.value_info} | {i.name for i in m.graph.initializer}
import onnx.numpy_helper as nh
inits = {i.name: nh.to_array(i) for i in m.graph.initializer}
if "model.seg_1.weight" not in inits or "mean_vec" not in inits:
    print("FEHLER: keine wespeaker-ResNet34-Struktur (model.seg_1/mean_vec fehlen) — diese "
          "Familie deckt barra-emb-port ab; andere Architekturen sind eigene Etappen."); sys.exit(1)

for d, v in zip(m.graph.input[0].type.tensor_type.shape.dim, (1, 300, 80)):
    d.dim_param = ""; d.dim_value = v
fixed = f"{D}/{NAME}_fixed.onnx"
onnx.save(m, fixed)
try:
    outils.extract_model(fixed, f"{D}/{NAME}_trunk.onnx", ["feats"], ["343"])
except Exception as e:
    print("FEHLER: Trunk-Schnitt '343' nicht moeglich (%s) — kein Standard-r34-Graph." % e); sys.exit(1)
print("Trunk extrahiert")

W = inits["model.seg_1.weight"].astype(np.float32)
b = inits["model.seg_1.bias"].astype(np.float32)
mv = inits["mean_vec"].astype(np.float32)
EMB, STAT = W.shape[0], W.shape[1] // 2
print("Kopf:", W.shape, "-> emb", EMB, "stat", STAT)

def wav_fbank(path):
    w = wave.open(path); pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32); w.close()
    o = knf.FbankOptions(); o.frame_opts.samp_freq = 16000; o.frame_opts.dither = 0.0; o.mel_opts.num_bins = 80
    f = knf.OnlineFbank(o); f.accept_waveform(16000, pcm.tolist()); f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)
fz = wav_fbank(f"{D}/0-four-speakers-zh.wav")
fe = wav_fbank(f"{D}/1-two-speakers-en.wav")
cal = [fz[s:s+300] for s in range(0, fz.shape[0]-300, 400)][:10] + \
      [fe[s:s+300] for s in range(0, fe.shape[0]-300, 400)][:4]
print(f"Kalibrier-Chunks: {len(cal)} (zh+en, kanonische WAVs)")

sm = f"{D}/{NAME}_trunk_tf"
if not os.path.isdir(sm):
    r = subprocess.run([sys.executable, "-m", "onnx2tf", "-i", f"{D}/{NAME}_trunk.onnx",
                        "-o", sm, "-fdosm"], capture_output=True, text=True)
    if not os.path.isdir(sm):
        print("onnx2tf FEHLGESCHLAGEN:", r.stderr[-800:]); sys.exit(1)
print("SavedModel da")
import tensorflow as tf
def rep():
    for c in cal:
        yield [c.T[None].astype(np.float32)]   # TRANSPONIEREN, nie reshapen (Layout-Falle!)
c = tf.lite.TFLiteConverter.from_saved_model(sm)
c.optimizations = [tf.lite.Optimize.DEFAULT]; c.representative_dataset = rep
c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
c.inference_input_type = tf.int16; c.inference_output_type = tf.int16
buf = c.convert()
open(f"{D}/{NAME}_trunk_16x8.tflite", "wb").write(buf)
it = tf.lite.Interpreter(model_content=buf); it.allocate_tensors()
di, do = it.get_input_details()[0], it.get_output_details()[0]
ISC, _ = di["quantization"]; OSC, _ = do["quantization"]
print(f"16x8: ISC={ISC!r} OSC={OSC!r} in={di['shape']} out={do['shape']}")

full = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
def head(fm):
    N = fm.shape[2]; mn = fm.mean(axis=2); v = fm.var(axis=2)*N/(N-1)
    x = np.concatenate([mn.ravel(), np.sqrt(v+1e-8).ravel()]).astype(np.float64)
    return W.astype(np.float64) @ x + b - mv
def cosd(a, bb):
    a, bb = np.asarray(a,np.float64).ravel(), np.asarray(bb,np.float64).ravel()
    return float(a@bb/(np.linalg.norm(a)*np.linalg.norm(bb)))
starts = list(range(0, fz.shape[0]-300, 300))[:8]
cs = []
for i, s0 in enumerate(starts):
    ch = fz[s0:s0+300]
    ef = full.run(None, {"feats": ch[None]})[0].ravel()
    xq = np.clip(np.round(ch.T/ISC), -32768, 32767).astype(np.int16)
    it.set_tensor(di["index"], xq[None]); it.invoke()
    fm = it.get_tensor(do["index"])[0].astype(np.float64)*OSC
    cs.append(cosd(head(fm), ef))
    if i == 0:
        xq[None].tofile(f"{D}/{NAME}_verify_in.bin")
        (fm.astype(np.float32)).tofile(f"{D}/{NAME}_verify_ref.f32")
print("cos(Sim-Emb, ONNX-Emb) je Chunk:", " ".join(f"{c:.4f}" for c in cs))
if min(cs) < 0.98:
    print("WARNUNG: Quantqualitaet unter r34-Niveau (min cos %.4f) — Ergebnis pruefen!" % min(cs))

with open(f"{D}/head_{NAME}.bin", "wb") as f:
    f.write(struct.pack("<III", 0x42454831, EMB, STAT))
    f.write(struct.pack("<dd", float(ISC), float(OSC)))
    f.write(W.tobytes()); f.write(b.tobytes()); f.write(mv.tobytes())
print("head_%s.bin: %d B  OSC fuer Verify: %r" % (NAME, os.path.getsize(f"{D}/head_{NAME}.bin"), float(OSC)))
print("R34_BUILD_DONE", NAME)
