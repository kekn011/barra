import numpy as np, tensorflow as tf, sys, os
import whisper_ref as W
# Whisper-Kuer: Conv-Frontend als TPU-Packages (ersetzt conv-graph 717ms auf Mali).
# Monolith (conv1+conv2) kompiliert NICHT (rc=2), conv2 direkt (k3 s2 @M3000) auch nicht ->
#   conv1: mel [1,3000,1,nmel] -> k=3 s=1 pad1 + GELU -> [1,3000,1,1280] (16x8)
#   conv2w: im2col-Form — Glue baut Fenster [x(2t-1)|x(2t)|x(2t+1)] = [1,1500,1,3840],
#           Package = 1x1-CONV 3840->1280 + GELU (16x8, proj-Klasse; Wflat[k*D+cin,cout]).
# Naht = CPU-Requant int16(c1osc)->int16(c2isc) im Fenster-Gather. PC-Ketten-Check vs float.
#   python gen_conv.py <ggml-bin> <test.wav> <outdir>

model, wav, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
hp, filt, t = W.parse_ggml(model)
D = hp["n_audio_state"]; S = 1500; T = 3000; M = hp["n_mels"]
print("D=%d n_mels=%d" % (D, M), flush=True)

mel = W.log_mel(wav, filt)
mel4 = mel.T.reshape(1, T, 1, M).astype(np.float32)
c1f = W.gelu(W.conv1d(mel, t["encoder.conv1.weight"], t["encoder.conv1.bias"], 1, 1))   # [1280,3000]
c2f = W.gelu(W.conv1d(c1f, t["encoder.conv2.weight"], t["encoder.conv2.bias"], 2, 1))
ref = c2f.T.astype(np.float32)                                                          # [1500,1280]
c1f4 = c1f.T.reshape(1, T, 1, D).astype(np.float32)

W1 = tf.constant(t["encoder.conv1.weight"].transpose(2, 1, 0).reshape(3, 1, M, D).astype(np.float32))
b1 = tf.constant(t["encoder.conv1.bias"].astype(np.float32))
Wflat = tf.constant(t["encoder.conv2.weight"].transpose(2, 1, 0).reshape(1, 1, 3*D, D).astype(np.float32))
b2 = tf.constant(t["encoder.conv2.bias"].astype(np.float32))

def windows_np(c1t):
    # c1t [3000, D] zeitmajor -> [1500, 3*D]: [x(2t-1)|x(2t)|x(2t+1)], t=0 links 0-Pad
    w = np.zeros((S, 3*D), c1t.dtype)
    w[1:, 0:D] = c1t[1:2*S-2:2]
    w[0, 0:D] = 0
    w[:, D:2*D] = c1t[0:2*S:2]
    w[:, 2*D:3*D] = c1t[1:2*S:2]
    return w

@tf.function(input_signature=[tf.TensorSpec([1, T, 1, M], tf.float32)])
def conv1_fn(x):
    x = tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]])
    return tf.nn.gelu(tf.nn.conv2d(x, W1, [1, 1, 1, 1], "VALID") + b1, approximate=False)

@tf.function(input_signature=[tf.TensorSpec([1, S, 1, 3*D], tf.float32)])
def conv2w_fn(x):
    return tf.nn.gelu(tf.nn.conv2d(x, Wflat, [1, 1, 1, 1], "VALID") + b2, approximate=False)

win = windows_np(c1f.T.astype(np.float32)).reshape(1, S, 1, 3*D)
rng = np.random.default_rng(31)
def rep1():
    yield [mel4]
    for sf in (0.9, 1.0, 1.1):
        for _ in range(3):
            yield [(mel4*sf + rng.standard_normal(mel4.shape).astype(np.float32)*0.05*mel4.std()).astype(np.float32)]
def rep2():
    yield [win]
    for sf in (0.9, 1.0, 1.1):
        for _ in range(3):
            yield [(win*sf + rng.standard_normal(win.shape).astype(np.float32)*0.05*win.std()).astype(np.float32)]

def convert(fn, rep, name):
    c = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()])
    c.optimizations = [tf.lite.Optimize.DEFAULT]; c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    c.inference_input_type = tf.int16; c.inference_output_type = tf.int16
    buf = c.convert()
    open(os.path.join(outdir, name + ".tflite"), "wb").write(buf)
    print("WROTE", name, len(buf), flush=True)
    return buf

def qp(buf):
    it = tf.lite.Interpreter(model_content=buf)
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    return (float(di["quantization"][0]), int(di["quantization"][1]),
            float(do["quantization"][0]), int(do["quantization"][1]))

def run_i16(buf, xq, outshape):
    it = tf.lite.Interpreter(model_content=buf); it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(di["index"], xq.reshape(di["shape"])); it.invoke()
    return it.get_tensor(do["index"]).reshape(outshape)

b1u = convert(conv1_fn, rep1, "conv1")
b2u = convert(conv2w_fn, rep2, "conv2w")
i1, z1, o1, zo1 = qp(b1u); i2, z2, o2, zo2 = qp(b2u)

# Ketten-Check wie der Treiber: mel->int16 -> conv1 -> Fenster-Gather mit Requant (o1->i2) -> conv2w
xq = np.clip(np.round(mel4/i1 + z1), -32768, 32767).astype(np.int16)
y1 = run_i16(b1u, xq, (T, D)).astype(np.float32)
y1r = np.clip(np.round((y1 - zo1)*(o1/i2) + z2), -32768, 32767).astype(np.int16)
wq = windows_np(y1r); wq[0, 0:D] = z2
y2 = run_i16(b2u, wq, (S, D)).astype(np.float32)
y = (y2 - zo2)*o2
cos = float((y*ref).sum())/(np.linalg.norm(y)*np.linalg.norm(ref)+1e-12)
rel = float(np.linalg.norm(y-ref))/(np.linalg.norm(ref)+1e-12)
print("CHECK conv-kette: cos=%.6f rel=%.4f" % (cos, rel), flush=True)

with open(os.path.join(outdir, "conv_params.txt"), "w") as f:
    f.write("conv=2\nC_nmel=%d\nC1_isc=%r\nC1_izp=%d\nC1_osc=%r\nC1_ozp=%d\nC2_isc=%r\nC2_izp=%d\nC2_osc=%r\nC2_ozp=%d\n"
            % (M, i1, z1, o1, zo1, i2, z2, o2, zo2))
print("DONE", flush=True)
