# numpy-Referenz des baj-tts HiFi-GAN-Decoders aus den gefalteten Gewichten.
# Dient (1) als Verifikation gegen ONNX, (2) als Kalibrierungs-Harness: erfasst pro Conv
# die Eingangs-Aktivierung (PRE-leaky, [Cin,T]) ueber den Kalibrierungssatz.
import json
import os

import numpy as np

LRELU = 0.1

DEC_DIR = os.environ.get("BARRA_DEC_DIR", "../baj-out")

def load(dec_dir=None):
    d = dec_dir or DEC_DIR
    W = np.load(os.path.join(d, "dec_weights.npz"))
    man = json.load(open(os.path.join(d, "dec_manifest.json")))
    def folded(prefix):
        if f"{prefix}.weight" in W:
            w = W[f"{prefix}.weight"]
        else:
            g = W[f"{prefix}.parametrizations.weight.original0"]
            v = W[f"{prefix}.parametrizations.weight.original1"]
            nrm = np.sqrt((v.astype(np.float64) ** 2).sum(axis=tuple(range(1, v.ndim)), keepdims=True))
            w = (g * (v / nrm)).astype(np.float32)
        b = W[f"{prefix}.bias"] if f"{prefix}.bias" in W else np.zeros(w.shape[-1] if False else w.shape[0], np.float32)
        return w.astype(np.float32), b.astype(np.float32)
    return W, man, folded

def lrelu(x, slope=None):
    a = LRELU if slope is None else slope
    return np.where(x >= 0, x, x * a)

def conv1d(x, w, b, dil):
    # x [Cin,T], w [Cout,Cin,K] -> [Cout,T], SAME (PyTorch pad = dil*(K-1)/2)
    Cout, Cin, K = w.shape
    pad = dil * (K - 1) // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    T = x.shape[1]
    y = np.zeros((Cout, T), np.float32)
    for k in range(K):
        y += w[:, :, k] @ xp[:, k * dil:k * dil + T]
    return y + b[:, None]

def conv_transpose1d(x, w, b, stride, pad):
    # PyTorch ConvTranspose1d: w [Cin,Cout,K]; out_len=(T-1)*stride - 2*pad + K
    Cin, Cout, K = w.shape
    T = x.shape[1]
    outlen = (T - 1) * stride + K
    # y[co, t*stride + k] += sum_ci w[ci,co,k] * x[ci,t]
    y = np.zeros((Cout, outlen), np.float32)
    for k in range(K):
        wk = w[:, :, k]                      # [Cin,Cout]
        pk = np.einsum("io,it->ot", wk, x)   # [Cout,T]
        y[:, k:k + (T - 1) * stride + 1:stride] += pk
    y = y[:, pad:pad + (T - 1) * stride - 2 * pad + K]
    return y + b[:, None]

class HiFiGAN:
    def __init__(self, capture=False, dec_dir=None):
        self.W, self.man, self.folded = load(dec_dir)
        self.rates = self.man["upsample_rates"]
        self.uks = self.man["upsample_kernel_sizes"]
        self.rks = self.man["resblock_kernel_sizes"]
        self.rds = self.man["resblock_dilation_sizes"]
        # ResBlock1 (Coqui/HiFi-GAN v1): convs1.{i} + convs2.{i} je Dilatation.
        # ResBlock2 (Piper/HiFi-GAN v2): convs.{i}, jeder ein eigener Residual-Schritt.
        self.rb_type = int(self.man.get("resblock_type", 1))
        # zwei verschiedene Steigungen: Koerper 0.1, vor conv_post 0.01 (PyTorch-Vorgabe)
        self.slope = float(self.man.get("lrelu_slope", LRELU))
        self.slope_final = float(self.man.get("lrelu_slope_final", 0.01))
        self.capture = capture
        self.caps = {}   # name -> list of [Cin,T] pre-leaky inputs

    def _cap(self, name, x):
        if self.capture:
            self.caps.setdefault(name, []).append(x.copy())

    def conv_pkg(self, name, x, prefix, dil):
        # Package = conv(lrelu(x)) + b ; erfasst PRE-leaky x
        self._cap(name, x)
        w, b = self.folded(prefix)
        return conv1d(lrelu(x, self.slope), w, b, dil)

    def resblock(self, stage, j, k, dils, x):
        rb = f"resblocks.{stage * len(self.rks) + j}"
        if self.rb_type == 2:
            # jeder Conv ist ein eigener Residual-Schritt
            for pi, d in enumerate(dils):
                xt = self.conv_pkg(f"s{stage}_rb{j}_p{pi}", x, f"{rb}.convs.{pi}", d)
                x = xt + x
            return x
        for pi, d in enumerate(dils):
            xt = self.conv_pkg(f"s{stage}_rb{j}_p{pi}_c1", x, f"{rb}.convs1.{pi}", d)
            xt = self.conv_pkg(f"s{stage}_rb{j}_p{pi}_c2", xt, f"{rb}.convs2.{pi}", 1)
            x = xt + x
        return x

    def forward(self, z):
        # z [192,T]
        w, b = self.folded("conv_pre")
        self._cap("pre", z)
        o = conv1d(z, w, b, 1)          # conv_pre hat KEIN pre-leaky
        for i, (r, uk) in enumerate(zip(self.rates, self.uks)):
            self._cap(f"up{i}", o)
            wu, bu = self.folded(f"ups.{i}")
            o = conv_transpose1d(lrelu(o, self.slope), wu, bu, r, (uk - r) // 2)
            zs = None
            for j, (k, dils) in enumerate(zip(self.rks, self.rds)):
                rb = self.resblock(i, j, k, dils, o)
                zs = rb if zs is None else zs + rb
            o = zs / len(self.rks)
        self._cap("post", o)
        wp, bp = self.folded("conv_post")
        o = conv1d(lrelu(o, self.slope_final), wp, bp, 1)
        return np.tanh(o[0])

if __name__ == "__main__":
    import sys
    hg = HiFiGAN()
    d = np.load("../baj-out/dec_calib.npz")
    for i in range(3):
        z = d[f"z{i}"].astype(np.float32)
        ref = d[f"w{i}"].astype(np.float32)
        y = hg.forward(z)
        n = min(len(ref), len(y))
        cos = float((y[:n] * ref[:n]).sum() / (np.linalg.norm(y[:n]) * np.linalg.norm(ref[:n]) + 1e-12))
        print(f"z{i}: numpy {len(y)} vs onnx {len(ref)}  cos={cos:.6f}")
