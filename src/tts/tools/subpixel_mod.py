# Verifiziert: ConvTranspose1d(stride r, kernel K, pad) == regulaerer Conv Cin->Cout*r
# (Phasen s=0..r-1, Tap j mit j*r+s<K, liest x[q-j]) + Pixel-Shuffle + Crop.
import numpy as np

from hifigan_np import conv_transpose1d, load

def subpixel_weights(w, r, K):
    # w [Cin,Cout,K] -> W_reg [Cout*r, Cin, J] mit Ausgangsindex co*r+s, W_reg[co*r+s, ci, j]=w[ci,co,j*r+s]
    Cin, Cout, K_ = w.shape
    assert K_ == K
    J = (K + r - 1) // r                      # max Taps ueber alle Phasen
    Wreg = np.zeros((Cout * r, Cin, J), np.float32)
    for s in range(r):
        for j in range(J):
            k = j * r + s
            if k < K:
                Wreg[np.arange(Cout) * r + s, :, j] = w[:, :, k].T   # [Cout,Cin]
    return Wreg, J

def subpixel_forward(x, Wreg, b, r, J, K, pad):
    # x [Cin,T]; regulaerer Conv: out[co*r+s, q] = sum_ci sum_j Wreg[.,ci,j]*x[ci,q-j], q=0..T-1+ (J-1)
    Cin, T = x.shape
    Ce = Wreg.shape[0]; Cout = Ce // r
    Q = T + (J - 1)
    xp = np.pad(x, ((0, 0), (J - 1, J - 1)))              # beidseitig J-1 (Tap j liest x[q-j], q bis Q-1)
    out = np.zeros((Ce, Q), np.float32)
    for j in range(J):
        # x[q-j] = xp[:, (J-1) + q - j], q=0..Q-1
        out += Wreg[:, :, j] @ xp[:, (J - 1) - j:(J - 1) - j + Q]
    # Pixel-Shuffle: out[co*r+s, q] -> y_full[co, q*r+s]
    yf = out.reshape(Cout, r, Q).transpose(0, 2, 1).reshape(Cout, Q * r)
    yf = yf + b[:, None]
    # y_full echte Laenge (T-1)*r+K; crop pad
    full = (T - 1) * r + K
    yf = yf[:, :full]
    outT = full - 2 * pad
    return yf[:, pad:pad + outT]

if __name__ == "__main__":
    # Selbsttest: Sub-Pixel-Umschreibung gegen die ConvTranspose-Referenz.
    #   python subpixel_mod.py <dec-dir>
    import sys
    W, man, folded = load(sys.argv[1] if len(sys.argv) > 1 else None)
    rates = man["upsample_rates"]; uks = man["upsample_kernel_sizes"]
    worst = 1.0
    for i in range(len(rates)):          # echte Stufenzahl, nicht fest 4
        w, b = folded(f"ups.{i}")
        r = rates[i]; K = uks[i]; pad = (K - r) // 2
        Cin = w.shape[0]
        x = np.random.default_rng(i).standard_normal((Cin, 200)).astype(np.float32) * 0.5
        ref = conv_transpose1d(x, w, b, r, pad)
        Wreg, J = subpixel_weights(w, r, K)
        y = subpixel_forward(x, Wreg, b, r, J, K, pad)
        n = min(ref.shape[1], y.shape[1])
        cos = float((y[:, :n] * ref[:, :n]).sum()
                    / (np.linalg.norm(y[:, :n]) * np.linalg.norm(ref[:, :n]) + 1e-12))
        worst = min(worst, cos)
        print(f"up{i} r{r} K{K} J{J} Cout*r={Wreg.shape[0]}: reflen {ref.shape[1]} ylen {y.shape[1]} cos={cos:.6f}")
    print("schlechtester Kosinus: %.6f -> %s" % (worst, "OK" if worst > 0.9999 else "ABWEICHUNG"))
