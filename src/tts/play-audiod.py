#!/usr/bin/env python3
# Spielt eine WAV ueber die audiod-alsa-Bruecke (audio.sock): resample -> 48kHz stereo S16LE.
# Aufruf: python play-audiod.py <wav> [audio.sock]
import sys, wave, socket
import numpy as np

wav_path = sys.argv[1]
sock = sys.argv[2] if len(sys.argv) > 2 else "/opt/hwbridge/audio.sock"

w = wave.open(wav_path, "rb")
ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
raw = w.readframes(n); w.close()
assert sw == 2, "nur S16"
a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
a = a.reshape(-1, ch).mean(axis=1) if ch > 1 else a   # -> mono

# resample sr -> 48000 (lineare Interpolation)
tgt = 48000
if sr != tgt:
    m = int(round(len(a) * tgt / sr))
    xi = np.linspace(0, len(a) - 1, m)
    a = np.interp(xi, np.arange(len(a)), a)

st = np.clip(a, -32768, 32767).astype(np.int16)
stereo = np.empty((len(st), 2), dtype=np.int16); stereo[:, 0] = st; stereo[:, 1] = st
pcm = stereo.tobytes()

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock)
off, CH = 0, 48000 * 2 * 2 // 10   # 0,1s Bloecke
while off < len(pcm):
    off += s.send(pcm[off:off + CH])
s.close()
print(f"gespielt: {wav_path} ({sr}Hz {ch}ch -> 48k stereo, {len(a)/tgt:.2f}s)")
