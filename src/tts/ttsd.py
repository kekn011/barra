#!/usr/bin/env python3
# barra TTS-Worker (Container): HTTP GET/POST /say?voice=<..>&text=<..> -> Synthese -> audiod-Lautsprecher.
# Stimmen datengetrieben aus $TTS_ROOT/voices/*/voice.json:
#   piper-gpu  espeak-ng+tokens -> front-onnx (ort warm) -> z -> gpudecd (GPU, warm) -> wav
#   coqui-gpu  derselbe Weg, IDs aus dem Coqui-Frontend
#   piper      sherpa-onnx-offline-tts (Rueckfall ohne GPU). Ausgabe: 48k stereo S16 an audio.sock.
# Env: TTS_ROOT (Default /opt/barra-tts), TTS_PORT (8095), TTS_GPUDEC_SOCK, TTS_AUDIO_SOCK, TTS_SHERPA.
import os, sys, json, time, socket, struct, subprocess, tempfile, wave, threading, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

ROOT   = os.environ.get("TTS_ROOT", "/opt/barra-tts")
PORT   = int(os.environ.get("TTS_PORT", "8095"))
GPUSOCK= os.environ.get("TTS_GPUDEC_SOCK", "/tmp/barra-tts/gpudec.sock")
AUDSOCK= os.environ.get("TTS_AUDIO_SOCK", "/opt/hwbridge/audio.sock")
SHERPA = os.environ.get("TTS_SHERPA", os.path.join(ROOT, "sherpa", "sherpa-onnx-offline-tts"))
VOICES = {}
_lock = threading.Lock()   # eine Synthese zur Zeit (GPU/audiod seriell)

def log(*a): print("[ttsd]", *a, file=sys.stderr, flush=True)

# ---------- Coqui-VITS ueber den GPU-Vokoder ----------
# (hiess frueher 'David'; diese Stimme ist entfallen, der Weg bleibt fuer eigene Coqui-Modelle)
class CoquiGpu:
    def __init__(self, d, cfg):
        import onnxruntime as ort
        sys.path.insert(0, cfg["site"])
        from bajtts.frontend import BajFrontend
        self.fe = BajFrontend(os.path.join(d, cfg["tokens"]))
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(d, cfg["front_onnx"]), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inames = [i.name for i in self.sess.get_inputs()]
        self.scales = np.array(cfg.get("scales", [0.667, 1.0, 0.8]), dtype=np.float32)
        self.sr = cfg.get("sample_rate", 22050)
    def synth(self, text):
        ids = np.asarray(self.fe.text_to_ids(text), dtype=np.int64)[None]
        feed = {self.inames[0]: ids, self.inames[1]: np.array([ids.shape[1]], dtype=np.int64),
                self.inames[2]: self.scales}
        z = np.asarray(self.sess.run(None, feed)[0], dtype=np.float32)[0]   # [192,T]
        T = z.shape[1]
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(GPUSOCK)
        s.sendall(struct.pack("i", T)); s.sendall(np.ascontiguousarray(z).tobytes())
        n = struct.unpack("i", _recvn(s, 4))[0]
        wav = np.frombuffer(_recvn(s, n * 4), dtype=np.float32); s.close()
        return wav, self.sr

# ---------- Piper ueber den GPU-Vokoder ----------
# Gleicher Weg wie David (Front-ONNX -> z -> gpudecd), aber die IDs kommen aus
# espeak-ng + tokens.txt statt aus dem Coqui-Frontend.
class PiperGpu:
    def __init__(self, d, cfg):
        import onnxruntime as ort
        sys.path.insert(0, os.path.join(ROOT, "bin"))
        from piper_ids import load_tokens, to_ids
        self._to_ids = to_ids
        self.tokens = load_tokens(os.path.join(d, cfg["tokens"]))
        self.espeak_voice = cfg.get("espeak_voice", "de")
        self.espeak = cfg.get("espeak", "espeak-ng")
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(d, cfg["front_onnx"]), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inames = [i.name for i in self.sess.get_inputs()]
        self.scales = np.array(cfg.get("scales", [0.667, 1.0, 0.8]), dtype=np.float32)
        self.sr = cfg.get("sample_rate", 22050)
        # eigener Socket je Stimme - siehe launch.sh
        self.sock = cfg.get("gpu_sock") or GPUSOCK
    def synth(self, text):
        ids, unk = self._to_ids(text, self.tokens, voice=self.espeak_voice, espeak=self.espeak)
        if unk:
            log("WARNUNG: Phoneme ohne Token uebersprungen:", sorted(set(unk)))
        ids = np.asarray(ids, dtype=np.int64)[None]
        feed = {self.inames[0]: ids, self.inames[1]: np.array([ids.shape[1]], dtype=np.int64),
                self.inames[2]: self.scales}
        z = np.asarray(self.sess.run(None, feed)[0], dtype=np.float32)
        while z.ndim > 2: z = z[0]
        T = z.shape[1]
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(self.sock)
        s.sendall(struct.pack("i", T)); s.sendall(np.ascontiguousarray(z).tobytes())
        n = struct.unpack("i", _recvn(s, 4))[0]
        wav = np.frombuffer(_recvn(s, n * 4), dtype=np.float32); s.close()
        return wav, self.sr

# ---------- Piper (sherpa) ----------
class Piper:
    def __init__(self, d, cfg):
        self.model = os.path.join(d, cfg["model"]); self.tokens = os.path.join(d, cfg["tokens"])
        self.data = os.path.join(d, cfg.get("data_dir", "espeak-ng-data"))
        self.sr = cfg.get("sample_rate", 22050); self.sid = cfg.get("sid", 0)
    def synth(self, text):
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tf.close()
        cmd = [SHERPA, f"--vits-model={self.model}", f"--vits-tokens={self.tokens}",
               f"--vits-data-dir={self.data}", f"--sid={self.sid}", "--num-threads=2",
               f"--output-filename={tf.name}", text]
        subprocess.run(cmd, capture_output=True)
        w = wave.open(tf.name, "rb"); n = w.getnframes(); sr = w.getframerate()
        a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0; w.close()
        os.unlink(tf.name)
        return a, sr

def _recvn(s, n):
    b = b""
    while len(b) < n:
        r = s.recv(min(n - len(b), 1 << 16))
        if not r: raise IOError("short read")
        b += r
    return b

def play_audiod(wav, sr):
    tgt = 48000
    a = wav
    if sr != tgt:
        m = int(round(len(a) * tgt / sr))
        a = np.interp(np.linspace(0, len(a) - 1, m), np.arange(len(a)), a)
    st = np.clip(a * 32767.0, -32768, 32767).astype(np.int16)
    stereo = np.empty((len(st), 2), dtype=np.int16); stereo[:, 0] = st; stereo[:, 1] = st
    pcm = stereo.tobytes()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(AUDSOCK)
    off, CH = 0, tgt * 2 * 2 // 10
    while off < len(pcm): off += s.send(pcm[off:off + CH])
    s.close()

def load_voices():
    vd = os.path.join(ROOT, "voices")
    for name in sorted(os.listdir(vd)):
        d = os.path.join(vd, name); cfgp = os.path.join(d, "voice.json")
        if not os.path.isfile(cfgp): continue
        cfg = json.load(open(cfgp))
        try:
            e = cfg["engine"]
            eng = CoquiGpu(d, cfg) if e == "coqui-gpu" else PiperGpu(d, cfg) if e == "piper-gpu" else Piper(d, cfg)
            VOICES[name] = eng; log("Stimme geladen:", name, f"({cfg['engine']})")
        except Exception as e:
            log("Stimme FEHLER", name, e)

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == "/health": return self._send(200, {"ok": True, "voices": list(VOICES)})
        if u.path == "/voices": return self._send(200, {"voices": list(VOICES)})
        if u.path == "/say":
            voice = (q.get("voice") or [next(iter(VOICES), "")])[0]
            text = (q.get("text") or [""])[0]
            play = (q.get("play") or ["1"])[0] != "0"
            if voice not in VOICES: return self._send(404, {"error": f"voice {voice} unbekannt", "voices": list(VOICES)})
            if not text.strip(): return self._send(400, {"error": "text leer"})
            try:
                with _lock:
                    t0 = time.time(); wav, sr = VOICES[voice].synth(text); t_s = time.time() - t0
                    dur = len(wav) / sr if sr else 0
                    if dur <= 0:
                        return self._send(500, {"error": "Synthese ergab kein Audio (leere Ausgabe?)"})
                    if play: play_audiod(wav, sr)
                return self._send(200, {"ok": True, "voice": voice, "audio_s": round(dur, 2),
                                        "synth_s": round(t_s, 2), "rtf": round(t_s / dur, 3)})
            except Exception as e:
                log("say FEHLER", e); return self._send(500, {"error": str(e)})
        self._send(404, {"error": "unbekannt"})
    do_POST = do_GET

if __name__ == "__main__":
    load_voices()
    if not VOICES: log("WARNUNG: keine Stimmen geladen aus", ROOT)
    log(f"bereit auf :{PORT}  Stimmen={list(VOICES)}")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
