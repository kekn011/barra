# Vokoder-Werkzeugkette (VITS-Decoder → GPU)

Diese vier Dateien erzeugen aus einem VITS-Checkpoint den GPU-Vokoder-Satz, den
`gpudecd` zur Laufzeit laedt. Sie lagen bis zum 26.8.2026 **nur im Sitzungs-Scratchpad**
und nicht im Repo — der GPU-Vokoder war damit eine Blackbox, die sich weder nachbauen
noch auf eine andere Stimme umziehen liess. Das ist behoben.

## Die Kette

```
VITS-Checkpoint (.pth + config.json)
   │
   │  dump-dec.py          Decoder-Gewichte + dec_manifest.json herausloesen
   ▼
baj-out/
   │
   │  hifigan_np.py        laedt sie als numpy; liefert load() -> (W, manifest, folded)
   │  subpixel_mod.py      rechnet Upsample-Convs in Sub-Pixel-Form um
   │
   │  gen-gpudec2.py       erzeugt den Satz fuer gpudecd:
   ▼                         program2.json  (Op-Liste: conv/shuffle/mrf, Offsets, Kacheln)
gpukit2/                     weights16.bin  (alle W-Matrizen f16, gepaddet, ein Blob)
                             bias16.bin     (alle Bias f16, ein Blob)
```

Parallel dazu erzeugt `../export-front.py` aus demselben Checkpoint das
**Front-ONNX** (alles bis zum Decoder, dessen Ausgabe `z[192,T]` ist). Zur Laufzeit
laeuft das Front auf der CPU (onnxruntime), `z` geht per Socket an `gpudecd`, und der
Decoder rechnet auf der Mali.

## Was modellabhaengig ist — und was nicht

`gen-gpudec2.py` ist **architektur-getrieben**: es liest `upsample_rates`,
`upsample_kernel_sizes`, `resblock_kernel_sizes`, `resblock_dilation_sizes` und
`upsample_initial_channel` aus dem Manifest und baut das Programm daraus. Ein anderer
HiFi-GAN-Decoder mit anderen Raten ist also grundsaetzlich abgedeckt, ohne die Shader
anzufassen.

Fest verdrahtet sind dagegen:

- **`192` in `gpudecd.c`** — die VITS-`inter_channels`, also die Kanalzahl von `z`.
  Bei Standard-VITS (auch Piper) ist das ebenfalls 192; ein Modell mit anderem Wert
  braucht eine Anpassung im Daemon.
- **`dump-dec.py` ist Coqui-TTS-spezifisch** (`from TTS.tts.models.vits import Vits`).
  Fuer Piper-Stimmen, die ein eigenes Checkpoint-Format haben, muss dieser eine Schritt
  neu geschrieben werden — der Rest der Kette bleibt.

Das ist der Stand fuer den geplanten Umzug des GPU-Vokoders auf eine Piper-Stimme:
machbar, aber `dump-dec.py` ist die Arbeit daran, nicht der Generator.

## Voraussetzungen

`torch`, `numpy`, `onnx`, `onnxruntime` und (nur fuer `dump-dec.py`/`export-front.py`)
`TTS` (Coqui). Laeuft auf dem PC, nicht auf dem Node.
