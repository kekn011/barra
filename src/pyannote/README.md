# pyannote-Diarization mit TPU-Embeddings (M4)

Sprecher-Diarization auf dem Node: sherpa-onnx (Segmentierung + Clustering auf der CPU),
Speaker-Embeddings (wespeaker-ResNet34) auf der Tensor-G3-TPU via libbarra.

## Bausteine
- `speaker-embedding-extractor-barra-impl.h` — neue Extractor-Impl fuer sherpa-onnx.
  Nutzt sherpa's eigene kaldi-fbank-Pipeline (normalize_samples=false, kein CMN),
  schickt 300-Frame-Fenster als int16 an den tpud (r34_trunk.package, TPZ2 Zero-Copy),
  akkumuliert mean/var ueber die Zeitspalten aller Fenster (Bessel) und rechnet den
  FC-Kopf in float. Segmente <3s: zyklisches Frame-Pad; >3s: 300er-Kacheln + Endanker.
- `barra.c/h` (aus src/experiments/gpu-attn/) — kompiliert unverändert unter glibc.
- head.bin — Kopfgewichte + Quant-Skalen (Generator: gen-head.py im Scratchpad-Muster;
  Quelle r34_head.npz): u32 magic 0x42454831, u32 dim=256, u32 stat=2560,
  f64 in_scale, f64 out_scale, f32 W[256*5120], f32 b[256], f32 mv[256].

## sherpa-onnx-Patch (3 Stellen, idempotent — patch-sherpa.py-Muster)
1. barra.c, barra.h, speaker-embedding-extractor-barra-impl.h nach sherpa-onnx/csrc/.
2. speaker-embedding-extractor-impl.cc: Include + Weiche am Anfang von
   `SpeakerEmbeddingExtractorImpl::Create(config)`:
   `if (getenv("BARRA_EMB")[0]=='1') return make_unique<...BarraImpl>(config);`
   (ONNX-Embedding-Modell wird dann NICHT geladen — nur der Pfad muss existieren.)
3. csrc/CMakeLists.txt: `barra.c` in die sources-Liste.

## Betrieb
- tpud (bionic) mit dem Trunk-Package auf eigenem Socket-Verzeichnis:
  `TPU_CPU=8 TPU_WARMUP=2 tpud_pipe4 /data/local/ubuntu/opt/hwbridge/pya/tpu.sock r34_trunk.package`
  (TPU_WARMUP gegen den Kaltstart-/Lazy-Init-Befund, siehe pyannote-Memory.)
- Diarization im Container:
  `BARRA_EMB=1 BARRA_SOCK_DIR=/opt/hwbridge/pya BARRA_EMB_HEAD=/root/pya/head.bin \
   sherpa-onnx-offline-speaker-diarization --clustering.num-clusters=K \
   --segmentation.pyannote-model=seg.onnx --embedding.model=resnet34.onnx wav`
- Ohne BARRA_EMB laeuft der unveraenderte CPU-Pfad (Vergleichsreferenz).
