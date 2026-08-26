# Models and their licenses

barra does **not** redistribute third-party models. The setup fetches each one from its
original source, pinned to a specific revision and verified against a SHA-256, after you
accept the terms — the same way it fetches Google's factory image.

Fetch them with `barra-setup\fetch-models.ps1`; `-List` shows what is present.

> This file is generated from `barra-setup/models.psd1` by `mk-model-docs.ps1`.
> Edit the manifest, not this file.

## Third-party models (downloaded)

### LLM (KI-Chat)

| File | Source | License |
|---|---|---|
| `gemma-4-e2b-q4_0.gguf` | **not yet pinned** | [Apache-2.0](https://ai.google.dev/gemma/apache_2) |
| `gemma-4-e4b-q3_k_s.gguf` | **not yet pinned** | [Apache-2.0](https://ai.google.dev/gemma/apache_2) |
| `glm-edge-4b-chat.gguf` | [link](https://huggingface.co/zai-org/glm-edge-4b-chat-gguf/resolve/e168d9e95cf9af8bfbe54e04df7123516a51415d/ggml-model-Q4_K_M.gguf) | [GLM-4 License (kein OSS-Standard)](https://huggingface.co/zai-org/glm-edge-4b-chat-gguf/blob/main/LICENSE) |
| `qwen2.5-1.5b.gguf` | [link](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf) | [Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) |
| `qwen3-4b.gguf` | **not yet pinned** | **unknown** |
| `qwen38-4b-distill.gguf` | [link](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF/resolve/391fc7d103e3942a408def3e4f51c2f85d464417/Qwen3.8-4B-Q4_K_M.gguf) | [Apache-2.0](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF) |

- **glm-edge-4b-chat.gguf** — Karten-Metadatum 'glm-4', eigener Lizenztext im Repo - NICHT als Apache/MIT behandeln.

### Spracherkennung

| File | Source | License |
|---|---|---|
| `ggml-base.bin` | [link](https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.bin) | [MIT](https://huggingface.co/ggerganov/whisper.cpp) |
| `ggml-medium-q5_0.bin` | [link](https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-medium-q5_0.bin) | [MIT](https://huggingface.co/ggerganov/whisper.cpp) |
| `ggml-small.bin` | [link](https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.bin) | [MIT](https://huggingface.co/ggerganov/whisper.cpp) |
| `ggml-tiny.bin` | [link](https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.bin) | [MIT](https://huggingface.co/ggerganov/whisper.cpp) |
| `ggml-large-v3-turbo-q5_0.bin` | [link](https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-q5_0.bin) | [MIT](https://huggingface.co/ggerganov/whisper.cpp) |

### Sprecher-Trennung

| File | Source | License |
|---|---|---|
| `eres2net.onnx` | [link](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx) | [Apache-2.0 (3D-Speaker)](https://github.com/modelscope/3D-Speaker) |
| `resnet34_zh.onnx` | [link](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_zh_cnceleb_resnet34_LM.onnx) | [Apache-2.0 (wespeaker)](https://github.com/wenet-e2e/wespeaker) |
| `titanet.onnx` | [link](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx) | [CC-BY-4.0 (NVIDIA NeMo)](https://huggingface.co/nvidia/speakerverification_en_titanet_large) |

- **eres2net.onnx** — Lizenz des Ursprungsprojekts; sherpa-onnx verteilt die Datei nur.
- **resnet34_zh.onnx** — Lizenz des Ursprungsprojekts; sherpa-onnx verteilt die Datei nur (selbst Apache-2.0).
- **titanet.onnx** — ABGELESEN AN titanet_LARGE - fuer die small-Variante gibt es keine eigene Karte auf HF. Vor Publish am NGC-Eintrag bestaetigen.

### Bildgenerator

| File | Source | License |
|---|---|---|
| `DreamShaper8_LCM.safetensors` | [link](https://huggingface.co/Lykon/dreamshaper-8-lcm/resolve/4645d8bc6a8e6b106d21606d63e8460cdad4f1a6/DreamShaper8_LCM.safetensors) | [CreativeML OpenRAIL-M](https://huggingface.co/Lykon/dreamshaper-8-lcm) |
| `taesd.safetensors` | [link](https://huggingface.co/madebyollin/taesd/resolve/614f76814bbe30edbe2e627ace1c2234c81a2c0e/diffusion_pytorch_model.safetensors) | [MIT](https://huggingface.co/madebyollin/taesd) |

- **DreamShaper8_LCM.safetensors** — OpenRAIL-M enthaelt Nutzungsbeschraenkungen, die WEITERGEGEBEN werden muessen - gehoert auf die Modellkarte im Wizard, nicht nur nach docs/.

## barra's own artifacts

These 25 files are our build output — TPU packages, kit archives and derived
float tails. They are Apache-2.0 like the rest of the project and ship as GitHub release
assets, not as downloads from third parties.

| File | Kit |
|---|---|
| `llm-attn-gemma-4-e2b-q4_0.tar` | LLM (KI-Chat) |
| `llm-attn-glm-edge-4b-chat.tar` | LLM (KI-Chat) |
| `llm-attn-qwen2.5-1.5b.tar` | LLM (KI-Chat) |
| `llm-attn-qwen3-4b.tar` | LLM (KI-Chat) |
| `eres_body.package` | Sprecher-Trennung |
| `head_eres.bin` | Sprecher-Trennung |
| `eres_tail.onnx` | Sprecher-Trennung |
| `img-kit.tar.gz` | Bildgenerator |
| `pyannote-kit.tar` | Sprecher-Trennung |
| `head_zh.bin` | Sprecher-Trennung |
| `r34zh_trunk.package` | Sprecher-Trennung |
| `tita_glue.bin` | Sprecher-Trennung |
| `tita_seg0.package` | Sprecher-Trennung |
| `tita_seg1.package` | Sprecher-Trennung |
| `tita_seg2.package` | Sprecher-Trennung |
| `tita_seg3.package` | Sprecher-Trennung |
| `tita_seg4.package` | Sprecher-Trennung |
| `tita_tail.onnx` | Sprecher-Trennung |
| `tts-kit.tar.gz` | Sprachausgabe |
| `wake-kit.tar.gz` | Weckwort |
| `whisper-kit-base.tar` | Spracherkennung |
| `whisper-kit-medium.tar` | Spracherkennung |
| `whisper-kit-small.tar` | Spracherkennung |
| `whisper-kit-tiny.tar` | Spracherkennung |
| `whisper-kit-turbo.tar` | Spracherkennung |

## Known gaps

Some of our kit archives still bundle third-party models. Those are **not** covered by the
table above and have to be split out before a release, so they can be fetched and licensed
like everything else:

- `pyannote-kit.tar` — buendelt ResNet34-en (wespeaker) + pyannote-Segmentierung 3.0 + das sherpa-onnx-Binary — diese drei brauchen eigene Manifest-Eintraege
- `tts-kit.tar.gz` — enthaelt die Piper-Stimmen Thorsten/Amy (MIT) — die gehoeren als Download ins Manifest; die geklonte Stimme David ist am 26.8. entfernt worden
- `wake-kit.tar.gz` — buendelt das sherpa-onnx-Keyword-Spotter-Modell (int8, 5 MB) — Herkunft und Lizenz noch offen

