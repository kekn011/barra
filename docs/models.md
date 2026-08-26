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
| `glm-edge-4b-chat.gguf` | [link](https://huggingface.co/zai-org/glm-edge-4b-chat-gguf/resolve/e168d9e95cf9af8bfbe54e04df7123516a51415d/ggml-model-Q4_K_M.gguf) | [GLM-4 License (kein OSS-Standard)](https://huggingface.co/zai-org/glm-edge-4b-chat-gguf/blob/main/LICENSE) |
| `qwen2.5-1.5b.gguf` | [link](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf) | [Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) |
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

These 37 files are our build output — TPU packages, kit archives and derived
float tails — and they ship as GitHub release assets, not as downloads from third parties.

**They are not automatically Apache-2.0.** A TPU package is not an original work: it is a
third-party model's weights, re-quantized and re-laid-out for the Tensor G3. It carries the
licence of the model it came from. The table below names that model and that licence for
every file; only our own code and shaders are Apache-2.0.

| File | Kit | Derived from | License |
|---|---|---|---|
| `llm-attn-gemma-4-e2b-q4_0.tar` | LLM (KI-Chat) | Google Gemma 4 E2B | [Apache-2.0](https://ai.google.dev/gemma/apache_2) |
| `llm-attn-glm-edge-4b-chat.tar` | LLM (KI-Chat) | GLM-Edge-4B-Chat | [GLM-Edge License](https://huggingface.co/zai-org/glm-edge-4b-chat-gguf/blob/main/LICENSE) |
| `llm-attn-qwen2.5-1.5b.tar` | LLM (KI-Chat) | Qwen2.5-1.5B-Instruct | [Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) |
| `llm-attn-qwen3-4b.tar` | LLM (KI-Chat) | Qwen3-4B | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-4B) |
| `barra-dev-kit.tar.gz` | Dev-Kit | barra (eigener Code) | Apache-2.0 |
| `eres_body.package` | Sprecher-Trennung | ERes2Net (3D-Speaker) | [Apache-2.0](https://github.com/modelscope/3D-Speaker) |
| `head_eres.bin` | Sprecher-Trennung | ERes2Net (3D-Speaker) | [Apache-2.0](https://github.com/modelscope/3D-Speaker) |
| `eres_tail.onnx` | Sprecher-Trennung | ERes2Net (3D-Speaker) | [Apache-2.0](https://github.com/modelscope/3D-Speaker) |
| `gemma-4-e2b-q4_0.gguf` | LLM (KI-Chat) | Google Gemma 4 E2B (unveraendert weitergegeben) | [Apache-2.0](https://ai.google.dev/gemma/apache_2) |
| `gemma-4-e4b-q3_k_s.gguf` | LLM (KI-Chat) | Google Gemma 4 E4B (unveraendert weitergegeben) | [Apache-2.0](https://ai.google.dev/gemma/apache_2) |
| `img-kit.tar.gz` | Bildgenerator | stable-diffusion.cpp (MIT) + Mali-Kernel von barra | [Apache-2.0 + MIT](https://github.com/leejet/stable-diffusion.cpp) |
| `barra-base.tar.gz` |  | barra + Ubuntu 24.04 Userland | [Apache-2.0 + Ubuntu-Paketlizenzen](https://github.com/kekn011/barra/blob/main/LICENSE) |
| `boot-lz4.img` |  | Android GKI 6.1 (modifiziert) | [GPLv2](https://github.com/kekn011/barra/tree/main/kernel) |
| `SHA256SUMS` |  | barra (eigener Code) | Apache-2.0 |
| `eres_params.txt` | Sprecher-Trennung | barra (eigener Code) | Apache-2.0 |
| `pyannote-kit.tar` | Sprecher-Trennung | wespeaker ResNet34 + pyannote/segmentation-3.0 + sherpa-onnx | [Apache-2.0 + MIT](https://huggingface.co/pyannote/segmentation-3.0) |
| `tita_params.txt` | Sprecher-Trennung | barra (eigener Code) | Apache-2.0 |
| `qwen3-4b.gguf` | LLM (KI-Chat) | Qwen3-4B (unveraendert weitergegeben) | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-4B) |
| `head_zh.bin` | Sprecher-Trennung | wespeaker ResNet34 CN-Celeb | [Apache-2.0](https://github.com/wenet-e2e/wespeaker) |
| `r34zh_trunk.package` | Sprecher-Trennung | wespeaker ResNet34 CN-Celeb | [Apache-2.0](https://github.com/wenet-e2e/wespeaker) |
| `sttserver.sh` | Spracherkennung | barra (eigener Code) | Apache-2.0 |
| `tita_glue.bin` | Sprecher-Trennung | NVIDIA NeMo TitaNet | [CC-BY-4.0](https://huggingface.co/nvidia/speakerverification_en_titanet_large) |
| `tita_seg0.package` | Sprecher-Trennung | NVIDIA NeMo TitaNet | [CC-BY-4.0](https://huggingface.co/nvidia/speakerverification_en_titanet_large) |
| `tita_seg1.package` | Sprecher-Trennung | NVIDIA NeMo TitaNet | [CC-BY-4.0](https://huggingface.co/nvidia/speakerverification_en_titanet_large) |
| `tita_seg2.package` | Sprecher-Trennung | NVIDIA NeMo TitaNet | [CC-BY-4.0](https://huggingface.co/nvidia/speakerverification_en_titanet_large) |
| `tita_seg3.package` | Sprecher-Trennung | NVIDIA NeMo TitaNet | [CC-BY-4.0](https://huggingface.co/nvidia/speakerverification_en_titanet_large) |
| `tita_seg4.package` | Sprecher-Trennung | NVIDIA NeMo TitaNet | [CC-BY-4.0](https://huggingface.co/nvidia/speakerverification_en_titanet_large) |
| `tita_tail.onnx` | Sprecher-Trennung | NVIDIA NeMo TitaNet | [CC-BY-4.0](https://huggingface.co/nvidia/speakerverification_en_titanet_large) |
| `tts-kit.tar.gz` | Sprachausgabe | Piper-Stimmen Thorsten (de) und Amy (en) + sherpa-onnx + CPython | [MIT + Apache-2.0 + PSF](https://huggingface.co/rhasspy/piper-voices) |
| `ttsserver.sh` | Sprachausgabe | barra (eigener Code) | Apache-2.0 |
| `barra-tts.service` | Sprachausgabe | barra (eigener Code) | Apache-2.0 |
| `wake-kit.tar.gz` | Weckwort | sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 (k2-fsa) | [Apache-2.0](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models) |
| `whisper-kit-base.tar` | Spracherkennung | Whisper base | [MIT](https://huggingface.co/ggerganov/whisper.cpp) |
| `whisper-kit-medium.tar` | Spracherkennung | Whisper medium | [MIT](https://huggingface.co/ggerganov/whisper.cpp) |
| `whisper-kit-small.tar` | Spracherkennung | Whisper small | [MIT](https://huggingface.co/ggerganov/whisper.cpp) |
| `whisper-kit-tiny.tar` | Spracherkennung | Whisper tiny | [MIT](https://huggingface.co/ggerganov/whisper.cpp) |
| `whisper-kit-turbo.tar` | Spracherkennung | Whisper large-v3-turbo | [MIT](https://huggingface.co/ggerganov/whisper.cpp) |

**Auflagen, die mit der Weitergabe uebernommen werden:**

- `llm-attn-glm-edge-4b-chat.tar` — Lizenztext beilegen (docs/licenses/GLM-Edge-LICENSE.txt); "Built with GLM-Edge" nennen; abgeleitete Modellnamen mit "GLM-Edge" praefixen; keine militaerische oder rechtswidrige Nutzung; kommerzielle Nutzung erfordert Registrierung beim Anbieter.
- `barra-base.tar.gz` — Enthaelt ein vollstaendiges Ubuntu-Userland; die Lizenztexte der Pakete liegen im Image unter /usr/share/doc. KEINE Vendor-Firmware (seit 26.8. ausgeschlossen und im Bake geprueft).
- `boot-lz4.img` — GPLv2 verlangt die Quellen: sie liegen im Repo unter kernel/.
- `pyannote-kit.tar` — Buendelt Fremdmodelle; deren Lizenztexte liegen im Kit. pyannote/segmentation-3.0 ist MIT (der Download-Gate auf HuggingFace ist eine Formalie, keine Weitergabebeschraenkung).
- `qwen3-4b.gguf` — Apache-2.0: Lizenz und Namensnennung weitergeben. Wir liefern die Datei selbst aus, weil der Upstream seit August eine andere Fassung unter derselben Adresse fuehrt und unsere Packages auf diese kalibriert sind.
- `tita_glue.bin` — Namensnennung erforderlich.
- `tita_seg0.package` — Namensnennung erforderlich.
- `tita_seg1.package` — Namensnennung erforderlich.
- `tita_seg2.package` — Namensnennung erforderlich.
- `tita_seg3.package` — Namensnennung erforderlich.
- `tita_seg4.package` — Namensnennung erforderlich.
- `tita_tail.onnx` — Namensnennung erforderlich.
- `tts-kit.tar.gz` — Buendelt Fremdmodelle und eine Python-Laufzeit; deren Lizenztexte liegen im Kit.
- `wake-kit.tar.gz` — Herkunft am 26.8. per SHA-256 bewiesen: encoder.int8.onnx ist bitgleich zur Datei im Release-Archiv.


## Known gaps

Some of our kit archives still bundle third-party models. Those are **not** covered by the
table above and have to be split out before a release, so they can be fetched and licensed
like everything else:

- `Magisk-30.7.apk` — wird beim Ablegen zu Magisk-30.7.apk umbenannt (patch-initboot.ps1 sucht Magisk-*.apk)
- `pyannote-kit.tar` — buendelt ResNet34-en (wespeaker) + pyannote-Segmentierung 3.0 + das sherpa-onnx-Binary — diese drei brauchen eigene Manifest-Eintraege
- `tts-kit.tar.gz` — enthaelt die Piper-Stimmen Thorsten/Amy (MIT) — die gehoeren als Download ins Manifest; die geklonte Stimme David ist am 26.8. entfernt worden; gpudecd am 26.8. abend neu gebaut - die ausgelieferte Binaerdatei war aelter als die leaky-Korrektur und liess die Aktivierung vor conv_post weg (cos 0,971 statt 1,000)
- `wake-kit.tar.gz` — buendelt das sherpa-onnx-Keyword-Spotter-Modell (int8, 5 MB) — Herkunft und Lizenz noch offen

