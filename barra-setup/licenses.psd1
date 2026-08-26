# licenses.psd1 — Lizenzkette der ABGELEITETEN Artefakte.
#
# Ein TPU-Package ist keine eigene Schoepfung, sondern sind die umgerechneten Gewichte eines
# fremden Modells. Es erbt dessen Lizenz. models.psd1 sagt, WOHER eine Datei kommt; diese
# Datei sagt, unter WELCHEN BEDINGUNGEN wir sie weitergeben duerfen.
#
# Schluessel = ID aus models.psd1. Wird von mk-model-docs.ps1 in docs/models.md gemischt.
# Alle Angaben am 26.8.2026 an der Quelle abgelesen (Modellkarte bzw. Projekt-Repo).

@{
  # ---------------- LLM: TPU-Attention-Packages ----------------
  'attn-qwen3-4b' = @{
    derivedFrom='Qwen3-4B'; license='Apache-2.0'
    licenseUrl='https://huggingface.co/Qwen/Qwen3-4B' }
  'attn-qwen2.5-1.5b' = @{
    derivedFrom='Qwen2.5-1.5B-Instruct'; license='Apache-2.0'
    licenseUrl='https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF' }
  'attn-gemma-e2b' = @{
    derivedFrom='Google Gemma 4 E2B'; license='Apache-2.0'
    licenseUrl='https://ai.google.dev/gemma/apache_2' }
  'attn-glm-edge-4b' = @{
    derivedFrom='GLM-Edge-4B-Chat'; license='GLM-Edge License'
    licenseUrl='https://huggingface.co/zai-org/glm-edge-4b-chat-gguf/blob/main/LICENSE'
    obligations='Lizenztext beilegen (docs/licenses/GLM-Edge-LICENSE.txt); "Built with GLM-Edge" nennen; abgeleitete Modellnamen mit "GLM-Edge" praefixen; keine militaerische oder rechtswidrige Nutzung; kommerzielle Nutzung erfordert Registrierung beim Anbieter.' }

  # ---------------- Spracherkennung: TPU-Encoder-Packages ----------------
  'wsp-turbo'  = @{ derivedFrom='Whisper large-v3-turbo'; license='MIT'; licenseUrl='https://huggingface.co/ggerganov/whisper.cpp' }
  'wsp-medium' = @{ derivedFrom='Whisper medium';         license='MIT'; licenseUrl='https://huggingface.co/ggerganov/whisper.cpp' }
  'wsp-small'  = @{ derivedFrom='Whisper small';          license='MIT'; licenseUrl='https://huggingface.co/ggerganov/whisper.cpp' }
  'wsp-base'   = @{ derivedFrom='Whisper base';           license='MIT'; licenseUrl='https://huggingface.co/ggerganov/whisper.cpp' }
  'wsp-tiny'   = @{ derivedFrom='Whisper tiny';           license='MIT'; licenseUrl='https://huggingface.co/ggerganov/whisper.cpp' }

  # ---------------- Sprecher-Trennung ----------------
  'eres-body'   = @{ derivedFrom='ERes2Net (3D-Speaker)'; license='Apache-2.0'; licenseUrl='https://github.com/modelscope/3D-Speaker' }
  'eres-tail'   = @{ derivedFrom='ERes2Net (3D-Speaker)'; license='Apache-2.0'; licenseUrl='https://github.com/modelscope/3D-Speaker' }
  'eres-head'   = @{ derivedFrom='ERes2Net (3D-Speaker)'; license='Apache-2.0'; licenseUrl='https://github.com/modelscope/3D-Speaker' }
  'r34zh-trunk' = @{ derivedFrom='wespeaker ResNet34 CN-Celeb'; license='Apache-2.0'; licenseUrl='https://github.com/wenet-e2e/wespeaker' }
  'r34zh-head'  = @{ derivedFrom='wespeaker ResNet34 CN-Celeb'; license='Apache-2.0'; licenseUrl='https://github.com/wenet-e2e/wespeaker' }
  'tita-seg0'   = @{ derivedFrom='NVIDIA NeMo TitaNet'; license='CC-BY-4.0'; licenseUrl='https://huggingface.co/nvidia/speakerverification_en_titanet_large'; obligations='Namensnennung erforderlich.' }
  'tita-seg1'   = @{ derivedFrom='NVIDIA NeMo TitaNet'; license='CC-BY-4.0'; licenseUrl='https://huggingface.co/nvidia/speakerverification_en_titanet_large'; obligations='Namensnennung erforderlich.' }
  'tita-seg2'   = @{ derivedFrom='NVIDIA NeMo TitaNet'; license='CC-BY-4.0'; licenseUrl='https://huggingface.co/nvidia/speakerverification_en_titanet_large'; obligations='Namensnennung erforderlich.' }
  'tita-seg3'   = @{ derivedFrom='NVIDIA NeMo TitaNet'; license='CC-BY-4.0'; licenseUrl='https://huggingface.co/nvidia/speakerverification_en_titanet_large'; obligations='Namensnennung erforderlich.' }
  'tita-seg4'   = @{ derivedFrom='NVIDIA NeMo TitaNet'; license='CC-BY-4.0'; licenseUrl='https://huggingface.co/nvidia/speakerverification_en_titanet_large'; obligations='Namensnennung erforderlich.' }
  'tita-tail'   = @{ derivedFrom='NVIDIA NeMo TitaNet'; license='CC-BY-4.0'; licenseUrl='https://huggingface.co/nvidia/speakerverification_en_titanet_large'; obligations='Namensnennung erforderlich.' }
  'tita-glue'   = @{ derivedFrom='NVIDIA NeMo TitaNet'; license='CC-BY-4.0'; licenseUrl='https://huggingface.co/nvidia/speakerverification_en_titanet_large'; obligations='Namensnennung erforderlich.' }
  'pya-kit'     = @{
    derivedFrom='wespeaker ResNet34 + pyannote/segmentation-3.0 + sherpa-onnx'
    license='Apache-2.0 + MIT'; licenseUrl='https://huggingface.co/pyannote/segmentation-3.0'
    obligations='Buendelt Fremdmodelle; deren Lizenztexte liegen im Kit. pyannote/segmentation-3.0 ist MIT (der Download-Gate auf HuggingFace ist eine Formalie, keine Weitergabebeschraenkung).' }

  # ---------------- Sprachausgabe / Weckwort / Bild ----------------
  'tts-kit'  = @{
    derivedFrom='Piper-Stimmen Thorsten (de) und Amy (en) + sherpa-onnx + CPython'
    license='MIT + Apache-2.0 + PSF'; licenseUrl='https://huggingface.co/rhasspy/piper-voices'
    obligations='Buendelt Fremdmodelle und eine Python-Laufzeit; deren Lizenztexte liegen im Kit.' }
  'wake-kit' = @{
    derivedFrom='sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 (k2-fsa)'
    license='Apache-2.0'; licenseUrl='https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models'
    obligations='Herkunft am 26.8. per SHA-256 bewiesen: encoder.int8.onnx ist bitgleich zur Datei im Release-Archiv.' }
  'img-kit'  = @{
    derivedFrom='stable-diffusion.cpp (MIT) + Mali-Kernel von barra'
    license='Apache-2.0 + MIT'; licenseUrl='https://github.com/leejet/stable-diffusion.cpp' }

  # ---------------- Payload ----------------
  'payload-base' = @{
    derivedFrom='barra + Ubuntu 24.04 Userland'
    license='Apache-2.0 + Ubuntu-Paketlizenzen'; licenseUrl='https://github.com/kekn011/barra/blob/main/LICENSE'
    obligations='Enthaelt ein vollstaendiges Ubuntu-Userland; die Lizenztexte der Pakete liegen im Image unter /usr/share/doc. KEINE Vendor-Firmware (seit 26.8. ausgeschlossen und im Bake geprueft).' }
  'payload-boot' = @{
    derivedFrom='Android GKI 6.1 (modifiziert)'
    license='GPLv2'; licenseUrl='https://github.com/kekn011/barra/tree/main/kernel'
    obligations='GPLv2 verlangt die Quellen: sie liegen im Repo unter kernel/.' }
  'qwen3-4b' = @{
    derivedFrom='Qwen3-4B (unveraendert weitergegeben)'; license='Apache-2.0'
    licenseUrl='https://huggingface.co/Qwen/Qwen3-4B'
    obligations='Apache-2.0: Lizenz und Namensnennung weitergeben. Wir liefern die Datei selbst aus, weil der Upstream seit August eine andere Fassung unter derselben Adresse fuehrt und unsere Packages auf diese kalibriert sind.' }
}
