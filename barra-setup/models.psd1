# models.psd1 — Herkunfts- und Lizenzmanifest aller Modelldateien.
#
# EINZIGE WAHRHEIT fuer drei Verbraucher:
#   1. fetch-models.ps1   laedt origin='upstream' gepinnt und prueft den SHA-256
#   2. barra-setup.ps1    zeigt Lizenz + Quelle auf der Modellkarte
#   3. mk-model-docs.ps1  erzeugt docs/models.md daraus
#
# origin:
#   'upstream'  Fremdmodell. barra liefert es NICHT mit, sondern laedt es beim Einrichten
#               von der Originalquelle — wie das Google-Factory-Image (fetch-stock.ps1).
#               Braucht: url, sha256, license, licenseUrl.
#   'barra'     unser eigenes Build-Ergebnis (TPU-Packages, Kit-Tars, abgeleitete Float-Tails).
#               Wird als GitHub-Release-Asset veroeffentlicht, Apache-2.0 wie der Rest.
#
# gated = $true: Quelle verlangt Login/Zustimmung, anonymer Download unmoeglich
#               -> der Wizard weist darauf hin, fetch-models.ps1 -Supply nimmt die Datei entgegen.
#
# ARBEITSSTAND: sha256 und bytes sind gemessen (26.8., aus den lokalen Dateien, vollstaendig).
# Lizenzen 26.8. an der Quelle abgelesen (HF-Karten-Metadatum bzw. Projekt-Repo), nicht geraten.
# HF-URLs zeigen auf eine feste REVISION statt auf main: 'resolve/main' ist beweglich - genau
# daran ist qwen3-4b gescheitert, und unsere TPU-Packages sind auf die konkrete Datei kalibriert.
# urlCandidate = Vermutung, noch nicht bestaetigt. Bestaetigung ist mechanisch:
#   .\fetch-models.ps1 -Verify <id>   laedt den Kandidaten und vergleicht gegen den gepinnten
#   SHA-256. Stimmt er, ist die Quelle bewiesen; stimmt er nicht, war es die falsche Datei.

@{
  # ============================== Auslieferung ==============================
  # Woher die Dateien kommen, die zu gross fuers Repo sind. base LEER = es gibt noch
  # keine Quelle; fetch-payload.ps1 sagt dann, was zu tun ist, statt stumm zu scheitern.
  # Eintragen, sobald das Release steht: 'https://github.com/kekn011/barra/releases/download/<tag>'
  # Beide Zeilen zieht release-please beim Versionswechsel selbst nach (extra-files in
  # release-please-config.json). Je Zeile genau EINE Versionsangabe - sonst weiss der
  # generische Ersetzer nicht, welche gemeint ist.
  '_release' = @{
    base='https://github.com/kekn011/barra/releases/download/v0.1.0' # x-release-please-version
    tag='v0.1.0' # x-release-please-version
  }

  # ---------- Kleinteile, die der Kit-Lauf ebenfalls uebertraegt ----------
  # Sie sind nicht im Repo (die Kit-Ordner sind ausgeignoriert) und hatten bisher keine
  # Bezugsquelle - der Kit-Lauf waere daran genauso gescheitert wie am ersten Paket.
  'pya-eres-params' = @{ kit='pya'; file='eres_params.txt'; origin='barra'; bytes=128
    sha256='f52abbb7b4b0949d6197ca3ff636615726bf2b358fedd1224bd0933575617837' }
  'pya-tita-params' = @{ kit='pya'; file='tita_params.txt'; origin='barra'; bytes=503
    sha256='481a599f3585a762625a60985e30ef0335b1a6265ab9efda9e40b5997f11ab31' }
  'tts-service' = @{ kit='tts'; file='barra-tts.service'; origin='barra'; bytes=305
    sha256='1e319014adb83d62511bcc7c6e8b8fc8274705d97fadf6304466433d7948331b' }
  'tts-server' = @{ kit='tts'; file='ttsserver.sh'; origin='barra'; bytes=2466
    sha256='332b320c195caa05d2ed3a2a1e676b70d43fa380b9ba91b985c2f4eefe76379b' }
  'stt-server' = @{ kit='stt'; file='sttserver.sh'; origin='barra'; bytes=5535
    sha256='261a85987f9680f55ddc97721c08eceb4b4f96f4ad4de6b4e16d5a151d5c511d' }
  'dev-kit' = @{ kit='dev'; file='barra-dev-kit.tar.gz'; origin='barra'; bytes=24877591
    sha256='a7fb69c4e9ec4088c41ae5d9bd90ea74edfcaadc32d74811831778a1d7a2349f' }

  # --- Payload: das, was der Flash zwingend braucht ---
  # Die Pruefsummenliste selbst: sie kommt aus dem Release wie die Dateien, die sie beschreibt.
  # Bewusst OHNE eigenen sha256 - sie ist die Referenz, nicht der Prueffall.
  'payload-sums' = @{ kit='payload'; file='SHA256SUMS'; origin='barra'; bytes=326
    license='Apache-2.0'; licenseUrl='https://github.com/kekn011/barra/blob/main/LICENSE' }
  'payload-base' = @{ kit='payload'; file='barra-base.tar.gz'; origin='barra'; bytes=429630384
    sha256='834e8901ec2b84435ad0fe01d10ff93782e1466762856a285fc64458b03d22f4'
    license='Apache-2.0'; licenseUrl='https://github.com/kekn011/barra/blob/main/LICENSE' }
  'payload-boot' = @{ kit='payload'; file='boot-lz4.img'; origin='barra'; bytes=53477376
    sha256='adb51190ffa908a12dd78e16e48a1478488f4d228dd32a8fc894ffd11145a667'
    license='Apache-2.0'; licenseUrl='https://github.com/kekn011/barra/blob/main/LICENSE' }
  # rfkill.ko ist mit dem Signaturschluessel DIESES Kernel-Builds signiert (GKI 'protected exports'):
  # es gehoert zum Kernel, nicht zum Base-Image. Ein rfkill.ko aus einem anderen Build wird vom
  # Kernel abgelehnt -> bcmdhd laedt nicht -> Node ohne WLAN (30.8. erlebt).
  'payload-rfkill' = @{ kit='payload'; file='wifi-rfkill.ko'; origin='barra'; bytes=63817
    sha256='31af0598543616971e5a9e77395705e6c4fd0249dcb3595db14d7c00ecc267c6'
    license='GPL-2.0'; licenseUrl='https://github.com/kekn011/barra/tree/main/kernel' }
  'magisk' = @{
    kit='payload'; file='Magisk-30.7.apk'; origin='upstream'; bytes=11613864
    sha256='e0d32d2123532860f97123d927b1bb86c4e08e6fd8a48bfc6b5bee0afae9ebd5'
    url='https://github.com/topjohnwu/Magisk/releases/download/v30.7/Magisk-v30.7.apk'
    urlSource='26.8. per -Verify bewiesen: geladene Datei SHA-gleich zur lokalen'
    license='GPL-3.0'; licenseUrl='https://github.com/topjohnwu/Magisk'; gated=$false
    note='wird beim Ablegen zu Magisk-30.7.apk umbenannt (patch-initboot.ps1 sucht Magisk-*.apk)' }

  # ============================== LLM ==============================
  'qwen3-4b' = @{
    kit='llm'; file='qwen3-4b.gguf'; origin='barra'; bytes=2497281120
    sha256='3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597'
    # Qwen3-4B steht unter Apache-2.0, wir DUERFEN die Datei also selbst weitergeben - und
    # tun es hier bewusst: die Datei hinter Qwen/Qwen3-4B-GGUF resolve/main wurde seit August
    # neu hochgeladen (26.8. geprueft, anderer Hash), und unsere TPU-Attention-Packages sind
    # gegen GENAU DIESE Fassung kalibriert. Aus dem Release bekommt der Nutzer die passende.
    #
    # 2,33 GB sprengen GitHubs Grenze fuer Release-Assets (2 GB je Datei). Deshalb liegt die
    # Datei geteilt im Release; fetch-models.ps1 laedt beide Teile, prueft jeden einzeln und
    # setzt sie zusammen - das Ergebnis wird gegen den sha256 oben geprueft (26.8. verifiziert:
    # part0+part1 ergeben bitgleich das Original). Teile erzeugen:
    #   split -b 1300M -d -a 1 qwen3-4b.gguf qwen3-4b.gguf.part
    parts=@(
      @{ file='qwen3-4b.gguf.part0'; bytes=1363148800; sha256='f14ec25c14ec2ddbbab03274346ae557f2a6ad2cdf3d70d88445b5c0f5f0af5d' }
      @{ file='qwen3-4b.gguf.part1'; bytes=1134132320; sha256='313ba6e9d71b76ed2de6ea8805874fb69afe42bf3b9a08891fcc6b9217a63647' }
    )
  }
    # AUSGESCHLOSSEN 26.8.: Qwen/Qwen3-4B-GGUF resolve/main/Qwen3-4B-Q4_K_M.gguf -> 7485fe6f11af2943...
    # Die GROESSE passt (2,5 GB angezeigt = unsere 2.497.281.120 B), der Hash nicht. Wahrscheinlichste
    # Erklaerung: die Datei hinter 'resolve/main' wurde seit August neu hochgeladen.
    # FOLGE, die groesser ist als dieser eine Eintrag: unsere TPU-Attention-Packages sind gegen eine
    # BESTIMMTE Modelldatei kalibriert. Ein abweichender Upstream kann den Offload verstimmen.
    # Deshalb sollten alle HF-Pins auf eine REVISION zeigen (resolve/<commit>/...), nicht auf main.
  'qwen2.5-1.5b' = @{
    kit='llm'; file='qwen2.5-1.5b.gguf'; origin='upstream'; bytes=1117320736
    sha256='6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
    url='https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf'; urlSource='dl-models.ps1, Sitzung 22.8. - Quelle dokumentiert, SHA-Pruefung trotzdem noetig (resolve/main ist beweglich)'
    license='Apache-2.0'; licenseUrl='https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF'; gated=$false
  }
  'glm-edge-4b' = @{
    kit='llm'; file='glm-edge-4b-chat.gguf'; origin='upstream'; bytes=2627488704
    sha256='8868cf22ae605bb294e40bfcb8924d94368e8c9f8741c4aedf21ecece7b1bdb1'
    url='https://huggingface.co/zai-org/glm-edge-4b-chat-gguf/resolve/e168d9e95cf9af8bfbe54e04df7123516a51415d/ggml-model-Q4_K_M.gguf'; urlSource='dl-models.ps1, Sitzung 22.8. - Quelle dokumentiert, SHA-Pruefung trotzdem noetig (resolve/main ist beweglich)'
    license='GLM-4 License (kein OSS-Standard)'; licenseUrl='https://huggingface.co/zai-org/glm-edge-4b-chat-gguf/blob/main/LICENSE'; gated=$false
    licenseNote='Karten-Metadatum ''glm-4'', eigener Lizenztext im Repo - NICHT als Apache/MIT behandeln.'
  }
  'qwen38-distill' = @{
    kit='llm'; file='qwen38-4b-distill.gguf'; origin='upstream'; bytes=2783446304
    sha256='dec96e8cf2e11b613bb46513dec485377f9ca5a351e71712ee0e244f287c6790'
    url='https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF/resolve/391fc7d103e3942a408def3e4f51c2f85d464417/Qwen3.8-4B-Q4_K_M.gguf'; urlSource='dl-models.ps1, Sitzung 22.8. - Quelle dokumentiert, SHA-Pruefung trotzdem noetig (resolve/main ist beweglich)'
    license='Apache-2.0'; licenseUrl='https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF'; gated=$false
  }
  'gemma-e2b' = @{
    kit='llm'; file='gemma-4-e2b-q4_0.gguf'; origin='barra'; bytes=3349516256
    sha256='fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634'
    # Gemma 4 steht unter Apache-2.0 (ai.google.dev/gemma/apache_2), wir duerfen die Datei also
    # selbst weitergeben. Ohne eine Bezugsquelle wuerde der Wizard das Modell gar nicht anbieten.
    # >2 GB -> geteilt im Release (split -b 1600M -d -a 1 <datei> <datei>.part).
    parts=@(
      @{ file='gemma-4-e2b-q4_0.gguf.part0'; bytes=1677721600; sha256='b22610c7e559ace590f93ff98e419a646bc80241ffcc62ab868003708cd16d3e' }
      @{ file='gemma-4-e2b-q4_0.gguf.part1'; bytes=1671794656; sha256='a0f97bff296bfb6663653819dd348b99ea010661721cb698574d9698f8b81557' }
    )
  }
    # LIZENZ GEPRUEFT 26.8. (ai.google.dev/gemma/terms): Gemma 1/2/3/3n stehen unter den
    # 'Gemma Terms of Use', GEMMA 4 dagegen unter Apache-2.0 ('For Gemma 4 terms, see the
    # Gemma 4 license'). Die Angabe in der GUI ist also RICHTIG — mein Auditbefund B3 war falsch.
    # gated war ebenfalls nur meine Annahme und ist zurueckgenommen; noch nicht am Upstream geprueft.
  'gemma-e4b' = @{
    kit='llm'; file='gemma-4-e4b-q3_k_s.gguf'; origin='barra'; bytes=3862379648
    sha256='fe6e02cdcd5f3287b4ac146ea68ea2c73363bf935159aa254a9d280a4ffe31aa'
    parts=@(
      @{ file='gemma-4-e4b-q3_k_s.gguf.part0'; bytes=1677721600; sha256='fb12fd5e2cffa948bce2ce1a9bc1a8586fbd547206f44dc039a584e438178c8c' }
      @{ file='gemma-4-e4b-q3_k_s.gguf.part1'; bytes=1677721600; sha256='c6d15a6f7be6051931491fd5295ca48a54004d043682c03527fbebc0a2519a52' }
      @{ file='gemma-4-e4b-q3_k_s.gguf.part2'; bytes=506936448;  sha256='833eeb198cd7a8208cbefc8d98ae1ff83995c0283fc68e0843a78f4247711808' }
    )
  }
    # LIZENZ GEPRUEFT 26.8. (ai.google.dev/gemma/terms): Gemma 1/2/3/3n stehen unter den
    # 'Gemma Terms of Use', GEMMA 4 dagegen unter Apache-2.0 ('For Gemma 4 terms, see the
    # Gemma 4 license'). Die Angabe in der GUI ist also RICHTIG — mein Auditbefund B3 war falsch.
    # gated war ebenfalls nur meine Annahme und ist zurueckgenommen; noch nicht am Upstream geprueft.

  # --- unsere TPU-Attention-Packages (Release-Assets) ---
  'attn-qwen3-4b'    = @{ kit='llm'; file='llm-attn-qwen3-4b.tar';        origin='barra'; bytes=955381760; sha256='5accb80125d8b88b02600522d512a0a92c06a530ca3d7d9a73b953b80928bf36' }
  'attn-qwen2.5-1.5b'= @{ kit='llm'; file='llm-attn-qwen2.5-1.5b.tar';    origin='barra'; bytes=160204800; sha256='5c9e82058523aaf0d2f48c710480d34ddfbf094e374128832bf31998a88ec8a3' }
  'attn-glm-edge-4b' = @{ kit='llm'; file='llm-attn-glm-edge-4b-chat.tar';origin='barra'; bytes=953866240; sha256='0ed2a137d4c8028467433bdea3a93db0defe3ba3073873987d9b0fe53e703430' }
  'attn-gemma-e2b'   = @{ kit='llm'; file='llm-attn-gemma-4-e2b-q4_0.tar';origin='barra'; bytes=286535680; sha256='7c1423eca696fafbece97a71c4e2ad9fdfa6bfe71cb320712718231deacfb164' }

  # ============================== STT (Whisper) ==============================
  'whisper-turbo' = @{
    kit='stt'; file='ggml-large-v3-turbo-q5_0.bin'; origin='upstream'; bytes=574041195
    sha256='394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2'
    url='https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-q5_0.bin'; urlSource='26.8. per -Verify bewiesen (SHA stimmt)'
    license='MIT'; licenseUrl='https://huggingface.co/ggerganov/whisper.cpp'; gated=$false
  }
  'whisper-medium' = @{ kit='stt'; file='ggml-medium-q5_0.bin'; origin='upstream'; bytes=539212467; sha256='19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f'; url='https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-medium-q5_0.bin'; urlSource='dl-models.ps1 (Sitzung 22.8.) - Quelle dokumentiert; SHA-Pruefung trotzdem noetig, resolve/main ist beweglich'; license='MIT'; licenseUrl='https://huggingface.co/ggerganov/whisper.cpp'; gated=$false }
  'whisper-small'  = @{ kit='stt'; file='ggml-small.bin';       origin='upstream'; bytes=487601967; sha256='1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b'; url='https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.bin'; urlSource='dl-models.ps1 (Sitzung 22.8.) - Quelle dokumentiert; SHA-Pruefung trotzdem noetig, resolve/main ist beweglich'; license='MIT'; licenseUrl='https://huggingface.co/ggerganov/whisper.cpp'; gated=$false }
  'whisper-base'   = @{ kit='stt'; file='ggml-base.bin';        origin='upstream'; bytes=147951465; sha256='60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe'; url='https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.bin'; urlSource='dl-models.ps1 (Sitzung 22.8.) - Quelle dokumentiert; SHA-Pruefung trotzdem noetig, resolve/main ist beweglich'; license='MIT'; licenseUrl='https://huggingface.co/ggerganov/whisper.cpp'; gated=$false }
  'whisper-tiny'   = @{ kit='stt'; file='ggml-tiny.bin';        origin='upstream'; bytes=77691713;  sha256='be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21'; url='https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.bin'; urlSource='26.8. per -Verify BESTAETIGT; dl-models.ps1 (Sitzung 22.8.) - Quelle dokumentiert; SHA-Pruefung trotzdem noetig, resolve/main ist beweglich'; license='MIT'; licenseUrl='https://huggingface.co/ggerganov/whisper.cpp'; gated=$false }

  # --- unsere Whisper-TPU-Encoder-Packages ---
  'wsp-turbo'  = @{ kit='stt'; file='whisper-kit-turbo.tar';  origin='barra'; bytes=672225280; sha256='68e42e1e436421f87082c424ba776da84cf560aadfe11c95992ce223f0e2dbe5' }
  'wsp-medium' = @{ kit='stt'; file='whisper-kit-medium.tar'; origin='barra'; bytes=390708224; sha256='da90413214dc87f8ac69daa70138c2b8b9d00b760e13fa1b9241a1152bf0b861' }
  'wsp-small'  = @{ kit='stt'; file='whisper-kit-small.tar';  origin='barra'; bytes=159334912; sha256='1b1e02a7e631dd20e6059525e799073952cb50efdc62f6663e62867c75cf1960' }
  'wsp-base'   = @{ kit='stt'; file='whisper-kit-base.tar';   origin='barra'; bytes=32187392;  sha256='9a614ef0d68ae78b9caa6c4043cdbceaecba62af2904de73312782328cb488d7' }
  'wsp-tiny'   = @{ kit='stt'; file='whisper-kit-tiny.tar';   origin='barra'; bytes=18156032;  sha256='74fe4efec0aa8f0052afda1b808ce96394b2a8f8d5205886d157f13f45118146' }

  # ============================== Diarisierung ==============================
  # Die Upstream-ONNX sind die EINGAENGE unserer TPU-Pipeline; die .package/.bin/-tail-ONNX
  # sind unsere Ableitungen daraus.
  'eres2net-zh' = @{
    kit='pya'; file='eres2net.onnx'; origin='upstream'; bytes=39593761
    sha256='1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b'
    url='https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx'; urlSource='26.8. per -Verify bewiesen (SHA stimmt)'
    license='Apache-2.0 (3D-Speaker)'; licenseUrl='https://github.com/modelscope/3D-Speaker'; gated=$false
    licenseNote='Lizenz des Ursprungsprojekts; sherpa-onnx verteilt die Datei nur.'
  }
  'resnet34-zh' = @{
    kit='pya'; file='resnet34_zh.onnx'; origin='upstream'; bytes=26530548
    sha256='87d1d5068397f3792c730570b53d66cd8be1da7ea22dd04f5b6706d96a3cd168'
    url='https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_zh_cnceleb_resnet34_LM.onnx'; urlCandidate=''
    # GEFUNDEN 26.8.: die _LM-Variante ist es (per -Verify bewiesen).
    # Vorher ausgeschlossen (SHA-abweichend):
    #   sherpa-onnx .../wespeaker_zh_cnceleb_resnet34.onnx -> f86cd6c509f331f0...
    #   wespeaker-CoS .../cnceleb/cnceleb_resnet34.onnx     -> 78817ca21a9707ad...
    license='Apache-2.0 (wespeaker)'; licenseUrl='https://github.com/wenet-e2e/wespeaker'; gated=$false
    licenseNote='Lizenz des Ursprungsprojekts; sherpa-onnx verteilt die Datei nur (selbst Apache-2.0).'
  }
  'titanet-en' = @{
    kit='pya'; file='titanet.onnx'; origin='upstream'; bytes=40257283
    sha256='ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e'
    url='https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx'; urlSource='26.8. per -Verify bewiesen (SHA stimmt)'
    license='CC-BY-4.0 (NVIDIA NeMo)'; licenseUrl='https://huggingface.co/nvidia/speakerverification_en_titanet_large'; gated=$false
    licenseNote='ABGELESEN AN titanet_LARGE - fuer die small-Variante gibt es keine eigene Karte auf HF. Vor Publish am NGC-Eintrag bestaetigen.'
  }
  # ACHTUNG: pyannote-kit.tar enthaelt derzeit das ResNet34-en-Modell UND die
  # pyannote-Segmentierung 3.0 UND das sherpa-onnx-Binary mit. Fuer den Download-Weg
  # muss es aufgetrennt werden (Befund: Fremdmodelle in unserem Kit-Tar).
  'pya-kit'      = @{ kit='pya'; file='pyannote-kit.tar';    origin='barra'; bytes=73799680; sha256='439b4fa5e1f544b5c29ff643bf5c85a3dc218254638f553df39592c719fe5e89'; note='buendelt ResNet34-en (wespeaker) + pyannote-Segmentierung 3.0 + das sherpa-onnx-Binary — diese drei brauchen eigene Manifest-Eintraege' }
  'eres-body'    = @{ kit='pya'; file='eres_body.package';   origin='barra'; bytes=6335168;  sha256='4ecf086f38bebe41c229b95a0e745bb7c1675b2a5441d7087ac53a8cd1b2a939' }
  'eres-tail'    = @{ kit='pya'; file='eres_tail.onnx';      origin='barra'; bytes=6892052;  sha256='b93082ec06906b96822e918eedff6db427a5919bde9b76f10814da826eb4a22d' }
  'eres-head'    = @{ kit='pya'; file='head_eres.bin';       origin='barra'; bytes=20975644; sha256='4fefde3591409b53020cd5b128daa24fed13ee82230979d7a4b110a7747fb4c0' }
  'r34zh-trunk'  = @{ kit='pya'; file='r34zh_trunk.package'; origin='barra'; bytes=5572160;  sha256='03fa582b373d756b0e6e3a60e75b4b766bc1c59ae73caa760d362e3d02235456' }
  'r34zh-head'   = @{ kit='pya'; file='head_zh.bin';         origin='barra'; bytes=5244956;  sha256='d62751a0fdb7a6d80457ebf222a5f8b20fbd18f3bc39aa12f718ddda5730f272' }
  'tita-seg0'    = @{ kit='pya'; file='tita_seg0.package';   origin='barra'; bytes=64640;   sha256='28693c174a54265ca980d97360a7e7592b6423649680ab15da7944cadbec254b' }
  'tita-seg1'    = @{ kit='pya'; file='tita_seg1.package';   origin='barra'; bytes=359808;   sha256='b87cbd26aef00b2f79ad1932414fe8d60c1648c36ab861475c3e0752cbfb0ead' }
  'tita-seg2'    = @{ kit='pya'; file='tita_seg2.package';   origin='barra'; bytes=362880;   sha256='413c34737b0b19b700293b08563b565b408158991d4dc3aa7d2f19fa685df723' }
  'tita-seg3'    = @{ kit='pya'; file='tita_seg3.package';   origin='barra'; bytes=365952;   sha256='f3da4e368ffe2595606cbb370fcb8449224534060e468b3d1037d0fd02198acb' }
  'tita-seg4'    = @{ kit='pya'; file='tita_seg4.package';   origin='barra'; bytes=865280;   sha256='7a1ea1838a9e3eccab72523086b25d777bb451ce14a58adf1459a27a12d34a57' }
  'tita-tail'    = @{ kit='pya'; file='tita_tail.onnx';      origin='barra'; bytes=11135229; sha256='e1b31f1afdf48db882927ec2935205be05d2c53433b49080feaddeb8239f5d93' }
  'tita-glue'    = @{ kit='pya'; file='tita_glue.bin';       origin='barra'; bytes=9699396;   sha256='581d3f4292e043cf349f6ed445198ef1b2fcb81743509fd19400877dd6aa2352' }

  # ============================== Bild ==============================
  'img-dreamshaper' = @{
    kit='img'; file='DreamShaper8_LCM.safetensors'; origin='upstream'; bytes=2133804992
    sha256='a4f3e1526c5dc4fcbe342f5c410d83ae202c7a415fcefcbb92e0f93fcd0a87c3'
    url='https://huggingface.co/Lykon/dreamshaper-8-lcm/resolve/4645d8bc6a8e6b106d21606d63e8460cdad4f1a6/DreamShaper8_LCM.safetensors'; urlSource='26.8. per -Verify bewiesen (SHA stimmt)'
    # WICHTIG: SD-1.5-Abkoemmlinge stehen ueblicherweise unter CreativeML OpenRAIL-M.
    # Diese Lizenz enthaelt Nutzungsbeschraenkungen, die WEITERGEGEBEN werden muessen —
    # sie gehoert damit sowohl nach docs/models.md als auch auf die Modellkarte.
    license='CreativeML OpenRAIL-M'; licenseUrl='https://huggingface.co/Lykon/dreamshaper-8-lcm'; gated=$false
    licenseNote='OpenRAIL-M enthaelt Nutzungsbeschraenkungen, die WEITERGEGEBEN werden muessen - gehoert auf die Modellkarte im Wizard, nicht nur nach docs/.'
  }
  'img-taesd' = @{
    kit='img'; file='taesd.safetensors'; origin='upstream'; bytes=9793292
    sha256='db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549'
    url='https://huggingface.co/madebyollin/taesd/resolve/614f76814bbe30edbe2e627ace1c2234c81a2c0e/diffusion_pytorch_model.safetensors'; urlSource='26.8. per -Verify bewiesen (SHA stimmt); Datei wird beim Ablegen zu taesd.safetensors umbenannt'
    license='MIT'; licenseUrl='https://huggingface.co/madebyollin/taesd'; gated=$false
  }
  'img-kit' = @{ kit='img'; file='img-kit.tar.gz'; origin='barra'; bytes=65459868; sha256='4441279f532d2dbd5b6eea4931836d608dda9a4a6fc6d5141b71a362fb98525f' }

  # ============================== TTS / Weckwort ==============================
  # tts-kit.tar.gz enthaelt derzeit die geklonte Stimme "David" (kein Upstream, keine Lizenz)
  # sowie Piper Thorsten/Amy. Entscheidung 26.8.: GPU-Vokoder auf eine Piper-Stimme umziehen,
  # David faellt raus. Danach ist das Kit rein 'barra' + Piper-Downloads.
  'tts-kit'  = @{ kit='tts';  file='tts-kit.tar.gz';  origin='barra'; bytes=332932765; sha256='27495ca3aa318c1953ec805703b15beffae930238d3f68c10d3628f53b36ce60'; note='enthaelt die Piper-Stimmen Thorsten/Amy (MIT) — die gehoeren als Download ins Manifest; die geklonte Stimme David ist am 26.8. entfernt worden; gpudecd am 26.8. abend neu gebaut - die ausgelieferte Binaerdatei war aelter als die leaky-Korrektur und liess die Aktivierung vor conv_post weg (cos 0,971 statt 1,000)' }
  'wake-kit' = @{ kit='wake'; file='wake-kit.tar.gz'; origin='barra'; bytes=16921642;  sha256='9c9fe572bfda709574108d245be34625dbd17e3dc4c1b2745bc85df7a1a4e4ff'; note='buendelt das sherpa-onnx-Keyword-Spotter-Modell (int8, 5 MB) — Herkunft und Lizenz noch offen' }
}
