# barra-llm — llama.cpp explizit auf die Pixel-Chips verteilt (Stand 16.8.2026)

**Was läuft (verifiziert auf barra-5599):** Qwen2.5-1.5B-Instruct Q4_K_M; Layer-Mathe auf der **GPU** (llama.cpp-Vulkan, `-ngl 99`, ~15 t/s decode) + **CPU**; **Token-Auswahl auf dem GXP-DSP** (greedy Argmax über 151 936 Logits, Kernel `src/dsp-kernels/k_argmax.S` via gxpd/barra) — jeder Token gegen CPU-Argmax geprüft: **0 Abweichungen**, kohärenter deutscher Text.

| Lauf | DSP-Argmax | Anmerkung |
|---|---|---|
| run2 | 95 ms/Token | Default-DSP-Puffer: speicherlatenz-gebunden (~0,8 µs/Element, auch `sum`) |
| run3 | **21 ms/Token** | gxpd mit `GXPD_CACHE=1` (`GxpCapi_BufferOptions_SetCacheability(1)`), 6× schneller; Fixkosten ~5 ms (Transport+CreateBuffer) |

Ehrlich: der DSP wählt den Token korrekt, kostet aber ~21 ms je Token (CPU-Argmax: 0,1 ms) → Gesamtrate ~11,7 statt 15,5 t/s. Beweis für „4 Chips, ein Modell“, kein Speed-Gewinn. Weitere Hebel: Argmax nur über Top-K-Vorauswahl (dann entscheidet der DSP nur noch unter Kandidaten), Kernel-Unroll, TCM.

**Bau/Deploy:** `build-android.sh` (WSL: NDK-clang++ gegen `~/llama.cpp/build-android-vulkan`-Statiklibs + `src/barra/barra.c`), Binary → `/data/local/tmp/llm/barra-llm` (Android-Seite, Bionic, braucht adb-root), Modell im Container-Home (`/data/local/ubuntu/home/barra/*.gguf`, per scp ohne root). Start: `run-llm.sh <name> -n 64 -ngl 99 --sampler dsp "Prompt"` (detached, `BARRA_SOCK_DIR=/data/local/ubuntu/opt/hwbridge` = Android-Sicht auf die Sockets). Fallen: adb reißt unter GPU-Last ab → immer `setsid` + Log pollen; Vulkan synchronisiert erst bei `llama_get_logits_ith` (Timer dahinter setzen).

**Nächste Phasen:** (2) TPU-Layer mit eingebackenen Gewichten (`tpu-toolchain/tf_layer.tflite` = kompletter Decoder-Layer in Qwen-1.5B-Form x[1,1536] + KV-Cache 12×256×128, kompilierte auf der TPU) → ggml-barra-Backend mit TPU-Gerät; (3) Placement-Messung GPU/TPU/CPU per `--override-tensor`.

## LLM-Chat-Server (16.8.) — testweise, Single-User
Optimale Pipeline fuer EINEN Chat-Stream = **GPU-Vulkan, alle Layer** (Messung: heterogen bringt fuer 1 Stream nichts). Aufsatz:
- **`llama-server`** (llama.cpp, GPU-Vulkan, aus build-android-vulkan, gestrippt 38 MB) laeuft Android-seitig auf `/data/local/tmp/llm/llama-server`, Modell Qwen2.5-1.5B Q4_K_M, `--host 0.0.0.0 --port 8080 -c 4096`, 4 Slots. Kontrolle: `sh /data/adb/hwbridge/llmserver.sh {start|stop|status}` (`src/boot/llmserver.sh`, braucht root/adb).
- **`chat`** = `src/barra-llm/barra-chat.c` (Terminal-Client, Roh-TCP+HTTP+SSE-Streaming, KEINE Libs, gcc im Container) → installiert nach `/usr/local/bin/chat` im Rootfs. Fuehrt History, `/reset` leert, `/bye` beendet.
- Container teilt Android-Netzwerk-NS → `127.0.0.1:8080` aus dem Container erreichbar; Port 8080 auch von aussen (Node-IP).
- **Nutzung:** `ssh barra@<node-ip>` → `chat`. Alternativ Web-UI im Browser: `http://<node-ip>:8080`.
- Offen: kein Boot-Autostart (testweise, manueller Start via adb); Server-Neustart braucht adb-root (Container-barra hat kein su).
