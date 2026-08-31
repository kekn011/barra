# --- ggml-rpc (Machbarkeitstest 30.8.): rpc-server auf der Android-Seite ---------------------
# Ein llama.cpp-rpc-server (Bionic, ggml-vulkan mit Mali-Patches + ggml-barra) lauscht auf
# 0.0.0.0:50052 (Node-IP, damit Pods anderer Nodes ihn erreichen). Ein glibc-Client im Container/Pod (llama-cli/llama-server --rpc 127.0.0.1:50052)
# laedt das Modell, schickt die Gewichte einmal (Tensor-Hash-Cache unter LLAMA_CACHE) und je Token
# nur den Graphen + Embedding/Logits; die GPU-Arbeit bleibt hier, wo die Vulkan-Blobs sind.
# Einbau: Inhalt dieser Datei in /data/adb/hwbridge/hw-bridges.sh vor der while-Schleife einfuegen
# und `start_rpc` in die Schleife aufnehmen (der Supervisor liest die Datei nur beim Start).
start_rpc(){
  pgrep -x ggml-rpc-server >/dev/null 2>&1 && return 0
  L=/data/adb/baseos/llm
  [ -x "$L/ggml-rpc-server" ] || return 1
  mkdir -p /data/adb/baseos/rpc-cache
  log "starte rpc-server (ggml-rpc, 127.0.0.1:50052, Cache /data/adb/baseos/rpc-cache)"
  ( . "$L/env.sh"; LLAMA_CACHE=/data/adb/baseos/rpc-cache \
    "$L/ggml-rpc-server" -H 0.0.0.0 -p 50052 -d Vulkan0 -c >>"$LOG" 2>&1 ) &
}
