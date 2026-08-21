#!/bin/bash
# Mali-llama.cpp auf einen Node installieren (Image-Pfad /data/adb/baseos/llm) — vom PC aus, WSL oder Git-Bash (MSYS_NO_PATHCONV=1).
#   ./install-node.sh [adb-serial] [build-bin-dir]
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
SER=${1:-}; [ -n "$SER" ] && ADB="adb -s $SER" || ADB=adb
BIN=${2:-$HOME/llama.cpp/build-android-vulkan-idp/stage}
case "$(uname -s)" in MINGW*|MSYS*) BIN=${2:-//wsl.localhost/Ubuntu-24.04/home/kevin/llama.cpp/build-android-vulkan-idp/stage}; SRC=$(cygpath -w "$SRC");; esac
T=/data/local/tmp/llm-stage
$ADB shell "mkdir -p $T"
for f in llama-server llama-bench llama-perplexity llama-quantize llama-cli barra-llm run-llm.sh \
         libggml-base.so libggml-cpu.so libggml-vulkan.so libggml-barra.so libggml.so libllama.so libllama-common.so libmtmd.so \
         libllama-server-impl.so libllama-bench-impl.so libllama-perplexity-impl.so libllama-quantize-impl.so libllama-cli-impl.so; do
  $ADB push "$BIN/$f" $T/ >/dev/null && echo "push $f"
done
$ADB push "$SRC/env.sh" "$SRC/llmserver.sh" "$SRC/llm-bench.sh" $T/ >/dev/null
$ADB shell "su -c 'mkdir -p /data/adb/baseos/llm /data/adb/baseos/bin; cp $T/* /data/adb/baseos/llm/; chmod 755 /data/adb/baseos/llm/*; mv /data/adb/baseos/llm/llmserver.sh /data/adb/baseos/bin/llmserver.sh; mv /data/adb/baseos/llm/llm-bench.sh /data/adb/baseos/bin/llm-bench.sh; chmod 755 /data/adb/baseos/bin/llmserver.sh /data/adb/baseos/bin/llm-bench.sh; rm -rf $T; ls /data/adb/baseos/llm | wc -l'"
echo "installiert: /data/adb/baseos/llm (+ bin/llmserver.sh, bin/llm-bench.sh)"
