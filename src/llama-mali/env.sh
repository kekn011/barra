# Laufzeit-Umgebung fuer das Mali-llama.cpp im Base (source /data/adb/baseos/llm/env.sh)
export LLM=/data/adb/baseos/llm
export LD_LIBRARY_PATH=$LLM:/system/lib64:/vendor/lib64
# Mali-Voreinstellungen stecken im Patch (coopmat aus, q4_K/q6_K-Mali-Kernel an). Schalter fuer Vergleiche:
#   GGML_VK_DISABLE_MMVQ=1 (alter Pfad), GGML_VK_ARM_NO_Q6K=1 (nur q4_K-Kernel), GGML_VK_ARM_COOPMAT=1 (coopmat an)
# Regeln: Kontext IMMER begrenzen (-c; Qwen3-4B hat 40k = 6 GB KV-Cache), Modelle unter /data/local/ubuntu/home/<user>/models/
