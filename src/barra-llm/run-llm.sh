#!/system/bin/sh
# run-llm.sh <logname> [barra-llm-Argumente...] — barra-llm (LLM ueber GPU+CPU+DSP-Argmax) detached starten, Log pollen.
#   Beispiel: run-llm.sh demo -n 64 -ngl 99 --sampler dsp "Erzaehl mir was."   -> /data/adb/baseos/run/demo.log
# Modell: erstes GGUF unter /data/local/ubuntu/home/*/models oder home/*, oder BARRA_MODEL=...
LLM=/data/adb/baseos/llm; R=/data/adb/baseos/run; mkdir -p $R
M=${BARRA_MODEL:-$(ls /data/local/ubuntu/home/*/models/*.gguf /data/local/ubuntu/home/*/*.gguf 2>/dev/null | head -1)}
[ -f "$M" ] || { echo "kein Modell (GGUF nach /data/local/ubuntu/home/<user>/models/ legen oder BARRA_MODEL setzen)"; exit 1; }
LOG=$R/${1:-barra-llm}.log; shift
export BARRA_SOCK_DIR=/data/local/ubuntu/opt/hwbridge   # Android-Sicht auf die Bruecken-Sockets
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
setsid timeout 600 $LLM/barra-llm -m "$M" "$@" </dev/null >$LOG 2>&1 &
echo "gestartet (PID $!): $(basename $M) -> $LOG"
