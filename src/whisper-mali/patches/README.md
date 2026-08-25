# whisper.cpp-Patch fuer den barra-TPU-Encoder (WHISPER_BARRA=1)
# Anwenden auf whisper.cpp (Commit siehe PIN unten):
#   git apply whisper-cpp-barra.patch
#   cp ../experiments/gpu-attn/barra.{c,h} src/ && cp ../tpu/whisper-barra.cpp src/
# (CMake-Ergaenzung barra.c+whisper-barra.cpp ist Teil des Patches.)
# Danach src/whisper-mali/build.sh. Laufzeit-Envs: WHISPER_BARRA=1 WSP_PKG_DIR WSP_SOCK_MAIN WSP_SOCK_CORE.
