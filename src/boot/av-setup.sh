#!/system/bin/sh
# ============================================================================
# av-setup - setzt die AoC-Audio-Routen fuer raw-ALSA (headless, OHNE audioserver/Framework).
# Vom hw-bridges-Supervisor beim Boot aufgerufen. Danach:
#   Playback: tinyplay X.wav -D 0 -d 1   (device1 = EP2 -> Cirrus-Speaker L+R)
#   Capture:  tinycap  X.wav -D 0 -d 8 -c 2 -r 48000 -b 16 -t N   (device8 = EP1 = Builtin-Mic)
# Verifiziert 15.8. (Speaker-Ton + Mic-Loopback von Kevin gehoert).
# ============================================================================
H=/data/adb/hwbridge
TM=$H/tinymix
[ -x "$TM" ] || TM=/data/local/tmp/tinymix
set_ctl(){ "$TM" set "$1" $2 >/dev/null 2>&1; }

# --- Speaker (Playback-Route EP2 + Cirrus CS35L41 L+R) ---
set_ctl "TDM_0_RX Mixer EP2" 1
set_ctl "Main AMP Enable Switch" 1
set_ctl "R Main AMP Enable Switch" 1
set_ctl "Fast Use Case Delta File" fast_switch3.txt
set_ctl "Fast Use Case Switch Enable" 1
set_ctl "R Fast Use Case Delta File" fast_switch4.txt
set_ctl "R Fast Use Case Switch Enable" 1
set_ctl "PCM Playback Volume" 800

# --- Mic (Capture-Route EP1 + Builtin-Mic) ---
set_ctl "EP1 TX Mixer INTERNAL_MIC_TX" 1
set_ctl "BUILDIN MIC ID CAPTURE LIST" "0 1 -1 -1"
set_ctl "MIC DC Blocker" 1
set_ctl "MIC Record Soft Gain (dB)" 30

echo "[av-setup] Audio-Routen gesetzt (Speaker EP2/device1, Mic EP1/device8)"
