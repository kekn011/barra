#!/bin/bash
# Selbst-enthaltenes barra-tts-Kit /opt/barra-tts aufbauen (runtime+site kit-intern, kein ~/tts noetig).
set -e
K=/opt/barra-tts
T=~/tts; G=~/gc
mkdir -p $K; rm -rf $K/* 2>/dev/null || true
mkdir -p $K/bin $K/voices/david/gpukit2 $K/voices/piper-de $K/voices/piper-en $K/sherpa

# --- Engine ---
cp $G/gpudecd $K/bin/gpudecd
cp $G/ttsd.py $K/bin/ttsd.py
cp $G/launch.sh $K/bin/launch.sh; chmod 755 $K/bin/launch.sh $K/bin/gpudecd

# --- Python-Runtime + site (kit-intern) ---
echo "kopiere runtime..."; cp -a $T/rt $K/runtime
echo "kopiere site...";    cp -a $T/site $K/site

# --- David ---
cp $G/david-front.onnx $K/voices/david/david-front.onnx
cp $G/gpukit2/program2.json $G/gpukit2/weights16.bin $G/gpukit2/bias16.bin $K/voices/david/gpukit2/
cp $T/site/tokens.txt $K/voices/david/tokens.txt
cat > $K/voices/david/voice.json <<'JSON'
{"engine":"david","front_onnx":"david-front.onnx","tokens":"tokens.txt",
 "gpukit":"gpukit2","site":"/opt/barra-tts/site","scales":[0.667,1.0,0.8],"sample_rate":22050,
 "label_de":"David (Erzaehlerstimme, GPU)","label_en":"David (narrator voice, GPU)"}
JSON

# --- Piper de/en ---
cp $T/vits-piper-de_DE-thorsten-medium/de_DE-thorsten-medium.onnx $K/voices/piper-de/model.onnx
cp $T/vits-piper-de_DE-thorsten-medium/tokens.txt $K/voices/piper-de/tokens.txt
cp -r $T/vits-piper-de_DE-thorsten-medium/espeak-ng-data $K/voices/piper-de/espeak-ng-data
cat > $K/voices/piper-de/voice.json <<'JSON'
{"engine":"piper","model":"model.onnx","tokens":"tokens.txt","data_dir":"espeak-ng-data","sample_rate":22050,"sid":0,
 "label_de":"Thorsten (Deutsch)","label_en":"Thorsten (German)"}
JSON
cp $T/vits-piper-en_US-amy-medium/en_US-amy-medium.onnx $K/voices/piper-en/model.onnx
cp $T/vits-piper-en_US-amy-medium/tokens.txt $K/voices/piper-en/tokens.txt
cp -r $T/vits-piper-en_US-amy-medium/espeak-ng-data $K/voices/piper-en/espeak-ng-data
cat > $K/voices/piper-en/voice.json <<'JSON'
{"engine":"piper","model":"model.onnx","tokens":"tokens.txt","data_dir":"espeak-ng-data","sample_rate":22050,"sid":0,
 "label_de":"Amy (Englisch)","label_en":"Amy (English)"}
JSON

# --- sherpa-offline-tts + libs ---
SB=$T/sherpa-onnx-v1.13.6-linux-aarch64-shared-cpu
cp $SB/bin/sherpa-onnx-offline-tts $K/sherpa/; cp -r $SB/lib $K/sherpa/lib

echo "=== Kit fertig ==="; du -sm $K; du -sm $K/runtime $K/site $K/voices $K/sherpa
