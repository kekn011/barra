# install-llm.ps1 — LLM-Kit auf einen barra-Node schieben (nach dem Flash, per USB/adb).
# Inhalt: qwen3-4b.gguf (Modell, ins Container-Home) + llm-attn-qwen3-4b.tar (144 TPU-Attention-
# Packages der v7-Pipeline + attn.meta + aux_attn.bin + pkglist.txt nach /data/local/barra-attn/
# qwen3-4b). Danach: llmserver.sh start erkennt das Kit und schaltet den TPU-Attention-Offload an
# (alle 36 Layer; +18 % Prefill, Qualitaet = Baseline).
$ErrorActionPreference = "Stop"
$kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$adb = Join-Path (Split-Path -Parent $kit) "tools\adb.exe"
if (-not (Test-Path $adb)) { $adb = "adb" }
# Native adb-Exit-Codes werfen unter -ErrorAction Stop NICHT -> explizit pruefen, sonst
# endet ein fehlgeschlagener Push/Install trotzdem mit "Fertig".
function Adb { & $adb @args; if ($LASTEXITCODE -ne 0) { Write-Error "adb $($args -join ' ') fehlgeschlagen (exit $LASTEXITCODE)"; exit 1 } }

Write-Host "== barra LLM-Kit: qwen3-4b =="
Adb wait-for-device | Out-Null

Write-Host "1/4 Attention-Packages pushen (911 MB) ..."
Adb push (Join-Path $kit "llm-attn-qwen3-4b.tar") /data/local/tmp/llm-kit.tar
Adb shell "su -c 'mkdir -p /data/local/barra-attn && cd /data/local/barra-attn && tar -xf /data/local/tmp/llm-kit.tar && chmod -R 755 /data/local/barra-attn && rm /data/local/tmp/llm-kit.tar'"

Write-Host "2/4 Modell pushen (2,4 GB) ..."
Adb push (Join-Path $kit "qwen3-4b.gguf") /data/local/tmp/qwen3-4b.gguf

Write-Host "3/4 Modell ins Container-Home legen ..."
Adb shell "su -c 'H=`$(ls -d /data/local/ubuntu/home/* | head -1); mkdir -p `$H/models; mv /data/local/tmp/qwen3-4b.gguf `$H/models/qwen3-4b.gguf; chown -R 1001:1001 `$H/models; ls -la `$H/models'"

Write-Host "4/4 LLM-Server starten ..."
Adb shell "su -c 'sh /data/adb/baseos/bin/llmserver.sh start'"

Write-Host ""
Write-Host "Fertig. Chat: http://<node-ip>:8080  (oder im Container: chat)"
Write-Host "Status:  adb shell su -c 'sh /data/adb/baseos/bin/llmserver.sh status'"
