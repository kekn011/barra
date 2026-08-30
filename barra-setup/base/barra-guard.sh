#!/system/bin/sh
# barra-guard.sh — gemeinsamer Koexistenz-Waechter aller Kit-Dienste (llm, stt, pya, img, tts, wake).
# Einbinden (defensiv, damit ein Kit auch auf einem aelteren Base laeuft):
#     G=/data/adb/baseos/bin/barra-guard.sh; [ -f "$G" ] && . "$G"
# Vor dem Start fragen:
#     guard_check llm "$(guard_need "$M" 1400)" || exit 1
# Nach dem Start den eigenen Prozess killbar machen:
#     guard_expendable "$P"
#
# WARUM (29.8.2026): auf einem Knoten mit Bildgenerator + Sprachausgabe + Weckwort hat ein
# zusaetzlicher llmserver-Start das TELEFON NEU GESTARTET. pstore: "Kernel panic - not syncing:
# System is deadlocked on memory", ausgeloest von llama-server beim Nachladen der Modellseiten.
# Zwei Ursachen, beide hier adressiert:
#  1. Es gab nur zwei handverlesene Paar-Verbote (llm<->stt, llm<->pya). img, tts und wake
#     waren ungeschuetzt — niemand hat den Speicher zusammengezaehlt.
#  2. ALLE unsere Dienste laufen mit oom_score_adj -1000 (geerbt: Android-seitig von su/adbd,
#     im Container vom systemd, das selbst mit -1000 gestartet wird) = fuer den Kernel
#     unkillbar. Der OOM-Killer hat erst journald (-250) und dbus (-900) erlegt und dann
#     mangels Opfer paniert. panic_on_oom steht auf 0 — der Schalter war NICHT das Problem.
#     guard_expendable dreht das fuer unsere Dienste um: lieber stirbt der Dienst als das Telefon.
#
# Gemessen am Knoten barra-5036 (29.8.2026), Spitze unter echter Last, als MemAvailable-Delta
# gegen den Leerlauf (6446-6515 MB frei, MemTotal 7570 MB):
#     llm  4031 MB (Modell 2654 + 1380 Laufzeit)   img 3281 MB (Modell 2035 + 1250)
#     stt  1636 MB (Modell  547 + 1090)            tts   270 MB
#     wake   80 MB                                 pya   100 MB (tpud RSS 10 MB)
# Die Modellgroesse steuert der Aufrufer bei (guard_need), der Aufschlag steht in seinem Skript.
#
# BARRA_GUARD=off schaltet den Waechter ab (Entwicklung, auf eigene Gefahr).

GUARD_RESERVE_MB=${GUARD_RESERVE_MB:-600}   # was fuer Basis-OS, Container und Puffer frei bleibt
GUARD_OOM_ADJ=${GUARD_OOM_ADJ:-500}         # unsere Dienste sind Opfer erster Wahl

guard_pat(){   # Erkennungsmuster je Dienst (pgrep -f). Android sieht auch Container-Prozesse.
  case "$1" in
    llm)  echo "llama-server";;
    stt)  echo "baseos/stt/whisper-server";;
    img)  echo "barra-img/bin/sd-server";;
    tts)  echo "barra-tts/bin/ttsd.py";;
    wake) echo "barra-wake/wake-bridge";;
    pya)  echo "tpud_pipe4 /data/local/ubuntu/opt/hwbridge/pya";;
  esac
}
guard_cmd(){   # wie man den Dienst wieder los wird (steht in der Meldung)
  case "$1" in
    llm)  echo "llmserver.sh stop";;
    stt)  echo "sttserver.sh stop";;
    img)  echo "imgserver.sh stop";;
    tts)  echo "ttsserver.sh stop";;
    wake) echo "wakeserver.sh stop";;
    pya)  echo "pyaserver.sh stop";;
  esac
}
guard_conflict(){   # echte Sachverbote, unabhaengig vom freien Speicher
  case "$1" in
    # llm+stt: zusaetzlich zum RAM sprengen die TPU-Graphen das Firmware-Limit (145+104 > ~157)
    llm)  echo "stt";;
    stt)  echo "llm";;
    pya)  echo "llm";;
    *)    echo "";;
  esac
}

guard_running(){ pgrep -f "$(guard_pat "$1")" >/dev/null 2>&1; }
guard_free_mb(){ awk '/^MemAvailable:/{printf "%d", $2/1024}' /proc/meminfo 2>/dev/null; }

guard_others(){   # laufende Dienste ausser $1, als Liste fuer die Meldung
  _l=""
  for _s in llm stt img tts wake pya; do
    [ "$_s" = "$1" ] && continue
    guard_running "$_s" && _l="$_l $_s"
  done
  echo "${_l# }"
}

guard_need(){   # $1 Modelldatei, $2 Aufschlag in MB -> Bedarf in MB
  # NICHT in Shell-Arithmetik rechnen: die Android-Shell rechnet mit 32 Bit, eine 2,78-GB-Datei
  # laeuft ueber und liefert einen NEGATIVEN Bedarf (29.8. genau so gemessen: -41 statt 4054).
  _b=$(stat -c %s "$1" 2>/dev/null || echo 0)
  awk -v b="$_b" -v a="${2:-0}" 'BEGIN{printf "%d", int(b/1048576) + a}'
}

guard_expendable(){   # $1.. PIDs: fuer den OOM-Killer angreifbar machen (sonst -1000 = unkillbar)
  for _p in "$@"; do
    [ -n "$_p" ] && [ -d "/proc/$_p" ] || continue
    echo "$GUARD_OOM_ADJ" > "/proc/$_p/oom_score_adj" 2>/dev/null
  done
}

guard_name(){ t "guard.name.$1"; }   # Klartextname aus dem i18n-Katalog

guard_check(){   # $1 Dienst, $2 Bedarf in MB -> 0 = darf starten, 1 = verweigert (Meldung ist raus)
  _s="$1"; _need="${2:-0}"
  [ "$BARRA_GUARD" = "off" ] && { t guard.off "$(guard_name "$_s")"; return 0; }
  for _c in $(guard_conflict "$_s"); do
    guard_running "$_c" && {
      t guard.conflict "$(guard_name "$_s")" "$(guard_name "$_c")" "$(guard_cmd "$_c")"
      return 1
    }
  done
  _free=$(guard_free_mb)
  [ -n "$_free" ] || return 0        # kein /proc/meminfo lesbar: nicht im Weg stehen
  if [ "$_free" -lt $(( _need + GUARD_RESERVE_MB )) ]; then
    t guard.no_mem "$(guard_name "$_s")" "$_free" "$_need" "$GUARD_RESERVE_MB"
    _o=$(guard_others "$_s")
    if [ -n "$_o" ]; then
      for _x in $_o; do t guard.stop_hint "$(guard_name "$_x")" "$(guard_cmd "$_x")"; done
    else
      t guard.nothing_else
    fi
    return 1
  fi
  return 0
}
