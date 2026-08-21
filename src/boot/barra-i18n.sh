#!/system/bin/sh
# barra-i18n.sh — Lokalisierungs-Loader fuer Shell-Skripte (Android toybox/mksh UND Container bash/dash).
# Nutzung:  . <pfad>/barra-i18n.sh   danach   t <key> [printf-args...]
# Kataloge: <dir>/{<lang>,en}.properties — Suchreihenfolge der Verzeichnisse:
#   $BARRA_I18N_DIR  ->  /data/adb/baseos/i18n (Android)  ->  /usr/share/barra/i18n (Container)
# Sprachwahl: $BARRA_LANG -> Config-Store LANG_UI (Android) -> $LANG-Prefix (Container) -> en
# Konvention: NUR UI-Texte lokalisieren; Logs/technische Ausgaben bleiben englisch im Code.
# Unbekannter Key -> der Key selbst wird ausgegeben (nie leer, nie Abbruch).

_bi18n_dir=""
for _d in "$BARRA_I18N_DIR" /data/adb/baseos/i18n /usr/share/barra/i18n; do
  [ -n "$_d" ] && [ -d "$_d" ] && { _bi18n_dir=$_d; break; }
done

_bi18n_lang=${BARRA_LANG:-}
if [ -z "$_bi18n_lang" ] && [ -r /etc/barra-lang ]; then
  _bi18n_lang=$(head -1 /etc/barra-lang 2>/dev/null)
fi
if [ -z "$_bi18n_lang" ] && [ -r /data/adb/baseos/config ]; then
  _bi18n_lang=$(grep -m1 '^LANG_UI=' /data/adb/baseos/config 2>/dev/null | cut -d= -f2)
fi
[ -z "$_bi18n_lang" ] && _bi18n_lang=${LANG%%_*}
[ -z "$_bi18n_lang" ] || [ "$_bi18n_lang" = "C" ] || [ "$_bi18n_lang" = "POSIX" ] && _bi18n_lang=en

t() {
  _k=$1; shift
  _v=""
  for _f in "$_bi18n_dir/$_bi18n_lang.properties" "$_bi18n_dir/en.properties"; do
    [ -r "$_f" ] || continue
    _v=$(grep -m1 "^$_k=" "$_f" 2>/dev/null)
    [ -n "$_v" ] && { _v=${_v#*=}; break; }
  done
  [ -n "$_v" ] || _v=$_k
  # shellcheck disable=SC2059  — Katalogwert IST das printf-Format (\033, %s gewollt)
  printf "$_v\n" "$@"
}
