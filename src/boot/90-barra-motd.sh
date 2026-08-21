# barra: kurzer Willkommenshinweis beim interaktiven Login (profile.d)
case $- in *i*) ;; *) return 0 2>/dev/null || exit 0;; esac
[ -n "$PS1" ] || return 0 2>/dev/null
[ -r /usr/share/barra/barra-i18n.sh ] && . /usr/share/barra/barra-i18n.sh
command -v t >/dev/null 2>&1 || t() { printf '%s\n' "$1"; }
_bip=$(ip -4 addr show wlan0 2>/dev/null | awk '/inet /{print $2; exit}' | cut -d/ -f1)
printf '\n  \033[1mbarra\033[0m  %s%s\n' "$(hostname)" "${_bip:+  ·  WLAN $_bip}"
if [ ! -f /var/lib/barra/configured ]; then
  printf '  '; t motd.unconfigured
fi
printf '\n'
unset _bip
