#!/bin/bash
# neueres glslang (mit GL_EXT_integer_dot_product) nach ~/glslang-new holen.
# Pinning (empfohlen): GLSLANG_TAG=<tag> und GLSLANG_SHA256=<sha256 des zip>. Ohne Pin
# wird 'latest' geholt und NUR ueber TLS abgesichert (dann Warnung).
set -e
WORK=$(mktemp -d)                       # keine festen /tmp-Namen (Symlink-/Race-Schutz)
trap 'rm -rf "$WORK"' EXIT
TAG="${GLSLANG_TAG:-latest}"
if [ "$TAG" = latest ]; then
  curl -s https://api.github.com/repos/KhronosGroup/glslang/releases/latest > "$WORK/rel.json"
else
  curl -s "https://api.github.com/repos/KhronosGroup/glslang/releases/tags/$TAG" > "$WORK/rel.json"
fi
RTAG=$(grep '"tag_name"' "$WORK/rel.json" | head -1 | sed 's/.*: "\(.*\)".*/\1/')
URL=$(grep browser_download_url "$WORK/rel.json" | grep -i linux | grep -i release | grep -v debug | head -1 | sed 's/.*"\(https[^"]*\)".*/\1/')
echo "tag=$RTAG url=$URL"
[ -n "$URL" ] || { echo "FEHLER: kein Linux-Release-Asset gefunden"; exit 1; }
curl -sL -o "$WORK/glslang.zip" "$URL"
file "$WORK/glslang.zip" | head -1
if [ -n "$GLSLANG_SHA256" ]; then
  have=$(sha256sum "$WORK/glslang.zip" | awk '{print $1}')
  [ "$have" = "$GLSLANG_SHA256" ] || { echo "FEHLER: SHA256 stimmt nicht ($have != $GLSLANG_SHA256)"; exit 1; }
  echo "SHA256 ok"
else
  echo "WARNUNG: kein GLSLANG_SHA256 gesetzt - nur TLS sichert die Toolchain ab. Zum Pinnen GLSLANG_TAG+GLSLANG_SHA256 setzen."
fi
rm -rf ~/glslang-new && mkdir -p ~/glslang-new && cd ~/glslang-new && unzip -q "$WORK/glslang.zip"
ls; ls bin 2>/dev/null
./bin/glslangValidator --version | head -1 || ./bin/glslang --version | head -1
