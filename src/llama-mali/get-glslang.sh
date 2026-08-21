#!/bin/bash
# neueres glslang (mit GL_EXT_integer_dot_product) nach ~/glslang-new holen
set -e
cd /tmp
curl -s https://api.github.com/repos/KhronosGroup/glslang/releases/latest > rel.json
TAG=$(grep '"tag_name"' rel.json | head -1 | sed 's/.*: "\(.*\)".*/\1/')
URL=$(grep browser_download_url rel.json | grep -i linux | grep -i release | grep -v debug | head -1 | sed 's/.*"\(https[^"]*\)".*/\1/')
echo "tag=$TAG url=$URL"
curl -sL -o glslang.zip "$URL"
file glslang.zip | head -1
rm -rf ~/glslang-new && mkdir -p ~/glslang-new && cd ~/glslang-new && unzip -q /tmp/glslang.zip
ls; ls bin 2>/dev/null
./bin/glslangValidator --version | head -1 || ./bin/glslang --version | head -1
