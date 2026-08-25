#!/bin/sh
# pack-dev-kit.sh - barra-dev-kit.tar.gz reproduzierbar bauen. IN WSL/git-bash laufen
# (tar erhaelt die Exec-Bits - Android braucht +x auf den .sh; Kit-Falle aus tts/pya).
# Assembliert aus: src/dev (unsere Skripte), src/gpu-kernels (Harnesses+Shader),
# dev-kit/tpu (tpuc1 + patch), dev-kit/third-party (fetch-devtools: frida/glslang).
# libcomp_std.so wird NICHT gepackt (on-device via extract-libedgetpu.sh generiert).
set -e
here=$(cd "$(dirname "$0")" && pwd)          # barra-setup/dev-kit
repo=$(cd "$here/../.." && pwd)
stage="$here/stage"
rm -rf "$stage"
mkdir -p "$stage/dev/bin" "$stage/dev/gpu-kernels" "$stage/dev/dsp-kernels" "$stage/dev/tpu" "$stage/dev/third-party/bin" "$stage/dev/harness" "$stage/base" "$stage/service.d"

cp "$repo/src/dev/devbuild.sh" "$repo/src/dev/devdoctor.sh"           "$stage/dev/bin/"
cp "$repo/src/dev/devkit-env.sh"                                       "$stage/dev/"
cp "$repo/src/gpu-kernels/"*                                           "$stage/dev/gpu-kernels/"
# NUR unser Code (.S/.sh/.md). KEINE *.elf: inbuilt.elf ist ein Vendor-Firmware-Template und
# ker_*.elf = inbuilt.elf mit Kernel gespleisst -> abgeleiteter Vendor-Blob (wie libcomp_std).
# inbuilt.elf wird on-device beschafft (P2, analog extract-libedgetpu.sh). tie/ (libgxp.so,
# fw_core*.bin) wird durch die Top-Level-Globs ohnehin nicht erfasst.
cp "$repo/src/dsp-kernels/"*.S "$repo/src/dsp-kernels/"*.sh "$repo/src/dsp-kernels/"*.md "$stage/dev/dsp-kernels/" 2>/dev/null || true
cp "$here/tpu/tpuc1" "$here/tpu/patch-compiler-standalone.py" "$here/tpu/extract-libedgetpu.sh" "$stage/dev/tpu/"
cp "$repo/src/dev/barra-dev-mode.sh" "$repo/src/dev/devdeploy.sh"      "$stage/base/"
cp "$repo/src/dev/55-barra-dev.sh"                                     "$stage/service.d/"
[ -d "$here/third-party/bin" ] && cp -a "$here/third-party/bin/." "$stage/dev/third-party/bin/" || true
echo P1 > "$stage/dev/VERSION"

# Exec-Bits sicherstellen (cp -a von Windows-FS bringt kein +x -> sonst laufen die
# Binaries/Skripte am Geraet nicht; Kit-Falle aus tts/pya)
chmod 755 "$stage"/dev/bin/*.sh "$stage"/dev/devkit-env.sh "$stage"/dev/tpu/*.sh "$stage"/dev/tpu/tpuc1 "$stage"/base/*.sh "$stage"/service.d/*.sh 2>/dev/null || true
[ -d "$stage/dev/third-party/bin" ] && chmod 755 "$stage"/dev/third-party/bin/* 2>/dev/null || true

# Warnungen bei fehlenden fetch-or-supply-Teilen
[ -x "$stage/dev/third-party/bin/glslang" ] || echo "  [WARN] glslang fehlt -> On-Device-Shaderbau erst nach -Supply/Build (P2)"
[ -x "$stage/dev/third-party/bin/frida-server" ] || echo "  [WARN] frida-server fehlt -> fetch-devtools.ps1 laufen lassen"

tar -C "$stage" -czf "$here/barra-dev-kit.tar.gz" dev base service.d
rm -rf "$stage"
echo "OK -> $here/barra-dev-kit.tar.gz"
tar -tzf "$here/barra-dev-kit.tar.gz" | sort
