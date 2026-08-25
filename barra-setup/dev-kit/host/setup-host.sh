#!/bin/sh
# setup-host.sh - P2 Host-Provisioning-Doctor (IN WSL). Prueft die Werkzeuge, mit denen die
# nicht-on-device-baubaren Teile entstehen: glslang(android) Cross-Build, Xtensa-DSP-Build,
# NDK-Cross-Builds, GKI-Kernel. Fetch-or-supply: was fehlt, wird hier benannt, nicht geraten.
ok(){ printf '  [ok]    %s\n' "$1"; }; no(){ printf '  [FEHLT] %s\n' "$1"; }
NDK=${ANDROID_NDK:-$HOME/android-sdk/ndk/27.2.12479018}
echo "barra Dev-Kit - P2 Host-Provisioning (WSL):"
[ -d "$NDK" ] && ok "NDK ($(basename $NDK))" || no "NDK r27c (~/android-sdk/ndk/27.2.12479018) - developer.android.com/ndk"
for t in cmake ninja git python3 clang; do command -v $t >/dev/null 2>&1 && ok "$t" || no "$t (apt install)"; done
X=$HOME/xtensa/xtensa-esp-elf/bin/xtensa-esp-elf-gcc
[ -x "$X" ] && ok "Xtensa esp-elf ($($X -dumpversion 2>/dev/null)) - DSP-Build" || no "Xtensa esp-elf (github.com/espressif/crosstool-NG releases; ~/xtensa/) - user-supplied"
command -v glslang >/dev/null 2>&1 && ok "glslang (host, .spv-Bau am PC)" || no "glslang-tools (host)"
echo "glslang(android)  : build-glslang-android.sh  -> fetch-devtools.ps1 -Supply glslang (on-device Shaderbau)"
echo "DSP-Build (Host!) : dsp-kernels/asmpatch2.sh k_x.S 14c ker_x.elf (Xtensa; braucht inbuilt.elf"
echo "                    = Vendor-Splice-Template, user-supplied) -> devdeploy dspkernel ker_x.elf"
# GKI: schwerer Vorlauf (AOSP repo sync ~30 GB), daher nur wiring:
if ls -d "$HOME"/aosp-akita "$HOME"/android-kernel 2>/dev/null >/dev/null; then ok "AOSP-akita-Baum"; else no "AOSP-Baum (repo init android-gs-akita-android16; s. kernel/README.md) - ~30 GB Sync"; fi
echo "GKI-Loop          : nach repo sync -> host/build-gki.sh <tree> (Patch+Bazel -> boot-lz4.img)"
echo "                    -> reflash via barra-setup Kernel-Schritt (oder fastboot flash boot)"
