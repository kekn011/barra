#!/system/bin/sh
# barra Dev-Kit Selbsttest — in der adb-Shell. Sagt, ob die Kernel-Werkbank bereit ist.
DEV=/data/adb/baseos/dev
ok(){ printf '  [ok]    %s\n' "$1"; }; no(){ printf '  [FEHLT] %s\n' "$1"; }
echo "barra Dev-Kit doctor (Android-Seite):"
{ command -v glslang >/dev/null 2>&1 || command -v glslangValidator >/dev/null 2>&1 || [ -x $DEV/third-party/bin/glslang ]; } && ok "glslang (Shader->SPIR-V)" || no "glslang (fetch-or-supply: fetch-devtools.ps1)"
ls $DEV/gpu-kernels/*.comp >/dev/null 2>&1 && ok "GPU-Shader-Quellen (gpu-kernels)" || no "gpu-kernels/*.comp"
ls $DEV/gpu-kernels/gpugemv.c $DEV/gpu-kernels/gpugemm.c >/dev/null 2>&1 && ok "Harness-Quellen (gpugemv/gpugemm)" || no "Harness-Quellen"
[ -x $DEV/tpu/tpuc1 ] && ok "tpuc1 (TPU-Standalone-Compiler)" || no "tpuc1 (aus RE-Strang)"
[ -f $DEV/tpu/libcomp_std.so ] && ok "libcomp_std.so" || no "libcomp_std.so (on-device: extract-libedgetpu.sh)"
ls $DEV/dsp-kernels/*.S >/dev/null 2>&1 && ok "DSP-Kernelquellen (Xtensa .S + asmpatch2)" || no "dsp-kernels/*.S"
[ -x $DEV/third-party/bin/frida-server ] && ok "frida-server (RE)" || no "frida-server (fetch-or-supply)"
[ -e /dev/mali0 ]        && ok "/dev/mali0 (GPU)" || no "/dev/mali0"
[ -e /dev/edgetpu-soc ]  && ok "/dev/edgetpu-soc (TPU)" || no "/dev/edgetpu-soc"
[ -e /dev/gxp ]          && ok "/dev/gxp (DSP)" || no "/dev/gxp"
echo "Geraetezustand: su -c 'sh /data/adb/baseos/bin/barra-dev-mode.sh status'"
