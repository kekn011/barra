#!/system/bin/sh
# barra Dev-Kit Umgebung — in der adb-Shell sourcen:  . /data/adb/baseos/dev/devkit-env.sh
# Legt PATH auf die Kit-Toolchains (glslang, tpuc1) und definiert Kurzbefehle. Die GPU/TPU-
# Kernelarbeit laeuft Android-seitig (bionic: /dev/mali0, /dev/edgetpu-soc, Vulkan). glslang
# kommt via fetch-or-supply nach third-party/bin/.
DEV=/data/adb/baseos/dev
export BARRA_DEV_DIR=$DEV
export PATH="$DEV/bin:$DEV/third-party/bin:$DEV/tpu:$PATH"
export LD_LIBRARY_PATH="$DEV/tpu:/system/lib64:/vendor/lib64:${LD_LIBRARY_PATH}"
export BARRA_DEV=1
alias devbuild="sh $DEV/bin/devbuild.sh"
alias devdoctor="sh $DEV/bin/devdoctor.sh"
alias devmode="su -c 'sh /data/adb/baseos/bin/barra-dev-mode.sh'"
alias devdeploy="su -c 'sh /data/adb/baseos/bin/devdeploy.sh'"
echo "barra Dev-Kit aktiv. 'devdoctor' prueft die Werkbank, 'devmode status' den Geraetezustand."
