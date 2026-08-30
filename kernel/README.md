# barra kernel (akita / Pixel 8a)

The barra base OS runs on a self-built GKI kernel `6.1.157-android14-11-g…` for
the Pixel 8a (akita, Tensor G3). It is the stock Google kernel tree with one
small patch on top; barra flashes **only** the resulting `boot` image
(`boot-lz4.img`). `vbmeta` and every vendor partition stay stock — the stock
vendor modules keep loading because the patch preserves the KMI.

This directory is the complete corresponding source for the shipped kernel
binary (GPL-2.0): the exact upstream base is pinned below and the only local
change is `patches/aosp.patch`.

## What the patch changes (`patches/aosp.patch`)

- `arch/arm64/configs/gki_defconfig`:
  - USB ethernet built in instead of modular (`MII`, `USBNET`, `RTL8152/8153`,
    `AX8817X/AX88179`, `CDC ether/NCM`, `AQC111`) — headless usb-network from
    the first boot moment, no module loading order issues.
  - `NFS`/`NFSD` (v3/v4.x), `IP_SET` + `xt_set`/`xt_addrtype`/`xt_rpfilter`,
    `VXLAN` — container/cluster networking (MicroK8s) and NFS shares.
  - `USER_NS=y`, `PID_NS` re-enabled, `FHANDLE=y` — required by systemd and
    unprivileged containers in the Ubuntu userland.
- `BUILD.bazel`: moves the now-builtin USB drivers out of the GKI module list
  and adds the new ipset/vxlan modules, so the Bazel GKI build stays consistent.
- `build.config.gki`: disables `check_defconfig` (the defconfig deliberately
  deviates from stock GKI).
- `drivers/usb/core/hub.h`: moves the `post_resume_work` member (added by the
  6.1.147 LTS merge) to the **end** of the private `struct usb_hub`. The stock
  Pixel vendor module `xhci_exynos` (built against 6.1.145) reads `hub->ports`
  at a fixed offset; `struct usb_hub` is not part of the KMI, so the LTS insert
  shifted that offset by `sizeof(struct delayed_work)` and every USB host
  attach (any OTG device, e.g. a USB-C ethernet adapter) ended in a kernel
  NULL-pointer panic in `xhci_exynos_early_stop_set`. The member itself is
  kept — only its position changes, so the LTS fix stays in.
- `kernel/cgroup/cgroup.c`: ignore `cgroup_disable=memory` from the kernel
  command line. The Pixel's stock `vendor_boot` (which barra does not
  touch) disables the memory cgroup controller for the whole system; the
  Ubuntu userland and its container runtime need memory accounting
  (kubelet/cadvisor, metrics-server, `kubectl top`). Everything else in
  `cgroup_disable=` is honoured as before. Measured on a node: no visible
  cost in idle memory (MemAvailable 5975 MB vs 5794 MB before, same load).

`gki_defconfig` here is the full resulting defconfig, for reference.
`build-gki-157-usbnet.sh` is the original build script that produced the
shipped image — it applies the same changes scriptally (sed/python against a
clean checkout) instead of via the patch, and documents the reasoning per
option; adjust its hardcoded paths if you use it.

## Reproducing the build

The tree is the standard repo-managed Pixel kernel workspace:

```sh
mkdir akita-kernel && cd akita-kernel
repo init -u https://android.googlesource.com/kernel/manifest \
          -b android-gs-akita-android16
repo sync -j4
```

Pin the checkouts to the exact revisions in `PINS.txt` (`git checkout <sha>` in
each listed project; only `aosp/` matters for the boot image, the others build
vendor modules that barra does not ship). Then build the pure GKI target —
barra does not use the Pixel device build (`build_akita.sh`), only the GKI
kernel + its boot image:

```sh
cd aosp && git apply ../path/to/patches/aosp.patch && cd ..
tools/bazel run //aosp:kernel_aarch64_dist -- --dist_dir=out/dist
```

`out/dist/boot-lz4.img` is the file `barra-setup` flashes in the kernel step
(see `docs/flashing.md`). Ship `out/dist/rfkill.ko` next to it as
`wifi-rfkill.ko`: it is signed with the module signing key of *this* kernel
build (GKI "protected exports" — the kernel refuses an `rfkill.ko` from any
other build, and without it the Wi-Fi driver `bcmdhd` cannot load). The
`wifi-rfkill.ko` inside `barra-base.tar.gz` is only a fallback for older
setups; the installer prefers the copy from the kernel payload.

Provenance of the shipped binaries: the `boot-lz4.img` in the setup payload has
SHA-256 `adb51190ffa908a12dd78e16e48a1478488f4d228dd32a8fc894ffd11145a667`
(`wifi-rfkill.ko`: `31af0598543616971e5a9e77395705e6c4fd0249dcb3595db14d7c00ecc267c6`)
and was built on 2026-08-30 from exactly the source state this directory pins
(base `aosp @ bde5fd109bd8` + `patches/aosp.patch`). Earlier images of the
same day: `8659acb5…` (hub.h fix, memory cgroup still disabled) and
`01938bbe…` (2026-08-11, panics on USB host attach).

**Kernel and modules are a set.** Every build signs its modules with a
build-time key and the GKI kernel refuses a foreign `rfkill.ko` (protected
exports). Ship `boot-lz4.img` together with the `wifi-rfkill.ko` (and, for
MicroK8s, the ipset/xt_set/xt_addrtype/vxlan `.ko`) of the *same* build.
`src/boot/barra-kernel-update.sh` applies such a set to a running node
without a cable. Pinning `CONFIG_MODULE_SIG_KEY` in the defconfig does not
survive the Kleaf build (open: pass `module_signing_key` to `kernel_build`).

## License

The kernel is GPL-2.0 (see the COPYING file in the kernel tree). The patch in
this directory is likewise GPL-2.0.
