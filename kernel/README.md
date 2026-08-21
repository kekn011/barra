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

`out/dist/boot-lz4.img` is the single file `barra-setup` flashes in the kernel
step (see `docs/flashing.md`). Provenance of the shipped binary: the
`boot-lz4.img` in the setup payload has SHA-256
`01938bbe72dc1f282b671673a0d01dd9f10a00f6674c6bef7b8c245e9da42949` and was
built on 2026-08-11 from exactly the source state this directory pins
(base `aosp @ bde5fd109bd8` + `patches/aosp.patch`).

## License

The kernel is GPL-2.0 (see the COPYING file in the kernel tree). The patch in
this directory is likewise GPL-2.0.
