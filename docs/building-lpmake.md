# Building lpmake.exe for Windows

barra's Windows setup builds `stock/super.img` locally with `lpmake` so it can
flash the system on the driver-free bootloader path (no Windows fastbootd driver
needed). `lpmake` is part of AOSP (Apache-2.0); Google ships no Windows build,
so barra builds its own and bundles it as `barra-setup/tools/lpmake.exe`.

We build it ourselves — rather than bundling a third-party binary — so the
artifact is auditable and its provenance is clear.

## The build (verified)

We cross-compile with **mingw-w64** from the standalone `lpunpack_and_lpmake`
tree (which carries the relevant AOSP sources: liblp, libsparse, libbase,
liblog, boringssl, zlib). No full AOSP checkout is needed.

```sh
# host: Ubuntu/WSL
sudo apt-get install -y g++-mingw-w64-x86-64 mingw-w64-tools
git clone https://github.com/LonelyFool/lpunpack_and_lpmake   # or the tree you build from
cp barra-setup/make-lpmake-win.sh lpunpack_and_lpmake/make.sh
cp barra-setup/atomic_shim.hpp    lpunpack_and_lpmake/
cd lpunpack_and_lpmake && bash make.sh                        # -> bin/lpmake.exe
```

`make-lpmake-win.sh` is the upstream `make.sh` with the barra patches applied.
Copy the result to `barra-setup/tools/lpmake.exe`.

## The four mingw portability fixes (what the patches do)

Cross-compiling these AOSP sources to Windows needs four changes, all captured
in the checked-in scripts:

1. **Toolchain + target.** `CC/CPP/AR/STRIP` point at `x86_64-w64-mingw32-*`;
   `OSTYPE`/`OS`/`HOSTTYPE` are forced so the sources pick their Windows
   branches (`errors_windows.cpp`, `utf8.cpp`, `LDFLAGS=-lws2_32`).
2. **boringssl without asm.** `-DOPENSSL_NO_ASM` + an empty asm source list, so
   no NASM is required — the C fallbacks are used.
3. **Missing Unix headers.** `atomic_shim.hpp` maps C11 `<stdatomic.h>` names
   onto C++ `<atomic>` (force-included into liblog); a tiny `sysexits.h` and the
   local `zlib.h` include path are added.
4. **Binary file mode (the important one).** Every `open()` in
   `liblp/images.cpp` gains `O_BINARY` — without it mingw opens partition images
   in *text* mode and mangles the binary data (symptom:
   `Invalid sparse file format at header magic`, `read failed`).

Only `lpmake` is built; `lpdump`/`lpadd`/`lpunpack` (which pull in protobuf) are
skipped.

## Verification

The produced `lpmake.exe` runs natively on Windows and builds a correct akita
`super.img`:

- `build-super.ps1` drives it on the pinned geometry (`super:8531214336`, two
  `google_dynamic_partitions_{a,b}` groups of 8527020032 bytes, metadata
  65536/3 slots, virtual A/B).
- The output is **byte-for-byte identical** (same MD5) to the reference
  `super.img` built during development, and is deterministic across runs.

That check — `build-super.ps1` producing a bit-identical `super.img` — is the
acceptance test for a rebuilt `lpmake.exe`.
