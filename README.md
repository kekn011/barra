# barra

**CUDA for your pocket.** barra turns a Google Pixel 8a into a headless Ubuntu
compute node and makes its Tensor G3 accelerators — the edge **TPU**, the Mali
**GPU** and the **DSP** — programmable from your own code, the way you'd use
CUDA on an NVIDIA card.

Flash a phone in about five minutes, `ssh` in, and you have a small ARM64 Linux
box with `gcc`, `apt`, an on-device TPU compiler, and a driver library
(`libbarra`) that shares zero-copy buffers across all four compute units. A
4-billion-parameter LLM runs on it out of the box, served over an
OpenAI-compatible API, with the attention math offloaded to the TPU.

> ⚠️ **Early and hardware-specific.** v0.x supports the **Pixel 8a (akita)**
> only. Flashing unlocks the bootloader, which **erases the phone** and voids
> warranty. Use at your own risk.

---

## What you get

| | |
|---|---|
| **A real Ubuntu node** | Ubuntu 24.04 userland, systemd, SSH, `apt`, full hardware access, headless. Configure it with `sudo barra-config`. |
| **The barra SDK** | `libbarra` (the driver API, `#include <barra.h>`), worked examples, and `barra-smi` — an `nvidia-smi`-style status view of the accelerators. |
| **On-device TPU toolchain** | `barrac model.tflite` compiles your own model into a TPU package, right on the phone. No PC toolchain, no cloud. |
| **An LLM, ready to serve** | `llama.cpp` with custom Mali GPU kernels **and** a TPU attention-offload path, behind an OpenAI-compatible server on port 8080. |
| **Localized end to end** | Setup wizard and on-device tools in English and German; adding a language is a one-file pull request. |

## Quick start

1. **Flash** — on a Windows PC, download a release, run `barra-setup`, fill in
   the form (user, Wi-Fi, SSH key, language), plug in the phone, hit Start.
   ~5 minutes later the node boots into your Wi-Fi and shows `ssh you@<ip>` on
   its screen. See [docs/flashing.md](docs/flashing.md).

2. **Explore the accelerators**

   ```sh
   ssh you@your-node
   barra-smi                 # GPU / TPU / DSP utilization, clocks, temps
   ```

3. **Run some code**

   ```sh
   cp -r /usr/share/barra/examples ~/ex && cd ~/ex
   sh build.sh && ./barra_demo         # CPU -> DSP kernel -> CPU, zero-copy
   ```

4. **Compile your own model onto the TPU**

   ```sh
   barrac mymodel.tflite               # -> mymodel.package
   ```

   Rules and limits: [docs/sdk/your-model-on-the-tpu.md](docs/sdk/your-model-on-the-tpu.md).

5. **Talk to the built-in LLM**

   ```sh
   # on the node (or via `adb forward tcp:8080 tcp:8080` over USB)
   curl http://<node>:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"hello from my pocket"}]}'
   ```

   There is also a built-in web UI at `http://<node>:8080`.

## The mental model

barra is **not** an auto-scheduling compiler. It is a thin, honest dispatch
layer: you decide which pipeline stage runs on which chip, and data flows
between them as shared **dmabuf** buffers — no copies.

```
   your code (C, glibc)
        │  #include <barra.h>
   ┌────┴─────────────────────────────────────────┐
   │  libbarra   — one buffer, four compute units  │
   └──┬─────────┬──────────┬──────────┬────────────┘
      │ CPU     │ TPU      │ GPU      │ DSP
      │ native  │ tpud     │ gpud     │ gxpd     (bridge daemons)
      │         │ packages │ SPIR-V   │ Xtensa kernels
```

- **TPU** runs precompiled graphs (`barrac`), best for dense matmul/attention.
- **GPU** runs SPIR-V compute shaders, best for programmable parallel math.
- **DSP** runs named Xtensa kernels, best for regular SIMD integer work.
- A `barra_zbuf` is one dmabuf that all of them can read and write in place.

The full API is documented in the header itself:
[`src/barra/barra.h`](src/barra/barra.h) (also at `/usr/include/barra.h` on a
node) and summarized in [docs/api-reference.md](docs/api-reference.md).

## Repository layout

| Path | What |
|---|---|
| `src/barra/` | `libbarra` — the driver API and its examples |
| `src/sdk/` | `barra-smi`, `barrac`, the `.deb` packaging (`mk-debs.sh`) |
| `src/boot/` | on-device system: boot chain, bridges config, `barra-config`, `llmserver.sh` |
| `src/i18n/` | translation catalogs (one `.properties` per language) |
| `src/ggml-barra/` | the llama.cpp attention-offload patch and TPU-package generators |
| `src/hwbridge/`, `src/gpu-kernels/`, `src/dsp-kernels/`, `src/tpu-runtime/` | the accelerator bridges and kernels |
| `barra-setup/` | the Windows flashing tool (PowerShell) |
| `kernel/` | the akita GKI kernel tree (GPLv2), config and patches |
| `docs/` | guides (CC BY 4.0) |

Large binary artifacts (the base image payload, the LLM kit) are published as
**GitHub release assets**, not committed to the repo.

## How it's built (honest notes)

- The **GPU** is only reachable Bionic-side (the Mali Vulkan driver is a vendor
  library), so GPU compute for your code goes through `gpud`/SPIR-V or the
  bundled llama.cpp — not native Vulkan inside the container. TPU and DSP are
  fully exposed through `libbarra`.
- Writing **your own DSP kernels** needs the Xtensa toolchain on a PC; using the
  bundled kernels does not. That's an advanced topic (v0.2).
- The base image ships a **modified GKI kernel** (6.1, GPLv2); its full source is
  in `kernel/`. Vendor firmware stays on the device and is never redistributed.

## Building from source

Everything except the Windows tool builds on the node. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the loop. To rebuild the flashable image
payload from a running node, use `src/boot/barra-bake.sh`.

## Security & expectations

- The LLM server listens on `0.0.0.0:8080` **without authentication** — keep it
  on a trusted LAN, or put it behind `--api-key` / a reverse proxy.
- First boot forces a password change; SSH password login is on by default.
- Unlocking the bootloader wipes the device. barra can also **undo** itself:
  the setup tool's "restore factory state" reflashes Google stock and
  (optionally) re-locks.

## License

- Code: **Apache-2.0** ([LICENSE](LICENSE), [NOTICE](NOTICE))
- Documentation: **CC BY 4.0**
- Kernel tree (`kernel/`): **GPLv2**
- Name & marks: see [TRADEMARKS.md](TRADEMARKS.md)

Contributions are under the [DCO](CONTRIBUTING.md#developer-certificate-of-origin-dco).

---

*barra is an independent project. Google, Pixel, Tensor, Android, Magisk, CUDA
and NVIDIA are trademarks of their respective owners, used here only
descriptively.*
