# Contributing to barra

Thanks for wanting to help. barra turns a Google Pixel 8a into a headless Ubuntu
compute node whose Tensor G3 accelerators (TPU / GPU / DSP) are programmable like
CUDA. It is early and hardware-specific — issues, fixes and new-language
translations are all welcome.

## Ground rules

- **Be honest about hardware claims.** This project lives or dies on measured,
  reproducible numbers. If you report a speedup or a working path, say how you
  measured it and on which build. "Works on my node" with the node details beats
  a confident guess.
- **One device for now.** v0.x supports **akita** (Pixel 8a) only. Pixel 8
  (shiba) / 8 Pro (husky) are separate bringup efforts — please don't
  file "doesn't boot on husky" as a bug yet.
- **Never commit binaries you don't own the rights to.** No Google factory
  images, no vendor blobs, no Magisk APK. Those are downloaded from their
  original sources at setup time. See [NOTICE](NOTICE).

## Developer Certificate of Origin (DCO)

We use the [DCO](https://developercertificate.org/) instead of a CLA. It is a
one-line certification that you wrote the change (or have the right to submit
it). Add a `Signed-off-by` line to every commit:

```
Signed-off-by: Your Name <your.email@example.com>
```

`git commit -s` adds it automatically. By signing off you agree to the DCO text.

## Licensing of contributions

- Code you contribute is licensed **Apache-2.0** (the project license).
- Documentation (`docs/`) is **CC BY 4.0**.
- Changes under `kernel/` stay **GPLv2** to match the upstream kernel.

## Translations

barra's setup wizard and `barra-config` are localized from flat catalogs. (The
dashboard and the kit services are not yet — wiring them up is a welcome
contribution.) To add a language:

1. Copy `src/i18n/en.properties` to `src/i18n/<code>.properties`
   (e.g. `fr.properties`) and translate the values — keys stay unchanged.
2. Copy the same file to `barra-setup/i18n/<code>.properties` (the Windows setup
   reads its own copy).
3. That's it — the loaders pick the file up automatically and fall back to
   English for any missing key. See `src/boot/barra-i18n.sh` for the lookup order.

A translation PR touching only `.properties` files is the easiest possible
contribution and always appreciated.

## Building and testing

Most of barra builds and runs **on the node itself** (the base image ships gcc,
dpkg and the on-device TPU compiler). The typical loop:

```sh
ssh you@your-node
# edit, then:
sh src/barra/build.sh          # libbarra + examples
sh src/sdk/mk-debs.sh          # rebuild + install the .deb packages
barra-smi                      # sanity-check the accelerators
```

The Windows setup tool (`barra-setup/`) is PowerShell; run
`barra-setup.ps1` and watch `%LOCALAPPDATA%\barra\setup.log`.

## Submitting

1. Fork, branch, make your change with signed-off commits.
2. Describe **what you measured** in the PR body if it's a behavior/perf change.
3. Open the PR against `main`. CI and a maintainer review follow.
