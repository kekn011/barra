# Releasing barra

Versioning is automated, publishing is not — on purpose.

## Why the release is created as a draft

barra's release assets are roughly 13 GB of local build artifacts: the base image, the
kernel, the kit archives, TPU packages. **None of them are produced in CI.** If
release-please published a release directly, that release would exist with no files
attached, and `fetch-models.ps1` on a user's machine would fail to find anything.

So release-please creates the GitHub release as a **draft** (`draft: true` in
`release-please-config.json`). A human attaches the assets and publishes.

## The loop

1. **Commit to `main` using [Conventional Commits](https://www.conventionalcommits.org/).**
   `fix:` bumps the patch version, `feat:` the minor, `feat!:` or a `BREAKING CHANGE:`
   footer the major.
2. **release-please opens a release PR** ("chore: release x.y.z") holding the updated
   `CHANGELOG.md`, `version.txt` and `.release-please-manifest.json`.
3. **Bump the release address in the PR.** `barra-setup/models.psd1` carries
   `_release.base` and `_release.tag`; both must name the new tag. The `checks` workflow
   fails the PR until they do — that is deliberate, not a nuisance: a manifest pointing at
   a release that does not exist breaks every fresh install.
4. **Merge the PR.** release-please tags the commit and creates the release **as a draft**.
5. **Attach the assets** from your working copy:

   ```powershell
   cd barra-setup
   .\upload-release.ps1 -VerifyOnly   # size + SHA-256 of every asset against models.psd1
   .\upload-release.ps1               # verifies, then uploads
   ```

   The script derives the tag from `version.txt`, refuses to run if `models.psd1` names a
   different one, and aborts on the first mismatch **before** uploading anything — a
   half-wrong release is worse than none.

   Files over 2 GB exceed GitHub's asset limit and are split
   (`split -b 1600M -d -a 1 <file> <file>.part`); the manifest pins each part separately
   and `fetch-models.ps1` reassembles and verifies the whole.
6. **Publish**: `gh release edit vX.Y.Z --draft=false -R kekn011/barra`.

`SHA256SUMS` is the one deliberate exception to the checksum rule: it is the reference list
itself and therefore carries no pin of its own.

## What CI guards

`.github/workflows/checks.yml` covers the two mistakes this project has actually shipped:

* **Every `.ps1`/`.psd1` carries a UTF-8 BOM.** PowerShell 5.1 reads a BOM-less file as
  ANSI, which turns German text in the setup wizard into mojibake.
* **`models.psd1` points at the current version**, matching `version.txt`.
* **`docs/models.md` is up to date** with the manifest (`mk-model-docs.ps1 -Check`). The
  license table silently lagged nine artifacts behind once.

## A note on binaries in the kits

Some kits ship a compiled binary (for example the GPU vocoder `gpudecd`). Its SPIR-V
shaders are compiled *into* the binary. If a shader source changes, the binary must be
rebuilt — otherwise a newly encoded parameter hits no branch in the old shader and a
computation step falls out **silently**, at full speed, with no error. That shipped once.
`src/tts/prebuilt/gpudecd.buildinfo` pins the checksums of every source the binary was
built from, and `repack-tts-kit.sh` refuses to package a binary that does not match.
