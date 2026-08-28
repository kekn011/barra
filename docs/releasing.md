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
3. **Check the release address.** `barra-setup/models.psd1` carries `_release.base` and
   `_release.tag`, and release-please updates both — they sit on their own annotated lines
   (`x-release-please-version`), one version per line so the generic updater cannot pick the
   wrong one. It leaves the rest of the file, its UTF-8 BOM included, untouched; that was
   verified against a real release PR, not assumed.

   Note that **no CI runs on that pull request**: GitHub does not trigger workflows for pull
   requests opened with the `GITHUB_TOKEN`. The guards below therefore run on `main` after
   the merge, and `upload-release.ps1` refuses to upload when the manifest and `version.txt`
   disagree — which is the point where it would actually hurt.
4. **Merge the PR.** release-please tags the commit and creates the release **as a draft**.
5. **Attach the assets** from your working copy:

   ```powershell
   cd barra-setup
   .\upload-release.ps1 -VerifyOnly   # size + SHA-256 against models.psd1, copies against src/
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
* **`models.psd1` points at the current version**, matching `version.txt`. release-please
  keeps these in step on its own; this is the net under it.
* **`docs/models.md` is up to date** with the manifest (`mk-model-docs.ps1 -Check`). The
  license table silently lagged nine artifacts behind once.

## A note on binaries in the kits

Some kits ship a compiled binary (for example the GPU vocoder `gpudecd`). Its SPIR-V
shaders are compiled *into* the binary. If a shader source changes, the binary must be
rebuilt — otherwise a newly encoded parameter hits no branch in the old shader and a
computation step falls out **silently**, at full speed, with no error. That shipped once.
`src/tts/prebuilt/gpudecd.buildinfo` pins the checksums of every source the binary was
built from, and `repack-tts-kit.sh` refuses to package a binary that does not match.

## A note on the copies inside the artifacts

The base image and every kit carry their own copy of the device scripts, and the base
image carries the i18n catalogue as well. A fix in `src/` therefore does **not** reach the
product on its own — the base image is a snapshot of a dev node, and a kit archive is
whatever was packed the day it was built.

This has shipped three times. Twice as the pre-i18n copy of a server script, and once as a
catalogue without the `stt.` and `pya.` keys, which made the device print raw keys like
`stt.stopped` — the loader answers an unknown key with the key itself, so nothing ever
reported an error.

`upload-release.ps1` now compares every such copy against `src/` and refuses to upload on a
difference. Correcting one does not require a re-bake:

```bash
# a file inside the base image (run in WSL, as root - see the head of the script)
sudo bash repack-base.sh -a ../src/i18n/de.properties=adb/baseos/i18n/de.properties old.tar.gz new.tar.gz

# a file inside a kit archive
python tools/replace-in-kit.py pyannote-kit/pyannote-kit.tar base/pyaserver.sh ../src/pyannote/pyaserver.sh
```

Both keep owner and mode of the entry they replace and verify the result before handing it
over — `repack-base.sh` re-checks the setuid list of a real Ubuntu 24.04, `replace-in-kit.py`
insists that exactly one member changed. Afterwards: pins in `models.psd1` (and
`payload/SHA256SUMS` for the base image), then `mk-model-docs.ps1`.
