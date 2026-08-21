# Flashing barra onto a Pixel 8a

> ⚠️ **This erases the phone.** Unlocking the bootloader wipes all data and
> voids the warranty. Only the **Pixel 8a (akita)** is supported in v0.x.

barra is flashed from a **Windows PC** with the tool in `barra-setup/`. No WSL,
no drivers to install, no admin rights — just `adb`/`fastboot`, which the setup
fetches from Google into `barra-setup\tools` on first run (you accept the
Android SDK terms when it asks).

## Before you start

- A Pixel 8a and its USB-C cable.
- ~20 GB free disk (the Google factory image is a ~3.5 GB download, its
  unpacked partitions ~7 GB, the assembled `super.img` ~6.3 GB).
- The phone's **OEM unlocking** and **USB debugging** toggles, under
  Settings → System → Developer options (tap the build number 7× to reveal it).

## One-time preparation (per release)

barra does not ship Google's firmware, tools, or a pre-patched boot image —
the setup fetches and builds them on the fly, **automatically**, the first time
you flash: platform-tools (adb/fastboot), the factory image (downloaded from
Google after you accept Google's terms, verified against a pinned SHA-256 that
is also the hash in Google's own URL, then assembled into `stock\super.img`
with lpmake), and `init_boot-magisk.img` (patched locally on the connected
phone with Magisk's own tooling — no root needed).

The same steps also exist as standalone scripts, if you prefer to prepare
everything up front:

```powershell
cd barra-setup
.\fetch-stock.ps1          # platform-tools + factory image + stock\super.img
.\patch-initboot.ps1       # payload\init_boot-magisk.img from stock init_boot
                           # + the Magisk APK (phone connected via adb)
```

## Flash

```powershell
.\barra-setup.ps1          # or double-click barra-setup.vbs
```

Then, in the window:

1. **Set up** — user name, password, hostname, Wi-Fi, timezone, an SSH public
   key (optional), and the interface language. This is saved and pre-filled next
   time.
2. **Flash** — plug in the phone and press *Start*. The tool walks through:
   connection → unlock bootloader → stock Android 16 → our kernel + Magisk →
   barra payload → first boot. Follow the on-phone prompts it shows you
   (confirm the unlock, tap through Magisk's one-time setup, etc.).
3. **Done** — the node boots into your Wi-Fi and the screen shows
   `ssh you@<ip>`. First login forces a password change.

Total time: about five minutes of work once the stock image is prepared.

## After flashing

- `ssh you@<node>` → you're in Ubuntu. `sudo barra-config` for settings
  (Wi-Fi, charge limits, display, language, …).
- The screen shows a dashboard; the power button toggles it.
- The LLM server and SDK are covered in the [README](../README.md).

## Undoing it (back to stock)

The setup tool has a **"restore factory state"** button: it reflashes Google
stock Android 16, wipes barra, and optionally re-locks the bootloader.

## Troubleshooting

- **Nothing detected** — replug the cable, turn the screen on, make sure USB
  debugging is on and you accepted the "allow this computer" dialog.
- **A step failed** — press *Start again*; it resumes where it stopped.
- **Details** — the *Details* pane shows the raw log; the full log is at
  `%LOCALAPPDATA%\barra\setup.log`.
- The flasher never downgrades the bootloader; factory-image upgrades are
  anti-rollback-safe.
