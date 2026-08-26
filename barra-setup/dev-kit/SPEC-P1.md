# barra Dev-Kit — P1-Spezifikation (On-Device Compute-Kernel)

Stand 2026-08-25. P1 = MVP aus dem Konzept: On-Device-Shaderbau + Harnesses + Dev-Mode +
Deploy, als 7. Wizard-Karte. Host-Provisioning (NDK/Xtensa/GKI) ist P2+.

**Status: ENTWURF, am Node noch nicht verifiziert.** Die Skripte folgen den Repo-Konventionen
(hw-bridges-Supervisor, base-boot/service.d, server.sh-Pins), sind aber bis zum E2E-Lauf
Hypothesen (arbeitsweise: HW-/Loop-Aussagen erst nach Geraetelauf „Durchbruch").

## 1. Kit-Inhalt (barra-dev-kit.tar.gz)

Android-Seite, weil GPU/TPU-Kernelarbeit in bionic lebt (/dev/mali0, /dev/edgetpu-soc, Vulkan) —
NICHT im Container wie die KI-Apps.

```
dev/            -> /data/adb/baseos/dev          (chmod 755)
  bin/devbuild.sh  bin/devdoctor.sh              [unser Code, src/dev/]
  devkit-env.sh                                  [unser Code]
  gpu-kernels/  (gemv.comp gemm_best.comp gpugemv.c gpugemm.c *build*.sh README)  [src/gpu-kernels/ 1:1]
  dsp-kernels/  (k_*.S vscale_*.S asmpatch2.sh gxpd-test-daemon.sh README)  [src/dsp-kernels, NUR .S/.sh/.md]
        [KEINE *.elf: inbuilt.elf = Vendor-Firmware-Template, ker_*.elf = Kernel darin gespleisst
         -> abgeleiteter Vendor-Blob. inbuilt.elf on-device beschaffen (P2, analog libedgetpu).
         tie/ (libgxp.so, fw_core*.bin) nie einbeziehen. DSP-Build braucht Xtensa-Toolchain = P2.]
  tpu/  tpuc1  patch-compiler-standalone.py  extract-libedgetpu.sh
        [tpuc1 = unser Binary (aus tpuc1.c). libcomp_std.so wird ON-DEVICE erzeugt
         (extract-libedgetpu.sh patcht die Geraete-libedgetpu via COMPILER_SO) und NIE
         mitgeliefert - es ist ein gepatchter Vendor-Blob. tpuc1 laedt es via COMPILER_SO=.../libcomp_std.so]
  third-party/bin/  glslang  frida-server        [LEER im Repo -> fetch-or-supply]
  harness/                                       [optional vorgebaute Binaries]
  VERSION  PINS.txt
base/           -> /data/adb/baseos/bin          (chmod 755)
  barra-dev-mode.sh  devdeploy.sh                [unser Code, src/dev/]
service.d/      -> /data/adb/service.d           (chmod 755)
  55-barra-dev.sh                                [Dev-Mode-Persistenz-Hook]
```

**Aufteilung nach fetch-or-supply (Kevin 25.8.):** gebuendelt = nur unser Code + die
gpu-kernels-Quellen. Gefetcht (fetch-devtools.ps1, gepinnt) = glslang, frida-server.
User-supplied = Xtensa (Vendor, P2). On-Device extrahiert = libedgetpu (extract-libedgetpu.sh,
verlaesst das Geraet nie).

## 2. Bauen (ein Befehl, reproduzierbar)

```
# 1) externe Werkzeuge holen (gepinnt) - schreibt nach dev-kit/third-party/bin/
powershell -File barra-setup/dev-kit/fetch-devtools.ps1
#    frida-server: gepinnt+verifiziert. glslang: -Supply noetig (kein Prebuilt, s. §5).
# 2) packen - MUSS ueber WSL laufen (git-bash chmod greift auf NTFS NICHT -> tpuc1/
#    frida-server/.sh landen als 644 und laufen am Geraet nicht; am 25.8. genau so erlebt):
MSYS_NO_PATHCONV=1 wsl.exe -e sh <repo>/barra-setup/dev-kit/pack-dev-kit.sh
```
`pack-dev-kit.sh` assembliert stage/ aus src/dev + src/gpu-kernels + dev-kit/tpu +
dev-kit/third-party, setzt +x und tart nach barra-dev-kit.tar.gz. **VERIFIZIERT 25.8.:**
tar gebaut (23,7 MB), alle Binaries/Skripte 755 (tpuc1, frida-server, alle .sh, patch-py).
Fehlt nur glslang -> per `fetch-devtools.ps1 -Supply glslang -Path <datei>` nachlegen, dann neu packen.

## 3. Skripte (src/dev/, fertig)

- **barra-dev-mode.sh** `on|off|status|apply` — permissive + Pins (Mali 890/rio 1119MHz/MIF/CPU
  performance) + Wakelock. Snapshot/Restore statt fester Defaults (merkt echten Vorzustand,
  legt ihn bei `off` 1:1 zurueck — genauer als die server.sh-pins_off). Flag $B/dev-mode.on.
- **55-barra-dev.sh** — service.d late_start-Hook: ruft nach Boot `barra-dev-mode apply`, das
  NUR bei gesetztem Flag re-applied -> Dev-Mode ueberdauert Reboot (Entscheidung 3), ohne
  base-boot.sh anzufassen. Reversibel via `off` (loescht Flag).
- **devbuild.sh** `spv|prog` — geraetelokaler Ersatz fuer gpu-kernels/build.sh (dessen ~/glslang-
  und NDK-Pfade sind host-gebunden). `spv` = glslang -V --target-env vulkan1.1 [-D...].
- **devdeploy.sh** `daemon|restart|status` — Daemon-Binary tauschen + killen; der hw-bridges-
  Supervisor respawnt (pgrep-guarded). Fuer den langsamen „Kernel in Daemon backen"-Pfad.
- **devkit-env.sh / devdoctor.sh** — PATH + Kurzbefehle; Selbsttest (Toolchains, /dev/*-Nodes).

Schneller Loop (P1-Kern): .comp editieren -> `devbuild spv` -> Harness laedt die .spv zur
Laufzeit -> messen. Kein Daemon-Neustart. Das ist die eigentliche Kernel-Werkbank.

## 4. Wizard-Anbindung (GETEILTE Dateien — Freigabe noetig, arbeitsweise §10)

Analog zu wake/img. Touch-Points:

**barra-setup/barra-setup.ps1**
- `$KitCatalog`: `@{ id='dev'; name={T 'setup.pkg.dev_name'}; models=@( @{ id='workbench';
  name={T 'setup.model.dev_wb'}; desc={T 'setup.model.dev_wb_desc'}; files=@('dev-kit\barra-dev-kit.tar.gz') } ) }`
- XAML: neue Karte `CardDev`+`ChkDev` auf dem Pakete-Screen; `PkgDev`-Checkbox aufs Fertig-Panel
  (Copy von CardWake/PkgWake).
- `Init-Pkg-Screen` foreach-Zeilen: `@('dev',$ui.CardDev,$ui.ChkDev)` ergaenzen.
- Checkbox-Coex-Schleife: `$ui.ChkDev` in die Liste.
- `BtnPkgNext`: `if ($ui.ChkDev.IsEnabled -and $ui.ChkDev.IsChecked) { $sel+='dev' }`
- `Detect-Kits`: `$script:kitDev = [bool](Get-KitModels 'dev').Count`
- `Show-Pkg-Panel`: `$ui.PkgDev.IsEnabled=$script:kitDev; $ui.PkgDev.IsChecked=$script:kitDev`
- `BtnPkg.Add_Click`: `if ($ui.PkgDev.IsChecked){ $sel+='dev' }` + PkgDev deaktivieren
- Timer-'pkg'-Zweig: `$ui.PkgDev.IsEnabled=$script:kitDev`
- Init-Zeile: `$script:kitDev=$false`

**barra-setup/barra-core.ps1** — `Run-KitsInner`, neuer Zweig (Muster wake/img):
```
elseif ($k -eq 'dev') {
  $name = T 'core.kit.dev_name'
  KitPush (Join-Path $script:Kit 'dev-kit\barra-dev-kit.tar.gz') '/data/local/tmp/barra-dev-kit.tar.gz' $name
  $r = AdbSuBg ($KitDiag + '<warte-auf-fs>; cd /data/local/tmp && rm -rf barra-dev && mkdir barra-dev && cd barra-dev && tar -xzf ../barra-dev-kit.tar.gz && B=/data/adb/baseos && mkdir -p $B/dev && cp -a dev/. $B/dev/ && chmod -R 755 $B/dev && cp base/barra-dev-mode.sh base/devdeploy.sh $B/bin/ && chmod 755 $B/bin/barra-dev-mode.sh $B/bin/devdeploy.sh && cp service.d/55-barra-dev.sh /data/adb/service.d/ && chmod 755 /data/adb/service.d/55-barra-dev.sh && cd /data/local/tmp && rm -rf barra-dev barra-dev-kit.tar.gz') 900 (T 'core.kit.extract' $name)
  # KEIN Auto-Start, KEIN Auto-Dev-Mode (opt-in).
}
```

**barra-setup/i18n/{de,en}.properties** — neue Schluessel (Modelltext nach dem festen Schema,
angepasst: Was installiert wird / Was es entsperrt / Einordnung):
- `setup.pkg.dev_name`  = "Dev-Kit (Kernel-Werkbank)" / "Dev-Kit (kernel workbench)"
- `setup.model.dev_wb`  = "Werkbank" / "Workbench"
- `setup.model.dev_wb_desc` = Herkunft (unser Dev-Setup) + „Installiert: glslang, Harnesses,
  TPU-Standalone-Compiler, Dev-Mode-Umschalter" + Einordnung (permissive nur fuer Dev-Geraete)
- `core.kit.dev_name`   = "Dev-Kit"

## 5. Pin-Manifest (fetch-devtools.ps1 `$Pins`)

- **frida-server — GEPINNT + VERIFIZIERT (25.8.):**
  URL `https://github.com/frida/frida/releases/download/17.17.0/frida-server-17.17.0-android-arm64.xz`,
  .xz-SHA256 `09d1fad867b27d69562a79289f4c412e85867f5d38ab72877036ed35e4223021` (16.168.344 B).
  Live geladen+entpackt; entpackte SHA `55ef78c3…588e7` = identisch zur RE-Archiv-Kopie
  (bestaetigt: Archiv-frida IST das offizielle Release). fetch entpackt via python-lzma.
- **glslang — P2 GELIEFERT + AM NODE VERIFIZIERT (25.8.).** aarch64-android via NDK r27c aus
  glslang @23076b37 cross-gebaut (host/build-glslang-android.sh; Android-Binary-Sperre im
  CMakeLists gepatcht, ENABLE_OPT=0, c++_static -> NEEDED nur libm/libdl/libc). Gestrippt 3,36 MB,
  sha256 1741431b...bac, via -Supply ins Kit. **devbuild spv gemv.comp -> gueltige .spv (Magic
  07230203) am Geraet** -> On-Device-Shaderbau grün. devbuild.sh laeuft Android-seitig, kein Umbau.
- **DSP (Xtensa) — P2 VERIFIZIERT (25.8.).** Bau ist HOST-seitig (esp-elf 14.2.0 = x86->Xtensa-Cross,
  kein On-Device-Xtensa): `dsp-kernels/asmpatch2.sh k_x.S 14c ker_x.elf` splict in inbuilt.elf
  (host-seitiges Vendor-Splice-Template, user-supplied — nicht mitgeliefert). **Host-Rebuild von
  ker_vadd.elf BIT-IDENTISCH zum Geraete-Kernel (sha 50e9d97f...)** -> Baukette korrekt. Deploy:
  `devdeploy dspkernel ker_x.elf` -> KDIR + gxpd3-Respawn (am Node getestet, PID-Wechsel, Kernel geladen).
  gxpd3 + die ker_*.elf laufen ohnehin schon im Base-Image.

**Scope-Praezisierung Baupfade:** GPU-Shader = ON-DEVICE (glslang aarch64); TPU = on-device (tpuc1
+ libcomp_std via extract-libedgetpu.sh, alles am Node bewiesen); DSP = HOST (asmpatch2/Xtensa) ->
ker_*.elf -> deploy; GKI = Host (AOSP). Alle drei Beschleuniger grün.

**GKI-Loop GEWIRED (nicht ausgefuehrt):** host/build-gki.sh automatisiert Patch+Bazel gegen einen
gesyncten AOSP-Baum -> boot-lz4.img -> reflash (barra-setup Kernel-Schritt). Der repo-sync (~30 GB)
+ Bazel-Build + Live-Reflash ist der schwere Einmal-Vorlauf des Entwicklers (Rezept kernel/README.md,
Pins kernel/PINS.txt) und wurde bewusst NICHT in-session gefahren (Zeit + Reflash-Risiko).

## 6. Abnahme am Node (E2E) — GRUEN am 25.8. (Pixel 8a)

1. **install + devdoctor: PASS.** DEV_OK; devdoctor grün (GPU-Shader/Harness, tpuc1, DSP-Quellen,
   frida, /dev/mali0|edgetpu-soc|gxp). Erwartet FEHLT: glslang (P2), libcomp_std (on-device-gen).
2. **dev-mode on/off + Snapshot-Restore: PASS (exakt).** on -> Permissive + Mali 890/890/890 +
   rio perf + MIF perf + Flag; off -> exakt Vorzustand (Enforcing, Mali 150000, MIF interactive,
   cpu sched_pixel), Flag+Saved weg. SAVED hielt den echten Vorzustand.
3. **Persistenz über Reboot: PASS (2 Reboots).** Hook re-applied (Log "dev-mode re-applied (Boot)"),
   nach Reboot weiter Permissive + voll gepinnt.
4. **devdeploy restart gpud-zc: PASS.** Supervisor respawnt (neue PID, Log "Vulkan bereit").
5. **Wizard-Karte + Modellscreen: PASS (Screenshot).** Karte "Dev-Kit (Kernel-Werkbank)" + Modell
   "Werkbank (GPU/TPU/DSP-Kernel)", i18n de sauber. (barra-setup/dev/shot-dev-pkg.png, shot-dev-model.png)
6. **devbuild spv: PASS (25.8., nach P2-glslang).** gemv.comp -> gemv_test.spv (6788 B, Magic
   07230203) am Geraet.
7. **libcomp_std On-Device-Gen + TPU-Standalone-Compile: PASS (25.8.).** extract-libedgetpu.sh
   (python-frei, dd-Patch der Geraete-libedgetpu, 4 Fixes mit Altwert-Verifikation, 0,34 s) ->
   libcomp_std.so; `COMPILER_SO=... tpuc1 t_add_D8.tflite` -> rc=0, status="OK", 12544-B-Package.

### Bugs via E2E gefunden + gefixt (25.8., alle verifiziert)
- install-dev.ps1 Schritt 3: brauchte `su` (/data/adb ist Magisk-root-700, Dev-Kit ist root-only).
- barra-dev-mode.sh: Helfer `r`/`w` kollidierten mit Android-mksh-Alias `r`=`fc -s` (History) ->
  alle sysfs-Reads leer (Snapshot/Status blind). Umbenannt zu `rd`/`wr` (+ unalias-Guard).
- devdeploy restart: 2s-Check zu kurz für den Supervisor-Loop -> 15s-Poll.
- 55-barra-dev.sh Hook: Mali-hint-Race (base-boot setzt hint nach boot_completed zurück) ->
  Hook wartet jetzt auf base 'fertig', dann apply (verifiziert: hint danach 890).
- DSP fehlte im Kit -> src/dsp-kernels gebündelt, devdoctor-Check, Modelltext GPU/TPU/**DSP**.
