# pixel-cluster-base — Pixel 8a als vollwertiger Linux-Cluster-Node

Ziel: Eine Basis, auf der ein Pixel 8a (`akita`, Tensor G3) unter einem Ubuntu-System
**alle Hardware nutzbar macht** — CPU, GPU (Mali-G715) und TPU (Tensor/EdgeTPU) — und dabei
**MicroK8s-fähig** ist, also echte Container und Orchestrierung trägt.

Fernziel: einen Raspberry-Pi-Cluster durch Pixel-Geräte ersetzen. Ein gebrauchtes Pixel 8a
bringt 8 Kerne, 8 GB RAM, UFS-Speicher und einen Akku als USV zum Preis eines Pi 5 mit
Netzteil, Gehäuse und SSD.

Vorgeschichte: [`pixel-whisper-node`](../pixel-whisper-node/README.md) — dort steht der
Transkriptionsdienst, die Benchmarks und der gesamte Erkenntnisstand zu Gerät, Flashen und
Kernel-Bau. **Dieses Projekt setzt darauf auf, ersetzt es nicht.**

Stand: 2026-08-09 — Planung, noch nichts umgesetzt.

## Die drei Blockaden

Aus `pixel-whisper-node` übernommen, alle experimentell belegt:

| # | Blockade | Ursache | Status |
|---|---|---|---|
| 1 | **TPU nicht nutzbar** | `/dev/rio` und `libedgetpu_litert.so` sind da, aber bionic-Binaries — im glibc-chroot nicht ladbar | offen |
| 2 | **GPU nicht nutzbar** | Mali-G715 hängt an `/dev/mali0` (proprietär). `/dev/dri/card0` ist `exynos-drm`, der Display-Controller. Kein `panfrost` → kein Mesa/PanVK | offen |
| 3 | **Keine Container** | `CONFIG_PID_NS`/`IPC_NS`/`USER_NS` nicht einkompiliert. `unshare --pid` → `EINVAL` | offen |

**1 und 2 hängen nicht am Kernel.** Der Kernel exponiert die Geräte bereits; das Hindernis ist
ausschließlich, dass bionic-Bibliotheken sich nicht in glibc-Prozesse laden lassen. Lösungsweg
wäre `libhybris` oder ein Build gegen das Android-NDK, der außerhalb des chroots läuft.
Beides ungetestet.

**3 hängt am Kernel** und ist der Grund für den Kernel-Bau.

## Warum der erste Kernel-Versuch scheiterte

Am 2026-08-09 wurde ein Kernel **6.1.124** aus dem AOSP-Manifest `android-gs-akita-android16`
gebaut, mit allen Container-Optionen, verifiziert im Image, korrekt geflasht — und er bootet
nicht. Das Gerät fällt in den Fastboot zurück.

**Methodischer Fehler dabei:** Es wurden *zwei* Dinge gleichzeitig geändert — die
Kernelversion (6.1.175 → 6.1.124) und die Konfiguration. Welche der beiden den Boot
verhindert, war danach nicht unterscheidbar.

**Regel für dieses Projekt: immer nur eine Variable ändern.**
1. Zuerst den **unveränderten** Kernel derselben Quelle bauen und booten lassen. Das beweist
   die Build- und Flash-Kette.
2. Erst danach Optionen hinzufügen, und bei Fehlschlag einzeln eingrenzen.

**Diagnosemittel, das bisher ungenutzt blieb:** Nach einem gescheiterten Boot liegen die
letzten Kernelmeldungen im pstore — beim nächsten erfolgreichen Start unter
`/sys/fs/pstore/`, typischerweise `console-ramoops`. Das ist der erste Griff nach einem
Fehlschlag, statt zu raten.

## Entscheidung 2026-08-09: GrapheneOS bleibt, Kernel aus derselben Quelle

Der Basiswechsel entfällt. `probe-graphene-branch.sh 16-kernel-3` zeigt: **24 von 25
Projekten lösen sauber auf**, inklusive `kernel_devices_google_akita` und aller Modul-Repos.
Es fehlt genau eines — `kernel_common-6.1` zeigt auf `refs/heads/16`, und dieser Branch ist
gelöscht. Vorhanden sind `16-qpr2` (6.1.174) und `17` (6.1.176).

Zusammen mit dem früheren Befund, dass der Commit des laufenden Kernels
(`6.1.175-…-g72524ede2ba4`) nachweislich in `GrapheneOS/kernel_common-6.1` liegt, ergibt sich
GrapheneOS' eigenes Baurezept:

```
Manifest 16-kernel-3  +  kernel_common-6.1 auf Branch 17
```

Sie haben dafür nur keinen eigenen Manifest-Branch geschnitten. Es genügt, **eine einzige
Revision umzubiegen** — dieselbe Technik, die `sync-kernel.sh` schon für das AOSP-Manifest
nutzt. `sync-graphene-kernel.sh` macht genau das und prüft danach, ob es gegriffen hat.

Damit bleibt alles erhalten, was gegen einen Wechsel sprach:

| | |
|---|---|
| GrapheneOS | bleibt — 228 Pakete, keine Play-Dienste (gemessen; Stock liegt bei 400–500) |
| Android-Version | bleibt 17, kein Downgrade |
| Anti-Rollback | kein Thema, Bootloader wird nicht angefasst |
| Kernel | 6.1.176 aus derselben Quelle wie der laufende 6.1.175 |

Offen bleibt allein, ob GrapheneOS' Härtung mit den Container-Optionen kollidiert — und das
wird diesmal in der richtigen Reihenfolge geprüft: erst der **unveränderte** 6.1.176.

## Verworfen: Basiswechsel auf Stock Android

Der Kernel muss zum Userspace passen. Genau daran ist Versuch 1 vermutlich gescheitert:
6.1.124 ist die AOSP-**Android-16**-Basis, das Gerät läuft aber unter **Android 17**.

| Variante | Kernelquelle | Bewertung |
|---|---|---|
| **A: GrapheneOS 17 behalten** | `kernel_common-6.1` Branch `17` → **6.1.176** | Quelle nachgewiesen vorhanden, Commit des laufenden Kernels liegt darin. Aber **kein Manifest-Branch** — muss von Hand gebaut werden. GrapheneOS-Härtung könnte mit Container-Optionen kollidieren |
| **B: Stock Android 16 flashen** | AOSP `android-gs-akita-android16` → **6.1.124** | **Das gepaarte Original.** Der bereits gebaute Kernel würde vermutlich booten. Kein Manifest-Basteln, keine Härtung. Risiko: **Anti-Rollback** beim Bootloader-Downgrade — vor dem Versuch zwingend prüfen |
| **C: Stock Android 17** | — | Für akita gibt es keine android17-Kernelquellen. Googles android17-Linie ist 6.18 und hat bisher nur ein Manifest für gs101 (Pixel 6) |

**B ist der aussichtsreichste Weg**, weil dort Kernel und Userspace aus derselben Quelle
stammen und der fertige Build von Versuch 1 direkt wiederverwendbar wäre. Er verlangt aber
einen vollständigen Reflash mit Wipe — und die Anti-Rollback-Prüfung vorher ist Pflicht, sonst
droht ein unbrauchbarer Bootloader.

## Anti-Rollback: gemessen 2026-08-09

`fastboot getvar all` im Bootloader:

```
ap-ar-ns:3      ap-ar-s:3
gsa-bl1-ar:3    gsa-fw-ar:3
```

**Der ARB-Index des Geräts ist 3.** Firmware mit niedrigerem Index lehnt der Bootloader ab —
und da es der Bootloader selbst ist, der dann nicht mehr startet, ist das der Fall, in dem ein
Gerät wirklich unbrauchbar wird. Welchen Index das Android-16-Image trägt, veröffentlicht
Google nicht direkt.

**Die Frage ist umgehbar.** Anti-Rollback betrifft nur Bootloader und Radio. Für einen
Basiswechsel genügt es, `boot`, `dtbo`, `vendor_boot`, `vendor_kernel_boot`, `super` und
`vbmeta` aus dem Android-16-Image zu flashen und den **Android-17-Bootloader zu behalten** —
dann kommt ARB nie ins Spiel.

Weitere Eckdaten aus demselben Auslesen: `secure-boot:PRODUCTION`, `unlocked:yes`,
`max-download-size:0xf900000`, `super` 0x1fc800000 (8,5 GB), `userdata` 0x1b6a215000 (110 GB),
Speicher Samsung 128 GB UFS 3.1, RAM 8 GB LPDDR5 Micron.

## Baseline-Test 2026-08-09: die Version ist schuld, nicht die Konfiguration

Der 6.1.124 wurde **unverändert** gebaut — unberührte `gki_defconfig`, keine einzige
Container-Option, verifiziert über die im Image eingebettete Config (`CONFIG_PID_NS` korrekt
*nicht* gesetzt). Vollständig geflasht in der richtigen Reihenfolge, alle Partitionen `OKAY`.

**Er bootet ebenfalls nicht.** Das Gerät landet wieder im Fastboot.

Damit ist die Frage entschieden: **Die Container-Optionen waren nie das Problem.** Ein 6.1.124
aus dem AOSP-Zweig `android-gs-akita-android16` läuft nicht unter dem Android-17-Userspace
dieses Geräts, gleich wie er konfiguriert ist. Variante B (Basis auf Android 16 wechseln) oder
ein Kernel aus der passenden Quelle ist damit begründet statt vermutet.

Kosten des Tests: ein Build (16 min) und ein Flash-Zyklus. Er hätte am Vortag als Erstes
laufen müssen — dann wäre die Fehlersuche über die Config-Optionen entfallen.

**Warum genau er nicht bootet, bleibt offen — der pstore ist leer.** Nach Wiederherstellung
von root (Magisk-APK installieren, App öffnen, Umgebung einrichten, neu starten) enthielt
`/sys/fs/pstore/` **keinen einzigen Eintrag**, weder `console-ramoops` noch `dmesg-ramoops-*`.
`/proc/last_kmsg` existiert nicht.

Der Kernel hinterlässt also **keinen Panic**. Entweder stirbt er, bevor pstore initialisiert
ist, oder er panickt gar nicht — er hängt, oder das Image wird vor der Ausführung abgewiesen.

**Besseres Diagnosemittel für den nächsten Fehlschlag**, direkt danach im Bootloader
abzufragen, nicht später:

```
fastboot getvar slot-retry-count:a
```

Steht er auf 0, hat der Bootloader den Kernel gestartet und nach drei Versuchen aufgegeben —
der Kernel läuft an und stirbt. Ist er unverändert (3), wurde das Image gar nicht erst
akzeptiert. Das trennt zwei völlig verschiedene Ursachen und kostet einen Befehl.

## GKI-Weg 2026-08-09: Kernel bitgenau reproduziert, bootet trotzdem nicht

Der wichtigste Fortschritt des Tages, auch ohne bootenden Kernel.

**Vendor-Module muss man gar nicht bauen.** GrapheneOS tut es nicht — sie bauen nur den
GKI-Kernel und behalten Googles vorgefertigte Module. Dafür ist GKI da. Die veröffentlichten
Modulquellen (`soc_gs` u. a.) sind vom **2024-08-29** und passen zu keinem verfügbaren
Kernelstand; sie gegen einen 2026er Kernel zu übersetzen geht niemand sonst. Zwei Patches
(dwc3-Makrokollisionen, VLA im Watchdog) waren nötig, bevor klar wurde, dass der Weg falsch ist.

**Damit genügt `//aosp:kernel_aarch64_dist` und das Flashen von `boot.img` allein** — kein
fastbootd, keine logischen Partitionen, kein `super`.

**Die Modulkompatibilität ist vollständig reproduzierbar.** Sie hängt an zwei Dingen, die
beide im `vermagic` stehen:

```
vermagic=6.1.175-android14-11-g72524ede2ba4 … RANDSTRUCT_c4416c5905e99ad4…
```

| Bestandteil | Herkunft |
|---|---|
| `g72524ede2ba4` | Git-Commit des Kernelbaums, über `--config=stamp` |
| `RANDSTRUCT_…` | sha256 des Build-Zeitstempels, dieser aus `SOURCE_DATE_EPOCH` = Commit-Zeitstempel |

`kernel_common-6.1` trägt **Release-Tags** (`2026080500` = der Build auf dem Gerät). Auf den
Tag gepinnt steht der Baum auf Commit `72524ede2ba4` mit Zeitstempel `1785881916` =
`Tue Aug  4 22:18:36 UTC 2026`. Daraus folgt exakt der RANDSTRUCT-Hash der Geräte-Module —
**nachgewiesen, Zeichen für Zeichen**.

Kuriosum am Rande: `gen-randstruct-seed.sh` hasht mit `echo -n $KBUILD_BUILD_TIMESTAMP`
**ohne Anführungszeichen**. Die Shell zerlegt den Wert und fügt ihn mit einfachen Leerzeichen
zusammen — der Doppelabstand aus `Aug  4` fällt weg, bevor gehasht wird. Mit dem korrekt
zitierten String trifft man den Hash nie.

**`boot.img` muss lz4-komprimiert sein.** Das dist-Verzeichnis liefert drei Varianten; die
richtige ist **`boot-lz4.img`**:

```
Stock:            02 21 4c 18   lz4
boot.img:         4d 5a         "MZ", unkomprimiert  <- falsch
boot-lz4.img:     02 21 4c 18   passt
```

**Es bootet trotzdem nicht.** Weder die unkomprimierte noch die lz4-Variante. Da der Kernel
aus demselben Commit mit derselben Config stammt und im `vermagic` bitgleich ist, liegt die
Ursache **außerhalb von Quelle und Konfiguration** — im Zusammenbau oder in der Signatur des
Boot-Image.

Zwei bekannte Unterschiede, beide ungeprüft:

| | Stock | unser Build |
|---|---|---|
| `os_version` | `17.0.0`, Patchlevel `2026-08` | **0** |
| AVB-Signatur | mit GrapheneOS' Schlüssel | keine |

### Die Reproduktion ist gelöst — vier Bedingungen, keine davon offensichtlich

Nach dem vollständigen Vergleich mit `/proc/config.gz` des laufenden Kernels:

| Prüfung | Ergebnis |
|---|---|
| Commit | `72524ede2ba4` — identisch |
| Config | **2141 Zeilen, null Abweichungen** in beide Richtungen |
| `vermagic` inkl. RANDSTRUCT | bitgleich |
| komprimierte Kernelgröße | `18.936.561` Byte — **byte-identisch** zum Stock-Image |

Nötig sind dafür vier Dinge, und fehlt eines, weicht das Ergebnis ab, **ohne dass eine
Fehlermeldung darauf hinweist**:

```
repo-Revision:  refs/tags/2026080500      Release-Tag, nicht Branch
                --config=stamp            scmversion + SOURCE_DATE_EPOCH -> RANDSTRUCT-Seed
                --lto=full                sonst CONFIG_LTO_NONE, 21 % Groessenunterschied
                --disable_slab_canary     aus GrapheneOS' build_akita.sh
```

**Das entscheidende Werkzeug war `/proc/config.gz`.** Der laufende Kernel trägt seine
Konfiguration selbst mit sich. Ein vollständiger Diff dagegen zeigte den Unterschied in
Sekunden — nach einem Umweg über Signaturprüfung, Kompressionsformat und Boot-Image-Header,
die alle nichts damit zu tun hatten. **Nicht gegen die Annahme prüfen, wie der Hersteller
baut, sondern gegen sein Ergebnis.**

### Es bootet trotzdem nicht — die Ursache liegt in der Verpackung

Auch die originalgetreue Reproduktion kommt nicht hoch. `slot-retry-count:a` fällt jedes Mal
von 3 auf 1: der Bootloader **startet** das Image und scheitert danach, er lehnt es nicht ab.

Geprüft und ausgeschlossen:

| Hypothese | Ergebnis |
|---|---|
| Falsche Kompression | `boot-lz4.img` ist formatgleich (`02 21 4c 18`) — half nicht |
| `fastboot oem disable-verification` | half nicht |
| `vbmeta` mit `--disable-verity --disable-verification` | half nicht |
| Kernel-Config | inzwischen identisch — war es also nicht |

**Einzig verbliebener bekannter Unterschied:** `os_version` im Boot-Header ist bei uns `0`,
beim Stock-Image `17.0.0` mit Patchlevel `2026-08`. Dazu trägt das aus dem Kernelbau
stammende `boot.img` keinen AVB-Hash-Footer — GrapheneOS signiert es erst im OS-Build,
nicht im Kernelbau.

### Alles Beobachtbare stimmt überein — es bootet trotzdem nicht

`repack-bootimg.sh` packt mit `mkbootimg --os_version 17.0.0 --os_patch_level 2026-08` und
setzt den Footer mit `avbtool add_hash_footer` neu. Danach ist **jedes prüfbare Feld gleich**:

| Feld | Stock | unser Repack |
|---|---|---|
| `kernel_size` | 18.936.561 | 18.936.561 |
| `os_version` | 570425768 | 570425768 |
| `header_version` | 4 | 4 |
| `signature_size` | 0 | 0 |
| cmdline | leer | leer |
| AVB-Footer | vorhanden | vorhanden |

Dazu identischer Commit, identische Config (2141 Zeilen, null Abweichungen), bitgleiches
`vermagic`, byte-identische Kernelgröße. Geflasht zusammen mit `vbmeta --disable-verity
--disable-verification`.

**Das Gerät bleibt im Fastboot.** `slot-retry-count:a` fällt jedes Mal von 3 auf 1.

Damit ist die Fehlersuche an ihrer Grenze: Es gibt keinen weiteren Unterschied mehr, den die
verfügbaren Werkzeuge zeigen. Was übrig bleibt, ist unsichtbar — denkbar wären PGO- oder
BOLT-Schritte auf dem Kernel, eine Signatur, die der Bootloader trotz deaktivierter
Verifikation erwartet, oder etwas, das GrapheneOS außerhalb des Kernel-Repos beisteuert.

**Bewertung:** Der GrapheneOS-Weg verlangt, einen Build exakt nachzubilden, den der Hersteller
nicht dokumentiert. Vier nicht offensichtliche Bedingungen waren nötig, um überhaupt so weit
zu kommen; die fünfte ist nicht auffindbar. Für ein Vorhaben, das auf **mehrere Geräte**
skalieren und Updates überleben soll, ist das die falsche Grundlage — jedes GrapheneOS-Update
würde die Übung wiederholen.

**Empfehlung: Basis auf Stock Android 16 wechseln.** Dort veröffentlicht Google Kernel *und*
Vendor-Module aus einem Guss (`android-gs-akita-android16`, 6.1.124), der Build vom
2026-08-08 liegt fertig und verifiziert vor, und die gesamte Reproduktionsakrobatik entfällt.
Preis: kein GrapheneOS mehr, dafür ein Debloat-Durchgang (Stock hat 400–500 Pakete,
GrapheneOS 228). Anti-Rollback ist umgehbar, indem Bootloader und Radio nicht angefasst werden.

## Umsetzung Basiswechsel: vorbereitet 2026-08-09

**Image:** `akita-bp4a.260205.001` (Android 16.0.0, Feb 2026) — der **letzte
Android-16-Build** für akita; ab `cp1a.260305` beginnt die Android-17-Linie.
Heruntergeladen nach `images/`, SHA-256 verifiziert (`661cb49b…`), entpackt nach
`images/akita-bp4a.260205.001/image/`.

**Befund: Stock BP4A trägt Kernel 6.1.145**, nicht 6.1.124:

```
Linux version 6.1.145-android14-11-gc1de4747ac59-ab14219743 … (+pgo, +bolt, +lto)
```

Zwei Konsequenzen:

1. Das Manifest `android-gs-akita-android16` (Quelle der fertigen Builds) ist auf
   dem Stand der *initialen* A16-Releases von Mitte 2025 eingefroren. Der
   6.1.124-Baseline-Test unter BP4A ist damit nicht mehr streng „eine Variable" —
   aber dank identischer KMI `android14-11` und eigener Module im Build-Set
   trotzdem der richtige erste (kostenlose) Versuch. Schlägt er fehl: Manifest für
   6.1.145 suchen (`ab14219743`), Baseline neu bauen (16 min), erneut testen.
2. `+pgo, +bolt` im Versionsstring: Googles Produktionskernel wird mit PGO und
   BOLT gebaut — genau eine der vermuteten unsichtbaren Differenzen, an denen die
   GrapheneOS-Reproduktion scheiterte. Der Verdacht war richtig.

**`android-info.txt` verlangt exakt `version-bootloader=akita-16.4`** — wir
behalten aber den 17.0er Bootloader (ARB). Deshalb manuelles Flashen statt
`fastboot update`; das prüft die Anforderung nicht.

**`scripts/flash-stock-base.sh`** (WSL, wiederaufnehmbar) folgt Googles eigenem
Rezept aus `fastboot-info.txt` im Image-Zip, ohne `bootloader`/`radio`:

1. Bootloader-Modus: `avb_custom_key` löschen (GrapheneOS-Schlüssel), dann
   `boot init_boot dtbo vendor_kernel_boot pvmfw vendor_boot vbmeta
   vbmeta_system vbmeta_vendor`.
2. `reboot fastboot` — **zugleich der Test**: fastbootd läuft schon auf dem neuen
   A16-Kernel. Kommt er nicht hoch, ist super noch unangetastet GrapheneOS und der
   Rückweg trivial. Erst danach wird super neu aufgebaut
   (`wipe-super` + `system system_dlkm system_ext product vendor vendor_dlkm`).
3. Wipe (`userdata` + `metadata`) nur mit `WIPE_OK=1` — ohne stoppt das Skript davor.

## Basiswechsel vollzogen: 2026-08-09

`flash-stock-base.sh` lief vollständig durch (Phase 1 aus Windows-fastboot, Phase 2+3
aus WSL, ~6,4 GB über usbipd bei ~5 MB/s ≈ 25 min). **Stock Android 16 bootet:**
`BP4A.260205.001`, Kernel 6.1.145, Patchlevel 2026-02-05, Bootloader unverändert
`akita-17.0` — die Mischkonfiguration A17-Bootloader + A16-System funktioniert,
ARB wurde nie berührt.

Bestätigt nebenbei die These von gestern: Der nicht bootende 6.1.124 lag am
Userspace, nicht an der Config. Und: fastbootd bootete bereits nach Phase 1 mit dem
A16-Kernel — der eingebaute Zwischentest hat sich gelohnt.

## 6.1.124-Baseline unter Stock: Kernel läuft, Userspace nicht

Der fertige Baseline-Build wurde geflasht (dlkm via fastbootd, dann Kernel — Skript
aus `pixel-whisper-node`). Ergebnis, sauber getrennt:

| Ebene | Ergebnis |
|---|---|
| Kernel 6.1.124 | **bootet** — die Recovery lief darauf, `uname` bewiesen |
| Android-Userspace | stirbt **früh** — adbd kam nie hoch (Beobachter-Log), Gerät fiel nach verbrauchten Retries in den Fastboot |
| fastbootd | kommt mit dem 6.1.124-Satz ebenfalls nicht hoch (`bootreason reboot,fastboot` → Recovery) |

Der Modulsatz aus dem eingefrorenen Mitte-2025-Manifest passt nicht zum
Feb-2026-Vendor-Userspace. **Rückweg:** `restore-stock-kernel.sh` — Reihenfolge
bewusst umgedreht (Boot-Kette zuerst aus dem Bootloader, denn fastbootd braucht
eine funktionierende Kette; das Übliche „erst fastbootd" gilt nur beim Flashen
*neuer* Module über eine noch intakte alte Kette). Stock bootet wieder.

## Der Durchbruch im vermagic: Module binden per KMI, nicht per Commit

`strings` über die Stock-dlkm-Images:

```
vendor_dlkm:  vermagic=6.1.145-android14-11-g8dd788b6a8b4-ab14305268 …
system_dlkm:  vermagic=6.1.145-android14-11-gc1de4747ac59-ab14219743 …
Boot-Kernel:                    6.1.145-android14-11-gc1de4747ac59-ab14219743
```

Drei Konsequenzen:

1. **Google mischt selbst Commits** — vendor_dlkm-Module aus einem anderen
   Kernel-Build als der Boot-Kernel. Modulkompatibilität hängt an der KMI
   `android14-6.1`, nicht am exakten vermagic. Die Commit-genaue Reproduktion
   ist unnötig.
2. **Kein RANDSTRUCT** (`CONFIG_RANDSTRUCT_NONE=y` in `/proc/config.gz`) — die
   Seed-Akrobatik war eine GrapheneOS-Härtung. Entfällt bei Stock komplett.
3. `CONFIG_LTO_NONE=y` — das „+lto" im Versionsstring beschreibt den
   Clang-*Toolchain*-Build. Stock-Kernel ist ohne LTO gebaut (`--lto=none`).

Dazu gemessen: Unser eigen gepacktes `boot.img` (os_version 0, ohne AVB-Signatur)
**bootet unter Stock anstandslos** — das unauffindbare fünfte Puzzleteil der
GrapheneOS-Sackgasse ist ein GrapheneOS-Spezifikum, kein generelles.

Der exakte Stock-Commit `c1de4747ac59` hängt übrigens an **keinem öffentlichen
Branch oder Tag** von `kernel/common` (Ancestry via Gitiles-Range-API geprüft:
android14-6.1, -lts, -sp, -2025-09/-12, -2026-03/-06 — alle negativ; direkter
SHA-Fetch verboten). Brauchen wir dank 1. aber auch nicht.

## Neuer Weg: GKI 6.1.157 aus ACK, nur boot.img

`android14-6.1-2025-12` ist der nächstgelegene **öffentliche, aktiv gepflegte**
Stand der KMI (6.1.157, letzter Commit 2026-08-04). `build-gki-157.sh`:
`//aosp:kernel_aarch64_dist`, `--config=stamp`, `--lto=none` — alle Google-Module
und dtbo bleiben unangetastet, geflasht wird **nur boot.img** aus dem
Bootloader-Modus. Kein fastbootd, kein Wipe, Iteration ≈ 2 min.

## DER DURCHBRUCH 2026-08-09: vbmeta war der Täter — die ganze Zeit

Der erste 6.1.157-Flash fiel wieder in die Recovery. Die Fehlersuche danach, in
Reihenfolge der Experimente:

| Experiment | Ergebnis |
|---|---|
| Recovery auf 6.1.157: `/proc/modules` | **212 Google-Module geladen** — KMI/CRC-Bindung funktioniert einwandfrei |
| `same_magic()` in `kernel/module/version.c` | Bei MODVERSIONS wird der **komplette Versionsteil des vermagic ignoriert** (Upstream-Verhalten!), Bindung läuft über Symbol-CRCs |
| Googles eigenen 6.1.145-Kernel byteidentisch extrahiert, mit unserem mkbootimg neu verpackt | **fällt ebenfalls in die Recovery** → Kernelinhalt vollständig entlastet |
| vbmeta mit `--disable-verity --disable-verification` geflasht | fällt in die Recovery |
| Kontrollversuch: **pures Stock-boot.img** | **fällt AUCH in die Recovery!** → das Image war gar nicht mehr der Diskriminator |
| `fastboot erase misc` (BCB!), Stock-vbmeta zurück, `set_active a` | **Stock bootet wieder** |
| **6.1.157 GKI, vbmeta unangetastet, misc sauber** | **ANDROID BOOTET.** 263 Module, zygote läuft |

**Wurzelursache:** Pixel bindet die FBE-/Keymint-Schlüssel für `/data` an den
Verified-Boot-Zustand. Jede vbmeta-Manipulation — `fastboot oem
disable-verification` genauso wie `--disable-verity --disable-verification` —
ändert diesen Zustand, `/data` bleibt zu, init fällt in die Recovery
(bzw. Bootschleife → Fastboot). Deshalb warnen Magisk-Guides für Pixel explizit
davor; Custom-Kernel werden dort **ohne** vbmeta-Änderung geflasht — der
entsperrte Bootloader toleriert die AVB-Abweichung, der vbmeta-Digest bleibt
unverändert, die Schlüssel passen.

**Verkettung der Fehldiagnosen:** `flash-kernel.sh` (pixel-whisper-node) führt
`oem disable-verification` aus — der 6.1.124-„Userspace-Fehlschlag" von heute
Vormittag und mutmaßlich **die gesamte GrapheneOS-Sackgasse** (bitgenaue
Reproduktion, os_version, AVB-Footer, Kompressions-Verdacht) waren Folgen dieser
einen Zeile. Dazu kam als zweite Falle der **Boot-Control-Block in `misc`**:
Nach einem Recovery-Fallback zwingt er *jeden* weiteren Boot in die Recovery und
vergiftet alle Folgeexperimente, bis man ihn löscht.

**Neue Standardprozedur für Kernel-Iterationen (≈3 min):**
```
fastboot flash boot boot-lz4.img        # vbmeta NIEMALS anfassen
fastboot reboot
# nach jedem Fehlversuch, VOR dem naechsten Experiment:
fastboot erase misc && fastboot set_active a
```

Offen/beobachten: `oem disable-verification` wurde einmal ausgeführt und wirkt
möglicherweise als persistenter Bootloader-Zustand unabhängig vom
vbmeta-Partitionsinhalt — nach Stock-vbmeta-Reflash bootete alles, also ist er
entweder dadurch neutralisiert oder war nie gesetzt. Bei unerklärlichen
Recovery-Fallbacks zuerst `misc` löschen, dann vbmeta prüfen.

## Blockade 3 GEFALLEN 2026-08-09: Container-Namespaces laufen

`build-gki-157-knews.sh STEP=1` setzt genau **eine** Option: `CONFIG_PID_NS=y`
(IPC/NET/UTS_NS sind in 6.1.157 schon aktiv — `IPC_NS` hat Google selbst über
`POSIX_MQUEUE` eingeschaltet, der lebende Beweis, dass Namespace-Optionen die
Modul-ABI nicht brechen). Gebaut, **nur `boot-lz4.img`** geflasht, bootet.

Wichtige Bau-Lektion: Optionen mit Kconfig-Default `y` (wie `PID_NS` unter
`NAMESPACES`) dürfen **nicht** als `=y`-Zeile angehängt werden — Kleafs
`check_defconfig` (`savedefconfig`) lässt Defaults weg und der Abgleich schlägt
fehl. Stattdessen nur die `# … is not set`-Opt-out-Zeile löschen.

Bewusst **nicht** gesetzt, weil sie mit Googles vorgebauten Modulen kollidieren:
`USER_NS` (macht `make_kuid`/`from_kuid` zu echten Symbolen → modpost-Abbruch,
fehlen in der akita-KMI-Liste), `SYSVIPC` (task_struct-Felder → CRC-Bruch),
`CGROUP_PIDS` (css_set-Layout). Container laufen hier als root, brauchen keins davon.

**Test als root** (`pidns-test.sh`):

```
uid=0(root) … context=u:r:magisk:s0
unshare -p -f -m sh -c 'mount -t proc proc /proc; echo $$; ls /proc'
  PID=1                      <- eigene PID im neuen Namespace ist 1
  sichtbare Prozesse: 3      <- statt Hunderten: Isolation greift
  pidns_exit=0
net/ipc/uts unshare: alle exit 0
```

Damit ist die dritte Blockade — die eigentliche Begründung für den ganzen
Kernel-Bau — **experimentell erledigt**. `unshare` ohne Root liefert jetzt EPERM
statt EINVAL, mit Root läuft es durch.

## Root via Magisk über init_boot

`boot_patch.sh` aus der Magisk-APK (v30.7) auf dem Gerät laufen lassen, Eingabe
ist **`init_boot.img`** (nicht `boot.img` — auf A13+-Geräten liegt die Ramdisk
dort), Ausgabe per `fastboot flash init_boot`. **vbmeta bleibt unangetastet**,
sonst greift wieder der FBE-Recovery-Fallback von oben.

Nötige Magiskboot-Bausteine aus der APK: `libmagiskboot.so`→`magiskboot`,
`libmagiskinit.so`→`magiskinit`, `libmagisk.so`→`magisk`, `libinit-ld.so`→`init-ld`,
dazu `boot_patch.sh`, `util_functions.sh`, `stub.apk`.

**Die eigentliche Root-Hürde war MagiskSU, nicht der Kernel** — und die Diagnose
kostete mehrere Fehlschläge, hier für die Zukunft festgehalten. Jede
`su`-Anfrage der ADB-Shell wurde sofort mit „Permission denied" abgelehnt,
`Magisk: su: request rejected (2000)`. Die Ursachenkette:

1. Die ersten `su`-Versuche liefen, während das **Display gesperrt** war —
   MagiskSU kann den Grant-Dialog nicht über den Sperrbildschirm legen und
   lehnt ab. Bildschirm entsperrt lassen.
2. Jede abgelehnte Anfrage schrieb einen **dauerhaften Deny-Eintrag für
   `com.android.shell` (uid 2000)** in Magisks Policy-DB. Der überstimmt danach
   *jede* Einstellung — auch „Automatisch beantworten: Gewähren". Sichtbar und
   umschaltbar nur im **Superuser-Tab** der App: Schalter auf an → Toast
   „Superuser-Rechte für Shell gewährt".
3. Nebenschauplatz, aber real: Androids **App-Freezer** fror die Magisk-App
   ein (`ActivityManager: sync unfroze com.topjohnwu.magisk`), was den
   Prompt-Pfad zusätzlich verzögert. `settings put global cached_apps_freezer
   disabled` (+ Reboot) und Doze-Whitelist entschärfen das.

Für einen headless-Node gehört danach `Automatisch beantworten` auf **Gewähren**
und die **Displaysperre auf „Keine"**, sonst wiederholt sich (2) nach jedem
Reboot. Die adb-Grants ließen sich nur per On-Screen-Automation setzen
(`input tap` + `screencap`), weil `magisk --sqlite` selbst Root braucht.

## Debloat 2026-08-09: RAM von 3,2 auf 5,1 GB frei

`debloat-node.sh` (reversibel über `restore-apps.sh`) macht das Stock-System
headless-tauglich, **ohne** das /system-Image anzufassen:

- **67 Google-Nutzer-Apps entfernt** (`pm uninstall --user 0`): Maps, Gmail,
  Fotos, Docs, Videos, Kamera, Meet, Messages, Files, YouTube/Music, Chrome,
  die Google-App/Assistant (`googlequicksearchbox`, allein ~700 MB), „Android
  System Intelligence" (`.as`), AICore, alle Wallpaper-/Pixel-Feature-Apps usw.
- **5 kern-nahe Pakete deaktiviert** (`pm disable-user`): `gms` (Play Services),
  `gsf`, `vending` (Play Store), `nexuslauncher`, `systemui`.
- **Tabu geblieben:** `com.android.shell` (adb), Magisk, `networkstack`,
  `wifi.*`, `telephony`, `permissioncontroller`, `media.module`, `webview` —
  alles, woran Boot/adb/Root/Netz hängen.

Ergebnis nach frischem Reboot (sauber, kein Bootloop, 40 s):

| | vorher | nachher |
|---|---|---|
| MemFree | 497 MB | **4.319 MB** |
| MemAvailable | 3.213 MB | **5.126 MB** |
| aktive Pakete | 333 | 266 |

Kernel und Root unverändert. `pm uninstall --user 0` löscht nichts aus /system
(nur die Nutzer-Registrierung), daher holt `restore-apps.sh` per
`cmd package install-existing` alles verlustfrei zurück.

Hinweis: `systemui` ist per Disable-Flag deaktiviert; eine früh im Boot
gestartete Instanz kann kurz im `dumpsys meminfo` auftauchen, wird aber nicht
dauerhaft neu gezogen. `cmd package list packages -d` zeigt den echten Zustand.

**Nachtrag — SystemUI/Launcher wieder AN:** SystemUI zu deaktivieren führt zum
Dauer-Ladebildschirm „Smartphone wird geladen" (nichts zeichnet die
Oberfläche). Beide wieder aktiviert; die eigentlichen RAM-Gewinne (GMS + 67
Apps) bleiben, MemAvailable ~4,6 GB. Der Launcher wird ohnehin gleich durch
NodeFace als Home-App ersetzt.

**Headless-Extra (`DO_EXTRA=1`):** Gboard (`inputmethod.latin`, deaktiviert →
**keine Bildschirmtastatur mehr**), Dialer (entfernt), `settings.intelligence`
(deaktiviert). Lehrreiche Ernüchterung: real nur **~60 MB** frei, obwohl die
PSS-Summe ~600 MB nahelegt — **PSS zählt geteilte Framework-Seiten mit**, die
via Zygote ohnehin für system_server/SystemUI im RAM bleiben. Nutzen liegt eher
in weniger Hintergrund-Wakeups als in RAM.

**Warum „nur" ~4,6 GB frei / ~40 % belegt:** `Used RAM` laut `dumpsys meminfo`
= ~1,5 GB Apps (PSS) + ~0,8 GB Kernel; dazu ~0,56 GB `Lost RAM`
(GPU/ION/DMA-Hardwarepuffer). Die 1,8 GB `Cached` sind wiederverwendbar und
zählen als verfügbar. Von 8 GB verwaltet der Kernel nur 7,4 GB (Rest:
Firmware/Modem/Secure-World-Carveout). Das ist der gesunde Android-Grundboden,
nicht Bloat — 4,6 GB stehen für Container bereit.

## NodeFace 2026-08-09: On-Demand-Status-Display als Home-App

Statt Launcher/Ladebildschirm zeigt das Gerät ein eigenes Vollbild-Dashboard
(`nodeface/`, ~17 KB APK): Hostname, Uptime, **CPU-Last** (+ loadavg, Farbe
grün→rot), **Temperatur** (Akku-°C + SoC-Thermalstatus), **RAM**, **Akku**
(+ Ladezustand).

**Bedienmodell (v2):** Bildschirm ist normalerweise **aus** (spart Strom, kein
Burn-in). Aufwecken (Doppeltipp/Power) zeigt die Stats; jede Interaktion
verlängert um **20 s**, danach legt die App das Gerät per Root
(`input keyevent 223` = SLEEP) wieder schlafen. `setShowWhenLocked(true)` +
`setTurnScreenOn(true)`, damit die Stats direkt über dem Sperrbildschirm
erscheinen (kein Entsperren nötig). Getestet: nach Wecken `Awake`/Screen an,
nach 20 s `Dozing`/Screen aus. Weck-Gesten gesetzt: `double_tap_to_wake=1`,
`doze_tap_gesture=1`, `doze_always_on=0` (Touch-Hardware-abhängig; Power-Taste
weckt garantiert).

> v1 war ein Dauer-Display mit langsamer Lissajous-Drift gegen Burn-in — durch
> das Screen-off-Modell überflüssig und wieder entfernt.

Zwei Erkenntnisse, die den Bau geprägt haben:

- **SELinux sperrt `/proc/stat`, `/proc/loadavg`, `/proc/uptime` für Apps**
  (Fingerprinting-Schutz) — nur `/proc/meminfo`, Battery- und Power-API sind
  frei. CPU-Last kommt daher über einen **einzelnen persistenten `su`-Prozess**
  (Magisk, Auto-Antwort „Gewähren" → grantet lautlos), der im Sekundentakt
  `/proc/stat`+`/proc/loadavg` ausgibt; die App parst den Stream. Ohne Root
  zeigt sie „-- · kein root". Uptime kommt aus `SystemClock` (kein Root nötig).
- **`/sys/class/thermal` ist selbst für die Shell gesperrt** → keine SoC-Die-°C
  ohne tieferen Eingriff. Stattdessen echte Akku-°C (Battery-API) plus
  `PowerManager.getCurrentThermalStatus()` als SoC-Label.

Gesetzt als Default-Home (`cmd package set-home-activity
com.knews.nodeface/.NodeActivity`) → erscheint automatisch beim Boot und hält
per `FLAG_KEEP_SCREEN_ON` den Bildschirm an. Zurück zum Pixel-Launcher:
`cmd package set-home-activity com.google.android.apps.nexuslauncher/...`.

Build ohne Gradle (`scripts/build-nodeface.sh`): javac (JDK 11 aus dem
Kernbaum) → d8 → aapt2 → zipalign → apksigner, gegen ein minimales SDK
(`scripts/setup-android-sdk.sh`, build-tools 34 + Plattform 34). Quellen in
`nodeface/src/`, Manifest registriert die Activity zusätzlich als Daydream
(`NodeDream`), falls man es doch als Bildschirmschoner statt als Home will.

## Ladesteuerung 2026-08-09: Akku als USV schonen

Ein 24/7 am Netz hängender Node würde bei Dauer-100 % den Akku aufblähen.
`scripts/charge-limit.sh` setzt über die Pixel-Ladesteuerung
(`/sys/devices/platform/google,charger/charge_{start,stop}_level`) ein Limit von
**60/65 %** — bewusst nicht niedriger, weil der Akku zugleich als USV dient
(bei Stromausfall läuft der Node bei 60–65 % noch mehrere Stunden). Übernommen
aus `pixel-whisper-node`, Sysfs-Pfad auf Stock A16 bestätigt vorhanden.

Persistenz über **Magisk `service.d`**: das Skript liegt als
`/data/adb/service.d/charge-limit.sh` (chmod 755) und läuft als root bei jedem
Boot, mit Retry-Schleife, falls der Charger-Treiber beim Boot noch nicht bereit
ist. Manuell: `su -c 'sh /data/adb/service.d/charge-limit.sh'`.

## WLAN-Fix 2026-08-09: rfkill=m vs =y — der Kernel-Config-Fallstrick

Nach dem Basiswechsel war **kein Netzwerk** da (chroot: „Network is unreachable",
Android: nur `lo`/`dummy0`, kein `wlan0`). Ursache tief im Kernel:

- Unser GKI-Build (plain `gki_defconfig` + PID_NS) hat **`CONFIG_RFKILL=m`**.
- Der **Stock-6.1.145-Kernel hat `CONFIG_RFKILL=y`** (eingebaut) — Googles
  akita-Build setzt das über ein Device-Fragment, das im plain GKI-defconfig fehlt.
- Folge: die Stock-`cfg80211.ko` (vendor_dlkm) findet ihre rfkill-Symbole nicht
  (`cfg80211: Unknown symbol rfkill_alloc`), lädt nicht → kein `bcmdhd` → kein
  `wlan0`. Die Stock-`modules.load` listet nur cfg80211+bcmdhd, keine rfkill.ko
  (die gab es nie, weil eingebaut).

**Fix ohne Kernel-Neubau:** Unser eigener Build hat wegen `=m` eine
`rfkill.ko` (6.1.157) erzeugt. Deren KMI-Symbole sind mit der Stock-cfg80211
(6.1.145) kompatibel — geladen in der Reihenfolge rfkill → cfg80211 → bcmdhd
kommt `wlan0`/`wlan1` hoch, Firmware lädt.

**Persistenz:** `scripts/wifi-modules-boot.sh` als Magisk
`post-fs-data.d/00-wifi-rfkill.sh` (rfkill.ko unter `/data/adb/wifi-rfkill.ko`)
lädt den Stack **früh im Boot**, bevor die WiFi-HAL startet — sonst latcht der
Framework auf „kein wlan0". Nach Reboot: `wlan0` von Anfang an da, `svc wifi
enable` funktioniert.

Lehre für künftige Kernel: entweder `CONFIG_RFKILL=y` in den Build ziehen (dann
entfällt der rfkill.ko-Umweg), oder besser gleich gegen die **Stock-device.config**
statt plain `gki_defconfig` bauen — sie enthält alle Device-Fragmente. Betrifft
potenziell weitere Module.

## systemd + MicroK8s 2026-08-09: Kubernetes auf dem Pixel

Der Chroot läuft (Ubuntu 24.04.3, `knews-node-7`). MicroK8s kommt nur als Snap,
Snap braucht **systemd als PID 1** und squashfuse (Kernel hat kein SQUASHFS,
aber FUSE=y). Ablauf:

1. **Chroot-Bootstrap** ohne systemd (`chroot-run.sh`, quoting-sicher: kopiert
   ein inneres Skript ins Rootfs und `chroot … bash /root/_inner.sh`). Erst
   `apt` (Netz via WLAN), dann `systemd systemd-sysv dbus snapd squashfuse fuse3`.
2. **systemd als PID 1 im PID-Namespace** (`boot-systemd.sh`): unser knews-Kernel
   mit `CONFIG_PID_NS` macht es erst möglich. Toybox-`unshare` kennt nur `-m -p
   -f` (kein `--propagation`/`--mount-proc`) → im neuen Mount+PID-NS manuell
   `mount -t proc` + `chroot … /lib/systemd/systemd --system`. `container=lxc`
   schaltet systemd in den Container-Modus. Ergebnis: `systemctl
   is-system-running` → **running**, snapd antwortet.
3. **Betreten** (`enter-systemd.sh`): Host-PID von systemd finden (cmdline
   BEGINNT mit `/lib/systemd/systemd`, nicht der unshare-Wrapper!), dann
   `nsenter -t <pid> --mount --pid --uts --ipc -- chroot … bash`.
4. `snap install microk8s --classic --channel=1.28/stable` → **v1.28.15**
   (rev 7988), genau die Zielversion.

Persistenz: `boot-node.sh` als Magisk `service.d/99-knews-node.sh` — wartet auf
Systemstart, **Partial Wakelock** (CPU wach trotz Display-aus, sonst frieren
alle k8s-Prozesse ein, sobald NodeFace den Screen schlafen legt), Doze aus,
`ip_forward`, dann `boot-systemd.sh`. Snap-Dienste sind systemd-Units und
starten damit automatisch mit.

## MicroK8s 1.28.15 läuft — Node registriert (NotReady)

Erster erfolgreicher Stand vor dem Kernel-Detour:
```
NAME           STATUS     VERSION    INTERNAL-IP       KERNEL                        RUNTIME
knews-node-7   NotReady   v1.28.15   192.168.178.118   6.1.157-...-dirty             containerd://1.6.28
```
API-Server, containerd, kubelite, k8s-dqlite alle up. **NotReady**, weil Calico
(`calico-node`) hängt. **SSH** funktioniert: `ssh knews@<ip>` (uid/gid 1000,
passwortloses sudo, in Gruppe microk8s, `kubectl`-Alias). sshd als `ssh.service`.

Wichtiger Routing-Fix (`fix-default-route.sh`): Android hält die Default-Route
nur in netz-spezifischen Tabellen → `ip route` (main) hat keine Default →
Container/Pods/microk8s finden keinen Weg raus. Fix: `ip route add default via
<gw> dev wlan0 table main`. Gehört in `boot-node.sh`.

## br_netfilter = SACKGASSE (KMI-Bruch)

Versuch, `CONFIG_BRIDGE_NETFILTER`/`VXLAN`/`IP_VS` in den knews-Kernel zu bauen
(`build-gki-157-net.sh`, `check_defconfig` via `POST_DEFCONFIG_CMDS="true"`
abgeschaltet): baut sauber, aber **bootet nicht**. pstore:
```
init: Failed to load kernel modules
Kernel panic - not syncing: Attempted to kill init!
```
**`CONFIG_BRIDGE_NETFILTER` fügt `nf_bridge` in `struct sk_buff` ein** → ändert
das Layout einer Kernstruktur, die fast jedes Vendor-Modul nutzt → KMI-CRC-Bruch
→ alle Module fallen aus → init-Panic. `=m` hilft nicht (Feld per `IS_ENABLED`
immer einkompiliert). **Mit Stock-Vendor-Modulen unmöglich** — bräuchte einen
Komplett-Rebuild aller Vendor-Module (das große Vermeidungsziel). Details:
Memory `kernel-kmi-grenzen`. Kernel auf knews1 (nur PID_NS) zurückgeflasht.

Konsequenz: **br_netfilter kommt nicht.** Die MicroK8s-inspect-Warnung ist
hinzunehmen; zu prüfen ist, ob Calico OHNE br_netfilter läuft (der eigentliche
Blocker war die Default-Route). VXLAN/IP_VS wären für sich KMI-safe, aber ohne
br_netfilter lohnt der Rebuild nicht.

Falle am Rande: die persistente `rfkill.ko` muss zum laufenden Kernel passen —
nach Kernel-Wechsel wieder die passende einspielen, sonst „Invalid argument"
beim Early-Load und kein WLAN.

## DURCHBRUCH 2026-08-09 Nacht: Container-Runtime läuft auf dem Android-Kernel

Die anfängliche „cgroup v1/v2 = unüberwindbar"-Diagnose war zu pessimistisch.
Mit **crun** statt runc + gezielten containerd-Flags fielen die Blocker
nacheinander. Die vollständige Fehlerkette (jeder Fix legte den nächsten frei):

| # | Fehler | Fix |
|---|---|---|
| 1 | API tot: `service IP family must match "fdab:..."` | IPv4-Default-Route + kubelet `--node-ip=192.168.178.118` |
| 2 | runc: `cpu.weight: no such file` | **crun** statt runc (`BinaryName`), da crun tolerant ist |
| 3 | crun: `/proc/self/uid_map: No such file` | **Kernel mit `CONFIG_USER_NS=y`** — und: USER_NS ist KEIN KMI-Bruch, Module laden, WiFi läuft |
| 4 | crun: `cgroup controller cpu not available` | containerd CRI `disable_cgroup = true` |
| 5 | crun: `pivot_root: Invalid argument` | runc.options `NoPivotRoot = true` (NICHT CRI-Level `no_pivot` — das zerlegt das CRI-Plugin!) |

**Ergebnis:** kubelite stabil (0 Restarts), **Container laufen** — calico-node
erster Init-Container (upgrade-ipam) durch. Der eigentliche Berg (OCI-Runtime auf
Android-GKI-Kernel mit Stock-Modulen) ist überwunden.

**Verbleibt (normales k8s-Networking):** `install-cni` scheitert, weil der Pod die
Kubernetes-ClusterIP **10.152.183.1:443 nicht erreicht** (`i/o timeout`) — ein
kube-proxy/apiserver-Binding-Thema, KEIN Runtime-Problem. Morgen dort weiter
(Memory `microk8s-durchbruch-stand`).

Kernel jetzt: `images/gki-6.1.157-userns` (PID_NS + USER_NS). MicroK8s frisch neu
installiert (v1.28.15). Aktive Config: kubelet `--cgroups-per-qos=false
--enforce-node-allocatable= --fail-swap-on=false --node-ip=…`; containerd
`snapshotter=native` + `disable_cgroup=true` + runc.options
`BinaryName=/usr/bin/crun` + `NoPivotRoot=true`.

## Stand Task 8 (2026-08-09 Abend, überholt): Control-Plane läuft, CNI hängt

**Erreicht:**
- MicroK8s **v1.28.15** installiert, Node `knews-node-7` registriert.
- Control-Plane-Daemons laufen: containerd (1.6.28), k8s-dqlite, kubelite,
  cluster-agent. **SSH** als `knews` funktioniert.
- **Default-Route-Fix ist der Schlüssel für kubelite**: ohne `default` in der
  main-Tabelle bleibt kubelite in „Waiting for default route to appear" hängen
  und startet die API nie. Mit Route (`fix-default-route.sh`) kommt die API hoch.

**Offene Blocker (die harte letzte Meile):**
1. **kubelite/API flappt** — `kubectl get nodes` bekommt zeitweise
   „connection refused". Instabil, Ursache noch offen (Ressourcen? dqlite nach
   unsauberem Shutdown? Container-Umgebung?).
2. **`calico-node` steckt bei `Init:0/2`** — auch mit Default-Route. Der
   CNI-Init kommt auf diesem Host nicht durch. Kandidaten: fehlendes
   `br_netfilter` (KMI-Sackgasse, s.o.) UND fehlendes `CGROUP_PIDS`
   (kubelet/Container-Limits), plus die ungewöhnliche Android-Kernel-Umgebung.
3. **Boot-Robustheit**: nach Reboot sind die squashfuse-**snap-Mounts** verwaist
   („transport endpoint is not connected"); Daemons + Route + Mounts müssen neu
   angestoßen werden. `boot-node.sh` muss das abhandeln, sonst kommt der Cluster
   nach Reboot nicht von allein hoch.

**Debugging-Kette 2026-08-09 Abend (drei Blocker nacheinander gelöst/diagnostiziert):**

1. **kubelite crasht** — `Failed to start ContainerManager: openat2
   /sys/fs/cgroup/kubepods/besteffort/cpu.weight: no such file` → gelöst mit
   kubelet-Args `--cgroups-per-qos=false --enforce-node-allocatable=
   --fail-swap-on=false`. Danach kubelite **stabil** (API konstant erreichbar).
2. **Pod-Sandbox scheitert** — `failed to mount rootfs overlay … invalid
   argument`. overlayfs geht auf Android/f2fs im Mount-NS nicht. → containerd
   auf **native-Snapshotter** umgestellt (MicroK8s' `snapshotter()` in
   `actions/common/utils.sh` wählt sonst hart overlayfs; Fix: im
   `containerd-template.toml` `snapshotter = "native"` hart eintragen). Overlay-
   Fehler weg.
3. **FUNDAMENTALER BLOCKER — Android-Hybrid-cgroups:** danach scheitert **runc**
   an `openat2 /sys/fs/cgroup/k8s.io/<id>/cpu.weight: no such file`. Ursache:
   Android legt `cpu`/`cpuset`/`blkio` auf cgroup **v1** (`/dev/cpuctl`,
   `/dev/cpuset`, `/dev/blkio`); die **v2**-Hierarchie (`/sys/fs/cgroup`, die
   runc/k8s nutzt) hat **null Controller**. Ein v2-Controller kann nur da sein,
   wenn ihn kein v1 belegt — `echo +cpu > cgroup.subtree_control` schlägt für
   alle fehl. Der Kernel HAT die Controller (`CGROUP_SCHED`, `FAIR_GROUP_SCHED`,
   `MEMCG` =y), sie hängen nur an v1.

**Das ist die harte Grenze.** k8s/runc braucht cgroup-v2 mit cpu-Controller;
Androids Hybrid-Layout gibt ihn nicht her, ohne Androids v1-Hierarchien
(`/dev/cpuctl` …) auszuhängen — was Androids Scheduling/Task-Profile bricht.
Optionen, alle nicht trivial:
- **A) Androids v1-cgroups aushängen** und die Controller in v2 delegieren
  (headless evtl. vertretbar, aber riskant — Android-Dienste können abstürzen;
  live remounten ist heikel). Sauberer wäre Boot-Config auf cgroup-v2-unified
  (`androidboot`/Kernel-cmdline), nicht live.
- **B) runc/containerd dazu bringen, cpu-cgroups zu ignorieren** (kein sauberer
  Standard-Schalter; Wrapper/Patch nötig).
- **C)** Node läuft bis zur Container-Erstellung; für echten Cluster-Betrieb ist
  A oder B nötig.

**Stand:** Control-Plane läuft stabil, native-Snapshotter aktiv, **letzter
Blocker präzise = cgroup-v1/v2**. Entscheidung A vs. B steht aus (riskant → mit
Kevin klären).

**Nächste Schritte:**
6. cgroup-Frage klären (A: v1-Hierarchien aushängen + v2 delegieren, am besten
   per Boot-Setup; oder B: runc-cgroup-Umgehung). Dann calico-node Sandbox →
   Node Ready.
7. `boot-node.sh` als Magisk-Service scharf schalten (Route + snap-Remount +
   Daemon-Restart + cgroup-Fix), Reboot-Test.
8. Cluster `knews-local` beitreten.
9. Danach TPU/GPU.

## Übernommenes Handwerkszeug

Alles in `pixel-whisper-node/scripts/` und dort dokumentiert:

| Thema | Wichtigste Erkenntnis |
|---|---|
| Flash-Reihenfolge | Erst fastbootd (`vendor_dlkm`, `system_dlkm`), dann Bootloader (`boot`, `dtbo`, `vendor_kernel_boot`). Umgekehrt sperrt man sich aus |
| fastbootd unter Windows | Enumeriert nicht (`ProblemCode 28`). Lösung: `usbipd` + WSL, Kabel nach dem Moduswechsel einmal neu stecken |
| Rückweg | `super` komplett: alle Chunks → roh → flashen. Einzelchunks scheitern am letzten (3,95 GB deklariert, Limit 249 MB) |
| Build-Verifikation | Ein grüner Build beweist nichts — Config **aus dem gebauten Image** extrahieren |
| IP/MAC | GrapheneOS randomisiert die MAC pro Verbindung. Auf „pro Netzwerk" stellen, sonst hält keine DHCP-Reservierung |
