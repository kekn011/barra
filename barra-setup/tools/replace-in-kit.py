#!/usr/bin/env python3
"""replace-in-kit.py - eine einzelne Datei in einem Kit-Archiv austauschen.

WARUM ES DIESES WERKZEUG GIBT
Die Kits tragen eigene Kopien der Server-Skripte. Eine Korrektur in src/ erreicht das
Produkt deshalb NICHT von selbst - am 27.8.2026 lief genau das auf: das Release lieferte
die Vor-i18n-Fassung von sttserver.sh aus, waehrend das Repo die reparierte hatte.

Ein Archiv dafuer auszupacken und neu zu packen ist die gefaehrliche Variante: auf NTFS
gibt es keine Ausfuehrbits, und ein Umpacken als falscher Benutzer verliert Eigentuemer
und Modi (so ist am 26.8.2026 ein kaputtes Base-Image ins Release gekommen). Dieses
Werkzeug fasst deshalb nur EINEN Eintrag an und uebernimmt alle Metadaten der Vorlage:
Modus, Eigentuemer, Reihenfolge. Danach vergleicht es beide Archive Eintrag fuer Eintrag
und besteht darauf, dass sich genau der gewuenschte geaendert hat.

  python replace-in-kit.py <archiv.tar[.gz]> <pfad-im-archiv> <neue-datei>

Das Archiv wird an Ort und Stelle ersetzt (erst nach bestandener Pruefung).
"""
import hashlib
import io
import os
import shutil
import sys
import tarfile
import tempfile


def sha256(pfad):
    h = hashlib.sha256()
    with open(pfad, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv):
    if len(argv) != 4:
        print(__doc__)
        return 1
    archiv, ziel, quelle = argv[1], argv[2], argv[3]
    for p in (archiv, quelle):
        if not os.path.isfile(p):
            print(f"FEHLER: {p} nicht gefunden")
            return 1

    modus = "r:gz" if archiv.endswith((".gz", ".tgz")) else "r:"
    schreib = "w:gz" if archiv.endswith((".gz", ".tgz")) else "w:"
    neu_daten = open(quelle, "rb").read()

    with tarfile.open(archiv, modus) as alt:
        namen = alt.getnames()
    if ziel not in namen:
        print(f"FEHLER: '{ziel}' ist nicht im Archiv. Vorhanden sind z.B.:")
        for n in namen[:20]:
            print("   ", n)
        return 1

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tar", dir=os.path.dirname(os.path.abspath(archiv)))
    tmp.close()
    with tarfile.open(archiv, modus) as alt, tarfile.open(tmp.name, schreib) as neu:
        for info in alt:
            if info.name == ziel:
                alt_groesse = info.size
                # Metadaten der Vorlage behalten (Modus, Eigentuemer, Namen) - nur Inhalt,
                # Groesse und Zeitstempel kommen von der neuen Datei.
                info.size = len(neu_daten)
                info.mtime = int(os.path.getmtime(quelle))
                neu.addfile(info, io.BytesIO(neu_daten))
            else:
                quelle_fd = alt.extractfile(info) if info.isreg() else None
                neu.addfile(info, quelle_fd)

    # Pruefen statt hoffen: beide Archive Eintrag fuer Eintrag vergleichen.
    fehler = []
    geaendert = []
    with tarfile.open(archiv, modus) as a, tarfile.open(tmp.name, modus) as b:
        ia = {i.name: i for i in a.getmembers()}
        ib = {i.name: i for i in b.getmembers()}
        if list(ia) != list(ib):
            fehler.append("Reihenfolge oder Bestand der Eintraege hat sich geaendert")
        for name, x in ia.items():
            y = ib.get(name)
            if y is None:
                fehler.append(f"fehlt im Ergebnis: {name}")
                continue
            if (x.mode, x.uid, x.gid, x.uname, x.gname, x.type) != (y.mode, y.uid, y.gid, y.uname, y.gname, y.type):
                fehler.append(f"Metadaten geaendert: {name}")
            if x.isreg():
                da = a.extractfile(x).read()
                db = b.extractfile(y).read()
                if da != db:
                    geaendert.append(name)
                    if name == ziel and db != neu_daten:
                        fehler.append(f"{name} traegt nicht den neuen Inhalt")
    if geaendert != [ziel]:
        fehler.append(f"geaendert wurden: {geaendert or 'nichts'} - erwartet war nur {ziel}")

    if fehler:
        os.unlink(tmp.name)
        print("ABBRUCH - das Ergebnis waere nicht das, was es sein soll:")
        for f in fehler:
            print("   ", f)
        return 1

    with tarfile.open(tmp.name, modus) as b:
        i = b.getmember(ziel)
        print(f"   {ziel}: {alt_groesse} -> {i.size} Bytes, Modus {oct(i.mode)}, "
              f"Eigentuemer {i.uname or i.uid}/{i.gname or i.gid} (unveraendert)")
    os.replace(tmp.name, archiv)
    print(f"   {archiv}: {os.path.getsize(archiv)} Bytes")
    print(f"   sha256 {sha256(archiv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
