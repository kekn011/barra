"""Text -> Phonem-IDs fuer Piper-VITS (espeak-ng + tokens.txt).

Bildet nach, was sherpa-onnx intern macht, damit eine Piper-Stimme ueber den
GPU-Vokoder laufen kann (Front-ONNX braucht IDs, nicht Text).

Konvention (piper_phonemize): BOS, dann je Phonem [id, PAD], am Ende EOS.
Das PAD zwischen den Phonemen ist nicht optional — VITS ist darauf trainiert.

Als Bibliothek nutzbar; direkt aufgerufen zeigt es die Zerlegung:
    python piper_ids.py <tokens.txt> <espeak-stimme> "<text>"
"""
import subprocess
import sys

PAD, BOS, EOS = "_", "^", "$"


def load_tokens(path):
    """tokens.txt: '<symbol> <id>' je Zeile. Das Leerzeichen-Symbol ist selbst ein Leerzeichen,
    deshalb wird von RECHTS getrennt."""
    t = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sym, _, num = line.rpartition(" ")
            if not num.isdigit():
                continue
            t[sym] = int(num)
    return t


def phonemize(text, voice="de", espeak="espeak-ng"):
    """espeak-ng -> IPA-Zeichenkette. --sep= liefert die Phoneme ohne Trenner,
    -q unterdrueckt die Tonausgabe (es wird NICHTS abgespielt)."""
    out = subprocess.run([espeak, "-q", "--ipa=3", "--sep=|", "-v", voice, text],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError("espeak-ng fehlgeschlagen: %s" % out.stderr.strip())
    return out.stdout


def to_ids(text, tokens, voice="de", espeak="espeak-ng", strict=False):
    """-> (ids, unbekannte Symbole). Unbekannte werden uebersprungen, aber gemeldet:
    stilles Verschlucken waere genau die Sorte Fehler, die man erst am Klang merkt."""
    raw = phonemize(text, voice, espeak)
    ids = [tokens[BOS]]
    unknown = []
    for line in raw.splitlines():
        for chunk in line.strip().split("|"):
            if not chunk:
                continue
            for ch in chunk:                    # espeak liefert Mehrzeichen-Phoneme
                if ch in tokens:
                    ids.append(tokens[ch])
                    ids.append(tokens[PAD])
                elif ch.strip() == "":
                    if " " in tokens:
                        ids.append(tokens[" "])
                        ids.append(tokens[PAD])
                else:
                    unknown.append(ch)
    ids.append(tokens[EOS])
    if unknown and strict:
        raise ValueError("Phoneme ohne Token: %s" % sorted(set(unknown)))
    return ids, unknown


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    tk = load_tokens(sys.argv[1])
    ids, unk = to_ids(sys.argv[3], tk, voice=sys.argv[2])
    print("Tokens im Satz : %d" % len(tk))
    print("IPA            : %s" % phonemize(sys.argv[3], sys.argv[2]).strip()[:200])
    print("IDs (%d)       : %s%s" % (len(ids), ids[:40], " ..." if len(ids) > 40 else ""))
    print("max ID         : %d" % max(ids))
    if unk:
        print("OHNE TOKEN     : %s" % sorted(set(unk)))
    else:
        print("OHNE TOKEN     : keine")
