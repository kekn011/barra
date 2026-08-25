#!/usr/bin/env python3
# Patcht eine KOPIE von libedgetpu_tflite_compiler.so, damit sie OHNE frida
# standalone kompiliert. Zwei Fixes (in .text ist file-offset == vaddr):
#  1. Config-Name: der Chip-Switch baut fuer Default-Chip (enum 0) "v1_config"
#     (nicht eingebettet). v1-Case bei 0x16b0978/097c auf den v6-Namen (0x54674b)
#     umbiegen -> baut "v6_config.binarypb" (registriert, mit Daten).
#  2. chip_type-Mapper @0xb7549c: Default liefert chip_type=0 -> "unrecognized
#     chip_type: 0". w0 am Einstieg auf 0x1a zwingen -> mappt auf id 6 (rio/G3).
import struct, sys

# Aufruf:  patch-compiler-standalone.py <vendor-libedgetpu.so> <ausgabe-libcomp_std.so>
# (Ohne Argumente die alten Scratchpad-Defaults, damit Altlaeufe nicht brechen.)
SRC = sys.argv[1] if len(sys.argv) > 1 else "/vendor/lib64/libedgetpu_tflite_compiler.so"
DST = sys.argv[2] if len(sys.argv) > 2 else "/data/adb/baseos/dev/tpu/libcomp_std.so"

# (offset, erwartet_alt, neu, name)
PATCHES = [
    (0x16b0978, 0xf0ff7149, 0xd0ff74a9, "config v1->v6 adrp"),
    (0x16b097c, 0x9137c129, 0x911d2d29, "config v1->v6 add"),
    (0xb7549c,  0x71003c1f, 0x52800340, "chip_type mov w0,#0x1a"),
    (0xb754a0,  0x5400024d, 0xd503201f, "chip_type nop"),
]

data = bytearray(open(SRC, "rb").read())
for off, old, new, name in PATCHES:
    cur = struct.unpack_from("<I", data, off)[0]
    assert cur == old, f"{name}: @0x{off:x} erwartet 0x{old:08x}, gefunden 0x{cur:08x}"
    struct.pack_into("<I", data, off, new)
    print(f"  0x{off:x}: 0x{old:08x} -> 0x{new:08x}  ({name})")
open(DST, "wb").write(data)
print(f"geschrieben: {DST}")
