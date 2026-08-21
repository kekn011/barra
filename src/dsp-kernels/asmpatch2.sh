#!/bin/bash
# asmpatch2.sh <asm.S> <vaddr-hex> <out.elf>
# Assembles base-ISA Xtensa, splices .text into inbuilt.elf at 0x100+vaddr,
# AND neutralizes any .rela.got relocation whose r_offset falls within the patched range.
set -e
SP=$(cd "$(dirname "$0")" && pwd)   # Kernel-Verzeichnis (src/dsp-kernels)
XT=/home/kevin/xtensa/xtensa-esp-elf/bin
cd "$SP"
ASM="$1"; VADDR="$2"; OUT="$3"
$XT/xtensa-esp-elf-as -o /tmp/probe.o "$ASM"
$XT/xtensa-esp-elf-objcopy -O binary --only-section=.text /tmp/probe.o /tmp/probe.bin
echo "assembled $(wc -c < /tmp/probe.bin) bytes:"; xxd /tmp/probe.bin
python3 - "$VADDR" "$OUT" <<'PY'
import sys,struct
vaddr=int(sys.argv[1],16); out=sys.argv[2]
d=bytearray(open("inbuilt.elf","rb").read())
code=open("/tmp/probe.bin","rb").read()
foff=0x100+vaddr
d[foff:foff+len(code)]=code
# neutralize overlapping relocations in .rela.got (file 0x30d8, size 0x1164, ent 12)
rf,rsz=0x30d8,0x1164
lo,hi=vaddr,vaddr+len(code)
n=0
for off in range(rf,rf+rsz,12):
    r_off=struct.unpack_from("<I",d,off)[0]
    if lo<=r_off<hi:
        struct.pack_into("<I",d,off+4,0)  # r_info=0 -> R_XTENSA_NONE
        n+=1
open(out,"wb").write(d)
print(f"patched {len(code)} bytes at vaddr {vaddr:#x}; neutralized {n} overlapping relocs -> {out}")
PY
