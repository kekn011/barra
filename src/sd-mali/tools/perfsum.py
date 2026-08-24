import re,sys
blocks=[];cur=None
for line in open(sys.argv[1],encoding='utf-8',errors='replace'):
    if line.startswith('Vulkan Timings'): cur=[]; blocks.append(cur); continue
    m=re.match(r'^(.*?): (\d+) x ([\d.]+) us = ([\d.e+]+) us(?: \(([\d.e+]+) GFLOPS/s\))?',line)
    if m and cur is not None: cur.append((m.group(1),int(m.group(2)),float(m.group(4))/1e3,m.group(5)))
for bi,b in enumerate(blocks):
    tot=sum(x[2] for x in b)
    print(f"\n=== Block {bi}: {len(b)} Eintraege, total {tot:.0f} ms")
    if tot<300: continue
    byop={}
    for name,n,ms,gf in b:
        op=name.split(' ')[0]; byop[op]=byop.get(op,0)+ms
    print("  nach Op:", ", ".join(f"{k} {v:.0f}" for k,v in sorted(byop.items(),key=lambda x:-x[1])[:10]))
    for name,n,ms,gf in sorted(b,key=lambda x:-x[2])[:22]:
        print(f"  {ms:7.1f} ms  {n:3d}x  {name[:70]}  {('%s GF'%gf) if gf else ''}")
