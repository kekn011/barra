#!/usr/bin/env python3
# Whisper-Decode-Kuer V1: f16-GEMV auf Mali mit 4 statt 2 Rows/Workgroup (mehr A-Loads
# in Flight; Befund 22.8.: lm_head 51866x1280 nur 24,8 GB/s bei Roofline 53).
# Patcht die beiden f16-Pipeline-Erzeugungen in ggml-vulkan.cpp (whisper-Baum). Idempotent.
import os
p = os.path.expanduser('~/whisper.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp'); s = open(p).read()
done = 0
for name in ('"mul_mat_vec_f16_f32_f32"', '"mul_mat_vec_f16_f16_f32"'):
    i = s.index(name)
    seg = s[i:i+400]
    if '{4, 1, 1}, {wg_size_subgroup, 4, i+1}' in seg:
        print(name, 'schon gepatcht'); done += 1; continue
    old = '{2, 1, 1}, {wg_size_subgroup, 2, i+1}'
    assert seg.count(old) == 1, name
    s = s[:i] + seg.replace(old, '{4, 1, 1}, {wg_size_subgroup, 4, i+1}', 1) + s[i+400:]
    done += 1
    print(name, '-> 4 rows/WG')
assert done == 2
open(p, 'w').write(s)
