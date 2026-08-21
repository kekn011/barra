#!/usr/bin/env python3
# q6_K: 16 Lanes je Zeile (Register-Druck -> Occupancy); Host: Zeilen/WG = wg/16 fuer Q6_K; q6_K-Pfad wieder Default (GGML_VK_ARM_NO_Q6K schaltet ab)
import os
root = os.path.expanduser('~/llama.cpp/ggml/src/ggml-vulkan/')
p = root + 'vulkan-shaders/mul_mat_vecq.comp'; s = open(p).read()
old = '#if defined(MALI_WIDE)\n#define MALI_LPR 8u'
new = '#if defined(MALI_WIDE)\n#if defined(MALI_Q6K_WIDE)\n#define MALI_LPR 16u\n#else\n#define MALI_LPR 8u\n#endif'
if old in s:
    s = s.replace(old, new); open(p, 'w').write(s); print('shader: LPR 16 fuer q6_K')
else:
    print('shader: schon' if 'MALI_LPR 16u' in s else 'MUSTER NICHT GEFUNDEN (shader)')
p = root + 'ggml-vulkan.cpp'; s = open(p).read()
n = s.count('const uint32_t mali_q6k_rows = (device->vendor_id == VK_VENDOR_ID_ARM) ? std::max(1u, wg_size_subgroup_int / 8) : 1*rm_kq_int;')
s = s.replace('const uint32_t mali_q6k_rows = (device->vendor_id == VK_VENDOR_ID_ARM) ? std::max(1u, wg_size_subgroup_int / 8) : 1*rm_kq_int;',
              'const uint32_t mali_q6k_rows = (device->vendor_id == VK_VENDOR_ID_ARM) ? std::max(1u, wg_size_subgroup_int / 16) : 1*rm_kq_int;')
old = 'return src0_type == GGML_TYPE_Q4_K || (src0_type == GGML_TYPE_Q6_K && getenv("GGML_VK_ARM_Q6K") != nullptr);   // q6_K-Mali-Pfad NUR Opt-in (riss akita 3x)'
new = 'return src0_type == GGML_TYPE_Q4_K || (src0_type == GGML_TYPE_Q6_K && getenv("GGML_VK_ARM_NO_Q6K") == nullptr);   // q6_K-Mali-Pfad v2 (224-B-Repack), Abschalter GGML_VK_ARM_NO_Q6K'
m = s.count(old); s = s.replace(old, new)
open(p, 'w').write(s); print('host: rows/16 Stellen', n, ', Default-Umschaltung', m)
