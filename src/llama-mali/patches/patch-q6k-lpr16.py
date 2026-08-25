#!/usr/bin/env python3
# q6_K: 16 Lanes je Zeile (Register-Druck -> Occupancy); Host: Zeilen/WG = wg/16 fuer Q6_K;
# q6_K-Pfad wieder Default (GGML_VK_ARM_NO_Q6K schaltet ab).
#
# Shader- und Host-Aenderung sind GEKOPPELT (Host-rows/WG setzt 16 Lanes/Zeile im Shader
# voraus). Daher: erst ALLE Anker pruefen, dann BEIDE Seiten schreiben — nie nur eine.
import os, sys

root = os.path.expanduser('~/llama.cpp/ggml/src/ggml-vulkan/')

# --- Shader ---
sp = root + 'vulkan-shaders/mul_mat_vecq.comp'; s = open(sp).read()
sh_old = '#if defined(MALI_WIDE)\n#define MALI_LPR 8u'
sh_new = '#if defined(MALI_WIDE)\n#if defined(MALI_Q6K_WIDE)\n#define MALI_LPR 16u\n#else\n#define MALI_LPR 8u\n#endif'
sh_done = 'MALI_LPR 16u' in s
if not sh_done and sh_old not in s:
    sys.exit('FEHLER: Shader-Anker in mul_mat_vecq.comp nicht gefunden — Abbruch, Host NICHT gepatcht')

# --- Host ---
hp = root + 'ggml-vulkan.cpp'; h = open(hp).read()
rows_old = 'const uint32_t mali_q6k_rows = (device->vendor_id == VK_VENDOR_ID_ARM) ? std::max(1u, wg_size_subgroup_int / 8) : 1*rm_kq_int;'
rows_new = 'const uint32_t mali_q6k_rows = (device->vendor_id == VK_VENDOR_ID_ARM) ? std::max(1u, wg_size_subgroup_int / 16) : 1*rm_kq_int;'
def_old = 'return src0_type == GGML_TYPE_Q4_K || (src0_type == GGML_TYPE_Q6_K && getenv("GGML_VK_ARM_Q6K") != nullptr);   // q6_K-Mali-Pfad NUR Opt-in (riss akita 3x)'
def_new = 'return src0_type == GGML_TYPE_Q4_K || (src0_type == GGML_TYPE_Q6_K && getenv("GGML_VK_ARM_NO_Q6K") == nullptr);   // q6_K-Mali-Pfad v2 (224-B-Repack), Abschalter GGML_VK_ARM_NO_Q6K'
host_done = ('wg_size_subgroup_int / 16' in h) and ('GGML_VK_ARM_NO_Q6K' in h)
n = h.count(rows_old); m = h.count(def_old)
if not host_done and (n < 1 or m < 1):
    sys.exit(f'FEHLER: Host-Anker in ggml-vulkan.cpp fehlen (rows={n} default={m}) — Abbruch, Shader NICHT gepatcht')

# --- Beide Anker ok (oder schon gepatcht): jetzt schreiben ---
if sh_done:
    print('shader: schon gepatcht')
else:
    open(sp, 'w').write(s.replace(sh_old, sh_new)); print('shader: LPR 16 fuer q6_K')

if host_done:
    print('host: schon gepatcht')
else:
    open(hp, 'w').write(h.replace(rows_old, rows_new).replace(def_old, def_new))
    print('host: rows/16 Stellen', n, ', Default-Umschaltung', m)
