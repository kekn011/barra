#!/usr/bin/env python3
# Fuegt in ggml-vulkan.cpp einen Env-Override fuer die scalar-Warptiles ein (Mali-Tuning-Sweep).
import io, sys
P = "/home/kevin/stable-diffusion.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp"
src = io.open(P, encoding="utf-8").read()
if "barra tile override" in src:
    print("schon gepatcht"); sys.exit(0)
anchor = "        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };\n        l_align = 128;\n        m_align =  64;\n        s_align =  32;\n"
assert src.count(anchor) == 1, "Anker nicht eindeutig: %d" % src.count(anchor)
block = anchor + """
        // barra tile override (Mali-Tuning): Warptiles per Env ueberschreiben — Sweep ohne Rebuild.
        // Format: GGML_VK_TILE_{L,M,S}=bs,BM,BN,BK,WM,WN,WMITER,TM,TN,TK,sg
        {
            auto barra_tile_env = [](const char * name, std::vector<uint32_t> & wt) {
                const char * v = getenv(name);
                if (v == nullptr) return false;
                std::vector<uint32_t> t; uint32_t x = 0; bool have = false;
                for (const char * p = v;; ++p) {
                    if (*p >= '0' && *p <= '9') { x = x * 10u + uint32_t(*p - '0'); have = true; }
                    else { if (have) { t.push_back(x); x = 0; have = false; } if (*p == 0) break; }
                }
                if (t.size() != 11) return false;
                wt = t;
                return true;
            };
            if (barra_tile_env("GGML_VK_TILE_L", l_warptile)) { l_wg_denoms = { l_warptile[1], l_warptile[2], 1 }; l_align = l_warptile[1]; std::cerr << "barra: TILE_L override" << std::endl; }
            if (barra_tile_env("GGML_VK_TILE_M", m_warptile)) { m_wg_denoms = { m_warptile[1], m_warptile[2], 1 }; m_align = m_warptile[1]; std::cerr << "barra: TILE_M override" << std::endl; }
            if (barra_tile_env("GGML_VK_TILE_S", s_warptile)) { s_wg_denoms = { s_warptile[1], s_warptile[2], 1 }; s_align = s_warptile[1]; std::cerr << "barra: TILE_S override" << std::endl; }
        }
"""
src = src.replace(anchor, block, 1)
io.open(P, "w", encoding="utf-8").write(src)
print("gepatcht")
