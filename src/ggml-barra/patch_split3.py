#!/usr/bin/env python3
# Patch 3: SmoothQuant-Skalen (Datei <meta ohne .meta>.smooth: je Layer s_h[D] f32, s_a[FF] f32) laden und im Split-Pfad
# h *= 1/s_h bzw. a *= 1/s_a vor der Zeilenquantisierung anwenden.
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert a in s, (path, a[:60]); s = s.replace(a, b, 1)
    open(path, "w", encoding="utf-8").write(s)
cpp = os.path.join(here, "ggml-barra.cpp"); c = open(cpp, encoding="utf-8").read()
if "inv_smooth" in c: print("schon gepatcht"); sys.exit(0)
patch(cpp, [
("""    barra_zbuf zi_d{}, zo_g{}, zo_u{}, zi_ff{}, zo_d{};""",
 """    barra_zbuf zi_d{}, zo_g{}, zo_u{}, zi_ff{}, zo_d{};
    std::vector<float> inv_smooth;          // SmoothQuant: je Layer 1/s_h[D], 1/s_a[FF] (leer = keine Glaettung)"""),
("""    if (const char * e = getenv("BARRA_MIN_BATCH")) ctx->min_batch = atoi(e);""",
 """    if (ctx->split == 3) {   // optionale SmoothQuant-Skalen: <meta>.smooth (gleicher Basisname)
        std::string sp = mp; size_t dot = sp.rfind(".meta"); if (dot != std::string::npos) sp = sp.substr(0, dot) + ".smooth";
        FILE * fs = fopen(sp.c_str(), "rb");
        if (fs) {
            size_t per = (size_t) ctx->D + ctx->FF; ctx->inv_smooth.resize(per * ctx->NL);
            size_t got = fread(ctx->inv_smooth.data(), sizeof(float), per * ctx->NL, fs); fclose(fs);
            if (got != per * ctx->NL) { GGML_LOG_WARN("barra: smooth-Datei %s zu kurz (%zu/%zu) - ignoriert\\n", sp.c_str(), got, per * ctx->NL); ctx->inv_smooth.clear(); }
            else { for (auto & v : ctx->inv_smooth) v = v != 0 ? 1.0f / v : 1.0f; GGML_LOG_WARN("barra: SmoothQuant-Skalen geladen (%s)\\n", sp.c_str()); }
        }
    }
    if (const char * e = getenv("BARRA_MIN_BATCH")) ctx->min_batch = atoi(e);"""),
])
inc = os.path.join(here, "split_path.inc")
patch(inc, [
("""    std::vector<float> hb((size_t) T * D);   // Kopie von h (dst kann h aliasen!)
    for (int r = 0; r < T; r++) memcpy(&hb[(size_t) r * D], (const char *) h->data + (size_t) r * h->nb[1], D * sizeof(float));
    std::vector<float> srow(B), srow2(B), a((size_t) B * FF);""",
 """    std::vector<float> hb((size_t) T * D);   // Kopie von h (dst kann h aliasen!)
    for (int r = 0; r < T; r++) memcpy(&hb[(size_t) r * D], (const char *) h->data + (size_t) r * h->nb[1], D * sizeof(float));
    const float * ish = ctx->inv_smooth.empty() ? nullptr : &ctx->inv_smooth[(size_t) L * (D + FF)];   // 1/s_h
    const float * isa = ish ? ish + D : nullptr;                                                          // 1/s_a
    std::vector<float> hs; if (ish) { hs.resize((size_t) T * D); for (int r = 0; r < T; r++) for (int d = 0; d < D; d++) hs[(size_t) r * D + d] = hb[(size_t) r * D + d] * ish[d]; }
    const float * hq = ish ? hs.data() : hb.data();   // geglaettete Eingabe fuer die TPU (Fallback/Check nutzen weiter hb = Original)
    std::vector<float> srow(B), srow2(B), a((size_t) B * FF);"""),
("""            barra_q8_row(&hb[(size_t) (t0i + r) * D], D, &srow[r], zi + (size_t) r * D, Qg.isc, Qg.izp);   // gate/up: gleiche Kalibrierung -> gleiches isc""",
 """            barra_q8_row(&hq[(size_t) (t0i + r) * D], D, &srow[r], zi + (size_t) r * D, Qg.isc, Qg.izp);   // gate/up: gleiche Kalibrierung -> gleiches isc"""),
("""                float g = (gq - Qg.ozp) * sg, u = (uq - Qu.ozp) * su; ar[j] = g / (1.0f + expf(-g)) * u;
            }
            clip[r] = c;""",
 """                float g = (gq - Qg.ozp) * sg, u = (uq - Qu.ozp) * su; ar[j] = g / (1.0f + expf(-g)) * u;
            }
            if (isa) for (int j = 0; j < FF; j++) ar[j] *= isa[j];   // a' = a / s_a (SmoothQuant fuer down)
            clip[r] = c;"""),
])
c = open(cpp, encoding="utf-8").read(); s = open(inc, encoding="utf-8").read()
a = c.index('// ---- M4 v2 "split"'); b = c.index("// fused FFN fuer Layer L: h [D,T] f32 -> dst [D,T] f32")
open(cpp, "w", encoding="utf-8").write(c[:a] + s + "\n" + c[b:]); print("gepatcht")
