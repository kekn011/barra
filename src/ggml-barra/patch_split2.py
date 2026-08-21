#!/usr/bin/env python3
# Patch 2 fuer den Split-Pfad: (a) Glue (silu*mul + Zeilenquantisierung) mehrfaedig, (b) Eingangsnormierung auf 100 statt 127
# (=27% Ausgangs-Headroom gegen Clipping/Fallback, ohne Packages neu zu bauen; env BARRA_QMAX ueberschreibt).
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
inc = os.path.join(here, "split_path.inc"); s = open(inc, encoding="utf-8").read()
if "barra_par_rows" in s: print("schon gepatcht"); sys.exit(0)
def rep(a, b):
    global s
    assert a in s, a[:70]; s = s.replace(a, b, 1)
# (b) qmax
rep("""static inline void barra_q8_row(const float * x, int n, float * s_out, int8_t * z, float isc, int izp) {
    float mx = 0; for (int d = 0; d < n; d++) { float a = fabsf(x[d]); if (a > mx) mx = a; }
    float srow = mx > 0 ? mx / 127.0f : 1.0f; *s_out = srow; const float k = 1.0f / (srow * isc);""",
"""static float barra_qmax(void) { static float q = 0; if (q == 0) { const char * e = getenv("BARRA_QMAX"); q = e ? (float) atof(e) : 100.0f; } return q; }
// einfache Zeilen-Parallelisierung ueber std::thread (Glue ist reines Elementwise; BARRA_GLUE_THREADS, Default 4)
#include <thread>
template <typename F> static void barra_par_rows(int n, F fn) {
    static int nt = 0; if (!nt) { const char * e = getenv("BARRA_GLUE_THREADS"); nt = e ? atoi(e) : 4; if (nt < 1) nt = 1; }
    if (nt == 1 || n < 2) { for (int r = 0; r < n; r++) fn(r); return; }
    std::vector<std::thread> th; int per = (n + nt - 1) / nt;
    for (int t = 0; t < nt; t++) { int lo = t * per, hi = std::min(n, lo + per); if (lo >= hi) break; th.emplace_back([=]() { for (int r = lo; r < hi; r++) fn(r); }); }
    for (auto & x : th) x.join();
}
static inline void barra_q8_row(const float * x, int n, float * s_out, int8_t * z, float isc, int izp) {
    float mx = 0; for (int d = 0; d < n; d++) { float a = fabsf(x[d]); if (a > mx) mx = a; }
    float srow = mx > 0 ? mx / barra_qmax() : 1.0f; *s_out = srow; const float k = 1.0f / (srow * isc);""")
# (a) parallel: input quant loop
rep("""        for (int r = 0; r < B; r++) {
            if (r >= n) { memset(zi + (size_t) r * D, 0, D); srow[r] = 0; continue; }
            barra_q8_row(&hb[(size_t) (t0i + r) * D], D, &srow[r], zi + (size_t) r * D, Qg.isc, Qg.izp);   // gate/up: gleiche Kalibrierung -> gleiches isc
        }""",
"""        barra_par_rows(B, [&](int r) {
            if (r >= n) { memset(zi + (size_t) r * D, 0, D); srow[r] = 0; return; }
            barra_q8_row(&hb[(size_t) (t0i + r) * D], D, &srow[r], zi + (size_t) r * D, Qg.isc, Qg.izp);   // gate/up: gleiche Kalibrierung -> gleiches isc
        });""")
# silu*mul loop
rep("""        for (int r = 0; r < n; r++) {
            const float sg = Qg.osc * srow[r], su = Qu.osc * srow[r]; float * ar = &a[(size_t) r * FF];
            const int8_t * g8 = zg + (size_t) r * FF, * u8 = zu + (size_t) r * FF;
            for (int j = 0; j < FF; j++) {
                int gq = g8[j], uq = u8[j]; if (gq >= 127 || gq <= -128 || uq >= 127 || uq <= -128) clip[r] = 1;
                float g = (gq - Qg.ozp) * sg, u = (uq - Qu.ozp) * su; ar[j] = g / (1.0f + expf(-g)) * u;
            }
        }""",
"""        barra_par_rows(n, [&](int r) {
            const float sg = Qg.osc * srow[r], su = Qu.osc * srow[r]; float * ar = &a[(size_t) r * FF];
            const int8_t * g8 = zg + (size_t) r * FF, * u8 = zu + (size_t) r * FF; char c = 0;
            for (int j = 0; j < FF; j++) {
                int gq = g8[j], uq = u8[j]; if (gq >= 127 || gq <= -128 || uq >= 127 || uq <= -128) c = 1;
                float g = (gq - Qg.ozp) * sg, u = (uq - Qu.ozp) * su; ar[j] = g / (1.0f + expf(-g)) * u;
            }
            clip[r] = c;
        });""")
rep("""        for (int r = 0; r < B; r++) { if (r >= n) { memset(zi2 + (size_t) r * FF, 0, FF); srow2[r] = 0; continue; } barra_q8_row(&a[(size_t) r * FF], FF, &srow2[r], zi2 + (size_t) r * FF, Qd.isc, Qd.izp); }""",
"""        barra_par_rows(B, [&](int r) { if (r >= n) { memset(zi2 + (size_t) r * FF, 0, FF); srow2[r] = 0; return; } barra_q8_row(&a[(size_t) r * FF], FF, &srow2[r], zi2 + (size_t) r * FF, Qd.isc, Qd.izp); });""")
open(inc, "w", encoding="utf-8").write(s)
# in ggml-barra.cpp: den alten Split-Block durch die neue inc ersetzen (zwischen Markern)
cpp = os.path.join(here, "ggml-barra.cpp"); c = open(cpp, encoding="utf-8").read()
a = c.index('// ---- M4 v2 "split"'); b = c.index("// fused FFN fuer Layer L: h [D,T] f32 -> dst [D,T] f32")
c = c[:a] + s + "\n" + c[b:]
open(cpp, "w", encoding="utf-8").write(c); print("gepatcht (inc + cpp)")
