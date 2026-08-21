#!/usr/bin/env python3
# Patch 4: out_div (Meta-Feld 7) + passes2-Zaehler im Kontext; Split-Block aus split_path.inc neu einsetzen.
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
cpp = os.path.join(here, "ggml-barra.cpp"); c = open(cpp, encoding="utf-8").read()
def rep(a, b):
    global c
    assert a in c, a[:60]; c = c.replace(a, b, 1)
if "out_div" not in c:
    rep("""    std::vector<float> inv_smooth;          // SmoothQuant: je Layer 1/s_h[D], 1/s_a[FF] (leer = keine Glaettung)""",
        """    std::vector<float> inv_smooth;          // SmoothQuant: je Layer 1/s_h[D], 1/s_a[FF] (leer = keine Glaettung)
    float out_div = 1.0f; long passes2 = 0;   // Meta-Feld 7: Ausgangsskala DIV-fach feiner kalibriert -> 2. Pass fuer clippende Zeilen""")
    rep("""    { int c; while ((c = fgetc(f)) == ' ') {} if (c >= '0' && c <= '9') { ungetc(c, f); if (fscanf(f, "%d", &ctx->split) != 1) ctx->split = 1; } }""",
        """    { int c; while ((c = fgetc(f)) == ' ') {} if (c >= '0' && c <= '9') { ungetc(c, f); if (fscanf(f, "%d", &ctx->split) != 1) ctx->split = 1; }
      while ((c = fgetc(f)) == ' ') {} if (c >= '0' && c <= '9') { ungetc(c, f); if (fscanf(f, "%f", &ctx->out_div) != 1) ctx->out_div = 1.0f; } }""")
    rep("""    GGML_LOG_WARN("barra: TPU-FFN aktiv: D=%d FF=%d B=%d bits=%d NL=%d split=%d min_batch=%d fallback=%s\\n", ctx->D, ctx->FF, ctx->B, ctx->bits, ctx->NL, ctx->split, ctx->min_batch, ctx->cpu ? "cpu" : "KEIN");""",
        """    GGML_LOG_WARN("barra: TPU-FFN aktiv: D=%d FF=%d B=%d bits=%d NL=%d split=%d out_div=%.3g min_batch=%d fallback=%s\\n", ctx->D, ctx->FF, ctx->B, ctx->bits, ctx->NL, ctx->split, ctx->out_div, ctx->min_batch, ctx->cpu ? "cpu" : "KEIN");""")
s = open(os.path.join(here, "split_path.inc"), encoding="utf-8").read()
a = c.index('// ---- M4 v2 "split"'); b = c.index("// fused FFN fuer Layer L: h [D,T] f32 -> dst [D,T] f32")
c = c[:a] + s + "\n" + c[b:]
open(cpp, "w", encoding="utf-8").write(c); print("gepatcht")
