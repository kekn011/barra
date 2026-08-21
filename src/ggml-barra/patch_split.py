#!/usr/bin/env python3
# einmaliger Patch: M4-v2-Split-Pfad in ggml-barra.cpp einbauen (idempotent ueber Marker)
import sys, os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggml-barra.cpp")
s = open(p, encoding="utf-8").read()
if "barra_ffn_split" in s: print("schon gepatcht"); sys.exit(0)
def rep(a, b):
    global s
    assert a in s, a[:60]
    s = s.replace(a, b, 1)
rep("""    barra_tpu tpu{}; barra_zbuf zi{}, zo{};
    ggml_backend_t cpu = nullptr;           // Fallback-Backend fuer Ausreisserzeilen""",
"""    barra_tpu tpu{}; barra_zbuf zi{}, zo{};
    // M4 v2 "split": je Layer 3 int8-Matmul-Packages (gate, up, down) @B (64) mit Per-Zeilen-Skalierung; q hat 3*NL Eintraege
    int split = 1;
    barra_zbuf zi_d{}, zo_g{}, zo_u{}, zi_ff{}, zo_d{};
    ggml_backend_t cpu = nullptr;           // Fallback-Backend fuer Ausreisserzeilen""")
rep("""    ctx->q.resize(ctx->NL);
    for (int i = 0; i < ctx->NL; i++) {""",
"""    { int c; while ((c = fgetc(f)) == ' ') {} if (c >= '0' && c <= '9') { ungetc(c, f); if (fscanf(f, "%d", &ctx->split) != 1) ctx->split = 1; } }
    if (ctx->split != 1 && ctx->split != 3) { fclose(f); GGML_LOG_WARN("barra: split=%d nicht unterstuetzt\\n", ctx->split); return false; }
    ctx->q.resize(ctx->NL * ctx->split);
    for (int i = 0; i < ctx->NL * ctx->split; i++) {""")
rep("""    if (barra_zc_alloc(&ctx->zi, (uint32_t) (ctx->B * ctx->D * bpe)) || barra_zc_alloc(&ctx->zo, (uint32_t) (ctx->B * ctx->D * bpe))) { GGML_LOG_WARN("barra: dmabuf alloc fehlgeschlagen\\n"); return false; }""",
"""    if (ctx->split == 1) {
        if (barra_zc_alloc(&ctx->zi, (uint32_t) (ctx->B * ctx->D * bpe)) || barra_zc_alloc(&ctx->zo, (uint32_t) (ctx->B * ctx->D * bpe))) { GGML_LOG_WARN("barra: dmabuf alloc fehlgeschlagen\\n"); return false; }
    } else {
        if (ctx->bits != 8) { GGML_LOG_WARN("barra: split-Modus erwartet int8\\n"); return false; }
        if (barra_zc_alloc(&ctx->zi_d, (uint32_t) (ctx->B * ctx->D)) || barra_zc_alloc(&ctx->zo_g, (uint32_t) (ctx->B * ctx->FF)) || barra_zc_alloc(&ctx->zo_u, (uint32_t) (ctx->B * ctx->FF))
            || barra_zc_alloc(&ctx->zi_ff, (uint32_t) (ctx->B * ctx->FF)) || barra_zc_alloc(&ctx->zo_d, (uint32_t) (ctx->B * ctx->D))) { GGML_LOG_WARN("barra: dmabuf alloc (split) fehlgeschlagen\\n"); return false; }
    }""")
rep("""    GGML_LOG_WARN("barra: TPU-FFN aktiv: D=%d FF=%d B=%d bits=%d NL=%d min_batch=%d fallback=%s\\n", ctx->D, ctx->FF, ctx->B, ctx->bits, ctx->NL, ctx->min_batch, ctx->cpu ? "cpu" : "KEIN");""",
"""    GGML_LOG_WARN("barra: TPU-FFN aktiv: D=%d FF=%d B=%d bits=%d NL=%d split=%d min_batch=%d fallback=%s\\n", ctx->D, ctx->FF, ctx->B, ctx->bits, ctx->NL, ctx->split, ctx->min_batch, ctx->cpu ? "cpu" : "KEIN");""")
split_fn = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "split_path.inc"), encoding="utf-8").read()
rep("// fused FFN fuer Layer L: h [D,T] f32 -> dst [D,T] f32\n", split_fn + "\n// fused FFN fuer Layer L: h [D,T] f32 -> dst [D,T] f32\n")
rep("""                        barra_ffn_fused(ctx, L, u->src[0], g->src[0], node->src[0], g->src[1], node);""",
"""                        if (ctx->split == 3) barra_ffn_split(ctx, L, u->src[0], g->src[0], node->src[0], g->src[1], node);
                        else                 barra_ffn_fused(ctx, L, u->src[0], g->src[0], node->src[0], g->src[1], node);""")
rep("""        barra_zc_free(&ctx->zi); barra_zc_free(&ctx->zo); barra_tpu_close(&ctx->tpu);""",
"""        if (ctx->split == 1) { barra_zc_free(&ctx->zi); barra_zc_free(&ctx->zo); }
        else { barra_zc_free(&ctx->zi_d); barra_zc_free(&ctx->zo_g); barra_zc_free(&ctx->zo_u); barra_zc_free(&ctx->zi_ff); barra_zc_free(&ctx->zo_d); }
        barra_tpu_close(&ctx->tpu);""")
open(p, "w", encoding="utf-8").write(s); print("gepatcht")
