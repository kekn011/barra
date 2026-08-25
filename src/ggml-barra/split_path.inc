// WARNUNG (Stand Review 2026-08-25): DIESER .inc IST VERALTET und divergiert vom gebauten
// ggml-barra.cpp. run_dn hier memset/indiziert eine volle B*FF-Kachel in zi_ff und liest B*D
// aus zo_d, waehrend das aktuelle ggml-barra.cpp diese dmabufs nur mit Bd*FF / Bd*D allokiert
// (Meta-Feld 9, Bd<=B). Ein erneutes Einspielen via patch_split*.py OHNE Anpassung an Bd
// erzeugt Out-of-bounds-Zugriffe, sobald Bd<B. NICHT blind re-applyen — gebautes Artefakt ist
// ggml-barra.cpp (CMakeLists); diesen .inc + patch_split*.py entweder loeschen oder aus dem
// aktuellen cpp neu generieren, damit beide nicht auseinanderlaufen.
// ---- M4 v2 "split": FFN als drei int8-Matmul-Packages @B (64er-Kacheln): gate/up: h -> [B,FF]; CPU: a = silu(g)*u
// (float); down: a -> [B,D]. Aktivierungszeilen werden per Zeile auf max|x| = QMAX (127) normiert (kein Saturieren, exakt
// fuer lineare Ops), Ergebnis zeilenweise zurueckskaliert. Ausgangsskala der Packages optional feiner kalibriert
// (Meta-Feld 7 = DIV, z.B. 3): Zeilen, deren Ausgang clippt, werden im 2. PASS mit Eingang/DIV nachgerechnet
// (grosse Zeilen -> grobe Aufloesung wie zuvor, kleine Zeilen -> DIV-fach feiner). Clippt es dann noch -> CPU-Fallback.
//
// AUSREISSER-AUSLAGERUNG (BARRA_OUTLIER, LLM.int8()-Prinzip): die k groessten Eingangskanaele werden im TPU-Eingang
// GENULLT und ihr Beitrag exakt in float auf der CPU addiert: y = TPU(h_rest) + CPU(h_out). Der Gewinn ist doppelt --
// der CPU-Anteil ist exakt, UND der TPU-Anteil hat ohne die Ausreisser einen 6-120x kleineren Zeilen-max, wodurch die
// Zeilennormierung auf 127 die uebrigen Kanaele entsprechend feiner trifft.
static float barra_qmax(void) { static float q = 0; if (q == 0) { const char * e = getenv("BARRA_QMAX"); q = e ? (float) atof(e) : 127.0f; } return q; }
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
// Persistenter Thread-Pool fuer den Glue (std::thread-Spawn pro Aufruf kostete ~ms je Start): NT Worker (BARRA_GLUE_THREADS,
// Default 4), Job = fn(r) fuer r in [0,n), dynamische Vergabe per Atomic-Zaehler; Worker spinnen kurz, dann cond-wait.
struct barra_pool {
    int nt = 1; std::vector<std::thread> th; std::mutex mu; std::condition_variable cv;
    std::atomic<int> gen{0}, next{0}, done{0}; int n = 0; const std::function<void(int)> * fn = nullptr; bool stop = false;
    void work() { for (;;) { int r = next.fetch_add(1); if (r >= n) break; (*fn)(r); } }
    void worker() { int seen = 0; for (;;) {
            int spins = 0; while (gen.load(std::memory_order_acquire) == seen && !stop) { if (++spins < 20000) continue; std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return gen.load() != seen || stop; }); }
            if (stop) return; seen = gen.load(); work(); done.fetch_add(1); } }
    barra_pool() { const char * e = getenv("BARRA_GLUE_THREADS"); nt = e ? atoi(e) : 4; if (nt < 1) nt = 1; for (int i = 1; i < nt; i++) th.emplace_back([this] { worker(); }); }
    ~barra_pool() { { std::lock_guard<std::mutex> lk(mu); stop = true; } cv.notify_all(); for (auto & t : th) if (t.joinable()) t.join(); }
    void run(int n_, const std::function<void(int)> & f) {
        if (nt == 1 || n_ < 2) { for (int r = 0; r < n_; r++) f(r); return; }
        n = n_; fn = &f; next.store(0); done.store(0);
        { std::lock_guard<std::mutex> lk(mu); gen.fetch_add(1, std::memory_order_release); } cv.notify_all();
        work(); while (done.load(std::memory_order_acquire) < nt - 1) std::this_thread::yield();
    }
};
static barra_pool & barra_pool_get() { static barra_pool p; return p; }
// Spaltenblöcke fuer das Rang-k-Update: 2 je Worker, damit ungleiche Bloecke sich ausgleichen
static int barra_nblk() { static int nb = 0; if (!nb) nb = std::max(1, barra_pool_get().nt * 2); return nb; }
template <typename F> static void barra_par_rows(int n, F fn) { std::function<void(int)> f = fn; barra_pool_get().run(n, f); }
// Zeile x[n] auf max|x| = qmax normieren und mit Package-Eingangsskala quantisieren; s_out = Zeilenskala (x = q*isc*s_out)
static inline void barra_q8_row(const float * x, int n, float qmax, float * s_out, int8_t * z, float isc, int izp) {
    float mx = 0; for (int d = 0; d < n; d++) { float a = fabsf(x[d]); if (a > mx) mx = a; }
    float srow = mx > 0 ? mx / qmax : 1.0f; *s_out = srow; const float k = 1.0f / (srow * isc);
    for (int d = 0; d < n; d++) { int v = (int) lrintf(x[d] * k) + izp; z[d] = (int8_t) (v > 127 ? 127 : v < -128 ? -128 : v); }
}
// y[n] += alpha * w[n]  (Ausreisser-Beitrag; der heisse Kern der Auslagerung)
static inline void barra_axpy(float * __restrict y, const float * __restrict w, float alpha, int n) {
    for (int j = 0; j < n; j++) y[j] += alpha * w[j];
}
// C[nrow][ncol] += A[nrow][k] * W[k][ncol]  (Rang-k-Update fuer die ausgelagerten Kanaele), spaltenblockweise ueber den
// Thread-Pool: Kanal aussen, Zeilen innen -- so wird jede Gewichtszeile EINMAL gelesen statt einmal je Zeile
// (zeilenweise waeren es bei k=128, FF=9728, n=64 rund 320 MB Gewichtsverkehr je Kachel statt 5 MB).
static void barra_rank_update(float * C, int nrow, int ncol, const float * A, int lda, const int32_t * idx, int k,
                              const ggml_fp16_t * W, int nblk) {
    if (k <= 0 || nrow <= 0) return;
    barra_par_rows(nblk, [&](int b) {
        const int j0 = (int) ((long) b * ncol / nblk), j1 = (int) ((long) (b + 1) * ncol / nblk);
        if (j1 <= j0) return;
        std::vector<float> wb(j1 - j0);
        for (int c = 0; c < k; c++) {
            const ggml_fp16_t * wsrc = W + (size_t) c * ncol + j0;
            bool used = false;
            for (int r = 0; r < nrow && !used; r++) if (A[(size_t) r * lda + idx[c]] != 0.0f) used = true;
            if (!used) continue;
            ggml_fp16_to_fp32_row(wsrc, wb.data(), j1 - j0);
            for (int r = 0; r < nrow; r++) {
                const float alpha = A[(size_t) r * lda + idx[c]];
                if (alpha == 0.0f) continue;
                barra_axpy(C + (size_t) r * ncol + j0, wb.data(), alpha, j1 - j0);
            }
        }
    });
}
static void barra_ffn_split(ggml_backend_barra_context * ctx, int L, const ggml_tensor * up_w, const ggml_tensor * gate_w, const ggml_tensor * down_w,
                            const ggml_tensor * h, ggml_tensor * dst) {
    const int D = ctx->D, FF = ctx->FF, B = ctx->B, T = (int) h->ne[1];
    const float DIV = ctx->out_div > 1.0f ? ctx->out_div : 1.0f, QMAX = barra_qmax();
    const barra_layer_q & Qg = ctx->q[3 * L], & Qu = ctx->q[3 * L + 1], & Qd = ctx->q[3 * L + 2];
    const barra_outl * ol = (L < (int) ctx->ol.size() && ctx->ol[L].kd > 0) ? &ctx->ol[L] : nullptr;
    std::vector<float> hb((size_t) T * D);   // Kopie von h (dst kann h aliasen!)
    for (int r = 0; r < T; r++) memcpy(&hb[(size_t) r * D], (const char *) h->data + (size_t) r * h->nb[1], D * sizeof(float));
    const float * ish = ctx->inv_smooth.empty() ? nullptr : &ctx->inv_smooth[(size_t) L * (D + FF)];   // SmoothQuant 1/s_h
    const float * isa = ish ? ish + D : nullptr;                                                          // 1/s_a
    std::vector<float> hs; if (ish) { hs.resize((size_t) T * D); for (int r = 0; r < T; r++) for (int d = 0; d < D; d++) hs[(size_t) r * D + d] = hb[(size_t) r * D + d] * ish[d]; }
    const float * hq = ish ? hs.data() : hb.data();
    std::vector<float> srow(B), a((size_t) B * FF);
    std::vector<char> clip(B), clip2(B);
    std::vector<float> hzt, cg, cu, cd;      // genullter TPU-Eingang + CPU-Korrekturen (nur bei Auslagerung belegt)
    std::vector<int> fb; double t0 = now_ms(), tq = 0, ti = 0, td = 0, tol = 0; int passes2 = 0;
    for (int t0i = 0; t0i < T; t0i += B) {
        const int n = std::min(B, T - t0i);
        const float * hsrc = hq + (size_t) t0i * D;   // Zeilenquelle fuer die Quantisierung (Kachel-lokal)
        // ---- Auslagerung gate/up: Ausreisserkanaele aus dem TPU-Eingang nehmen und ihren Beitrag exakt vorrechnen
        if (ol) {
            double o0 = now_ms();
            const float * horig = hq + (size_t) t0i * D;
            hzt.assign(horig, horig + (size_t) n * D);
            cg.assign((size_t) n * FF, 0.0f); cu.assign((size_t) n * FF, 0.0f);
            barra_rank_update(cg.data(), n, FF, horig, D, ol->od.data(), ol->kd, ol->wg.data(), barra_nblk());
            barra_rank_update(cu.data(), n, FF, horig, D, ol->od.data(), ol->kd, ol->wu.data(), barra_nblk());
            for (int r = 0; r < n; r++) for (int c = 0; c < ol->kd; c++) hzt[(size_t) r * D + ol->od[c]] = 0.0f;
            hsrc = hzt.data(); tol += now_ms() - o0;
        }
        // ---- gate/up: rows = zu rechnende Zeilen (Kachel-lokal), div = Eingangs-Teiler (1 = Pass 1). Elementweise Zustand:
        // a[] = bisher bestes Ergebnis (float), clipe[] = Element noch geclippt -> naechster Pass ersetzt nur diese Elemente.
        std::vector<char> clipe((size_t) B * FF, 1), clipd((size_t) B * D, 1);
        auto run_gu = [&](const std::vector<int> & rows, float div) -> int {
            double a0 = now_ms(); barra_zc_cpu_begin(&ctx->zi_d); int8_t * zi = (int8_t *) ctx->zi_d.map;
            memset(zi, 0, (size_t) B * D);
            barra_par_rows((int) rows.size(), [&](int k) { int r = rows[k]; barra_q8_row(hsrc + (size_t) r * D, D, QMAX / div, &srow[r], zi + (size_t) r * D, Qg.isc, Qg.izp); });
            barra_zc_cpu_end(&ctx->zi_d); double a1 = now_ms(); tq += a1 - a0;
            uint32_t us = 0; int rc = barra_tpu_infer(&ctx->tpu, (uint32_t) (3 * L), &ctx->zi_d, &ctx->zo_g, &us);
            rc |= barra_tpu_infer(&ctx->tpu, (uint32_t) (3 * L + 1), &ctx->zi_d, &ctx->zo_u, &us);
            double a2 = now_ms(); ti += a2 - a1; if (rc) return rc;
            barra_zc_cpu_begin(&ctx->zo_g); barra_zc_cpu_begin(&ctx->zo_u);
            const int8_t * zg = (const int8_t *) ctx->zo_g.map, * zu = (const int8_t *) ctx->zo_u.map;
            barra_par_rows((int) rows.size(), [&](int k) { int r = rows[k];
                const float sg = Qg.osc * srow[r], su = Qu.osc * srow[r]; float * ar = &a[(size_t) r * FF]; char * ce = &clipe[(size_t) r * FF];
                const float * cgr = ol ? &cg[(size_t) r * FF] : nullptr, * cur = ol ? &cu[(size_t) r * FF] : nullptr;
                const int8_t * g8 = zg + (size_t) r * FF, * u8 = zu + (size_t) r * FF; char c = 0;
                for (int j = 0; j < FF; j++) {
                    if (!ce[j]) continue;                                    // schon aufgeloest (frueherer Pass, feiner)
                    int gq = g8[j], uq = u8[j]; bool cl = (gq >= 127 || gq <= -128 || uq >= 127 || uq <= -128);
                    float g = (gq - Qg.ozp) * sg, u = (uq - Qu.ozp) * su;
                    if (ol) { g += cgr[j]; u += cur[j]; }                     // exakter CPU-Anteil der Ausreisserkanaele
                    ar[j] = g / (1.0f + expf(-g)) * u;
                    ce[j] = cl; if (cl) c = 1;
                }
                clip[r] = c; });
            barra_zc_cpu_end(&ctx->zo_g); barra_zc_cpu_end(&ctx->zo_u); tq += now_ms() - a2; return 0;
        };
        // ---- down (o[] in dst = bisher bestes Ergebnis, clipd[] = Element noch geclippt)
        auto run_dn = [&](const std::vector<int> & rows, float div) -> int {
            double a0 = now_ms(); barra_zc_cpu_begin(&ctx->zi_ff); int8_t * zi2 = (int8_t *) ctx->zi_ff.map;
            memset(zi2, 0, (size_t) B * FF);
            barra_par_rows((int) rows.size(), [&](int k) { int r = rows[k];
                barra_q8_row(&a[(size_t) r * FF], FF, QMAX / div, &srow[r], zi2 + (size_t) r * FF, Qd.isc, Qd.izp); });
            barra_zc_cpu_end(&ctx->zi_ff); double a1 = now_ms(); tq += a1 - a0;
            uint32_t us = 0; int rc = barra_tpu_infer(&ctx->tpu, (uint32_t) (3 * L + 2), &ctx->zi_ff, &ctx->zo_d, &us);
            double a2 = now_ms(); ti += a2 - a1; if (rc) return rc;
            barra_zc_cpu_begin(&ctx->zo_d); const int8_t * zd = (const int8_t *) ctx->zo_d.map;
            for (int r : rows) {
                float * o = (float *) ((char *) dst->data + (size_t) (t0i + r) * dst->nb[1]); const float sd = Qd.osc * srow[r];
                const int8_t * d8 = zd + (size_t) r * D; char * cdp = &clipd[(size_t) r * D]; char c = 0;
                const float * cdr = ol ? &cd[(size_t) r * D] : nullptr;
                for (int d = 0; d < D; d++) { if (!cdp[d]) continue; int v = d8[d]; bool cl = (v >= 127 || v <= -128);
                    o[d] = (v - Qd.ozp) * sd + (ol ? cdr[d] : 0.0f); cdp[d] = cl; if (cl) c = 1; }
                clip2[r] = c;
            }
            barra_zc_cpu_end(&ctx->zo_d); td += now_ms() - a2; return 0;
        };
        static int maxpass = 0; if (!maxpass) { const char * e = getenv("BARRA_MAX_PASS"); maxpass = e ? atoi(e) : 3; if (maxpass < 1) maxpass = 1; }
        std::vector<int> rows(n); for (int r = 0; r < n; r++) rows[r] = r;
        bool failed = false; float div = 1.0f;
        for (int pass = 1; pass <= maxpass && !rows.empty(); pass++, div *= DIV) {
            if (pass > 1) passes2++;
            if (run_gu(rows, div)) { for (int r : rows) fb.push_back(t0i + r); failed = true; break; }
            std::vector<int> nxt; for (int r : rows) if (clip[r]) nxt.push_back(r); rows.swap(nxt);
            if (DIV <= 1.0f) break;
        }
        if (failed) continue;
        std::vector<int> okrows; for (int r = 0; r < n; r++) { if (clip[r]) fb.push_back(t0i + r); else okrows.push_back(r); }
        if (okrows.empty()) continue;
        // ---- a fuer den down-Schritt vorbereiten: SmoothQuant einmal, dann Ausreisserkanaele auslagern
        if (isa) barra_par_rows((int) okrows.size(), [&](int k) { int r = okrows[k]; float * ar = &a[(size_t) r * FF]; for (int j = 0; j < FF; j++) ar[j] *= isa[j]; });
        if (ol) {
            double o0 = now_ms(); cd.assign((size_t) n * D, 0.0f);
            // Zeilen, die schon in den CPU-Fallback gefallen sind, tragen hier nichts bei -> ihre a-Werte auf 0
            for (int r = 0, k = 0; r < n; r++) {
                if (k < (int) okrows.size() && okrows[k] == r) { k++; continue; }
                for (int c = 0; c < ol->kf; c++) a[(size_t) r * FF + ol->of[c]] = 0.0f;
            }
            barra_rank_update(cd.data(), n, D, a.data(), FF, ol->of.data(), ol->kf, ol->wd.data(), barra_nblk());
            for (int r : okrows) for (int c = 0; c < ol->kf; c++) a[(size_t) r * FF + ol->of[c]] = 0.0f;
            tol += now_ms() - o0;
        }
        rows = okrows; div = 1.0f;
        for (int pass = 1; pass <= maxpass && !rows.empty(); pass++, div *= DIV) {
            if (pass > 1) passes2++;
            if (run_dn(rows, div)) { for (int r : rows) fb.push_back(t0i + r); failed = true; break; }
            std::vector<int> nxt; for (int r : rows) if (clip2[r]) nxt.push_back(r); rows.swap(nxt);
            if (DIV <= 1.0f) break;
        }
        if (failed) continue;
        for (int r : okrows) if (clip2[r]) fb.push_back(t0i + r);
    }
    ctx->ms_tpu += now_ms() - t0; ctx->calls++; ctx->rows += T; ctx->passes2 += passes2; ctx->ms_ol += tol;
    if (!fb.empty()) barra_ffn_fallback(ctx, up_w, gate_w, down_w, hb.data(), dst, fb);
    if (getenv("BARRA_CHECK") && ctx->cpu && ctx->calls <= 48) {
        std::vector<float> tpu((size_t) T * D); for (int r = 0; r < T; r++) memcpy(&tpu[(size_t) r * D], (char *) dst->data + (size_t) r * dst->nb[1], D * sizeof(float));
        std::vector<int> allr(T); for (int r = 0; r < T; r++) allr[r] = r;
        barra_ffn_fallback(ctx, up_w, gate_w, down_w, hb.data(), dst, allr);
        double worst = 0, sumrel = 0; int worst_r = -1;
        for (int r = 0; r < T; r++) { const float * ref = (const float *) ((char *) dst->data + (size_t) r * dst->nb[1]); double e = 0, nn = 0;
            for (int d = 0; d < D; d++) { double df = tpu[(size_t) r * D + d] - ref[d]; e += df * df; nn += (double) ref[d] * ref[d]; }
            double rel = sqrt(e / (nn + 1e-20)); sumrel += rel; if (rel > worst) { worst = rel; worst_r = r; } }
        GGML_LOG_WARN("barra: CHECK(split) L%d T=%d rel-Fehler mittel %.4f, max %.4f (Zeile %d) pass2=%d fb=%d ol=%d/%d\n", L, T, sumrel / T, worst, worst_r, passes2, (int) fb.size(), ol ? ol->kd : 0, ol ? ol->kf : 0);
    }
    if (ctx->log) GGML_LOG_WARN("barra: L%d T=%d fb=%d pass2=%d  %.2f ms [quant+glue %.2f infer %.2f dequant %.2f ausreisser %.2f] (kum: %ld Aufrufe, %ld Zeilen, fb %ld, pass2 %ld, tpu %.1f ms, fb %.1f ms, ol %.1f ms)\n", L, T, (int) fb.size(), passes2, now_ms() - t0, tq, ti, td, tol, ctx->calls, ctx->rows, ctx->fb_rows, ctx->passes2, ctx->ms_tpu, ctx->ms_fb, ctx->ms_ol);
}
