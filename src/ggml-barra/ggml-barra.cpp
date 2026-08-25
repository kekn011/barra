// ggml-barra: ggml-Backend, das die Feed-Forward-Bloecke (ffn_up/ffn_gate/swiglu/ffn_down je Layer) als EIN
// vorkompiliertes TPU-Package auf der Tensor-G3-TPU rechnet (ueber tpud/libbarra, Zero-Copy dmabuf).
// Muster: BLAS-Backend (Host-Buffer, supports_op selektiv). Nur Prefill/Batch (T >= BARRA_MIN_BATCH); Decode bleibt CPU.
//
// Packages: pro Layer f(h) = Wd(silu(Wg h) * (Wu h)) mit h = normierter Eingang (rmsnorm bleibt auf der CPU, Residual auch),
// Batch B Zeilen (Tiles), int16 (16x8) oder int8 I/O; Skalen aus <meta>: "D FF B bits NL" + je Layer "isc izp osc ozp".
// Ausreisserzeilen (Eingang saturiert / Ausgang clippt) -> CPU-Backend (Mini-Graph auf den Original-Gewichten).
// Env: BARRA_FFN_META=<meta-Datei> (Pflicht, sonst passiv), BARRA_MIN_BATCH (8), BARRA_FFN_LOG=1, BARRA_SOCK_DIR (libbarra).
#include "ggml-barra.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
extern "C" {
#include "barra.h"
}
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>

struct barra_layer_q { float isc, osc; int izp, ozp; };
// Ausreisser-Auslagerung je Layer: die kd groessten h-Kanaele (gate/up) und kf groessten a-Kanaele (down) samt der
// zugehoerigen Gewichtsspalten. Datei: gen_outliers.py (Format BOL1); Kanaele stehen nach Wichtigkeit sortiert,
// deshalb ist jedes Praefix k' <= k wieder eine gueltige Auswahl (BARRA_OUTLIER_FRAC).
struct barra_outl {
    int kd = 0, kf = 0;
    std::vector<int32_t> od, of;
    std::vector<ggml_fp16_t> wg, wu, wd;   // [kd][FF], [kd][FF], [kf][D] -- f16, erst im Kernel blockweise entpackt
};
struct ggml_backend_barra_context {
    bool ready = false;
    int D = 0, FF = 0, B = 0, bits = 16, NL = 0, min_batch = 8, log = 0;
    int Bd = 0;   // Kachelgroesse fuer down (Meta-Feld 9). gate/up laufen bei B (bis 160 amortisierbar), down cappt
                  // bei 80 im Compile -> eigene, kleinere Kachel; down wird je B-Tile in Bd-Subkacheln gerechnet.
    std::vector<barra_layer_q> q;
    barra_tpu tpu{}; barra_zbuf zi{}, zo{};
    // M4 v2 "split": je Layer 3 int8-Matmul-Packages (gate, up, down) @B (64) mit Per-Zeilen-Skalierung; q hat 3*NL Eintraege
    int split = 1;
    barra_zbuf zi_d{}, zo_g{}, zo_u{}, zi_ff{}, zo_d{};
    std::vector<float> inv_smooth;          // SmoothQuant: je Layer 1/s_h[D], 1/s_a[FF] (leer = keine Glaettung)
    float out_div = 1.0f, out_div_gu = 1.0f; long passes2 = 0;   // Meta-Feld 7: Ausgangsskala DIV-fach feiner kalibriert -> 2. Pass fuer clippende Zeilen
    std::vector<barra_outl> ol;             // je Layer; leer / kd=0 = keine Auslagerung
    ggml_backend_t cpu = nullptr;           // Fallback-Backend fuer Ausreisserzeilen
    long calls = 0, rows = 0, fb_rows = 0; double ms_tpu = 0, ms_fb = 0, ms_ol = 0;
};

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// "blk.N.ffn_gate.weight" -> layer N, kind 0=up 1=gate 2=down, sonst -1
static int barra_ffn_kind(const ggml_tensor * w, int * layer) {
    if (!w) return -1;
    int L = -1; char kind[16] = {0};
    if (sscanf(w->name, "blk.%d.ffn_%15[a-z].weight", &L, kind) != 2) return -1;
    if (layer) *layer = L;
    if (!strcmp(kind, "up")) return 0;
    if (!strcmp(kind, "gate")) return 1;
    if (!strcmp(kind, "down")) return 2;
    return -1;
}

// BARRA_OUTLIER=<datei.bol> laden; BARRA_OUTLIER_FRAC (Default: alles, was in der Datei steht) waehlt ein Praefix.
// Es werden nur die tatsaechlich genutzten Gewichtszeilen gelesen -- ein 5-%-File taugt damit auch fuer 0,5 %.
static void barra_load_outliers(ggml_backend_barra_context * ctx) {
    const char * op = getenv("BARRA_OUTLIER");
    if (!op) return;
    FILE * f = fopen(op, "rb");
    if (!f) { GGML_LOG_WARN("barra: Ausreisser-Datei %s nicht lesbar - Auslagerung aus\n", op); return; }
    char magic[4] = {0}; int32_t hdr[6] = {0};
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BOL1", 4) || fread(hdr, sizeof(int32_t), 6, f) != 6) {
        fclose(f); GGML_LOG_WARN("barra: %s hat keinen BOL1-Kopf - Auslagerung aus\n", op); return; }
    const int fD = hdr[0], fFF = hdr[1], fNL = hdr[2], fkd = hdr[3], fkf = hdr[4], dtype = hdr[5];
    if (fD != ctx->D || fFF != ctx->FF || fNL < ctx->NL || dtype != 1) {
        fclose(f); GGML_LOG_WARN("barra: %s passt nicht (D=%d FF=%d NL=%d dtype=%d, erwartet %d/%d/>=%d/1)\n", op, fD, fFF, fNL, dtype, ctx->D, ctx->FF, ctx->NL); return; }
    int kd = fkd, kf = fkf;
    if (const char * e = getenv("BARRA_OUTLIER_FRAC")) {
        // "0.05" = beide Seiten, "0.05,0" = nur h (gate/up). Letzteres ist der belegte Fall: die Ausreisser von h
        // sind ueber Texte hinweg dieselben Kanaele, die von a wandern mit dem Eingang -> statische Auswahl bringt
        // dort nichts (gemessen 19.8.: selbst eine statische Obermenge von 50 % der Kanaele nur 1,7x).
        const double fr = atof(e);
        const char * comma = strchr(e, ',');
        const double fr_f = comma ? atof(comma + 1) : fr;
        kd = (int) lround(fr * fD); kf = (int) lround(fr_f * fFF);
        if (kd > fkd) kd = fkd; if (kf > fkf) kf = fkf; if (kd < 0) kd = 0; if (kf < 0) kf = 0;
    }
    if (kd <= 0 && kf <= 0) { fclose(f); GGML_LOG_WARN("barra: Ausreisser-Praefix leer - Auslagerung aus\n"); return; }
    const long hdr_bytes = 4 + 6 * (long) sizeof(int32_t);
    const long per_layer = (long) (fkd + fkf) * (long) sizeof(int32_t)      // od + of
                         + (2 * (long) fkd * fFF + (long) fkf * fD) * (long) sizeof(uint16_t);   // wg + wu + wd
    std::vector<int32_t> idx;
    auto rd_f16 = [&](std::vector<ggml_fp16_t> & dst, long off, size_t nrow, size_t ncol) -> bool {
        if (nrow == 0) { dst.clear(); return true; }
        if (fseek(f, off, SEEK_SET)) return false;
        dst.resize(nrow * ncol);
        return fread(dst.data(), sizeof(ggml_fp16_t), nrow * ncol, f) == nrow * ncol;
    };
    ctx->ol.assign(ctx->NL, barra_outl());
    size_t bytes = 0; bool ok = true;
    for (int L = 0; L < ctx->NL && ok; L++) {
        const long base = hdr_bytes + (long) L * per_layer;
        barra_outl & o = ctx->ol[L]; o.kd = kd; o.kf = kf;
        idx.resize(fkd);
        ok = ok && !fseek(f, base, SEEK_SET) && fread(idx.data(), sizeof(int32_t), fkd, f) == (size_t) fkd;
        if (ok) o.od.assign(idx.begin(), idx.begin() + kd);
        idx.resize(fkf);
        ok = ok && !fseek(f, base + (long) fkd * sizeof(int32_t), SEEK_SET) && fread(idx.data(), sizeof(int32_t), fkf, f) == (size_t) fkf;
        if (ok) o.of.assign(idx.begin(), idx.begin() + kf);
        const long wbase = base + (long) (fkd + fkf) * sizeof(int32_t);
        const long wg_off = wbase, wu_off = wbase + (long) fkd * fFF * (long) sizeof(uint16_t);
        const long wd_off = wu_off + (long) fkd * fFF * (long) sizeof(uint16_t);
        ok = ok && rd_f16(o.wg, wg_off, kd, fFF) && rd_f16(o.wu, wu_off, kd, fFF) && rd_f16(o.wd, wd_off, kf, fD);
        for (int c = 0; ok && c < kd; c++) if (o.od[c] < 0 || o.od[c] >= fD) ok = false;
        for (int c = 0; ok && c < kf; c++) if (o.of[c] < 0 || o.of[c] >= fFF) ok = false;
        bytes += (o.wg.size() + o.wu.size() + o.wd.size()) * sizeof(ggml_fp16_t);
    }
    fclose(f);
    if (!ok) { ctx->ol.clear(); GGML_LOG_WARN("barra: %s unvollstaendig/ungueltig - Auslagerung aus\n", op); return; }
    GGML_LOG_WARN("barra: Ausreisser-Auslagerung aktiv: kd=%d/%d (%.2f%% von D) kf=%d/%d (%.2f%% von FF), %d Layer, %.0f MB\n",
                  kd, fkd, 100.0 * kd / fD, kf, fkf, 100.0 * kf / fFF, ctx->NL, bytes / 1e6);
}

static bool barra_load_meta(ggml_backend_barra_context * ctx) {
    const char * mp = getenv("BARRA_FFN_META");
    if (!mp) { GGML_LOG_WARN("barra: BARRA_FFN_META nicht gesetzt - TPU-Backend passiv\n"); return false; }
    FILE * f = fopen(mp, "r");
    if (!f) { GGML_LOG_WARN("barra: meta %s nicht lesbar\n", mp); return false; }
    if (fscanf(f, "%d %d %d %d %d", &ctx->D, &ctx->FF, &ctx->B, &ctx->bits, &ctx->NL) != 5) { fclose(f); GGML_LOG_WARN("barra: meta-Kopf ungueltig\n"); return false; }
    // Optionale Kopf-Felder 6..9 (split, out_div, out_div_gu, Bd) stehen im REST
    // DERSELBEN Zeile. Wir lesen den Zeilenrest komplett und parsen ihn mit EINEM
    // sscanf; so kann ein fehlendes Feld nicht den Zeilenumbruch verschlucken und in
    // die Layer-Daten der naechsten Zeile hineinlesen (Feld 8 = eigener DIV fuer
    // gate/up, Feld 9 = eigene down-Kachel).
    ctx->split = 1; ctx->out_div = 1.0f; ctx->out_div_gu = 1.0f; ctx->Bd = 0;
    { char rest[256];
      if (fgets(rest, sizeof rest, f)) {
          float od = 1.0f, odg = 1.0f; int sp = 1, bd = 0;
          int nf = sscanf(rest, "%d %f %f %d", &sp, &od, &odg, &bd);
          if (nf >= 1) ctx->split      = sp;
          if (nf >= 2) ctx->out_div    = od;
          ctx->out_div_gu = (nf >= 3) ? odg : ctx->out_div;   // Feld 8 fehlt -> gleicher DIV wie down
          if (nf >= 4) ctx->Bd         = bd;
      } else ctx->out_div_gu = ctx->out_div;
    }
    if (ctx->Bd <= 0 || ctx->Bd > ctx->B) ctx->Bd = ctx->B;   // Default: down laeuft in derselben Kachel wie gate/up
    if (ctx->split != 1 && ctx->split != 3) { fclose(f); GGML_LOG_WARN("barra: split=%d nicht unterstuetzt\n", ctx->split); return false; }
    ctx->q.resize(ctx->NL * ctx->split);
    for (int i = 0; i < ctx->NL * ctx->split; i++) {
        if (fscanf(f, "%f %d %f %d", &ctx->q[i].isc, &ctx->q[i].izp, &ctx->q[i].osc, &ctx->q[i].ozp) != 4) { fclose(f); GGML_LOG_WARN("barra: meta Layer %d ungueltig\n", i); return false; }
    }
    fclose(f);
    if (ctx->split == 3) {   // optionale SmoothQuant-Skalen: <meta>.smooth (gleicher Basisname)
        std::string sp = mp; size_t dot = sp.rfind(".meta"); if (dot != std::string::npos) sp = sp.substr(0, dot) + ".smooth";
        FILE * fs = fopen(sp.c_str(), "rb");
        if (fs) {
            size_t per = (size_t) ctx->D + ctx->FF; ctx->inv_smooth.resize(per * ctx->NL);
            size_t got = fread(ctx->inv_smooth.data(), sizeof(float), per * ctx->NL, fs); fclose(fs);
            if (got != per * ctx->NL) { GGML_LOG_WARN("barra: smooth-Datei %s zu kurz (%zu/%zu) - ignoriert\n", sp.c_str(), got, per * ctx->NL); ctx->inv_smooth.clear(); }
            else { for (auto & v : ctx->inv_smooth) v = v != 0 ? 1.0f / v : 1.0f; GGML_LOG_WARN("barra: SmoothQuant-Skalen geladen (%s)\n", sp.c_str()); }
        }
    }
    if (ctx->split == 3) barra_load_outliers(ctx);
    if (const char * e = getenv("BARRA_MIN_BATCH")) ctx->min_batch = atoi(e);
    if (getenv("BARRA_FFN_LOG")) ctx->log = 1;
    if (barra_tpu_open(&ctx->tpu)) { GGML_LOG_WARN("barra: tpud nicht erreichbar (BARRA_SOCK_DIR?)\n"); return false; }
    uint32_t nm = 0, isz = 0, osz = 0;
    if (barra_tpu_info(&ctx->tpu, 0, &isz, &osz, &nm) == 0 && (int) nm < ctx->NL) { GGML_LOG_WARN("barra: tpud hat %u Modelle, meta erwartet %d\n", nm, ctx->NL); return false; }
    size_t bpe = ctx->bits == 16 ? 2 : 1;
    if (ctx->split == 1) {
        if (barra_zc_alloc(&ctx->zi, (uint32_t) (ctx->B * ctx->D * bpe)) || barra_zc_alloc(&ctx->zo, (uint32_t) (ctx->B * ctx->D * bpe))) { GGML_LOG_WARN("barra: dmabuf alloc fehlgeschlagen\n"); return false; }
    } else {
        if (ctx->bits != 8) { GGML_LOG_WARN("barra: split-Modus erwartet int8\n"); return false; }
        if (barra_zc_alloc(&ctx->zi_d, (uint32_t) (ctx->B * ctx->D)) || barra_zc_alloc(&ctx->zo_g, (uint32_t) (ctx->B * ctx->FF)) || barra_zc_alloc(&ctx->zo_u, (uint32_t) (ctx->B * ctx->FF))
            || barra_zc_alloc(&ctx->zi_ff, (uint32_t) (ctx->Bd * ctx->FF)) || barra_zc_alloc(&ctx->zo_d, (uint32_t) (ctx->Bd * ctx->D))) { GGML_LOG_WARN("barra: dmabuf alloc (split) fehlgeschlagen\n"); return false; }
    }
    // CPU-Fallback direkt aus ggml-cpu (kein Umweg ueber die Registry in libggml -> keine Link-Zyklen im Shared-Build)
    ctx->cpu = ggml_backend_cpu_init();
    if (ctx->cpu) {   // Threads des Fallback-CPU-Backends (Default 4)
        int nt = 4; if (const char * e = getenv("BARRA_FB_THREADS")) nt = atoi(e);
        if (nt > 0) ggml_backend_cpu_set_n_threads(ctx->cpu, nt);
    }
    GGML_LOG_WARN("barra: TPU-FFN aktiv: D=%d FF=%d B=%d/Bd=%d bits=%d NL=%d split=%d out_div=%.3g/%.3g min_batch=%d fallback=%s\n", ctx->D, ctx->FF, ctx->B, ctx->Bd, ctx->bits, ctx->NL, ctx->split, ctx->out_div_gu, ctx->out_div, ctx->min_batch, ctx->cpu ? "cpu" : "KEIN");
    ctx->ready = true;
    return true;
}

// Ausreisserzeilen (Indizes rows[]) ueber das CPU-Backend: Mini-Graph auf den Original-Gewichten
// hb = KOPIE von h (T x D, contiguous). WICHTIG: der ggml-Allocator legt dst oft auf den Speicher von h (h ist nach
// gate/up tot) - wer h nach dem ersten dst-Schreiben liest, liest dst. Deshalb arbeiten Fallback+Check auf der Kopie.
static void barra_ffn_fallback(ggml_backend_barra_context * ctx, const ggml_tensor * up_w, const ggml_tensor * gate_w, const ggml_tensor * down_w,
                               const float * hb, ggml_tensor * dst, const std::vector<int> & rows) {
    if (!ctx->cpu || rows.empty()) return;
    double t0 = now_ms();
    const int n = (int) rows.size(); const int D = ctx->D;
    ggml_init_params ip = { /*mem_size*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(), /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * gc = ggml_init(ip);
    ggml_tensor * hin = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, n);
    ggml_tensor * g = ggml_mul_mat(gc, (ggml_tensor *) gate_w, hin);
    ggml_tensor * u = ggml_mul_mat(gc, (ggml_tensor *) up_w, hin);
    ggml_tensor * s = ggml_swiglu_split(gc, g, u);
    ggml_tensor * d = ggml_mul_mat(gc, (ggml_tensor *) down_w, s);
    ggml_cgraph * gf = ggml_new_graph(gc);
    ggml_build_forward_expand(gf, d);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gc, ctx->cpu);
    for (int i = 0; i < n; i++) ggml_backend_tensor_set(hin, hb + (size_t) rows[i] * D, (size_t) i * D * sizeof(float), D * sizeof(float));
    ggml_backend_graph_compute(ctx->cpu, gf);
    for (int i = 0; i < n; i++) ggml_backend_tensor_get(d, (char *) dst->data + (size_t) rows[i] * dst->nb[1], (size_t) i * D * sizeof(float), D * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(gc);
    ctx->fb_rows += n; ctx->ms_fb += now_ms() - t0;
}

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
// C[nrow][ncol] += A[nrow][k] * W[k][ncol]  (Rang-k-Update fuer die ausgelagerten Kanaele). Parallel ueber KLEINE
// Spaltenbloecke (BLK): je Block werden alle k Gewichtsbloecke EINMAL entpackt (Wloc), dann pro Zeile die k Kanaele
// in den 2-KB-cr-Block akkumuliert -- der bleibt ueber den ganzen k-Sweep im L1. Vorher lief die Schleife Kanal-aussen,
// wodurch C k-mal aus dem Speicher gestrichen wurde (speichergebunden, in Konkurrenz zur TPU-DMA). Reihenfolge der
// Akkumulation (c aufsteigend je (r,j)) ist unveraendert -> bit-identisches Ergebnis.
static void barra_rank_update(float * C, int nrow, int ncol, const float * A, int lda, const int32_t * idx, int k,
                              const ggml_fp16_t * W, int nblk) {
    (void) nblk;
    if (k <= 0 || nrow <= 0) return;
    const int BLK = 512;                       // cr-Block je Zeile = 2 KB -> L1-resident ueber den k-Sweep
    const int nb = (ncol + BLK - 1) / BLK;
    barra_par_rows(nb, [&](int b) {
        const int j0 = b * BLK, jn = (ncol - j0 < BLK) ? ncol - j0 : BLK;
        static thread_local std::vector<float> Wloc;   // je Worker-Thread wiederverwendet (keine Alloc im Hot-Loop)
        Wloc.resize((size_t) k * jn);
        for (int c = 0; c < k; c++) ggml_fp16_to_fp32_row(W + (size_t) c * ncol + j0, &Wloc[(size_t) c * jn], jn);
        for (int r = 0; r < nrow; r++) {
            float * __restrict cr = C + (size_t) r * ncol + j0;
            for (int c = 0; c < k; c++) {
                const float alpha = A[(size_t) r * lda + idx[c]];
                if (alpha == 0.0f) continue;
                const float * __restrict wl = &Wloc[(size_t) c * jn];
                for (int j = 0; j < jn; j++) cr[j] += alpha * wl[j];
            }
        }
    });
}
static void barra_ffn_split(ggml_backend_barra_context * ctx, int L, const ggml_tensor * up_w, const ggml_tensor * gate_w, const ggml_tensor * down_w,
                            const ggml_tensor * h, ggml_tensor * dst) {
    const int D = ctx->D, FF = ctx->FF, B = ctx->B, Bd = ctx->Bd > 0 ? ctx->Bd : ctx->B, T = (int) h->ne[1];
    const float DIV = ctx->out_div > 1.0f ? ctx->out_div : 1.0f, QMAX = barra_qmax();
    const float DIV_GU = ctx->out_div_gu > 1.0f ? ctx->out_div_gu : 1.0f;   // gate/up eigener Teiler (s. Meta-Feld 8)
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
        const float * horig = hq + (size_t) t0i * D;
        bool corr_todo = false;   // Rang-k-Update steht noch aus -> wird UNTER der laufenden Inferenz gerechnet
        if (ol && ol->kd > 0) {
            double o0 = now_ms();
            hzt.assign(horig, horig + (size_t) n * D);
            for (int r = 0; r < n; r++) for (int c = 0; c < ol->kd; c++) hzt[(size_t) r * D + ol->od[c]] = 0.0f;
            cg.assign((size_t) n * FF, 0.0f); cu.assign((size_t) n * FF, 0.0f);
            hsrc = hzt.data(); corr_todo = true; tol += now_ms() - o0;
        }
        // ---- gate/up: rows = zu rechnende Zeilen (Kachel-lokal), div = Eingangs-Teiler (1 = Pass 1). Elementweise Zustand:
        // a[] = bisher bestes Ergebnis (float), clipe[] = Element noch geclippt -> naechster Pass ersetzt nur diese Elemente.
        std::vector<char> clipe((size_t) B * FF, 1), clipd((size_t) B * D, 1);
        auto run_gu = [&](const std::vector<int> & rows, float div) -> int {
            double a0 = now_ms(); barra_zc_cpu_begin(&ctx->zi_d); int8_t * zi = (int8_t *) ctx->zi_d.map;
            memset(zi, 0, (size_t) B * D);
            barra_par_rows((int) rows.size(), [&](int k) { int r = rows[k]; barra_q8_row(hsrc + (size_t) r * D, D, QMAX / div, &srow[r], zi + (size_t) r * D, Qg.isc, Qg.izp); });
            barra_zc_cpu_end(&ctx->zi_d); double a1 = now_ms(); tq += a1 - a0;
            // Absenden/Warten getrennt: das Rang-k-Update der ausgelagerten Kanaele laeuft, WAEHREND die TPU rechnet
            // (vorher spann die CPU hier nur am Fence). Es fasst nur cg/cu/horig an, nie die TPU-Puffer.
            uint32_t us = 0;
            int rc = barra_tpu_submit(&ctx->tpu, (uint32_t) (3 * L), &ctx->zi_d, &ctx->zo_g);
            if (!rc && corr_todo) { double c0 = now_ms(); barra_rank_update(cg.data(), n, FF, horig, D, ol->od.data(), ol->kd, ol->wg.data(), barra_nblk()); tol += now_ms() - c0; }
            if (!rc) rc = barra_tpu_wait(&ctx->tpu, &ctx->zi_d, &ctx->zo_g, &us);
            if (!rc) rc = barra_tpu_submit(&ctx->tpu, (uint32_t) (3 * L + 1), &ctx->zi_d, &ctx->zo_u);
            if (!rc && corr_todo) { double c0 = now_ms(); barra_rank_update(cu.data(), n, FF, horig, D, ol->od.data(), ol->kd, ol->wu.data(), barra_nblk()); tol += now_ms() - c0; }
            if (!rc) rc = barra_tpu_wait(&ctx->tpu, &ctx->zi_d, &ctx->zo_u, &us);
            corr_todo = false;
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
        // rows liegen in [base, base+Bd); die down-Puffer sind Bd breit und werden mit (r-base) indiziert.
        auto run_dn = [&](const std::vector<int> & rows, float div, int base) -> int {
            double a0 = now_ms(); barra_zc_cpu_begin(&ctx->zi_ff); int8_t * zi2 = (int8_t *) ctx->zi_ff.map;
            memset(zi2, 0, (size_t) Bd * FF);
            barra_par_rows((int) rows.size(), [&](int k) { int r = rows[k];
                barra_q8_row(&a[(size_t) r * FF], FF, QMAX / div, &srow[r], zi2 + (size_t) (r - base) * FF, Qd.isc, Qd.izp); });
            barra_zc_cpu_end(&ctx->zi_ff); double a1 = now_ms(); tq += a1 - a0;
            uint32_t us = 0; int rc = barra_tpu_infer(&ctx->tpu, (uint32_t) (3 * L + 2), &ctx->zi_ff, &ctx->zo_d, &us);
            double a2 = now_ms(); ti += a2 - a1; if (rc) return rc;
            barra_zc_cpu_begin(&ctx->zo_d); const int8_t * zd = (const int8_t *) ctx->zo_d.map;
            for (int r : rows) {
                float * o = (float *) ((char *) dst->data + (size_t) (t0i + r) * dst->nb[1]); const float sd = Qd.osc * srow[r];
                const int8_t * d8 = zd + (size_t) (r - base) * D; char * cdp = &clipd[(size_t) r * D]; char c = 0;
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
        for (int pass = 1; pass <= maxpass && !rows.empty(); pass++, div *= DIV_GU) {
            if (pass > 1) passes2++;
            if (run_gu(rows, div)) { for (int r : rows) fb.push_back(t0i + r); failed = true; break; }
            std::vector<int> nxt; for (int r : rows) if (clip[r]) nxt.push_back(r); rows.swap(nxt);
            if (DIV_GU <= 1.0f) break;
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
        // down in Bd-Subkacheln (down-Package cappt im Compile bei ~80, gate/up laufen bei groesserem B):
        for (int base = 0; base < n; base += Bd) {
            std::vector<int> rows; for (int r : okrows) if (r >= base && r < base + Bd) rows.push_back(r);
            if (rows.empty()) continue;
            div = 1.0f; bool sfailed = false;
            for (int pass = 1; pass <= maxpass && !rows.empty(); pass++, div *= DIV) {
                if (pass > 1) passes2++;
                if (run_dn(rows, div, base)) { for (int r : rows) fb.push_back(t0i + r); sfailed = true; break; }
                std::vector<int> nxt; for (int r : rows) if (clip2[r]) nxt.push_back(r); rows.swap(nxt);
                if (DIV <= 1.0f) break;
            }
            if (sfailed) continue;
            for (int r : rows) fb.push_back(t0i + r);   // nach dem letzten Pass noch clippende Zeilen -> CPU-Fallback
        }
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
    if (ctx->log) GGML_LOG_WARN("barra: L%d T=%d fb=%d pass2=%d  %.2f ms [quant+glue %.2f infer %.2f (davon ausreisser %.2f ueberlappt) dequant %.2f] (kum: %ld Aufrufe, %ld Zeilen, fb %ld, pass2 %ld, tpu %.1f ms, fb %.1f ms, ol %.1f ms)\n", L, T, (int) fb.size(), passes2, now_ms() - t0, tq, ti, tol, td, ctx->calls, ctx->rows, ctx->fb_rows, ctx->passes2, ctx->ms_tpu, ctx->ms_fb, ctx->ms_ol);
}

// fused FFN fuer Layer L: h [D,T] f32 -> dst [D,T] f32
static void barra_ffn_fused(ggml_backend_barra_context * ctx, int L, const ggml_tensor * up_w, const ggml_tensor * gate_w, const ggml_tensor * down_w,
                            const ggml_tensor * h, ggml_tensor * dst) {
    const int D = ctx->D, B = ctx->B, T = (int) h->ne[1];
    const barra_layer_q & Q = ctx->q[L];
    const float qi = 1.0f / Q.isc; const bool i16 = ctx->bits == 16;
    const int qmax = i16 ? 32767 : 127, qmin = i16 ? -32768 : -128;
    const size_t bpe = i16 ? 2 : 1;
    std::vector<int> fb;
    double t0 = now_ms(), tq = 0, ti = 0, td = 0;
    std::vector<float> hb((size_t) T * D);   // Kopie von h (dst kann h aliasen!)
    for (int r = 0; r < T; r++) memcpy(&hb[(size_t) r * D], (const char *) h->data + (size_t) r * h->nb[1], D * sizeof(float));
    static int force_fb = -1; if (force_fb < 0) force_fb = getenv("BARRA_FORCE_FB") != nullptr;
    if (force_fb) { for (int r = 0; r < T; r++) fb.push_back(r); barra_ffn_fallback(ctx, up_w, gate_w, down_w, hb.data(), dst, fb); ctx->calls++; ctx->rows += T; return; }
    for (int t0i = 0; t0i < T; t0i += B) {
        const int n = std::min(B, T - t0i);
        double a0 = now_ms();
        barra_zc_cpu_begin(&ctx->zi);
        for (int r = 0; r < B; r++) {
            void * zrow = (char *) ctx->zi.map + (size_t) r * D * bpe;
            if (r >= n) { memset(zrow, 0, (size_t) D * bpe); continue; }
            const float * hr = &hb[(size_t) (t0i + r) * D];
            float mx = 0; for (int d = 0; d < D; d++) { float a = fabsf(hr[d]); if (a > mx) mx = a; }
            if (mx * qi + Q.izp > qmax || -mx * qi + Q.izp < qmin) { fb.push_back(t0i + r); memset(zrow, 0, (size_t) D * bpe); continue; }
            if (i16) { int16_t * z = (int16_t *) zrow; for (int d = 0; d < D; d++) { int v = (int) lrintf(hr[d] * qi) + Q.izp; z[d] = (int16_t) (v > qmax ? qmax : v < qmin ? qmin : v); } }
            else     { int8_t  * z = (int8_t  *) zrow; for (int d = 0; d < D; d++) { int v = (int) lrintf(hr[d] * qi) + Q.izp; z[d] = (int8_t)  (v > qmax ? qmax : v < qmin ? qmin : v); } }
        }
        barra_zc_cpu_end(&ctx->zi);
        double a1 = now_ms(); tq += a1 - a0;
        uint32_t us = 0;
        int rc = barra_tpu_infer(&ctx->tpu, (uint32_t) L, &ctx->zi, &ctx->zo, &us);
        double a2 = now_ms(); ti += a2 - a1;
        if (rc) {
            GGML_LOG_ERROR("barra: TPU-Inferenz Layer %d fehlgeschlagen -> CPU\n", L);
            for (int r = 0; r < n; r++) fb.push_back(t0i + r);
            continue;
        }
        barra_zc_cpu_begin(&ctx->zo);
        for (int r = 0; r < n; r++) {
            float * o = (float *) ((char *) dst->data + (size_t) (t0i + r) * dst->nb[1]);
            const void * zrow = (const char *) ctx->zo.map + (size_t) r * D * bpe;
            bool clip = false;
            if (i16) { const int16_t * z = (const int16_t *) zrow; for (int d = 0; d < D; d++) { if (z[d] >= qmax || z[d] <= qmin) clip = true; o[d] = (z[d] - Q.ozp) * Q.osc; } }
            else     { const int8_t  * z = (const int8_t  *) zrow; for (int d = 0; d < D; d++) { if (z[d] >= qmax || z[d] <= qmin) clip = true; o[d] = (z[d] - Q.ozp) * Q.osc; } }
            if (clip) fb.push_back(t0i + r);
        }
        barra_zc_cpu_end(&ctx->zo);
        td += now_ms() - a2;
    }
    ctx->ms_tpu += now_ms() - t0; ctx->calls++; ctx->rows += T;
    if (!fb.empty()) barra_ffn_fallback(ctx, up_w, gate_w, down_w, hb.data(), dst, fb);
    if (getenv("BARRA_CHECK") && ctx->cpu && ctx->calls <= 48) {   // Selbsttest: TPU-Ergebnis vs CPU-Referenz (alle Zeilen)
        std::vector<float> tpu((size_t) T * D); for (int r = 0; r < T; r++) memcpy(&tpu[(size_t) r * D], (char *) dst->data + (size_t) r * dst->nb[1], D * sizeof(float));
        std::vector<int> all(T); for (int r = 0; r < T; r++) all[r] = r;
        barra_ffn_fallback(ctx, up_w, gate_w, down_w, hb.data(), dst, all);
        double worst = 0, sumrel = 0; int worst_r = -1;
        for (int r = 0; r < T; r++) { const float * ref = (const float *) ((char *) dst->data + (size_t) r * dst->nb[1]); double e = 0, n = 0;
            for (int d = 0; d < D; d++) { double df = tpu[(size_t) r * D + d] - ref[d]; e += df * df; n += (double) ref[d] * ref[d]; }
            double rel = sqrt(e / (n + 1e-20)); sumrel += rel; if (rel > worst) { worst = rel; worst_r = r; } }
        GGML_LOG_WARN("barra: CHECK L%d T=%d rel-Fehler mittel %.4f, max %.4f (Zeile %d, fb=%s)\n", L, T, sumrel / T, worst, worst_r, std::find(fb.begin(), fb.end(), worst_r) != fb.end() ? "ja" : "nein");
    }
    if (ctx->log) GGML_LOG_WARN("barra: L%d T=%d fb=%d  %.2f ms [quant %.2f infer %.2f dequant %.2f fbk %.2f] (kum: %ld Aufrufe, %ld Zeilen, fb %ld, tpu %.1f ms, fb %.1f ms)\n", L, T, (int) fb.size(), now_ms() - t0, tq, ti, td, now_ms() - t0 - tq - ti - td, ctx->calls, ctx->rows, ctx->fb_rows, ctx->ms_tpu, ctx->ms_fb);
}

// ganzen Knoten (Nicht-Fusions-Fall) ueber das CPU-Backend rechnen
static void barra_node_on_cpu(ggml_backend_barra_context * ctx, ggml_tensor * node) {
    if (!ctx->cpu) { GGML_ABORT("barra: kein CPU-Fallback fuer %s\n", ggml_op_desc(node)); }
    if (ctx->log) GGML_LOG_WARN("barra: Knoten %s (%s) einzeln auf CPU\n", node->name, ggml_op_desc(node));
    // NUR diesen Knoten rechnen (ggml_build_forward_expand wuerde alle Vorgaenger bis zu den Graph-Eingaengen
    // mit hineinziehen und den gesamten Praefix erneut rechnen -> falsch/teuer)
    ggml_init_params ip = { ggml_graph_overhead_custom(1, false), nullptr, true };
    ggml_context * gc = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 1, false);
    ggml_graph_add_node(gf, node);
    ggml_backend_graph_compute(ctx->cpu, gf);
    ggml_free(gc);
}

static enum ggml_status ggml_backend_barra_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_barra_context * ctx = (ggml_backend_barra_context *) backend->context;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        switch (node->op) {
            case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: break;
            case GGML_OP_MUL_MAT: {
                int L = -1; int kind = barra_ffn_kind(node->src[0], &L);
                if (kind == 2) {   // down: Fusionspunkt
                    ggml_tensor * glu = node->src[1];
                    ggml_tensor * g = glu && glu->op == GGML_OP_GLU ? glu->src[0] : nullptr;
                    ggml_tensor * u = glu && glu->op == GGML_OP_GLU ? glu->src[1] : nullptr;
                    int Lg = -1, Lu = -1;
                    if (g && u && g->op == GGML_OP_MUL_MAT && u->op == GGML_OP_MUL_MAT && barra_ffn_kind(g->src[0], &Lg) == 1 && barra_ffn_kind(u->src[0], &Lu) == 0
                        && Lg == L && Lu == L && g->src[1] == u->src[1]) {
                        if (ctx->split == 3) barra_ffn_split(ctx, L, u->src[0], g->src[0], node->src[0], g->src[1], node);
                        else                 barra_ffn_fused(ctx, L, u->src[0], g->src[0], node->src[0], g->src[1], node);
                        break;
                    }
                    barra_node_on_cpu(ctx, node); break;
                }
                // up/gate: nur rechnen, wenn KEIN down in diesem Graph sie ueber ein GLU konsumiert
                bool consumed = false;
                for (int j = i + 1; j < cgraph->n_nodes && !consumed; j++) {
                    ggml_tensor * m = cgraph->nodes[j];
                    if (m->op == GGML_OP_MUL_MAT && barra_ffn_kind(m->src[0], nullptr) == 2 && m->src[1] && m->src[1]->op == GGML_OP_GLU && (m->src[1]->src[0] == node || m->src[1]->src[1] == node)) consumed = true;
                }
                if (!consumed) barra_node_on_cpu(ctx, node);
                break;
            }
            case GGML_OP_GLU: {
                bool consumed = false;
                for (int j = i + 1; j < cgraph->n_nodes && !consumed; j++) { ggml_tensor * m = cgraph->nodes[j]; if (m->op == GGML_OP_MUL_MAT && m->src[1] == node && barra_ffn_kind(m->src[0], nullptr) == 2) consumed = true; }
                if (!consumed) barra_node_on_cpu(ctx, node);
                break;
            }
            default: GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }
    return GGML_STATUS_SUCCESS;
}

// ---- Backend-Objekt
static const char * ggml_backend_barra_get_name(ggml_backend_t backend) { GGML_UNUSED(backend); return "BARRA"; }
static void ggml_backend_barra_free(ggml_backend_t backend) {
    ggml_backend_barra_context * ctx = (ggml_backend_barra_context *) backend->context;
    if (ctx->ready) {
        GGML_LOG_WARN("barra: Ende: %ld Aufrufe, %ld Zeilen, %ld Fallback-Zeilen, tpu %.1f ms, fb %.1f ms, ausreisser %.1f ms\n", ctx->calls, ctx->rows, ctx->fb_rows, ctx->ms_tpu, ctx->ms_fb, ctx->ms_ol);
        if (ctx->split == 1) { barra_zc_free(&ctx->zi); barra_zc_free(&ctx->zo); }
        else { barra_zc_free(&ctx->zi_d); barra_zc_free(&ctx->zo_g); barra_zc_free(&ctx->zo_u); barra_zc_free(&ctx->zi_ff); barra_zc_free(&ctx->zo_d); }
        barra_tpu_close(&ctx->tpu);
    }
    if (ctx->cpu) ggml_backend_free(ctx->cpu);
    delete ctx; delete backend;
}
static struct ggml_backend_i barra_backend_i = {
    /* .get_name                = */ ggml_backend_barra_get_name,
    /* .free                    = */ ggml_backend_barra_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_barra_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};
static ggml_guid_t ggml_backend_barra_guid(void) {
    static ggml_guid guid = { 0xba, 0x77, 0xa0, 0x01, 0x54, 0x50, 0x55, 0x00, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2e };
    return &guid;
}
static ggml_backend_barra_context * g_barra_ctx = nullptr;   // fuer supports_op (Device kennt den Kontext)
ggml_backend_t ggml_backend_barra_init(void) {
    ggml_backend_barra_context * ctx = new ggml_backend_barra_context;
    barra_load_meta(ctx);
    g_barra_ctx = ctx;
    ggml_backend_t backend = new ggml_backend { ggml_backend_barra_guid(), barra_backend_i, ggml_backend_reg_dev_get(ggml_backend_barra_reg(), 0), ctx };
    return backend;
}
bool ggml_backend_is_barra(ggml_backend_t backend) { return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_barra_guid()); }

// ---- Device
static const char * ggml_backend_barra_device_get_name(ggml_backend_dev_t dev) { GGML_UNUSED(dev); return "BARRA"; }
static const char * ggml_backend_barra_device_get_description(ggml_backend_dev_t dev) { GGML_UNUSED(dev); return "Tensor-G3-TPU FFN (barra/tpud)"; }
static void ggml_backend_barra_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) { GGML_UNUSED(dev); *free = 0; *total = 0; }
static enum ggml_backend_dev_type ggml_backend_barra_device_get_type(ggml_backend_dev_t dev) { GGML_UNUSED(dev); return GGML_BACKEND_DEVICE_TYPE_ACCEL; }
static void ggml_backend_barra_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_barra_device_get_name(dev); props->description = ggml_backend_barra_device_get_description(dev);
    props->type = ggml_backend_barra_device_get_type(dev); ggml_backend_barra_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { /* .async */ false, /* .host_buffer */ false, /* .buffer_from_host_ptr */ true, /* .events */ false, /* .mmap_support */ true };
}
static ggml_backend_t ggml_backend_barra_device_init_backend(ggml_backend_dev_t dev, const char * params) { GGML_UNUSED(dev); GGML_UNUSED(params); return ggml_backend_barra_init(); }
static ggml_backend_buffer_type_t ggml_backend_barra_device_get_buffer_type(ggml_backend_dev_t dev) { GGML_UNUSED(dev); return ggml_backend_cpu_buffer_type(); }
static ggml_backend_buffer_t ggml_backend_barra_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) { GGML_UNUSED(dev); GGML_UNUSED(max_tensor_size); return ggml_backend_cpu_buffer_from_ptr(ptr, size); }

static bool ggml_backend_barra_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    ggml_backend_barra_context * ctx = g_barra_ctx;
    // NUR die vier FFN-Ops annehmen. (BLAS sagt auch fuer VIEW/RESHAPE/... ja - das zieht hier Views/Reshapes in
    // eigene Splits mit Eingangs-Kopien und bremst den CPU-Pfad massiv; wir brauchen es nicht.)
    switch (op->op) {
        case GGML_OP_MUL_MAT: {
            if (!ctx || !ctx->ready) return false;
            const ggml_tensor * w = op->src[0]; const ggml_tensor * x = op->src[1];
            int L = -1; int kind = barra_ffn_kind(w, &L);
            if (kind < 0 || L < 0 || L >= ctx->NL) return false;
            const int64_t din = kind == 2 ? ctx->FF : ctx->D, dout = kind == 2 ? ctx->D : ctx->FF;
            bool ok = x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[2] == 1 && x->ne[3] == 1 && x->ne[1] >= ctx->min_batch
                      && w->ne[0] == din && w->ne[1] == dout && x->ne[0] == din;
            if (ctx->log) { static int dbg = 0; if (dbg < 12) { dbg++;
                GGML_LOG_WARN("barra: supports_op %s x[%ld,%ld,%ld,%ld] type=%d contig=%d w[%ld,%ld] -> %s\n", w->name, (long) x->ne[0], (long) x->ne[1], (long) x->ne[2], (long) x->ne[3], (int) x->type, (int) ggml_is_contiguous(x), (long) w->ne[0], (long) w->ne[1], ok ? "JA" : "nein"); } }
            return ok;
        }
        case GGML_OP_GLU: {
            if (!ctx || !ctx->ready) return false;
            if (ctx->log) { static int dbg2 = 0; if (dbg2 < 6) { dbg2++; GGML_LOG_WARN("barra: supports_op GLU op=%d src1=%p ne[%ld,%ld]\n", (int) ggml_get_glu_op(op), (void *) op->src[1], (long) op->ne[0], (long) op->ne[1]); } }
            if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU || !op->src[1]) return false;
            const ggml_tensor * g = op->src[0]; const ggml_tensor * u = op->src[1];
            int Lg = -1, Lu = -1;
            if (g->op != GGML_OP_MUL_MAT || u->op != GGML_OP_MUL_MAT) return false;
            if (barra_ffn_kind(g->src[0], &Lg) != 1 || barra_ffn_kind(u->src[0], &Lu) != 0 || Lg != Lu) return false;
            return op->ne[1] >= ctx->min_batch && op->ne[0] == ctx->FF;
        }
        default: return false;
    }
}
// Host-Buffer UND die CPU-Extra-Buffer-Typen (CPU_REPACK fuer Q8_0/Q4_0-Gewichte: is_host=NULL, aber Host-Speicher).
// Der Scheduler fragt supports_buft fuer den Buffer der GEWICHTE - wir lesen die Gewichte nie (nur Name/Shape),
// die Aktivierungen liegen in normalen CPU-Buffern.
static bool ggml_backend_barra_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    if (ggml_backend_buft_is_host(buft)) return true;
    ggml_backend_dev_t bd = ggml_backend_buft_get_device(buft);
    return bd && ggml_backend_dev_type(bd) == GGML_BACKEND_DEVICE_TYPE_CPU;
}
static const struct ggml_backend_device_i ggml_backend_barra_device_i = {
    /* .get_name             = */ ggml_backend_barra_device_get_name,
    /* .get_description      = */ ggml_backend_barra_device_get_description,
    /* .get_memory           = */ ggml_backend_barra_device_get_memory,
    /* .get_type             = */ ggml_backend_barra_device_get_type,
    /* .get_props            = */ ggml_backend_barra_device_get_props,
    /* .init_backend         = */ ggml_backend_barra_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_barra_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_barra_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_barra_device_supports_op,
    /* .supports_buft        = */ ggml_backend_barra_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};
// ---- Registry
static const char * ggml_backend_barra_reg_get_name(ggml_backend_reg_t reg) { GGML_UNUSED(reg); return "BARRA"; }
static size_t ggml_backend_barra_reg_get_device_count(ggml_backend_reg_t reg) { GGML_UNUSED(reg); return 1; }
static ggml_backend_dev_t ggml_backend_barra_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = { /* .iface */ ggml_backend_barra_device_i, /* .reg */ reg, /* .context */ nullptr };
    return &dev;
}
static void * ggml_backend_barra_get_proc_address(ggml_backend_reg_t reg, const char * name) { GGML_UNUSED(reg); GGML_UNUSED(name); return NULL; }
static const struct ggml_backend_reg_i ggml_backend_barra_reg_i = {
    /* .get_name         = */ ggml_backend_barra_reg_get_name,
    /* .get_device_count = */ ggml_backend_barra_reg_get_device_count,
    /* .get_device       = */ ggml_backend_barra_reg_get_device,
    /* .get_proc_address = */ ggml_backend_barra_get_proc_address,
};
ggml_backend_reg_t ggml_backend_barra_reg(void) {
    static struct ggml_backend_reg reg = { /* .api_version */ GGML_BACKEND_API_VERSION, /* .iface */ ggml_backend_barra_reg_i, /* .context */ NULL };
    return &reg;
}
GGML_BACKEND_DL_IMPL(ggml_backend_barra_reg)
