// ggml-gpud (M1): ggml-Backend ueber den gpud-zc-v3-Socket. Tensoren in dmabufs (barra_zbuf), Ops als
// Stufenliste (Shader-Handle + Bindings mit Offset + Push-Constants) in einem Roundtrip je Batch.
// Korrektheit vor Tempo: die Shader sind einfach (M2 bringt mmvq/GEMM/FA). Nicht unterstuetzte Ops laufen
// ohne Kopie auf der CPU (Buffer ist is_host: UMA, der dmabuf ist gemappt).
#include "ggml-gpud.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "barra.h"
#include "gpud_shaders.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <mutex>
#include <functional>
#include <chrono>
static inline double gpud_now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#define GPUD_ALIGN 64u                       // minStorageBufferOffsetAlignment der Mali (gpud-zc meldet 64)
#define GPUD_MAX_BUF (3u * 1024u * 1024u * 1024u)
#define GPUD_MAX_STAGE 4096
#define GPUD_MAX_PC 128

// ---------------------------------------------------------------- Shader-Tabelle (Push-Constant-Layouts wie in shaders/*.comp)
enum { SH_BINARY, SH_UNARY, SH_RMS, SH_CPY, SH_GET_ROWS, SH_MUL_MAT, SH_ROPE, SH_SOFTMAX, SH_GLU, SH_SET_ROWS, SH_GEMV_Q, SH_QUANT_Q81, SH_GEMV_Q4K_IQ, SH_FLASH, SH_N };
struct pc_binary   { uint32_t ne[4], nb0[4], nb1[4], nbd[4], ne1[4], offa, offb, offd, op, n; };
struct pc_unary    { uint32_t n, offa, offd, op; float s, b; };
struct pc_rms      { uint32_t ne00, nrows, nb01, nbd1, offa, offw, offd, fuse; float eps; };
struct pc_cpy      { uint32_t ne[4], nes[4], nbs[4], nbd[4], offa, offd, st, dt, n; };
struct pc_get_rows { uint32_t ne00, ne10, ne11, ne12, nb01, nb10, nb11, nb12, nbd1, nbd2, nbd3, off0, off1, offd, type, nrows; };
struct pc_mul_mat  { uint32_t ne00, ne01, ne11, ne12, nb01, nb02, nb03, nb11, nb12, nb13, nbd1, nbd2, nbd3, off0, off1, offd, type, r2, r3; };
struct pc_quant    { uint32_t ne00, ncols, nb32, offx, offq, nblk; };
struct pc_gemv_iq  { uint32_t ne00, ne01, ne11, ne12, nb01v4, nb02v4, nb03v4, nb11, nb12, nb13, nbd1, nbd2, nbd3, off0v4, offxq, offd, r2, r3, n256, xqcols; };
struct pc_flash    { uint32_t D, DV, M, nbq1, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3, nbm1, nbm2, nbd1, nbd2, nbd3, offq, offk, offv, offm, offd, qt, maskt, gqa, n_head_log2, nem2; float scale, softcap, m0, m1; };
struct pc_gemv     { uint32_t ne00, ne01, ne11, ne12, nb01, nb02, nb03, nb11, nb12, nb13, nbd1, nbd2, nbd3, off0, off1, offd, type, r2, r3, nblk; };
struct pc_rope     { uint32_t ne00, ne01, ne02, n_dims, nb01, nb02, nb03, nbd1, nbd2, nbd3, offa, offp, offf, offd, neox, has_ff, npairs; float theta_scale, freq_scale, attn_factor; };
struct pc_softmax  { uint32_t ne00, ne01, ne02, ne11, ne12, ne13, nb01, nb02, nb03, nbm1, nbm2, nbm3, nbd1, nbd2, nbd3, offa, offm, offd, mask, nrows; float scale; };
struct pc_glu      { uint32_t ne0, nrows, nb01, nb11, nbd1, offa, offb, offd, split, swapped, op, n; };
struct pc_set_rows { uint32_t ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nbd1, nbd2, nbd3, offa, offi, offd, i64, dt, nrows; };

struct shader_def { const uint8_t * spv; uint32_t len; uint32_t nbind; uint32_t pcsize; const char * name; };
static const shader_def g_shaders[SH_N] = {
    { gpud_spv_binary,   gpud_spv_binary_len,   3, sizeof(pc_binary),   "binary"   },
    { gpud_spv_unary,    gpud_spv_unary_len,    2, sizeof(pc_unary),    "unary"    },
    { gpud_spv_rms_norm, gpud_spv_rms_norm_len, 3, sizeof(pc_rms),      "rms_norm" },
    { gpud_spv_cpy,      gpud_spv_cpy_len,      4, sizeof(pc_cpy),      "cpy"      },
    { gpud_spv_get_rows, gpud_spv_get_rows_len, 3, sizeof(pc_get_rows), "get_rows" },
    { gpud_spv_mul_mat,  gpud_spv_mul_mat_len,  3, sizeof(pc_mul_mat),  "mul_mat"  },
    { gpud_spv_rope,     gpud_spv_rope_len,     4, sizeof(pc_rope),     "rope"     },
    { gpud_spv_soft_max, gpud_spv_soft_max_len, 4, sizeof(pc_softmax),  "soft_max" },
    { gpud_spv_glu,      gpud_spv_glu_len,      3, sizeof(pc_glu),      "glu"      },
    { gpud_spv_set_rows, gpud_spv_set_rows_len, 4, sizeof(pc_set_rows), "set_rows" },
    { gpud_spv_gemv_q,   gpud_spv_gemv_q_len,   4, sizeof(pc_gemv),     "gemv_q"   },
    { gpud_spv_quantize_q8_1, gpud_spv_quantize_q8_1_len, 2, sizeof(pc_quant),  "quant_q81" },
    { gpud_spv_gemv_q4k_iq,   gpud_spv_gemv_q4k_iq_len,   3, sizeof(pc_gemv_iq), "gemv_q4k_iq" },
    { gpud_spv_flash_attn,    gpud_spv_flash_attn_len,    7, sizeof(pc_flash),   "flash_attn" },
};

// ---------------------------------------------------------------- Session (eine je Prozess; gpud-zc serialisiert ohnehin)
struct gpud_session {
    barra_gpu3 g{};
    bool open = false;
    int sh[SH_N];
    std::mutex mtx;
    bool log = false;
    // Ein Batch mit ~7 s GPU-Zeit hat das Geraet HART aufgehaengt (30.8., kein pstore, Strom ziehen;
    // vermutlich Job-Timeout/GPU-Reset im Mali-Treiber). Deshalb wird der Batch nach VORHERGESAGTER
    // GPU-Zeit begrenzt (31.8.): je Workgroup wird eine gemessene ms-Schaetzung gefuehrt (est_wg, EMA:
    // sofort HOCH bei Ueberraschung, langsam runter), der Encoder flusht bei pred_ms >= budget_ms.
    // Workgroups skalieren mit der Arbeit (Prefill ~30x Decode je Stufe) -> Prefill drosselt a priori.
    // GGML_GPUD_MAXSTAGE=<n> schaltet auf das alte FIXE Verhalten (Reproduzierbarkeit alter Messungen).
    int maxstage = 512;              // harte Stufen-Obergrenze; im Fix-Modus die Batchgroesse
    bool fixed = false;              // GGML_GPUD_MAXSTAGE gesetzt -> kein Zeit-Budget
    double budget_ms = 100.0;        // GGML_GPUD_BUDGET_MS ueberschreibt (1..2000); 100 ms = der am
                                     // 30./31.8. vermessene sichere Decode-Betriebspunkt (MS=512, med 103 ms)
    double est_wg = 0.0;             // gemessene ms je Workgroup (0 = noch keine Messung -> Bootstrap-Cap 64)
    barra_zbuf scratch{}; bool scratch_ok = false; uint32_t scratch_sz = 0;   // x-q8_1 fuer int-dot-GEMV
    // Scratch ist GLOBAL, Backends gibt es MEHRERE (llama-bench/Server ueberlappen Kontexte beim
    // Wechsel). 31.8.: gpud_free der ALTEN Instanz hat den Scratch unter der neuen weggezogen ->
    // Re-Import mit fd=-1 -> EBADF -> Session tot. Deshalb Refcount: Scratch stirbt mit der letzten.
    int nbackends = 0;
    FILE * trace = nullptr;          // GGML_GPUD_TRACE=<datei>: je Batch Stufen + Ops, sofort geflusht
};
static gpud_session g_s;

static bool gpud_session_open() {
    std::lock_guard<std::mutex> lock(g_s.mtx);
    if (g_s.open) return true;
    g_s.log = getenv("GGML_GPUD_LOG") != nullptr;
    if (const char * e = getenv("GGML_GPUD_MAXSTAGE")) { int v = atoi(e); if (v >= 1 && v <= GPUD_MAX_STAGE) { g_s.maxstage = v; g_s.fixed = true; } }
    if (const char * e = getenv("GGML_GPUD_BUDGET_MS")) { double v = atof(e); if (v >= 1.0 && v <= 2000.0) g_s.budget_ms = v; }
    if (const char * e = getenv("GGML_GPUD_TRACE")) g_s.trace = fopen(e, "w");
    if (barra_gpu3_open(&g_s.g)) { GGML_LOG_ERROR("gpud: gpuzc.sock nicht erreichbar (gpud-zc v3 laeuft?)\n"); return false; }
    for (int i = 0; i < SH_N; i++) {
        g_s.sh[i] = barra_gpu3_load(&g_s.g, g_shaders[i].spv, g_shaders[i].len, g_shaders[i].nbind, g_shaders[i].pcsize);
        if (g_s.sh[i] < 0) { GGML_LOG_ERROR("gpud: Shader %s laedt nicht\n", g_shaders[i].name); barra_gpu3_close(&g_s.g); return false; }
    }
    barra_gpu3_set_autosync(&g_s.g, 0);   // ggml-gpud synct selbst (nur CPU<->GPU-Uebergang)
    if (getenv("GGML_GPUD_NOBAR")) { barra_gpu3_flags(&g_s.g, 1); GGML_LOG_INFO("gpud: NOBAR-Messmodus (Ergebnisse undefiniert)\n"); }
    g_s.open = true;
    if (g_s.log) GGML_LOG_INFO("gpud: Session offen, %d Shader geladen\n", SH_N);
    return true;
}

// Scratch-dmabuf fuer x-q8_1 (int-dot-GEMV): waechst nur; Aufrufer flusht VOR dem Wachsen.
static bool gpud_scratch_ensure(uint32_t bytes) {
    if (g_s.scratch_ok && g_s.scratch_sz >= bytes) return true;
    uint32_t sz = (bytes + 0xFFFFFu) & ~0xFFFFFu;
    std::lock_guard<std::mutex> lock(g_s.mtx);
    if (g_s.scratch_ok) { barra_zbuf * pz = &g_s.scratch; barra_gpu3_release(&g_s.g, &pz, 1); barra_zc_free(&g_s.scratch); g_s.scratch_ok = false; }
    if (barra_zc_alloc(&g_s.scratch, sz)) { GGML_LOG_ERROR("gpud: Scratch alloc %u B fehlgeschlagen\n", sz); return false; }
    barra_zbuf * pz = &g_s.scratch;
    if (barra_gpu3_import(&g_s.g, &pz, 1)) { GGML_LOG_ERROR("gpud: Scratch-Import fehlgeschlagen\n"); barra_zc_free(&g_s.scratch); return false; }
    g_s.scratch_ok = true; g_s.scratch_sz = sz;
    if (g_s.log) GGML_LOG_INFO("gpud: Scratch %u KB\n", sz >> 10);
    return true;
}

// ---------------------------------------------------------------- Buffer (dmabuf)
struct gpud_buffer_ctx { barra_zbuf z; int state = 0; };   // 0 = CPU darf zugreifen, 1 = Geraet besitzt (nach cpu_end)

static const char * gpud_buft_get_name(ggml_backend_buffer_type_t) { return "GPUD"; }

// Puffer in CPU- bzw. GPU-Besitz bringen (CPU-Cache-Pflege nur beim echten Uebergang, nicht je Batch).
static void gpud_to_cpu(gpud_buffer_ctx * c) { if (c->state == 1) { barra_zc_cpu_begin(&c->z); c->state = 0; } }
static void gpud_to_gpu(gpud_buffer_ctx * c) { if (c->state == 0) { barra_zc_cpu_end(&c->z);  c->state = 1; } }
static void gpud_buffer_free(ggml_backend_buffer_t buffer) {
    gpud_buffer_ctx * c = (gpud_buffer_ctx *) buffer->context;
    { std::lock_guard<std::mutex> lock(g_s.mtx); barra_zbuf * p = &c->z; if (g_s.open) barra_gpu3_release(&g_s.g, &p, 1); }
    barra_zc_free(&c->z);
    delete c;
}
static void * gpud_buffer_get_base(ggml_backend_buffer_t buffer) { return ((gpud_buffer_ctx *) buffer->context)->z.map; }
static void gpud_buffer_memset_tensor(ggml_backend_buffer_t b, ggml_tensor * t, uint8_t v, size_t off, size_t size) { gpud_to_cpu((gpud_buffer_ctx *) b->context); memset((char *) t->data + off, v, size); }
static void gpud_buffer_set_tensor(ggml_backend_buffer_t b, ggml_tensor * t, const void * data, size_t off, size_t size) { gpud_to_cpu((gpud_buffer_ctx *) b->context); memcpy((char *) t->data + off, data, size); }
static void gpud_buffer_get_tensor(ggml_backend_buffer_t b, const ggml_tensor * t, void * data, size_t off, size_t size) { gpud_to_cpu((gpud_buffer_ctx *) b->context); memcpy(data, (const char *) t->data + off, size); }
static bool gpud_buffer_cpy_tensor(ggml_backend_buffer_t, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) { gpud_to_cpu((gpud_buffer_ctx *) dst->buffer->context); memcpy(dst->data, src->data, ggml_nbytes(src)); return true; }
    return false;
}
static void gpud_buffer_clear(ggml_backend_buffer_t buffer, uint8_t v) { gpud_buffer_ctx * c = (gpud_buffer_ctx *) buffer->context; gpud_to_cpu(c); memset(c->z.map, v, c->z.size); }

static const ggml_backend_buffer_i gpud_buffer_i = {
    /* .free_buffer   = */ gpud_buffer_free,
    /* .get_base      = */ gpud_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ gpud_buffer_memset_tensor,
    /* .set_tensor    = */ gpud_buffer_set_tensor,
    /* .get_tensor    = */ gpud_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ gpud_buffer_cpy_tensor,
    /* .clear         = */ gpud_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t gpud_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (!gpud_session_open()) return nullptr;
    if (size == 0) size = GPUD_ALIGN;
    if (size > GPUD_MAX_BUF) { GGML_LOG_ERROR("gpud: Puffer %zu B > %u B\n", size, GPUD_MAX_BUF); return nullptr; }
    size_t sz = (size + 4095) & ~(size_t) 4095;
    gpud_buffer_ctx * c = new gpud_buffer_ctx();
    if (barra_zc_alloc(&c->z, (uint32_t) sz)) { GGML_LOG_ERROR("gpud: dmabuf %zu B fehlgeschlagen\n", sz); delete c; return nullptr; }
    { std::lock_guard<std::mutex> lock(g_s.mtx); barra_zbuf * p = &c->z; if (barra_gpu3_import(&g_s.g, &p, 1)) { GGML_LOG_ERROR("gpud: Import fehlgeschlagen\n"); barra_zc_free(&c->z); delete c; return nullptr; } }
    if (g_s.log) GGML_LOG_INFO("gpud: Puffer %zu MB (fd %d, gpu_h %d)\n", sz >> 20, c->z.fd, c->z.gpu_h);
    return ggml_backend_buffer_init(buft, gpud_buffer_i, c, sz);
}
static size_t gpud_buft_get_alignment(ggml_backend_buffer_type_t) { return GPUD_ALIGN; }
static size_t gpud_buft_get_max_size(ggml_backend_buffer_type_t) { return GPUD_MAX_BUF; }
static bool   gpud_buft_is_host(ggml_backend_buffer_type_t) { return false; }   // false: Scheduler haelt GPUD-Ops zusammen (is_host=true zersplitterte den Graphen, 30.8.)

ggml_backend_buffer_type_t ggml_backend_gpud_buffer_type(void) {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ gpud_buft_get_name,
            /* .alloc_buffer   = */ gpud_buft_alloc_buffer,
            /* .get_alignment  = */ gpud_buft_get_alignment,
            /* .get_max_size   = */ gpud_buft_get_max_size,
            /* .get_alloc_size = */ nullptr,
            /* .is_host        = */ gpud_buft_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &buft;
}

// ---------------------------------------------------------------- Encoder
struct gpud_encoder {
    std::vector<barra_gpu3_stage> st;
    std::vector<barra_gpu3_bind>  bd;      // reserviert, Zeiger bleiben gueltig
    std::vector<uint8_t>          pcs;     // GPUD_MAX_PC je Stufe
    size_t nst = 0, nbd = 0;
    uint64_t wgs = 0;                      // Workgroups im Batch (fuer die est_wg-Messung)
    double pred_ms = 0;                    // vorhergesagte GPU-Zeit des Batches (wgs * est_wg)
    gpud_encoder() { st.resize(GPUD_MAX_STAGE); bd.resize(GPUD_MAX_STAGE * 4); pcs.resize((size_t) GPUD_MAX_STAGE * GPUD_MAX_PC); }
    void reset() { nst = 0; nbd = 0; wgs = 0; pred_ms = 0; opnames.clear(); }
    bool full(int nbind) const {
        if ((int) nst >= g_s.maxstage || nbd + nbind > bd.size()) return true;
        if (g_s.fixed) return false;
        if (g_s.est_wg <= 0) return (int) nst >= 64;            // Bootstrap: klein anfangen bis zur ersten Messung
        return nst > 0 && pred_ms >= g_s.budget_ms;             // Zeit-Budget (>=1 Stufe je Batch garantiert)
    }
    std::vector<const char *> opnames;   // je Stufe (nur Trace)
    // Binding fuer einen Tensor: Basis 64-aligned, Rest als Byte-Offset zurueck
    static bool bind(const ggml_tensor * t, barra_gpu3_bind & b, uint32_t & rem) {
        if (!t || !t->buffer || t->buffer->buft != ggml_backend_gpud_buffer_type()) return false;
        gpud_buffer_ctx * c = (gpud_buffer_ctx *) t->buffer->context;
        uint64_t off = (uint64_t) ((const char *) t->data - (const char *) c->z.map);
        if (off >= c->z.size) return false;
        b.buf = &c->z; b.off = off & ~(uint64_t) (GPUD_ALIGN - 1); b.range = 0; rem = (uint32_t) (off & (GPUD_ALIGN - 1));
        return true;
    }
    // Stufe anlegen: binds[0..nbind-1] (Tensoren, nullptr = wie binds[0]), pc kopieren
    bool push(int sh, uint32_t gx, uint32_t gy, uint32_t gz, const ggml_tensor * const * ts, int nbind, const void * pc, uint32_t pcsize, uint32_t * rems) {
        if (full(nbind)) return false;
        barra_gpu3_bind * b = &bd[nbd];
        for (int i = 0; i < nbind; i++) {
            const ggml_tensor * t = ts[i] ? ts[i] : ts[0];
            if (!bind(t, b[i], rems[i])) return false;
        }
        // Bereichspruefung: Tensor muss komplett im dmabuf liegen (sonst GPU-Seitenfehler -> Geraet haengt)
        for (int i = 0; i < nbind; i++) {
            const ggml_tensor * t = ts[i] ? ts[i] : ts[0];
            gpud_buffer_ctx * c = (gpud_buffer_ctx *) t->buffer->context;
            uint64_t end = (uint64_t) ((const char *) t->data - (const char *) c->z.map) + ggml_nbytes(t);
            if (end > c->z.size) { GGML_LOG_ERROR("gpud: Tensor %s ragt aus dem Puffer (%llu > %u)" "\n", t->name, (unsigned long long) end, c->z.size); return false; }
        }
        uint8_t * p = &pcs[nst * GPUD_MAX_PC]; memcpy(p, pc, pcsize);
        st[nst] = barra_gpu3_stage{ g_s.sh[sh], gx, gy, gz, b, nbind, p, pcsize };
        nst++; nbd += nbind; opnames.push_back(g_shaders[sh].name);
        uint64_t wg = (uint64_t) gx * gy * gz; wgs += wg; pred_ms += wg * g_s.est_wg;
        return true;
    }
    bool push_raw(int sh, uint32_t gx, uint32_t gy, uint32_t gz, const barra_gpu3_bind * src, int nbind, const void * pc, uint32_t pcsize) {
        if (full(nbind)) return false;
        barra_gpu3_bind * bb = &bd[nbd];
        for (int i = 0; i < nbind; i++) { if (!src[i].buf || (src[i].off % GPUD_ALIGN)) return false; bb[i] = src[i]; }
        uint8_t * pp = &pcs[nst * GPUD_MAX_PC]; memcpy(pp, pc, pcsize);
        st[nst] = barra_gpu3_stage{ g_s.sh[sh], gx, gy, gz, bb, nbind, pp, pcsize };
        nst++; nbd += nbind; opnames.push_back(g_shaders[sh].name);
        uint64_t wg = (uint64_t) gx * gy * gz; wgs += wg; pred_ms += wg * g_s.est_wg;
        return true;
    }
};

static uint32_t sh_type(ggml_type t) {
    switch (t) { case GGML_TYPE_F32: return 0; case GGML_TYPE_F16: return 1; case GGML_TYPE_Q4_K: return 2; case GGML_TYPE_Q6_K: return 3; case GGML_TYPE_Q8_0: return 4; default: return 0xFFFFFFFFu; }
}
static bool weight_type_ok(ggml_type t) { return sh_type(t) != 0xFFFFFFFFu; }
static inline uint32_t E(size_t nb, size_t es) { return (uint32_t) (nb / es); }   // Stride in Elementen
static inline uint32_t GRP(uint64_t n, uint32_t w) { return (uint32_t) ((n + w - 1) / w); }
// Vulkan/gpud-zc: hoechstens 65535 Workgroups je Achse -> Gruppen auf x/y verteilen (Shader rechnen linear zurueck)
static inline void grid2(uint64_t n, uint32_t & gx, uint32_t & gy) { if (n <= 65535) { gx = (uint32_t) n; gy = 1; } else { gx = 32768; gy = (uint32_t) ((n + 32767) / 32768); } }

// Der Batch braucht die Byte-Reste der Bindings erst NACH bind(): deshalb PC in zwei Schritten (Rest-Felder nachtragen)
#define GPUD_PUSH3(sh, gx, gy, gz, ts, nb, pc, fix)  do { uint32_t rems_[4] = {0,0,0,0}; \
    if (enc.full(nb)) { if (!flush()) return false; } \
    if (!enc.push(sh, gx, gy, gz, ts, nb, &pc, sizeof(pc), rems_)) return false; \
    { auto * P = (decltype(&pc)) &enc.pcs[(enc.nst - 1) * GPUD_MAX_PC]; (void) P; fix; } } while (0)
#define GPUD_PUSH(sh, ngroups, ts, nb, pc, fix) do { uint32_t gx_, gy_; grid2(ngroups, gx_, gy_); GPUD_PUSH3(sh, gx_, gy_, 1, ts, nb, pc, fix); } while (0)

// ---------------------------------------------------------------- Backend
struct gpud_backend_ctx {
    gpud_encoder enc;
    ggml_backend_t cpu = nullptr;   // Rueckfall fuer nicht kodierbare Knoten (in-place, is_host)
    int n_gpu = 0, n_cpu = 0, n_batches = 0, n_graphs = 0;
    double ms_batch = 0, ms_cpu = 0, ms_graph = 0;
};

static bool gpud_supports_op(const ggml_tensor * op);

// Liste der Ops, die keinen Kernel brauchen
static bool op_is_noop(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || ggml_is_empty(t);
}

static bool encode_node(gpud_backend_ctx * ctx, ggml_cgraph * cg, int i, int & skip, const std::function<bool()> & flush);

static void node_on_cpu(gpud_backend_ctx * ctx, ggml_tensor * node) {
    if (!ctx->cpu) ctx->cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    static std::vector<uint8_t> buf;
    size_t need = ggml_graph_overhead_custom(2, false) + 256;
    if (buf.size() < need) buf.resize(need);
    ggml_init_params ip = { buf.size(), buf.data(), true };
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 2, false);
    gf->nodes[0] = node; gf->n_nodes = 1;
    double t0 = gpud_now_ms();
    ggml_backend_graph_compute(ctx->cpu, gf);
    ctx->ms_cpu += gpud_now_ms() - t0;
    ggml_free(c);
    ctx->n_cpu++;
    if (g_s.trace) { fprintf(g_s.trace, "t=%.1f cpu: %s %s [%lld,%lld,%lld,%lld] %.2f ms" "\n", gpud_now_ms(), ggml_op_name(node->op), node->name, (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3], gpud_now_ms() - t0); }
}

static enum ggml_status gpud_graph_compute(ggml_backend_t backend, ggml_cgraph * cg) {
    gpud_backend_ctx * ctx = (gpud_backend_ctx *) backend->context;
    gpud_encoder & enc = ctx->enc;
    enc.reset();
    if (getenv("GGML_GPUD_GC")) fprintf(stderr, "GC nodes=%d\n", cg->n_nodes);
    double tg0 = gpud_now_ms();
    {   // Alle benutzten gpud-Puffer EINMAL flushen (CPU->GPU); Gewichte bleiben danach im GPU-Besitz.
        std::vector<gpud_buffer_ctx *> seen;
        auto touch = [&](const ggml_tensor * t) {
            if (!t || !t->buffer || t->buffer->buft != ggml_backend_gpud_buffer_type()) return;
            gpud_buffer_ctx * c = (gpud_buffer_ctx *) t->buffer->context;
            if (c->state == 1) return;
            for (auto * x : seen) if (x == c) return;
            seen.push_back(c);
        };
        for (int i = 0; i < cg->n_nodes; i++) { ggml_tensor * n = cg->nodes[i]; touch(n); for (int j = 0; j < GGML_MAX_SRC; j++) touch(n->src[j]); }
        for (auto * c : seen) gpud_to_gpu(c);
    }
    auto flush = [&]() -> bool {
        if (enc.nst == 0) return true;
        if (g_s.trace) { fprintf(g_s.trace, "t=%.1f batch #%d: %zu Stufen:", gpud_now_ms(), ctx->n_batches + 1, enc.nst); for (size_t k = 0; k < enc.nst; k++) fprintf(g_s.trace, " %s", enc.opnames[k]); fprintf(g_s.trace, "\n"); fflush(g_s.trace); }
        int r; double t0 = gpud_now_ms();
        { std::lock_guard<std::mutex> lock(g_s.mtx); r = barra_gpu3_batch(&g_s.g, enc.st.data(), (int) enc.nst); }
        double dt = gpud_now_ms() - t0;
        ctx->ms_batch += dt;
        ctx->n_batches++;
        // est_wg nachfuehren: bei Ueberraschung nach oben SOFORT uebernehmen (Sicherheit gegen den
        // Geraete-Hang), nach unten nur langsam (10 %/Batch) — Ausreisser druecken das Tempo kurz,
        // reissen aber nie das Budget.
        if (!g_s.fixed && r == 0 && enc.wgs > 0) {
            double per = dt / (double) enc.wgs;
            if (g_s.est_wg <= 0 || per > g_s.est_wg) g_s.est_wg = per;
            else g_s.est_wg = 0.9 * g_s.est_wg + 0.1 * per;
        }
        if (g_s.trace) { fprintf(g_s.trace, "batch #%d fertig r=%d %.2f ms (pred %.1f, wgs %llu, est_wg %.4g)" "\n", ctx->n_batches, r, dt, enc.pred_ms, (unsigned long long) enc.wgs, g_s.est_wg); fflush(g_s.trace); }
        if (r) { fprintf(stderr, "gpud: Batch mit %zu Stufen fehlgeschlagen (%d)\n", enc.nst, r); enc.reset(); return false; }   // fprintf: llama-bench schluckt GGML_LOG
        enc.reset();
        return true;
    };
    static int fl_full = 0, fl_calls = 0;
    (void) fl_full;
    for (int i = 0; i < cg->n_nodes; i++) {
        ggml_tensor * node = cg->nodes[i];
        if (op_is_noop(node)) continue;
        int skip = 0;
        if (gpud_supports_op(node) && encode_node(ctx, cg, i, skip, flush)) { ctx->n_gpu++; i += skip; continue; }
        // Rueckfall: erst die GPU-Arbeit abschliessen (Reihenfolge!), dann CPU in-place
        if (!flush()) return GGML_STATUS_FAILED;
        node_on_cpu(ctx, node);
    }
    if (!flush()) return GGML_STATUS_FAILED;
    ctx->ms_graph += gpud_now_ms() - tg0; ctx->n_graphs++;
    if (g_s.log && (ctx->n_graphs % 8 == 1)) GGML_LOG_INFO("gpud: Graph #%d (%d Knoten): gesamt %d GPU-Knoten / %d CPU-Knoten / %d Batches; Zeit %.0f ms Graphen = %.0f ms Batches + %.0f ms CPU-Rueckfall" "\n", ctx->n_graphs, cg->n_nodes, ctx->n_gpu, ctx->n_cpu, ctx->n_batches, ctx->ms_graph, ctx->ms_batch, ctx->ms_cpu);
    return GGML_STATUS_SUCCESS;
}

// ---- Kodierung der einzelnen Ops
static bool encode_node(gpud_backend_ctx * ctx, ggml_cgraph * cg, int i, int & skip, const std::function<bool()> & flush) {
    gpud_encoder & enc = ctx->enc;
    ggml_tensor * t = cg->nodes[i];
    const ggml_tensor * s0 = t->src[0], * s1 = t->src[1], * s2 = t->src[2];
    switch (t->op) {
    case GGML_OP_ADD: case GGML_OP_MUL: case GGML_OP_SUB: case GGML_OP_DIV: {
        pc_binary pc{};
        for (int k = 0; k < 4; k++) { pc.ne[k] = (uint32_t) t->ne[k]; pc.nb0[k] = E(s0->nb[k], 4); pc.nb1[k] = E(s1->nb[k], 4); pc.nbd[k] = E(t->nb[k], 4); pc.ne1[k] = (uint32_t) s1->ne[k]; }
        pc.op = t->op == GGML_OP_ADD ? 0 : t->op == GGML_OP_MUL ? 1 : t->op == GGML_OP_SUB ? 2 : 3;
        pc.n = (uint32_t) ggml_nelements(t);
        const ggml_tensor * ts[3] = { s0, s1, t };
        GPUD_PUSH(SH_BINARY, GRP(pc.n, 256), ts, 3, pc, { P->offa = rems_[0] / 4; P->offb = rems_[1] / 4; P->offd = rems_[2] / 4; });
        return true;
    }
    case GGML_OP_SCALE: case GGML_OP_UNARY: {
        pc_unary pc{}; pc.n = (uint32_t) ggml_nelements(t);
        if (t->op == GGML_OP_SCALE) { pc.op = 0; pc.s = ggml_get_op_params_f32(t, 0); pc.b = ggml_get_op_params_f32(t, 1); }
        else switch (ggml_get_unary_op(t)) {
            case GGML_UNARY_OP_SILU: pc.op = 1; break; case GGML_UNARY_OP_GELU: pc.op = 2; break; case GGML_UNARY_OP_RELU: pc.op = 3; break;
            case GGML_UNARY_OP_NEG: pc.op = 4; break; case GGML_UNARY_OP_GELU_QUICK: pc.op = 5; break; case GGML_UNARY_OP_SIGMOID: pc.op = 6; break;
            default: return false; }
        const ggml_tensor * ts[2] = { s0, t };
        GPUD_PUSH(SH_UNARY, GRP(pc.n, 256), ts, 2, pc, { P->offa = rems_[0] / 4; P->offd = rems_[1] / 4; });
        return true;
    }
    case GGML_OP_RMS_NORM: {
        pc_rms pc{}; pc.ne00 = (uint32_t) t->ne[0]; pc.nrows = (uint32_t) ggml_nrows(t); pc.nb01 = E(s0->nb[1], 4); pc.nbd1 = E(t->nb[1], 4);
        pc.eps = ggml_get_op_params_f32(t, 0);
        const ggml_tensor * w = nullptr; ggml_tensor * out = t;
        // Fusion mit dem folgenden MUL(norm, weight[ne00]) - nur wenn dieser Knoten sonst niemanden speist
        if (i + 1 < cg->n_nodes) {
            ggml_tensor * nx = cg->nodes[i + 1];
            if (nx->op == GGML_OP_MUL && nx->src[0] == t && nx->src[1]->type == GGML_TYPE_F32 && nx->src[1]->ne[0] == t->ne[0] && ggml_nelements(nx->src[1]) == t->ne[0]
                && ggml_node_get_use_count(cg, i) == 1 && ggml_is_contiguous(nx) && nx->buffer && nx->buffer->buft == ggml_backend_gpud_buffer_type()) {
                w = nx->src[1]; out = nx; pc.fuse = 1; pc.nbd1 = E(nx->nb[1], 4); skip = 1;
            }
        }
        const ggml_tensor * ts[3] = { s0, w, out };
        GPUD_PUSH(SH_RMS, pc.nrows, ts, 3, pc, { P->offa = rems_[0] / 4; P->offw = rems_[1] / 4; P->offd = rems_[2] / 4; });
        return true;
    }
    case GGML_OP_CPY: case GGML_OP_CONT: case GGML_OP_DUP: {
        pc_cpy pc{}; size_t es = ggml_type_size(s0->type), ed = ggml_type_size(t->type);
        for (int k = 0; k < 4; k++) { pc.ne[k] = (uint32_t) t->ne[k]; pc.nes[k] = (uint32_t) s0->ne[k]; pc.nbs[k] = E(s0->nb[k], es); pc.nbd[k] = E(t->nb[k], ed); }
        pc.st = s0->type == GGML_TYPE_F16; pc.dt = t->type == GGML_TYPE_F16; pc.n = (uint32_t) ggml_nelements(t);
        const ggml_tensor * ts[4] = { s0, s0, t, t };
        GPUD_PUSH(SH_CPY, GRP(pc.n, 256), ts, 4, pc, { P->offa = rems_[0] / (uint32_t) es; P->offd = rems_[2] / (uint32_t) ed; });
        return true;
    }
    case GGML_OP_GET_ROWS: {
        pc_get_rows pc{}; pc.ne00 = (uint32_t) s0->ne[0]; pc.ne10 = (uint32_t) s1->ne[0]; pc.ne11 = (uint32_t) s1->ne[1]; pc.ne12 = (uint32_t) s1->ne[2];
        pc.nb01 = (uint32_t) s0->nb[1]; pc.nb10 = E(s1->nb[0], 4); pc.nb11 = E(s1->nb[1], 4); pc.nb12 = E(s1->nb[2], 4);
        pc.nbd1 = E(t->nb[1], 4); pc.nbd2 = E(t->nb[2], 4); pc.nbd3 = E(t->nb[3], 4); pc.type = sh_type(s0->type);
        pc.nrows = pc.ne10 * pc.ne11 * pc.ne12;
        const ggml_tensor * ts[3] = { s0, s1, t };
        GPUD_PUSH(SH_GET_ROWS, pc.nrows, ts, 3, pc, { P->off0 = rems_[0]; P->off1 = rems_[1] / 4; P->offd = rems_[2] / 4; });
        return true;
    }
    case GGML_OP_MUL_MAT: {
        // M2b: int-dot-GEMV fuer q4_K (dotPacked4x8EXT + Subgroup-Cluster) - x wird zuerst nach q8_1 quantisiert.
        if (s0->type == GGML_TYPE_Q4_K && s0->ne[0] % 256 == 0 && ggml_is_contiguous(s1) && !getenv("GGML_GPUD_NOIQ")) {
            uint32_t ncols = (uint32_t) (s1->ne[1] * s1->ne[2] * s1->ne[3]);
            uint32_t nb32 = (uint32_t) s1->ne[0] / 32;
            uint32_t need = nb32 * ncols * 36u;                       // q8_1: 36 B je 32er-Block
            if (need > g_s.scratch_sz) { if (g_s.log) GGML_LOG_INFO("gpud: flush wg scratch-grow (nst=%zu, need=%u > %u)\n", ctx->enc.nst, need, g_s.scratch_sz); if (!flush()) return false; if (!gpud_scratch_ensure(need)) return false; }
            barra_zbuf * sc = &g_s.scratch;
            // Stufe 1: quantisieren (x -> scratch)
            {
                pc_quant pc{}; pc.ne00 = (uint32_t) s1->ne[0]; pc.ncols = ncols; pc.nb32 = nb32; pc.nblk = nb32 * ncols;
                uint32_t remx = 0; barra_gpu3_bind bx;
                if (!gpud_encoder::bind(s1, bx, remx)) return false;
                pc.offx = remx / 4;
                barra_gpu3_bind bq{ sc, 0, 0 };
                barra_gpu3_bind binds[2] = { bx, bq };
                if (enc.full(2)) { if (g_s.log) GGML_LOG_INFO("gpud: flush wg full(2) nst=%zu maxstage=%d\n", enc.nst, g_s.maxstage); if (!flush()) return false; }
                uint32_t gx, gy; grid2(pc.nblk, gx, gy);
                if (!enc.push_raw(SH_QUANT_Q81, gx, gy, 1, binds, 2, &pc, sizeof(pc))) return false;
            }
            // Stufe 2: int-dot-GEMV (W, scratch, dst)
            {
                pc_gemv_iq pc{}; pc.ne00 = (uint32_t) s0->ne[0]; pc.ne01 = (uint32_t) s0->ne[1]; pc.ne11 = (uint32_t) s1->ne[1]; pc.ne12 = (uint32_t) s1->ne[2];
                pc.nb01v4 = (uint32_t) (s0->nb[1] / 16); pc.nb02v4 = (uint32_t) (s0->nb[2] / 16); pc.nb03v4 = (uint32_t) (s0->nb[3] / 16);
                pc.nb11 = E(s1->nb[1], 4); pc.nb12 = E(s1->nb[2], 4); pc.nb13 = E(s1->nb[3], 4);
                pc.nbd1 = E(t->nb[1], 4); pc.nbd2 = E(t->nb[2], 4); pc.nbd3 = E(t->nb[3], 4);
                pc.r2 = (uint32_t) (s1->ne[2] / s0->ne[2]); pc.r3 = (uint32_t) (s1->ne[3] / s0->ne[3]);
                pc.n256 = pc.ne00 / 256; pc.xqcols = nb32;
                uint32_t remw = 0, remd = 0; barra_gpu3_bind bw, bd;
                if (!gpud_encoder::bind(s0, bw, remw) || !gpud_encoder::bind(t, bd, remd)) return false;
                pc.off0v4 = remw / 16; pc.offxq = 0; pc.offd = remd / 4;
                barra_gpu3_bind bq{ sc, 0, 0 };
                barra_gpu3_bind binds[3] = { bw, bq, bd };
                if (enc.full(3)) { if (g_s.log) GGML_LOG_INFO("gpud: flush wg full(3) nst=%zu maxstage=%d\n", enc.nst, g_s.maxstage); if (!flush()) return false; }
                uint32_t gx, gy; grid2((pc.ne01 + 7) / 8, gx, gy);   // 8 Zeilen je Workgroup
                if (!enc.push_raw(SH_GEMV_Q4K_IQ, gx, pc.ne11, (uint32_t) (s1->ne[2] * s1->ne[3]), binds, 3, &pc, sizeof(pc))) return false;
            }
            return true;
        }
        // M2: schneller GEMV fuer q4_K/q6_K (32-Bit-Lasten, vec4-Dot); Rest ueber den generischen Kernel
        if ((s0->type == GGML_TYPE_Q4_K || s0->type == GGML_TYPE_Q6_K) && s0->ne[0] % 256 == 0 && !getenv("GGML_GPUD_NOGEMV")) {
            pc_gemv pc{}; pc.ne00 = (uint32_t) s0->ne[0]; pc.ne01 = (uint32_t) s0->ne[1]; pc.ne11 = (uint32_t) s1->ne[1]; pc.ne12 = (uint32_t) s1->ne[2];
            pc.nb01 = (uint32_t) s0->nb[1]; pc.nb02 = (uint32_t) s0->nb[2]; pc.nb03 = (uint32_t) s0->nb[3];
            pc.nb11 = E(s1->nb[1], 4); pc.nb12 = E(s1->nb[2], 4); pc.nb13 = E(s1->nb[3], 4);
            pc.nbd1 = E(t->nb[1], 4); pc.nbd2 = E(t->nb[2], 4); pc.nbd3 = E(t->nb[3], 4); pc.type = sh_type(s0->type);
            pc.r2 = (uint32_t) (s1->ne[2] / s0->ne[2]); pc.r3 = (uint32_t) (s1->ne[3] / s0->ne[3]); pc.nblk = pc.ne00 / 256;
            const ggml_tensor * ts[4] = { s0, s1, s1, t };
            GPUD_PUSH3(SH_GEMV_Q, GRP(pc.ne01, 8), pc.ne11, (uint32_t) (s1->ne[2] * s1->ne[3]), ts, 4, pc, { P->off0 = rems_[0]; P->off1 = rems_[1] / 4; P->offd = rems_[3] / 4; });
            return true;
        }
        pc_mul_mat pc{}; pc.ne00 = (uint32_t) s0->ne[0]; pc.ne01 = (uint32_t) s0->ne[1]; pc.ne11 = (uint32_t) s1->ne[1]; pc.ne12 = (uint32_t) s1->ne[2];
        pc.nb01 = (uint32_t) s0->nb[1]; pc.nb02 = (uint32_t) s0->nb[2]; pc.nb03 = (uint32_t) s0->nb[3];
        pc.nb11 = E(s1->nb[1], 4); pc.nb12 = E(s1->nb[2], 4); pc.nb13 = E(s1->nb[3], 4);
        pc.nbd1 = E(t->nb[1], 4); pc.nbd2 = E(t->nb[2], 4); pc.nbd3 = E(t->nb[3], 4); pc.type = sh_type(s0->type);
        pc.r2 = (uint32_t) (s1->ne[2] / s0->ne[2]); pc.r3 = (uint32_t) (s1->ne[3] / s0->ne[3]);
        const ggml_tensor * ts[3] = { s0, s1, t };
        GPUD_PUSH3(SH_MUL_MAT, GRP(pc.ne01, 8), pc.ne11, (uint32_t) (s1->ne[2] * s1->ne[3]), ts, 3, pc, { P->off0 = rems_[0]; P->off1 = rems_[1] / 4; P->offd = rems_[2] / 4; });
        return true;
    }
    case GGML_OP_ROPE: {
        pc_rope pc{}; const int n_dims = ggml_get_op_params_i32(t, 1), mode = ggml_get_op_params_i32(t, 2);   // [0] = n_past (alt)
        const float freq_base = ggml_get_op_params_f32(t, 5), freq_scale = ggml_get_op_params_f32(t, 6), attn_factor = ggml_get_op_params_f32(t, 8);
        pc.ne00 = (uint32_t) s0->ne[0]; pc.ne01 = (uint32_t) s0->ne[1]; pc.ne02 = (uint32_t) s0->ne[2]; pc.n_dims = (uint32_t) n_dims;
        pc.nb01 = E(s0->nb[1], 4); pc.nb02 = E(s0->nb[2], 4); pc.nb03 = E(s0->nb[3], 4); pc.nbd1 = E(t->nb[1], 4); pc.nbd2 = E(t->nb[2], 4); pc.nbd3 = E(t->nb[3], 4);
        pc.neox = (mode & GGML_ROPE_TYPE_NEOX) ? 1 : 0; pc.has_ff = s2 ? 1 : 0;
        pc.npairs = (uint32_t) (ggml_nelements(s0) / 2);
        pc.theta_scale = powf(freq_base, -2.0f / n_dims); pc.freq_scale = freq_scale; pc.attn_factor = attn_factor;
        const ggml_tensor * ts[4] = { s0, s1, s2 ? s2 : s1, t };
        GPUD_PUSH(SH_ROPE, GRP(pc.npairs, 256), ts, 4, pc, { P->offa = rems_[0] / 4; P->offp = rems_[1] / 4; P->offf = rems_[2] / 4; P->offd = rems_[3] / 4; });
        return true;
    }
    case GGML_OP_SOFT_MAX: {
        pc_softmax pc{}; pc.ne00 = (uint32_t) s0->ne[0]; pc.ne01 = (uint32_t) s0->ne[1]; pc.ne02 = (uint32_t) s0->ne[2];
        pc.nb01 = E(s0->nb[1], 4); pc.nb02 = E(s0->nb[2], 4); pc.nb03 = E(s0->nb[3], 4); pc.nbd1 = E(t->nb[1], 4); pc.nbd2 = E(t->nb[2], 4); pc.nbd3 = E(t->nb[3], 4);
        pc.scale = ggml_get_op_params_f32(t, 0);
        pc.ne11 = pc.ne12 = pc.ne13 = 1; pc.mask = 0;
        if (s1) { size_t em = ggml_type_size(s1->type); pc.mask = s1->type == GGML_TYPE_F16 ? 1 : 2; pc.ne11 = (uint32_t) s1->ne[1]; pc.ne12 = (uint32_t) s1->ne[2]; pc.ne13 = (uint32_t) s1->ne[3]; pc.nbm1 = E(s1->nb[1], em); pc.nbm2 = E(s1->nb[2], em); pc.nbm3 = E(s1->nb[3], em); }
        const ggml_tensor * m = s1 ? s1 : s0;
        const ggml_tensor * ts[4] = { s0, m, m, t };
        pc.nrows = (uint32_t) ggml_nrows(s0);
        GPUD_PUSH(SH_SOFTMAX, pc.nrows, ts, 4, pc, { P->offa = rems_[0] / 4; P->offm = s1 ? rems_[1] / (uint32_t) ggml_type_size(s1->type) : 0; P->offd = rems_[3] / 4; });
        return true;
    }
    case GGML_OP_GLU: {
        pc_glu pc{}; pc.ne0 = (uint32_t) t->ne[0]; pc.nrows = (uint32_t) ggml_nrows(t); pc.nb01 = E(s0->nb[1], 4); pc.nbd1 = E(t->nb[1], 4);
        pc.split = s1 ? 1 : 0; pc.nb11 = s1 ? E(s1->nb[1], 4) : 0; pc.swapped = ggml_get_op_params_i32(t, 1) ? 1 : 0;
        switch (ggml_get_glu_op(t)) { case GGML_GLU_OP_REGLU: pc.op = 0; break; case GGML_GLU_OP_GEGLU: pc.op = 1; break; case GGML_GLU_OP_SWIGLU: pc.op = 2; break; default: return false; }
        pc.n = pc.ne0 * pc.nrows;
        const ggml_tensor * ts[3] = { s0, s1 ? s1 : s0, t };
        GPUD_PUSH(SH_GLU, GRP(pc.n, 256), ts, 3, pc, { P->offa = rems_[0] / 4; P->offb = rems_[1] / 4; P->offd = rems_[2] / 4; });
        return true;
    }
    case GGML_OP_SET_ROWS: {
        pc_set_rows pc{}; size_t ei = ggml_type_size(s1->type), ed = ggml_type_size(t->type);
        pc.ne00 = (uint32_t) s0->ne[0]; pc.ne01 = (uint32_t) s0->ne[1]; pc.ne02 = (uint32_t) s0->ne[2]; pc.ne03 = (uint32_t) s0->ne[3];
        pc.nb01 = E(s0->nb[1], 4); pc.nb02 = E(s0->nb[2], 4); pc.nb03 = E(s0->nb[3], 4);
        pc.ne11 = (uint32_t) s1->ne[1]; pc.ne12 = (uint32_t) s1->ne[2]; pc.nb11 = E(s1->nb[1], ei); pc.nb12 = E(s1->nb[2], ei);
        pc.nbd1 = E(t->nb[1], ed); pc.nbd2 = E(t->nb[2], ed); pc.nbd3 = E(t->nb[3], ed);
        pc.i64 = s1->type == GGML_TYPE_I64; pc.dt = t->type == GGML_TYPE_F16;
        pc.nrows = (uint32_t) ggml_nrows(s0);
        const ggml_tensor * ts[4] = { s0, s1, t, t };
        GPUD_PUSH(SH_SET_ROWS, pc.nrows, ts, 4, pc, { P->offa = rems_[0] / 4; P->offi = rems_[1] / (uint32_t) ei; P->offd = rems_[2] / (uint32_t) ed; });
        return true;
    }
    case GGML_OP_FLASH_ATTN_EXT: {
        const ggml_tensor * q = t->src[0], * k = t->src[1], * v = t->src[2], * msk = t->src[3];
        pc_flash pc{}; size_t esq = ggml_type_size(q->type), esm = msk ? ggml_type_size(msk->type) : 4;
        pc.D = (uint32_t) q->ne[0]; pc.DV = (uint32_t) v->ne[1]; pc.M = (uint32_t) k->ne[1];
        pc.nbq1 = E(q->nb[1], esq); pc.nbq2 = E(q->nb[2], esq); pc.nbq3 = E(q->nb[3], esq);
        pc.nbk1 = E(k->nb[1], 2); pc.nbk2 = E(k->nb[2], 2); pc.nbk3 = E(k->nb[3], 2);
        pc.nbv1 = E(v->nb[1], 2); pc.nbv2 = E(v->nb[2], 2); pc.nbv3 = E(v->nb[3], 2);
        pc.nbm1 = msk ? E(msk->nb[1], esm) : 0; pc.nbm2 = msk ? E(msk->nb[2], esm) : 0;
        pc.nbd1 = E(t->nb[1], 4); pc.nbd2 = E(t->nb[2], 4); pc.nbd3 = E(t->nb[3], 4);
        pc.qt = q->type == GGML_TYPE_F16 ? 1 : 0;
        pc.maskt = msk ? (msk->type == GGML_TYPE_F16 ? 1 : 2) : 0;
        pc.gqa = (uint32_t) (q->ne[2] / k->ne[2]);
        pc.nem2 = msk ? (uint32_t) msk->ne[2] : 1; if (pc.nem2 == 0) pc.nem2 = 1;
        pc.scale = ggml_get_op_params_f32(t, 0); pc.softcap = ggml_get_op_params_f32(t, 2);
        float max_bias = ggml_get_op_params_f32(t, 1);
        if (max_bias > 0.0f) { uint32_t nh = (uint32_t) q->ne[2]; uint32_t nhl = 1; while (nhl * 2 <= nh) nhl *= 2; pc.n_head_log2 = nhl;
            pc.m0 = powf(2.0f, -(max_bias) / (float) nhl); pc.m1 = powf(2.0f, -(max_bias / 2.0f) / (float) nhl); }
        else pc.n_head_log2 = 0;
        const ggml_tensor * mm = msk ? msk : q;
        const ggml_tensor * ts[7] = { q, q, k, v, mm, mm, t };
        GPUD_PUSH3(SH_FLASH, (uint32_t) q->ne[1], (uint32_t) q->ne[2], (uint32_t) q->ne[3], ts, 7, pc,
            { P->offq = rems_[0] / (uint32_t) esq; P->offk = rems_[2] / 2; P->offv = rems_[3] / 2; P->offm = msk ? rems_[4] / (uint32_t) esm : 0; P->offd = rems_[6] / 4; });
        return true;
    }
    default: return false;
    }
}

// ---- supports_op: nur was die Shader wirklich koennen (der Rest laeuft ohne Kopie auf der CPU)
static bool gpud_supports_op(const ggml_tensor * t) {
    const ggml_tensor * s0 = t->src[0], * s1 = t->src[1], * s2 = t->src[2];
    auto f32 = [](const ggml_tensor * x) { return x && x->type == GGML_TYPE_F32 && (x->nb[0] == 4); };
    switch (t->op) {
    case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_RESHAPE: case GGML_OP_TRANSPOSE: return true;
    case GGML_OP_ADD: case GGML_OP_MUL: case GGML_OP_SUB: case GGML_OP_DIV:
        return f32(s0) && f32(s1) && f32(t) && ggml_can_repeat(s1, s0);
    case GGML_OP_SCALE: return f32(s0) && f32(t) && ggml_is_contiguous(s0) && ggml_is_contiguous(t);
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(t)) { case GGML_UNARY_OP_SILU: case GGML_UNARY_OP_GELU: case GGML_UNARY_OP_RELU: case GGML_UNARY_OP_NEG: case GGML_UNARY_OP_GELU_QUICK: case GGML_UNARY_OP_SIGMOID: break; default: return false; }
        return f32(s0) && f32(t) && ggml_is_contiguous(s0) && ggml_is_contiguous(t);
    case GGML_OP_RMS_NORM: return f32(s0) && f32(t) && ggml_is_contiguous_rows(s0) && ggml_is_contiguous(s0) && ggml_is_contiguous(t);
    case GGML_OP_CPY: case GGML_OP_CONT: case GGML_OP_DUP:
        return s0 && (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16) && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && ggml_nelements(s0) == ggml_nelements(t);
    case GGML_OP_GET_ROWS:
        return s0 && weight_type_ok(s0->type) && s1 && s1->type == GGML_TYPE_I32 && t->type == GGML_TYPE_F32 && s0->ne[2] == 1 && s0->ne[3] == 1 && s0->nb[0] == ggml_type_size(s0->type);
    case GGML_OP_MUL_MAT:
        return s0 && weight_type_ok(s0->type) && s0->nb[0] == ggml_type_size(s0->type) && (s0->ne[0] % ggml_blck_size(s0->type) == 0) && f32(s1) && f32(t)
            && s1->ne[2] % s0->ne[2] == 0 && s1->ne[3] % s0->ne[3] == 0
            && (s0->ne[1] + 7) / 8 <= 65535 && s1->ne[1] <= 65535 && s1->ne[2] * s1->ne[3] <= 65535;
    case GGML_OP_ROPE: {
        const int mode = ggml_get_op_params_i32(t, 2); const float ext = ggml_get_op_params_f32(t, 7);
        bool ok = f32(s0) && f32(t) && s1 && s1->type == GGML_TYPE_I32 && (mode == 0 || mode == GGML_ROPE_TYPE_NEOX) && ext == 0.0f && (!s2 || f32(s2)) && (ggml_get_op_params_i32(t, 1) % 2 == 0);
        if (!ok && getenv("GGML_GPUD_LOG")) GGML_LOG_INFO("gpud: ROPE abgelehnt: s0=%d t=%d s1=%p/%d mode=%d ext=%g s2=%p ndims=%d\n", (int) f32(s0), (int) f32(t), (void *) s1, s1 ? (int) s1->type : -1, mode, ext, (void *) s2, ggml_get_op_params_i32(t, 1));
        return ok;
    }
    case GGML_OP_SOFT_MAX:
        return f32(s0) && f32(t) && (!s1 || s1->type == GGML_TYPE_F16 || s1->type == GGML_TYPE_F32) && !s2 && ggml_get_op_params_f32(t, 1) == 0.0f;
    case GGML_OP_GLU:
        switch (ggml_get_glu_op(t)) { case GGML_GLU_OP_REGLU: case GGML_GLU_OP_GEGLU: case GGML_GLU_OP_SWIGLU: break; default: return false; }
        return f32(s0) && f32(t) && (!s1 || f32(s1)) && ggml_is_contiguous_rows(s0) && ggml_is_contiguous(t);
    case GGML_OP_SET_ROWS:
        return f32(s0) && s1 && (s1->type == GGML_TYPE_I64 || s1->type == GGML_TYPE_I32) && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && ggml_is_contiguous_rows(s0);
    case GGML_OP_FLASH_ATTN_EXT: {
        // Kernel vorhanden (shaders/flash_attn.comp), aber Messung 30.8.: LANGSAMER als der unfused Pfad
        // (-fa 0: mul_mat+softmax+mul_mat) UND noch numerisch falsch -> standardmaessig AUS. GGML_GPUD_FA=1
        // aktiviert ihn wieder (experimentell). Betriebspunkt bleibt -fa 0.
        if (!getenv("GGML_GPUD_FA")) return false;
        const ggml_tensor * q = t->src[0], * k = t->src[1], * v = t->src[2], * msk = t->src[3];
        if (t->src[4]) return false;   // attention sinks: nicht unterstuetzt
        bool ok = q && k && v && t->type == GGML_TYPE_F32
            && (q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16)
            && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16
            && (!msk || msk->type == GGML_TYPE_F16 || msk->type == GGML_TYPE_F32)
            && q->ne[0] <= 256 && v->ne[1] <= 256 && k->ne[2] != 0 && (q->ne[2] % k->ne[2] == 0);
        return ok;
    }
    default: return false;
    }
}

// ---------------------------------------------------------------- Backend-Objekt
static const char * gpud_get_name(ggml_backend_t) { return "GPUD"; }
static void gpud_free(ggml_backend_t backend) {
    gpud_backend_ctx * ctx = (gpud_backend_ctx *) backend->context;
    if (g_s.log) GGML_LOG_INFO("gpud: %d Knoten GPU, %d Knoten CPU-Rueckfall, %d Batches\n", ctx->n_gpu, ctx->n_cpu, ctx->n_batches);
    if (ctx->cpu) ggml_backend_free(ctx->cpu);
    { std::lock_guard<std::mutex> lock(g_s.mtx);
      if (--g_s.nbackends <= 0 && g_s.scratch_ok) { barra_zbuf * pz = &g_s.scratch; barra_gpu3_release(&g_s.g, &pz, 1); barra_zc_free(&g_s.scratch); g_s.scratch_ok = false; g_s.scratch_sz = 0; } }
    delete ctx; delete backend;
}
static void gpud_synchronize(ggml_backend_t) {}   // Batch ist synchron
static const ggml_backend_i gpud_backend_i = {
    /* .get_name           = */ gpud_get_name,
    /* .free               = */ gpud_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .set_tensor_2d_async= */ nullptr,
    /* .get_tensor_2d_async= */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ gpud_synchronize,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ gpud_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};
static ggml_guid_t gpud_guid(void) { static ggml_guid g = { 0x9d, 0x0b, 0x47, 0x50, 0x55, 0x44, 0x33, 0x2e, 0xba, 0x77, 0xa1, 0x30, 0x08, 0x26, 0x01, 0x02 }; return &g; }

ggml_backend_t ggml_backend_gpud_init(void) {
    if (!gpud_session_open()) return nullptr;
    { std::lock_guard<std::mutex> lock(g_s.mtx); g_s.nbackends++; }
    ggml_backend_t b = new ggml_backend { gpud_guid(), gpud_backend_i, ggml_backend_reg_dev_get(ggml_backend_gpud_reg(), 0), new gpud_backend_ctx() };
    return b;
}
bool ggml_backend_is_gpud(ggml_backend_t backend) { return backend && ggml_guid_matches(backend->guid, gpud_guid()); }

// ---------------------------------------------------------------- Device / Reg
static const char * gpud_dev_get_name(ggml_backend_dev_t) { return "GPUD"; }
static const char * gpud_dev_get_description(ggml_backend_dev_t) { return "Mali-G715 via gpud-zc (dmabuf zero-copy)"; }
static void gpud_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *total = (size_t) 4 << 30; *free = (size_t) 3 << 30; }
static enum ggml_backend_dev_type gpud_dev_get_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void gpud_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * p) {
    p->name = gpud_dev_get_name(dev); p->description = gpud_dev_get_description(dev); p->type = gpud_dev_get_type(dev);
    gpud_dev_get_memory(dev, &p->memory_free, &p->memory_total);
    p->caps = { /* async */ false, /* host_buffer */ false, /* buffer_from_host_ptr */ false, /* events */ false };
}
static ggml_backend_t gpud_dev_init_backend(ggml_backend_dev_t, const char *) { return ggml_backend_gpud_init(); }
static ggml_backend_buffer_type_t gpud_dev_get_buffer_type(ggml_backend_dev_t) { return ggml_backend_gpud_buffer_type(); }
static bool gpud_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    bool r = gpud_supports_op(op);
    if (!r && getenv("GGML_GPUD_LOGREJECT")) { const ggml_tensor* a=op->src[0]; fprintf(stderr, "REJ %s dt=%s s0=%s cont0=%d same=%d ne=%ld,%ld,%ld,%ld\n", ggml_op_name(op->op), ggml_type_name(op->type), a?ggml_type_name(a->type):"-", a?(int)ggml_is_contiguous(a):-1, a?(int)ggml_are_same_shape(a,op):-1, (long)op->ne[0],(long)op->ne[1],(long)op->ne[2],(long)op->ne[3]); }
    if (false) {
        const ggml_tensor * s0 = op->src[0], * s1 = op->src[1];
        GGML_LOG_INFO("gpud: ABLEHNT %s  dst=%d[%ld,%ld,%ld,%ld] s0=%s s1=%s\n", ggml_op_name(op->op), (int)op->type,
            (long)op->ne[0],(long)op->ne[1],(long)op->ne[2],(long)op->ne[3],
            s0?ggml_type_name(s0->type):"-", s1?ggml_type_name(s1->type):"-");
    }
    return r;
}
static bool gpud_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == ggml_backend_gpud_buffer_type(); }
static const ggml_backend_device_i gpud_device_i = {
    /* .get_name             = */ gpud_dev_get_name,
    /* .get_description      = */ gpud_dev_get_description,
    /* .get_memory           = */ gpud_dev_get_memory,
    /* .get_type             = */ gpud_dev_get_type,
    /* .get_props            = */ gpud_dev_get_props,
    /* .init_backend         = */ gpud_dev_init_backend,
    /* .get_buffer_type      = */ gpud_dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ gpud_dev_supports_op,
    /* .supports_buft        = */ gpud_dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};
static const char * gpud_reg_get_name(ggml_backend_reg_t) { return "GPUD"; }
static size_t gpud_reg_get_device_count(ggml_backend_reg_t) { return 1; }
static ggml_backend_dev_t gpud_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = { /* .iface */ gpud_device_i, /* .reg */ reg, /* .context */ nullptr };
    return &dev;
}
static void * gpud_reg_get_proc_address(ggml_backend_reg_t, const char *) { return nullptr; }
static const ggml_backend_reg_i gpud_reg_i = {
    /* .get_name         = */ gpud_reg_get_name,
    /* .get_device_count = */ gpud_reg_get_device_count,
    /* .get_device       = */ gpud_reg_get_device,
    /* .get_proc_address = */ gpud_reg_get_proc_address,
};
ggml_backend_reg_t ggml_backend_gpud_reg(void) {
    static ggml_backend_reg reg = { /* .api_version */ GGML_BACKEND_API_VERSION, /* .iface */ gpud_reg_i, /* .context */ nullptr };
    ggml_backend_gpud_buffer_type()->device = gpud_reg_get_device(&reg, 0);
    return &reg;
}
GGML_BACKEND_DL_IMPL(ggml_backend_gpud_reg)
