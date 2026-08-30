// ggml-vulkan-barra.inc — ggml-barra v2.1: Attention-Block-Offload im Prefill (TPU-Projektionen als
// GETRENNTE q/k/v/o-Packages mit ZEILEN-NORMIERTER Quantisierung + eigene GPU-Kernel), eingebunden am
// Ende von ggml-vulkan.cpp (nach ggml_backend_vk_graph_compute_impl).
//
// Pro Layer (T >= BARRA_ATTN_MIN_BATCH, n_past==0): h_norm f32 (Vulkan-UMA-Puffer) -> zeilennormiert int8
// -> TPU q/k/v-Packages -> GPU rope_qk_l2 (RMSNorm+RoPE, int8->f16; Norm ist skaleninvariant, srow exakt)
// -> qkt_f16 -> softmax_causal -> av_rows (f16 x int8-V*srow -> ctx f16) -> CPU-Zeilenquant -> TPU o-Package
// -> f32*srow_ctx in den Wo-dst. KV-Handoff: k f16 aus bQK, v dequant aus bV, per ids in die Cache-Zeilen.
//
// Env: BARRA_ATTN_META (Pflicht: attn2.meta v2; sonst passiv), BARRA_ATTN_DIR (spv + aux_attn.bin,
// Default=Verzeichnis der Meta), BARRA_SOCK_DIR (libbarra/tpud, 4 Packages je Layer: q=4L k=4L+1 v=4L+2
// o=4L+3), BARRA_ATTN_MIN_BATCH (32), BARRA_ATTN_LAYERS ("0-35"/"0,2,7"), BARRA_ATTN_LOG=1,
// BARRA_ATTN_THREADS (4, CPU-Glue), BARRA_ATTN_OFF=1 (Killschalter).
extern "C" {
#include "barra.h"
}
#include <set>
#include <map>
#include <chrono>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>

static double now_ms_battn() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct battn_qgrp { float isc, osc; int izp, ozp; };
struct battn_layerq { battn_qgrp q, q2, k, v, o, g, g2; };   // q2 nur bei gechunktem q (meta v4);
                                                              // g/g2 nur bei gegateter Attention (qwen3.5)
struct battn_kern { VkPipeline pipe; VkPipelineLayout pll; VkDescriptorSetLayout dsl; VkDescriptorSet ds; };
struct battn_state {
    int env = -1;
    bool init_done = false, ready = false;
    int D=0,H=0,NKV=0,HD=0,FF=0,NL=0,S=0,min_batch=32,log=0,nthreads=4,bits=8;
    bool qchunk=false;                                 // meta v4: q in 2x2048-Packages
    bool kvmerge=false;                                // meta v5: k+v EIN Package (Zeile [k|v], 4 Pkg/Layer wg. Firmware-Graphlimit ~157)
    bool glm=false;                                    // meta v6: GLM-Edge (kein q/k-Norm, NORM-RoPE, q ungechunkt, 3 Pkg/Layer)
    bool gen=false;                                    // meta v7: generisch (Geometrie/RoPE aus Meta, Spec-Constant-Kernel, Bias in Packages)
    int rmode=2, bias=0, qknorm=0;                     // v7-Flags (rmode 0=NORM/2=NEOX)
    bool gemma=false; int HDS=0;                       // meta v8 (Gemma-4): 2 Layer-Typen (full HD / swa HDS), KV-Sharing, V-Norm
    std::vector<uint8_t> ltype;                        // je Layer: 0=swa+kv 1=full+kv 2=swa-shared 3=full-shared
    std::vector<uint32_t> midq, midkv, mido;           // Praefixsummen-Modell-IDs (variable Paketzahl je Layer)
    // Paket-IDs sind POSITIONEN in pkglist.txt. Die feste Arithmetik (4*L o.ae.) stimmt nur,
    // wenn jede Schicht Pakete hat. Hybride Modelle (qwen3.5: 8 Attention-Schichten von 33)
    // brechen das. Ist pkglist.txt lesbar, wird daraus eine Tabelle gebaut; sonst bleibt es
    // bei der Arithmetik (Altkits). (29.8.2026)
    std::vector<uint32_t> pq, pq2, pg, pg2, pkv, pv, po;
    bool haspkgmap = false;
    barra_zbuf bKs{},bVs{},bKf{},bVf{};                // persistente f16-Typ-Puffer (KV-Sharing)
    VkBuffer vbKs{},vbVs{},vbKf{},vbVf{};
    VkDeviceMemory vmKs{},vmVs{},vmKf{},vmVf{};
    battn_kern krp8s{},krp8f{},kqk8s{},kqk8f{},kav8s{},kav8f{};
    std::vector<battn_layerq> q;
    std::vector<uint8_t> layer_on;
    barra_tpu tpu{};
    barra_zbuf bX{},bQ{},bQ2{},bK{},bV{},bQK{},bSc{},bAoF{},bAo{},bO{},bAux{},bSrow{};
    barra_zbuf bG{},bG2{};   // Gate-Projektion (qwen3.5): sigmoid(gate) wird NACH der
                             // Attention und VOR der o-Projektion multipliziert.
    bool hasgate = false;
    VkInstance inst{}; VkPhysicalDevice phys{}; VkDevice dev{}; VkQueue queue{}; uint32_t qfam=0;
    VkCommandPool pool{}; VkDescriptorPool dp{}; VkFence fen{}; VkCommandBuffer cb{};
    battn_kern krp{},kqk{},ksm{},kav{};
    VkBuffer vbQ{},vbQ2{},vbK{},vbV{},vbQK{},vbSc{},vbAoF{},vbAux{},vbSrow{};
    VkDeviceMemory vmQ{},vmQ2{},vmK{},vmV{},vmQK{},vmSc{},vmAoF{},vmAux{},vmSrow{};
    PFN_vkGetMemoryFdPropertiesKHR pGetMemFdProps{};
    std::vector<float> srow_c;                     // Zeilenskala ctx (CPU-seitig)
    long calls=0, layers=0, fb=0; double ms_total=0, ms_tpu=0, ms_gpu=0, ms_glue=0;
};
static battn_state g_battn;

static bool ggml_vk_battn_env_on(void) {
    battn_state & st = g_battn;
    if (st.env < 0) st.env = ((getenv("BARRA_ATTN_META") && !getenv("BARRA_ATTN_OFF")) || getenv("BARRA_ATTN_CALDUMP")) ? 1 : 0;
    return st.env == 1;
}

#define BALOG(...) GGML_LOG_WARN(__VA_ARGS__)

// ---------- Persistenter Threadpool fuer den CPU-Glue (Zeilen-Quant/-Dequant) ----------
// Frueher wurde je Aufruf ein std::thread-Satz gespawnt (~ms je Start) — im Prefill
// hunderte Male. Jetzt persistente Worker (BARRA_ATTN_THREADS), Job = fn(r) fuer r in
// [0,n), dynamische Vergabe per Atomic-Zaehler; Worker spinnen kurz, dann cond-wait.
// (Gespiegelt von barra_pool in ggml-barra.cpp; eigene Instanz, da eigene TU.)
struct battn_pool {
    int nt = 1; std::vector<std::thread> th; std::mutex mu; std::condition_variable cv;
    std::atomic<int> gen{0}, next{0}, done{0}; int n = 0; const std::function<void(int)> * fn = nullptr; bool stop = false;
    void work() { for (;;) { int r = next.fetch_add(1); if (r >= n) break; (*fn)(r); } }
    void worker() { int seen = 0; for (;;) {
            int spins = 0; while (gen.load(std::memory_order_acquire) == seen && !stop) { if (++spins < 20000) continue; std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return gen.load() != seen || stop; }); }
            if (stop) return; seen = gen.load(); work(); done.fetch_add(1); } }
    explicit battn_pool(int nthreads) { nt = nthreads < 1 ? 1 : nthreads; for (int i = 1; i < nt; i++) th.emplace_back([this] { worker(); }); }
    ~battn_pool() { { std::lock_guard<std::mutex> lk(mu); stop = true; } cv.notify_all(); for (auto & t : th) if (t.joinable()) t.join(); }
    void run(int n_, const std::function<void(int)> & f) {
        if (nt == 1 || n_ < 2) { for (int r = 0; r < n_; r++) f(r); return; }
        n = n_; fn = &f; next.store(0); done.store(0);
        { std::lock_guard<std::mutex> lk(mu); gen.fetch_add(1, std::memory_order_release); } cv.notify_all();
        work(); while (done.load(std::memory_order_acquire) < nt - 1) std::this_thread::yield();
    }
};
static battn_pool & battn_pool_get() { static battn_pool p(g_battn.nthreads); return p; }
static void battn_par(int n, const std::function<void(int)> & fn) {
    if (g_battn.nthreads <= 1 || n < 4) { for (int i = 0; i < n; i++) fn(i); return; }
    battn_pool_get().run(n, fn);
}

// ---------- eigenes Vulkan-Geraet (dmabuf-Import via barra_zc) ----------
static uint32_t battn_pick_mem(uint32_t tb, uint32_t want) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g_battn.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((tb & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}
static bool battn_gpu_init(void) {
    battn_state & st = g_battn;
    VkApplicationInfo app = {}; app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici = {}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, 0, &st.inst) != VK_SUCCESS) return false;
    uint32_t n = 1; vkEnumeratePhysicalDevices(st.inst, &n, &st.phys); if (!n) return false;
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(st.phys, &qn, 0);
    std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(st.phys, &qn, qf.data());
    st.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < qn; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { st.qfam = i; break; }
    if (st.qfam == UINT32_MAX) return false;
    const char * exts[] = { VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; qci.queueFamilyIndex = st.qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceShaderFloat16Int8Features f16f = {}; f16f.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES; f16f.shaderFloat16 = VK_TRUE;
    VkPhysicalDevice16BitStorageFeatures s16 = {}; s16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES; s16.pNext = &f16f; s16.storageBuffer16BitAccess = VK_TRUE;
    VkDeviceCreateInfo dci = {}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.pNext = &s16;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.enabledExtensionCount = 3; dci.ppEnabledExtensionNames = exts;
    if (vkCreateDevice(st.phys, &dci, 0, &st.dev) != VK_SUCCESS) return false;
    vkGetDeviceQueue(st.dev, st.qfam, 0, &st.queue);
    st.pGetMemFdProps = (PFN_vkGetMemoryFdPropertiesKHR) vkGetDeviceProcAddr(st.dev, "vkGetMemoryFdPropertiesKHR");
    if (!st.pGetMemFdProps) return false;
    VkCommandPoolCreateInfo pci = {}; pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; pci.queueFamilyIndex = st.qfam; pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(st.dev, &pci, 0, &st.pool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cbai = {}; cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cbai.commandPool = st.pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(st.dev, &cbai, &st.cb) != VK_SUCCESS) return false;
    VkFenceCreateInfo fci = {}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(st.dev, &fci, 0, &st.fen) != VK_SUCCESS) return false;
    return true;
}
static bool battn_import(int fd, uint32_t size, VkBuffer * b, VkDeviceMemory * m) {
    battn_state & st = g_battn;
    VkMemoryFdPropertiesKHR fp = {}; fp.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
    if (st.pGetMemFdProps(st.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp) != VK_SUCCESS) return false;
    uint32_t mt = battn_pick_mem(fp.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) mt = battn_pick_mem(fp.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (mt == UINT32_MAX) return false;
    VkExternalMemoryBufferCreateInfo ext = {}; ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO; ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bci = {}; bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bci.pNext = &ext; bci.size = size; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(st.dev, &bci, 0, b) != VK_SUCCESS) return false;
    int dfd = dup(fd); if (dfd < 0) return false;
    VkImportMemoryFdInfoKHR imp = {}; imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR; imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT; imp.fd = dfd;
    VkMemoryAllocateInfo mai = {}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.pNext = &imp; mai.allocationSize = size; mai.memoryTypeIndex = mt;
    if (vkAllocateMemory(st.dev, &mai, 0, m) != VK_SUCCESS) { close(dfd); return false; }
    if (vkBindBufferMemory(st.dev, *b, *m, 0) != VK_SUCCESS) return false;
    return true;
}
static bool battn_mkkern(const char * path, uint32_t nbind, uint32_t pushsz, battn_kern * k, const VkSpecializationInfo * spec = nullptr) {
    battn_state & st = g_battn;
    FILE * f = fopen(path, "rb"); if (!f) { BALOG("barra-attn: %s fehlt\n", path); return false; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code((len + 3) / 4);
    if (fread(code.data(), 1, len, f) != (size_t) len) { fclose(f); return false; }
    fclose(f);
    VkShaderModuleCreateInfo smci = {}; smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; smci.codeSize = len; smci.pCode = code.data();
    VkShaderModule sm; if (vkCreateShaderModule(st.dev, &smci, 0, &sm) != VK_SUCCESS) return false;
    VkDescriptorSetLayoutBinding b[10];   // v8-Kernel binden bis zu 8 Buffer
    for (uint32_t i = 0; i < nbind && i < 10; i++) b[i] = VkDescriptorSetLayoutBinding{ i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0 };
    VkDescriptorSetLayoutCreateInfo dlci = {}; dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; dlci.bindingCount = nbind; dlci.pBindings = b;
    if (vkCreateDescriptorSetLayout(st.dev, &dlci, 0, &k->dsl) != VK_SUCCESS) return false;
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, pushsz };
    VkPipelineLayoutCreateInfo plci = {}; plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; plci.setLayoutCount = 1; plci.pSetLayouts = &k->dsl; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(st.dev, &plci, 0, &k->pll) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpci = {}; cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main"; cpci.stage.pSpecializationInfo = spec; cpci.layout = k->pll;
    if (vkCreateComputePipelines(st.dev, VK_NULL_HANDLE, 1, &cpci, 0, &k->pipe) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dsai = {}; dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; dsai.descriptorPool = st.dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &k->dsl;
    if (vkAllocateDescriptorSets(st.dev, &dsai, &k->ds) != VK_SUCCESS) return false;
    return true;
}
static void battn_bind(battn_kern * k, VkBuffer * bufs, int n) {
    VkDescriptorBufferInfo bi[10]; VkWriteDescriptorSet w[10];   // v8: bis zu 8 Bindings
    for (int i = 0; i < n && i < 10; i++) {
        bi[i] = VkDescriptorBufferInfo{ bufs[i], 0, VK_WHOLE_SIZE };
        w[i] = VkWriteDescriptorSet{}; w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = k->ds; w[i].dstBinding = (uint32_t) i;
        w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(g_battn.dev, n, w, 0, 0);
}

// ---------- Init (lazy, einmal) ----------
static void battn_parse_layers(const char * s) {
    battn_state & st = g_battn;
    std::fill(st.layer_on.begin(), st.layer_on.end(), 0);
    while (*s) {
        int a = (int) strtol(s, (char **) &s, 10), b = a;
        if (*s == '-') { s++; b = (int) strtol(s, (char **) &s, 10); }
        for (int i = a; i <= b && i < st.NL; i++) if (i >= 0) st.layer_on[i] = 1;
        if (*s == ',') s++;
    }
}
static void battn_pkgmap(const char * dir);   // Definition weiter unten (vor battn_scan)
static bool battn_init(void) {
    battn_state & st = g_battn;
    if (st.init_done) return st.ready;
    st.init_done = true;
    const char * mp = getenv("BARRA_ATTN_META");
    FILE * f = fopen(mp, "r");
    if (!f) { BALOG("barra-attn: meta %s nicht lesbar - Offload aus\n", mp); return false; }
    int ver = 0;
    if (fscanf(f, "%d %d %d %d %d %d %d %d", &st.D, &st.H, &st.NKV, &st.HD, &st.FF, &st.NL, &st.S, &ver) != 8 || (ver != 2 && ver != 4 && ver != 5 && ver != 6 && ver != 7 && ver != 8)) {
        fclose(f); BALOG("barra-attn: meta-Kopf ungueltig/keine v2/v4/v5/v6/v7/v8 - Offload aus\n"); return false; }
    st.bits = ver >= 4 ? 16 : 8;   // Version 4/5/6/7 = 16x8 (Aktivierungen int16 symmetrisch)
    st.qchunk = ver == 4 || ver == 5;
    st.kvmerge = ver >= 5;         // Version 5 = k+v gemergt (16 Werte/Layer: q1 q2 kv o)
    st.glm = ver == 6;             // Version 6 = GLM-Edge: q ungechunkt + kv gemergt, kein q/k-Norm, NORM-RoPE (12 Werte/Layer: q kv o)
    st.gen = ver == 7;             // Version 7 = generisch: Zeile 2 "rmode bias qknorm [qchunk]", 12/16 Werte/Layer, Spec-Constant-Kernel
    st.gemma = ver == 8;           // Version 8 = Gemma-4: 2 Layer-Typen + KV-Sharing + V-Norm, immer 16 Werte/Layer
    if (st.gen || st.gemma) {
        char fl[512];
        if (!fgets(fl, sizeof fl, f) || !fgets(fl, sizeof fl, f)) { fclose(f); BALOG("barra-attn: v7/v8-Flagzeile fehlt\n"); return false; }
        int qch = 0;
        if (sscanf(fl, "%d %d %d %d", &st.rmode, &st.bias, &st.qknorm, &qch) < 3) {
            fclose(f); BALOG("barra-attn: v7/v8-Flagzeile ungueltig\n"); return false; }
        st.qchunk = qch != 0;
    }
    if (st.gemma) {
        char fl[512];
        int vnorm = 0, nkeep = 0;
        if (!fgets(fl, sizeof fl, f) || sscanf(fl, "%d %d %d", &st.HDS, &vnorm, &nkeep) != 3) {
            fclose(f); BALOG("barra-attn: v8-Typzeile (hd_swa vnorm nkeep) fehlt\n"); return false; }
        if (!fgets(fl, sizeof fl, f)) { fclose(f); BALOG("barra-attn: v8-Layertypen fehlen\n"); return false; }
        st.ltype.assign(st.NL, 0);
        { const char * s = fl;
          for (int i = 0; i < st.NL; i++) { while (*s == ' ') s++; if (*s < '0' || *s > '3') { fclose(f); BALOG("barra-attn: v8-Layertyp %d ungueltig\n", i); return false; } st.ltype[i] = (uint8_t) (*s - '0'); s++; } }
        // Praefixsummen-mids: swa+kv q,kv,o | full+kv q1,q2,kv,o | swa-sh q,o | full-sh q1,q2,o
        st.midq.assign(st.NL, 0); st.midkv.assign(st.NL, 0); st.mido.assign(st.NL, 0);
        uint32_t off = 0;
        for (int i = 0; i < st.NL; i++) {
            const int ty = st.ltype[i];
            st.midq[i] = off;
            if (ty == 0)      { st.midkv[i] = off + 1; st.mido[i] = off + 2; off += 3; }
            else if (ty == 1) { st.midkv[i] = off + 2; st.mido[i] = off + 3; off += 4; }
            else if (ty == 2) { st.mido[i] = off + 1; off += 2; }
            else              { st.mido[i] = off + 2; off += 3; }
        }
    }
    st.q.resize(st.NL);
    for (int i = 0; i < st.NL; i++) {
        battn_qgrp * g4[4] = { &st.q[i].q, &st.q[i].k, &st.q[i].v, &st.q[i].o };
        battn_qgrp * g5[5] = { &st.q[i].q, &st.q[i].q2, &st.q[i].k, &st.q[i].v, &st.q[i].o };
        battn_qgrp * g6[4] = { &st.q[i].q, &st.q[i].q2, &st.q[i].k, &st.q[i].o };
        battn_qgrp * g7[3] = { &st.q[i].q, &st.q[i].k, &st.q[i].o };
        battn_qgrp ** g = (st.gemma || (st.gen && st.qchunk)) ? g6 : (st.gen ? g7 : (st.glm ? g7 : (st.kvmerge ? g6 : (st.qchunk ? g5 : g4))));
        const int ng = st.gemma ? 4 : (st.gen ? (st.qchunk ? 4 : 3) : (st.glm ? 3 : ((st.qchunk && !st.kvmerge) ? 5 : 4)));
        for (int j = 0; j < ng; j++) {
            float isc, izp, osc, ozp;
            if (fscanf(f, "%f %f %f %f", &isc, &izp, &osc, &ozp) != 4) { fclose(f); BALOG("barra-attn: meta Layer %d ungueltig\n", i); return false; }
            g[j]->isc = isc; g[j]->izp = (int) lrintf(izp); g[j]->osc = osc; g[j]->ozp = (int) lrintf(ozp);
        }
        if (!st.qchunk) st.q[i].q2 = st.q[i].q;
        if (st.kvmerge) st.q[i].v = st.q[i].k;   // v teilt die kv-Package-Skalen
    }
    fclose(f);
    const bool geom_ok = st.gemma
        ? (st.HD == 512 && st.HDS == 256 && st.H == 8 && st.NKV == 1 && st.S == 512)
        // HD=256 fuer den generischen Zweig aufgemacht (qwen3.5): die Kernel bekommen HD als
        // Spezialisierungskonstante. OB sie 256 tragen, wird am Geraet gegen die reine
        // GPU-Ausgabe geprueft - nicht angenommen. (29.8.2026)
        : ((st.HD == 128 || (st.gen && st.HD == 256)) && st.S == 512 &&
           (st.gen ? (st.NKV > 0 && st.H % st.NKV == 0 && st.H <= 32)
                   : ((!st.glm && st.H == 32 && st.NKV == 8) || (st.glm && st.H == 24 && st.NKV == 6))));
    if (!geom_ok) { BALOG("barra-attn: Geometrie %d/%d/%d/%d nicht unterstuetzt\n", st.H, st.NKV, st.HD, st.S); return false; }
    st.layer_on.assign(st.NL, 1);
    // q/k/v teilen den quantisierten Eingang -> ihre Eingangs-Skalen muessen uebereinstimmen (gleiche Kalibrierdaten)
    for (int i = 0; i < st.NL; i++) {
        const battn_layerq & Q = st.q[i];
        const bool g_haskv = !st.gemma || st.ltype[i] <= 1;
        const bool g_qch   = st.gemma ? (st.ltype[i] == 1 || st.ltype[i] == 3) : st.qchunk;
        if (g_haskv && (fabsf(Q.k.isc - Q.q.isc) > 1e-9f * fabsf(Q.q.isc) + 1e-12f || Q.k.izp != Q.q.izp ||
            fabsf(Q.v.isc - Q.q.isc) > 1e-9f * fabsf(Q.q.isc) + 1e-12f || Q.v.izp != Q.q.izp)) {
            st.layer_on[i] = 0;
            BALOG("barra-attn: L%d Eingangs-Skalen q/k/v ungleich (%.6e/%.6e/%.6e, zp %d/%d/%d) - Layer aus\n",
                  i, Q.q.isc, Q.k.isc, Q.v.isc, Q.q.izp, Q.k.izp, Q.v.izp);
        }
        if (g_qch && (fabsf(Q.q2.isc - Q.q.isc) > 1e-9f * fabsf(Q.q.isc) + 1e-12f || Q.q2.izp != Q.q.izp)) {
            st.layer_on[i] = 0;
            BALOG("barra-attn: L%d Eingangs-Skalen q1/q2 ungleich - Layer aus\n", i);
        }
        if (st.bits == 16 && (Q.q.izp || Q.q.ozp || Q.q2.ozp || Q.k.ozp || Q.v.ozp || Q.o.izp || Q.o.ozp)) {
            st.layer_on[i] = 0;
            BALOG("barra-attn: L%d int16 mit Zero-Point != 0 - Layer aus\n", i);
        }
    }
    if (const char * e = getenv("BARRA_ATTN_LAYERS")) battn_parse_layers(e);
    if (const char * e = getenv("BARRA_ATTN_MIN_BATCH")) st.min_batch = atoi(e);
    if (const char * e = getenv("BARRA_ATTN_THREADS")) st.nthreads = atoi(e);
    st.log = getenv("BARRA_ATTN_LOG") ? 1 : 0;
    std::string dir;
    if (const char * e = getenv("BARRA_ATTN_DIR")) dir = e;
    else { dir = mp; size_t sl = dir.rfind('/'); dir = sl == std::string::npos ? "." : dir.substr(0, sl); }
    if (barra_tpu_open(&st.tpu)) { BALOG("barra-attn: tpud nicht erreichbar (BARRA_SOCK_DIR?) - Offload aus\n"); return false; }
    battn_pkgmap(dir.c_str());
    {   // gate.meta (qwen3.5): je aktiver Schicht die Quantisierung der beiden Gate-Pakete.
        // Fehlt die Datei, bleibt alles wie bisher - Altkits sind nicht betroffen.
        char gp[512]; snprintf(gp, sizeof(gp), "%s/gate.meta", dir.c_str());
        FILE * gf = fopen(gp, "r");
        if (gf) {
            char ln[512]; int n_gate = 0;
            while (fgets(ln, sizeof(ln), gf)) {
                if (ln[0] == '#') continue;
                int L = -1; float i1, o1_, i2, o2_; int z1, zo1, z2, zo2;
                if (sscanf(ln, "%d %f %d %f %d %f %d %f %d", &L, &i1, &z1, &o1_, &zo1, &i2, &z2, &o2_, &zo2) != 9) continue;
                if (L < 0 || L >= st.NL) continue;
                st.q[L].g.isc = i1; st.q[L].g.izp = z1; st.q[L].g.osc = o1_; st.q[L].g.ozp = zo1;
                st.q[L].g2.isc = i2; st.q[L].g2.izp = z2; st.q[L].g2.osc = o2_; st.q[L].g2.ozp = zo2;
                n_gate++;
            }
            fclose(gf);
            st.hasgate = n_gate > 0;
            if (st.hasgate) BALOG("barra-attn: gate.meta gelesen, %d Schichten mit Gate\n", n_gate);
        }
    }
    int total_pkg_map = 0;
    if (st.haspkgmap) {   // gezaehlt statt gerechnet: Teilmengen-Kits haben nicht NL*ppl Pakete
        const uint32_t NONE = 0xffffffffu;
        for (int i = 0; i < st.NL; i++)
            for (uint32_t v : { st.pq[i], st.pq2[i], st.pg[i], st.pg2[i], st.pkv[i], st.pv[i], st.po[i] })
                if (v != NONE) total_pkg_map++;
    }
    const int total_pkg = st.haspkgmap ? total_pkg_map
                        : (st.gemma ? (int) (st.mido[st.NL - 1] + 1)
                                    : (st.gen ? (st.qchunk ? 4 : 3) : (st.glm ? 3 : (st.kvmerge ? 4 : (st.qchunk ? 5 : 4)))) * st.NL);
    uint32_t isz = 0, osz = 0, nm = 0;
    if (barra_tpu_info(&st.tpu, 0, &isz, &osz, &nm) || (int) nm < total_pkg) { BALOG("barra-attn: tpud hat %u Modelle, erwartet %d - Offload aus\n", nm, total_pkg); return false; }
    const uint32_t bpe = st.bits / 8;
    const uint32_t qout = st.gemma ? (uint32_t) (st.H * ((st.ltype[0] & 1) ? st.HD : st.HDS) / (((st.ltype[0] & 1)) ? 2 : 1))
                                   : (st.qchunk ? st.H * st.HD / 2 : st.H * st.HD);
    if ((int) isz != (int) (st.S * st.D * bpe) || (int) osz != (int) (st.S * qout * bpe)) { BALOG("barra-attn: Package-Groessen %u/%u passen nicht (q-Package, bits=%d qchunk=%d?)\n", isz, osz, st.bits, (int) st.qchunk); return false; }
    // gemma: bQ/bQ2 je S*2048 int16 (swa-voll == full-Haelfte), bK = kv int16 max S*1024,
    // QK-Puffer nur q f16 (S*H*HD-full), Typ-Puffer separat
    const uint32_t szX = st.S * st.D * bpe;
    const uint32_t szQ = st.gemma ? (uint32_t) (st.S * st.H * st.HDS * bpe) : st.S * qout * bpe;
    const uint32_t szKV = st.gemma ? (uint32_t) (st.S * st.HD * bpe) : st.S * st.NKV * st.HD * bpe;
    const uint32_t szK = st.kvmerge ? 2 * szKV : szKV;   // kvmerge: bK haelt [k|v]-Zeilen (Stride 2*NKV*HD)
    const uint32_t szQK = st.gemma ? (uint32_t) (st.S * st.H * st.HD * 2) : (st.S * st.H * st.HD + st.S * st.NKV * st.HD) * 2;
    const uint32_t szSc = st.H * st.S * st.S * 2;
    const uint32_t szAoF = st.S * st.H * st.HD * 2, szAo = st.S * st.H * st.HD * bpe, szO = st.S * st.D * bpe;
    const uint32_t szAux = st.gemma ? (uint32_t) ((2 * st.NL * st.HD + 2 * st.S * st.HDS / 2 + 2 * st.S * st.HD / 2) * 4)
                                    : (2 * st.NL * st.HD + 2 * st.S * st.HD / 2) * 4;
    const uint32_t szSrow = st.S * 4;
    if ((st.hasgate && (barra_zc_alloc(&st.bG, szQ) || barra_zc_alloc(&st.bG2, szQ))) ||
        barra_zc_alloc(&st.bX, szX) || barra_zc_alloc(&st.bQ, szQ) || barra_zc_alloc(&st.bK, szK) || (!st.kvmerge && barra_zc_alloc(&st.bV, szKV)) ||
        barra_zc_alloc(&st.bQK, szQK) || barra_zc_alloc(&st.bSc, szSc) || barra_zc_alloc(&st.bAoF, szAoF) ||
        barra_zc_alloc(&st.bAo, szAo) || barra_zc_alloc(&st.bO, szO) || barra_zc_alloc(&st.bAux, szAux) || barra_zc_alloc(&st.bSrow, szSrow) ||
        (st.qchunk && barra_zc_alloc(&st.bQ2, szQ))) {
        BALOG("barra-attn: dmabuf-Alloc fehlgeschlagen - Offload aus\n"); return false;
    }
    if (st.gemma) {
        const uint32_t szKs = st.S * st.HDS * 2, szKf = st.S * st.HD * 2;
        if (barra_zc_alloc(&st.bKs, szKs) || barra_zc_alloc(&st.bVs, szKs) ||
            barra_zc_alloc(&st.bKf, szKf) || barra_zc_alloc(&st.bVf, szKf)) {
            BALOG("barra-attn: Typ-Puffer-Alloc fehlgeschlagen - Offload aus\n"); return false; }
    }
    { std::string ap = dir + "/aux_attn.bin"; FILE * fa = fopen(ap.c_str(), "rb");
      if (!fa || fread(st.bAux.map, 1, szAux, fa) != szAux) { if (fa) fclose(fa); BALOG("barra-attn: %s fehlt/zu kurz - Offload aus\n", ap.c_str()); return false; }
      fclose(fa); barra_zc_cpu_end(&st.bAux); }
    if (!battn_gpu_init()) { BALOG("barra-attn: eigenes Vulkan-Geraet fehlgeschlagen - Offload aus\n"); return false; }
    VkDescriptorPoolSize pss = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 48 };
    VkDescriptorPoolCreateInfo dpci = {}; dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dpci.maxSets = 12; dpci.poolSizeCount = 1; dpci.pPoolSizes = &pss;
    if (vkCreateDescriptorPool(st.dev, &dpci, 0, &st.dp) != VK_SUCCESS) return false;
    if (!battn_import(st.bQ.fd, szQ, &st.vbQ, &st.vmQ) || !battn_import(st.bK.fd, szK, &st.vbK, &st.vmK) ||
        (!st.kvmerge && !battn_import(st.bV.fd, szKV, &st.vbV, &st.vmV)) || !battn_import(st.bQK.fd, szQK, &st.vbQK, &st.vmQK) ||
        !battn_import(st.bSc.fd, szSc, &st.vbSc, &st.vmSc) || !battn_import(st.bAoF.fd, szAoF, &st.vbAoF, &st.vmAoF) ||
        !battn_import(st.bAux.fd, szAux, &st.vbAux, &st.vmAux) || !battn_import(st.bSrow.fd, szSrow, &st.vbSrow, &st.vmSrow) ||
        (st.qchunk && !battn_import(st.bQ2.fd, szQ, &st.vbQ2, &st.vmQ2))) {
        BALOG("barra-attn: dmabuf-Import fehlgeschlagen - Offload aus\n"); return false; }
    if (st.gemma) {
        const uint32_t szKs = st.S * st.HDS * 2, szKf = st.S * st.HD * 2;
        if (!battn_import(st.bKs.fd, szKs, &st.vbKs, &st.vmKs) || !battn_import(st.bVs.fd, szKs, &st.vbVs, &st.vmVs) ||
            !battn_import(st.bKf.fd, szKf, &st.vbKf, &st.vmKf) || !battn_import(st.bVf.fd, szKf, &st.vbVf, &st.vmVf)) {
            BALOG("barra-attn: Typ-Puffer-Import fehlgeschlagen - Offload aus\n"); return false; }
    }
    if (st.gemma) {
        // v8: je Layer-Typ eigene Pipeline-Instanzen desselben Spec-Kernels (swa HD=256 / full HD=512+qchunk)
        uint32_t sr_s[5] = { (uint32_t) st.H, (uint32_t) (st.H * st.HDS), (uint32_t) st.HDS, 0u, (uint32_t) st.HD };
        uint32_t sr_f[5] = { (uint32_t) st.H, (uint32_t) (st.H * st.HD), (uint32_t) st.HD, 1u, (uint32_t) st.HD };
        uint32_t sq_s[2] = { (uint32_t) (st.H * st.HDS), (uint32_t) st.HDS };
        uint32_t sq_f[2] = { (uint32_t) (st.H * st.HD), (uint32_t) st.HD };
        VkSpecializationMapEntry sme8[5];
        for (uint32_t i = 0; i < 5; i++) sme8[i] = VkSpecializationMapEntry{ i, i * 4, 4 };
        VkSpecializationInfo si_rs = { 5, sme8, sizeof sr_s, sr_s }, si_rf = { 5, sme8, sizeof sr_f, sr_f };
        VkSpecializationInfo si_qs = { 2, sme8, sizeof sq_s, sq_s }, si_qf = { 2, sme8, sizeof sq_f, sq_f };
        if (!battn_mkkern((dir + "/rope_qk_v8.spv").c_str(), 8, 40, &st.krp8s, &si_rs) ||
            !battn_mkkern((dir + "/rope_qk_v8.spv").c_str(), 8, 40, &st.krp8f, &si_rf) ||
            !battn_mkkern((dir + "/qkt_v8.spv").c_str(), 3, 4, &st.kqk8s, &si_qs) ||
            !battn_mkkern((dir + "/qkt_v8.spv").c_str(), 3, 4, &st.kqk8f, &si_qf) ||
            !battn_mkkern((dir + "/av_v8.spv").c_str(), 3, 4, &st.kav8s, &si_qs) ||
            !battn_mkkern((dir + "/av_v8.spv").c_str(), 3, 4, &st.kav8f, &si_qf) ||
            !battn_mkkern((dir + "/softmax_causal.spv").c_str(), 1, 4, &st.ksm)) {
            BALOG("barra-attn: v8-Kernel-Setup fehlgeschlagen - Offload aus\n"); return false; }
        { VkBuffer b[8] = { st.vbQ, st.vbQ, st.vbK, st.vbQK, st.vbKs, st.vbVs, st.vbAux, st.vbSrow }; battn_bind(&st.krp8s, b, 8); }
        { VkBuffer b[8] = { st.vbQ, st.vbQ2, st.vbK, st.vbQK, st.vbKf, st.vbVf, st.vbAux, st.vbSrow }; battn_bind(&st.krp8f, b, 8); }
        { VkBuffer b3[3] = { st.vbQK, st.vbKs, st.vbSc }; battn_bind(&st.kqk8s, b3, 3); }
        { VkBuffer b3[3] = { st.vbQK, st.vbKf, st.vbSc }; battn_bind(&st.kqk8f, b3, 3); }
        { VkBuffer b1[1] = { st.vbSc }; battn_bind(&st.ksm, b1, 1); }
        { VkBuffer b3[3] = { st.vbSc, st.vbVs, st.vbAoF }; battn_bind(&st.kav8s, b3, 3); }
        { VkBuffer b3[3] = { st.vbSc, st.vbVf, st.vbAoF }; battn_bind(&st.kav8f, b3, 3); }
        st.srow_c.resize(st.S);
        int non8 = 0, nkv8 = 0;
        for (int i = 0; i < st.NL; i++) { non8 += st.layer_on[i]; nkv8 += st.ltype[i] <= 1 ? 1 : 0; }
        BALOG("barra-attn: AKTIV v2.7 (v8 Gemma-4) D=%d H=%d HD=%d/%d NL=%d S=%d Layer an=%d (kv=%d shared=%d)\n",
              st.D, st.H, st.HDS, st.HD, st.NL, st.S, non8, nkv8, st.NL - nkv8);
        st.ready = true;
        return true;
    }
    const char * krope = st.gen ? "/rope_qk_gen.spv" : (st.glm ? "/rope_qk_l6.spv" : (st.kvmerge ? "/rope_qk_l5.spv" : (st.qchunk ? "/rope_qk_l4.spv" : (st.bits == 16 ? "/rope_qk_l3.spv" : "/rope_qk_l2.spv"))));
    const char * kavn  = st.gen ? "/av_gen.spv" : (st.glm ? "/av_glm.spv" : (st.kvmerge ? "/av_rows3.spv" : (st.bits == 16 ? "/av_rows2.spv" : "/av_rows.spv")));
    const char * kqkn  = st.gen ? "/qkt_gen.spv" : (st.glm ? "/qkt_glm.spv" : "/qkt_f16.spv");
    // v7: Geometrie als Vulkan-Spezialisierungskonstanten (ein Kernel-Satz fuer alle Standard-GQA-Transformer HD=128)
    const uint32_t QW = (uint32_t) (st.H * st.HD), KVROW = (uint32_t) (st.NKV * st.HD), KVW = 2 * KVROW;
    const uint32_t KOFFv = (uint32_t) st.S * QW, GRPv = (uint32_t) (st.H / st.NKV);
    uint32_t sd_rope[8] = { (uint32_t) st.H, (uint32_t) st.NKV, QW, KVW, KOFFv, (uint32_t) st.rmode, (uint32_t) st.qknorm, st.qchunk ? 1u : 0u };
    uint32_t sd_qkt[4]  = { GRPv, QW, KVROW, KOFFv };
    uint32_t sd_av[4]   = { GRPv, QW, KVROW, KVW };
    VkSpecializationMapEntry sme[8];
    for (uint32_t i = 0; i < 8; i++) sme[i] = VkSpecializationMapEntry{ i, i * 4, 4 };
    VkSpecializationInfo si_rope = { 8, sme, sizeof sd_rope, sd_rope };
    VkSpecializationInfo si_qkt  = { 4, sme, sizeof sd_qkt,  sd_qkt };
    VkSpecializationInfo si_av   = { 4, sme, sizeof sd_av,   sd_av };
    if (!battn_mkkern((dir + krope).c_str(), (st.gen || st.qchunk) ? 6 : 5, (st.gen || st.qchunk) ? 40 : 32, &st.krp, st.gen ? &si_rope : nullptr) ||
        !battn_mkkern((dir + kqkn).c_str(), 2, 4, &st.kqk, st.gen ? &si_qkt : nullptr) ||
        !battn_mkkern((dir + "/softmax_causal.spv").c_str(), 1, 4, &st.ksm) ||
        !battn_mkkern((dir + kavn).c_str(), 4, 8, &st.kav, st.gen ? &si_av : nullptr)) { BALOG("barra-attn: Kernel-Setup fehlgeschlagen - Offload aus\n"); return false; }
    if (st.gen)         { VkBuffer b6[6] = { st.vbQ, st.qchunk ? st.vbQ2 : st.vbQ, st.vbK, st.vbQK, st.vbAux, st.vbSrow }; battn_bind(&st.krp, b6, 6); }
    else if (st.qchunk) { VkBuffer b6[6] = { st.vbQ, st.vbQ2, st.vbK, st.vbQK, st.vbAux, st.vbSrow }; battn_bind(&st.krp, b6, 6); }
    else                { VkBuffer b5[5] = { st.vbQ, st.vbK, st.vbQK, st.vbAux, st.vbSrow }; battn_bind(&st.krp, b5, 5); }
    { VkBuffer b2[2] = { st.vbQK, st.vbSc };                             battn_bind(&st.kqk, b2, 2); }
    { VkBuffer b1[1] = { st.vbSc };                                      battn_bind(&st.ksm, b1, 1); }
    { VkBuffer b4[4] = { st.vbSc, st.kvmerge ? st.vbK : st.vbV, st.vbAoF, st.vbSrow }; battn_bind(&st.kav, b4, 4); }
    st.srow_c.resize(st.S);
    int non = 0; for (int i = 0; i < st.NL; i++) non += st.layer_on[i];
    BALOG("barra-attn: AKTIV v2.%s (Split, bits=%d) D=%d H=%d NKV=%d HD=%d NL=%d S=%d rmode=%d bias=%d qknorm=%d qchunk=%d min_batch=%d threads=%d Layer an=%d\n",
          st.gen ? "6 (v7 generisch)" : (st.glm ? "5 (GLM-Edge 16x8)" : (st.kvmerge ? "4 (16x8 kv-merge)" : (st.bits == 16 ? "3 (16x8)" : "2 (geklemmte Zeilennorm)"))), st.bits, st.D, st.H, st.NKV, st.HD, st.NL, st.S, st.rmode, st.bias, st.qknorm, (int) st.qchunk, st.min_batch, st.nthreads, non);
    st.ready = true;
    return true;
}

// ---------- Graph-Mustererkennung ----------
struct battn_range {
    int start, end, L, T;
    ggml_tensor *hnorm, *wo, *setk, *setv, *ropeq;
    float scale, eps;
};
static ggml_tensor * battn_skipv(ggml_tensor * t, std::set<ggml_tensor *> * claim) {
    while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_CONT)) {
        if (claim) claim->insert(t);
        t = t->src[0];
    }
    return t;
}
static int battn_wname(const ggml_tensor * w, const char * kind) {
    if (!w) return -1;
    int L = -1; char k[24] = { 0 };
    if (sscanf(w->name, "blk.%d.attn_%23[a-z_].weight", &L, k) != 2) return -1;
    return strcmp(k, kind) == 0 ? L : -1;
}
static void * battn_host_ptr(const ggml_tensor * t) {
    if (!t || !t->buffer || !t->buffer->context) return nullptr;
    ggml_backend_vk_buffer_context * bctx = (ggml_backend_vk_buffer_context *) t->buffer->context;
    vk_buffer & vb = bctx->dev_buffer;
    if (!vb || !vb->ptr) return nullptr;
    if (!(vb->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) ||
        !(vb->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent)) return nullptr;
    return (uint8_t *) vb->ptr + vk_tensor_dev_offset(t);
}
// pkglist.txt lesen und Name -> Position abbilden. Namensschema des Generators:
//   q_L<i> / q1_L<i> / q2_L<i> / g1_L<i> / g2_L<i> / kv_L<i> / k_L<i> / v_L<i> / o_L<i>
static void battn_pkgmap(const char * dir) {
    battn_state & st = g_battn;
    st.haspkgmap = false;
    if (!dir) return;
    char path[512]; snprintf(path, sizeof(path), "%s/pkglist.txt", dir);
    FILE * f = fopen(path, "r"); if (!f) return;
    const uint32_t NONE = 0xffffffffu;
    st.pq.assign(st.NL, NONE); st.pq2.assign(st.NL, NONE); st.pg.assign(st.NL, NONE);
    st.pg2.assign(st.NL, NONE); st.pkv.assign(st.NL, NONE); st.pv.assign(st.NL, NONE);
    st.po.assign(st.NL, NONE);
    char line[256]; uint32_t pos = 0; int n = 0;
    while (fgets(line, sizeof(line), f)) {
        char kind[8] = {0}; int L = -1;
        if (sscanf(line, "%7[a-z0-9]_L%d", kind, &L) != 2) continue;
        if (L < 0 || L >= st.NL) { fclose(f); st.haspkgmap = false; return; }
        if      (!strcmp(kind, "q") || !strcmp(kind, "q1")) st.pq[L]  = pos;
        else if (!strcmp(kind, "q2"))                       st.pq2[L] = pos;
        else if (!strcmp(kind, "g") || !strcmp(kind, "g1")) st.pg[L]  = pos;
        else if (!strcmp(kind, "g2"))                       st.pg2[L] = pos;
        else if (!strcmp(kind, "kv") || !strcmp(kind, "k")) st.pkv[L] = pos;
        else if (!strcmp(kind, "v"))                        st.pv[L]  = pos;
        else if (!strcmp(kind, "o"))                        st.po[L]  = pos;
        else { continue; }
        pos++; n++;
    }
    fclose(f);
    st.haspkgmap = n > 0;
    if (st.haspkgmap) BALOG("barra-attn: pkglist gelesen, %d Pakete zugeordnet\n", n);
}
static void battn_scan(ggml_cgraph * cg, std::vector<battn_range> & out) {
    battn_state & st = g_battn;
    std::map<ggml_tensor *, int> idx;
    for (int i = 0; i < cg->n_nodes; i++) idx[cg->nodes[i]] = i;
    std::map<int, ggml_tensor *> setk_by_layer, setv_by_layer;
    for (int i = 0; i < cg->n_nodes; i++) {
        ggml_tensor * n = cg->nodes[i];
        if (n->op != GGML_OP_SET_ROWS) continue;
        ggml_tensor * src = battn_skipv(n->src[0], nullptr);
        if (src && src->op == GGML_OP_ROPE) {
            ggml_tensor * mm = battn_skipv(src->src[0], nullptr);
            if (mm && mm->op == GGML_OP_ADD) mm = battn_skipv(mm->src[0], nullptr);   // qkv-Bias (v7)
            if (mm && mm->op == GGML_OP_MUL) mm = battn_skipv(mm->src[0], nullptr);
            if (mm && mm->op == GGML_OP_RMS_NORM) mm = battn_skipv(mm->src[0], nullptr);
            int L = mm && mm->op == GGML_OP_MUL_MAT ? battn_wname(mm->src[0], "k") : -1;
            if (L >= 0) setk_by_layer[L] = n;
        } else if (src && (src->op == GGML_OP_MUL_MAT || src->op == GGML_OP_ADD || src->op == GGML_OP_RMS_NORM)) {
            ggml_tensor * mm = src->op == GGML_OP_MUL_MAT ? src : battn_skipv(src->src[0], nullptr);
            int L = mm && mm->op == GGML_OP_MUL_MAT ? battn_wname(mm->src[0], "v") : -1;
            if (L >= 0) setv_by_layer[L] = n;
        }
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        ggml_tensor * wo = cg->nodes[i];
        if (wo->op != GGML_OP_MUL_MAT) continue;
        int L = battn_wname(wo->src[0], "output");
        if (L < 0 || L >= st.NL || !st.layer_on[L]) continue;
        std::set<ggml_tensor *> claim; claim.insert(wo);
        ggml_tensor * fa = battn_skipv(wo->src[1], &claim);
        // src[4] (sinks) nicht unterstuetzt; src[3] (Maske) MUSS vorhanden sein — der
        // Offload wendet feste kausale Maskierung an, eine Null-Maske (bidirektional)
        // waere nicht-kausal. ALiBi/logit-softcap werden weiter unten (fap[1]/fap[2]) abgelehnt.
        // qwen3.5: gegatete Attention - zwischen FA und o-Projektion steht
        // ggml_mul(attn, sigmoid(gate)). Nur ueberspringen reicht NICHT: beide Knoten
        // muessen in den Bereich beansprucht werden, sonst gilt er als "nicht sauber"
        // und der Offload faellt still auf den Vulkan-Pfad zurueck. Das Gate selbst
        // rechnet die Laufzeit aus den g-Paketen (st.hasgate), deshalb nur dann.
        if (st.hasgate && fa && fa->op == GGML_OP_MUL) {
            ggml_tensor * a0 = battn_skipv(fa->src[0], nullptr);
            ggml_tensor * s0 = battn_skipv(fa->src[1], nullptr);
            if (a0 && a0->op == GGML_OP_FLASH_ATTN_EXT && s0) {
                claim.insert(fa); claim.insert(s0);
                fa = a0;
            }
        }
        if (!fa || fa->op != GGML_OP_FLASH_ATTN_EXT || fa->src[4] || !fa->src[3]) continue;
        claim.insert(fa);
        ggml_tensor * rq = battn_skipv(fa->src[0], &claim);
        if (!rq || rq->op != GGML_OP_ROPE) continue;
        claim.insert(rq);
        ggml_tensor * nq = nullptr, * wq = nullptr;
        if (st.glm || st.gen || st.gemma) {
            // GLM-Edge/v7/v8: ROPE auf der q-Projektion; optional q/k-RMSNorm (Qwen3/Gemma) und/oder Bias-ADD
            ggml_tensor * t = battn_skipv(rq->src[0], &claim);
            if ((st.gen || st.gemma) && st.qknorm) {
                if (!t || t->op != GGML_OP_MUL) continue;
                claim.insert(t);
                nq = battn_skipv(t->src[0], &claim);
                if (!nq || nq->op != GGML_OP_RMS_NORM) continue;
                claim.insert(nq);
                t = battn_skipv(nq->src[0], &claim);
            }
            if (st.gen && st.bias) {
                if (!t || t->op != GGML_OP_ADD) continue;
                claim.insert(t); t = battn_skipv(t->src[0], &claim);
            }
            wq = t;
            if (!wq || wq->op != GGML_OP_MUL_MAT || battn_wname(wq->src[0], "q") != L) continue;
            claim.insert(wq);
        } else {
            ggml_tensor * mq = battn_skipv(rq->src[0], &claim);
            if (!mq || mq->op != GGML_OP_MUL) continue;
            claim.insert(mq);
            nq = battn_skipv(mq->src[0], &claim);
            if (!nq || nq->op != GGML_OP_RMS_NORM) continue;
            claim.insert(nq);
            wq = battn_skipv(nq->src[0], &claim);
            if (!wq || wq->op != GGML_OP_MUL_MAT || battn_wname(wq->src[0], "q") != L) continue;
            claim.insert(wq);
        }
        ggml_tensor * hnorm = wq->src[1];
        const int lhd = st.gemma ? ((st.ltype[L] & 1) ? st.HD : st.HDS) : st.HD;   // Kopfbreite dieses Layers
        const bool lshared = st.gemma && st.ltype[L] >= 2;                          // KV-Sharing-Layer: kein setk/setv
        auto itk = setk_by_layer.find(L), itv = setv_by_layer.find(L);
        if (!lshared && (itk == setk_by_layer.end() || itv == setv_by_layer.end())) continue;
        if (lshared && (itk != setk_by_layer.end() || itv != setv_by_layer.end())) continue;   // Typkonflikt Meta vs Graph
        ggml_tensor * setk = lshared ? nullptr : itk->second, * setv = lshared ? nullptr : itv->second;
        if (lshared) {
            // shared: keine k/v-Ketten; FA liest den Cache des letzten kv-Layers gleichen Typs
        } else if (st.glm || st.gen || st.gemma) {
            ggml_tensor * t = battn_skipv(setk->src[0], &claim);
            claim.insert(t); t = battn_skipv(t->src[0], &claim);
            if ((st.gen || st.gemma) && st.qknorm) {
                if (!t || t->op != GGML_OP_MUL) continue; claim.insert(t);
                t = battn_skipv(t->src[0], &claim);
                if (!t || t->op != GGML_OP_RMS_NORM) continue; claim.insert(t);
                t = battn_skipv(t->src[0], &claim);
            }
            if (st.gen && st.bias) {
                if (!t || t->op != GGML_OP_ADD) continue;
                claim.insert(t); t = battn_skipv(t->src[0], &claim);
            }
            if (!t || t->op != GGML_OP_MUL_MAT || battn_wname(t->src[0], "k") != L || t->src[1] != hnorm) continue; claim.insert(t);
        } else {
          ggml_tensor * t = battn_skipv(setk->src[0], &claim);
          claim.insert(t); t = battn_skipv(t->src[0], &claim);
          if (!t || t->op != GGML_OP_MUL) continue; claim.insert(t);
          t = battn_skipv(t->src[0], &claim); if (!t || t->op != GGML_OP_RMS_NORM) continue; claim.insert(t);
          t = battn_skipv(t->src[0], &claim); if (!t || t->op != GGML_OP_MUL_MAT || battn_wname(t->src[0], "k") != L || t->src[1] != hnorm) continue; claim.insert(t); }
        if (!lshared) { ggml_tensor * t = battn_skipv(setv->src[0], &claim);
          if (st.gemma) {
              // Gemma: V wird RMS-normiert (ohne Gamma) vor SET_ROWS
              if (t && t->op == GGML_OP_RMS_NORM) { claim.insert(t); t = battn_skipv(t->src[0], &claim); }
              else continue;
          }
          if (st.gen && st.bias) {
              if (!t || t->op != GGML_OP_ADD) continue;
              claim.insert(t); t = battn_skipv(t->src[0], &claim);
          }
          if (!t || t->op != GGML_OP_MUL_MAT || battn_wname(t->src[0], "v") != L || t->src[1] != hnorm) continue; claim.insert(t); }
        if (!lshared) { claim.insert(setk); claim.insert(setv); }
        const int T = (int) hnorm->ne[1];
        if (hnorm->type != GGML_TYPE_F32 || !ggml_is_contiguous(hnorm)) continue;
        if (T < st.min_batch || T > st.S) continue;
        if (wo->type != GGML_TYPE_F32) continue;
        if (!lshared) {
            if (setk->type != GGML_TYPE_F16 || setv->type != GGML_TYPE_F16) continue;
            if ((int) setk->ne[0] != st.NKV * lhd || (int) setv->ne[0] != st.NKV * lhd) continue;
            // Die SET_ROWS-Index-Tensoren werden im Exec als int64 gelesen. ggml erlaubt
            // auch I32-ids; ein Upstream-Wechsel darauf wuerde die memcpys in wilde
            // Offsets schreiben. Nur bei I64 uebernehmen, sonst Vulkan-Fallback.
            if (!setk->src[1] || setk->src[1]->type != GGML_TYPE_I64) continue;
            if (!setv->src[1] || setv->src[1]->type != GGML_TYPE_I64) continue;
        }
        if (fa->src[1]->ne[1] < T) continue;
        float fap[3]; memcpy(fap, fa->op_params, sizeof(fap));
        if (fap[1] != 0.0f || fap[2] != 0.0f) continue;
        int32_t ndims; memcpy(&ndims, (const char *) rq->op_params + 4, 4);
        if (ndims != lhd) continue;
        int32_t rmode; memcpy(&rmode, (const char *) rq->op_params + 8, 4);
        if (st.glm && rmode != 0) continue;   // GLM braucht NORM-RoPE (Paar-Rotation); NEOX waere ein anderes Modell
        if ((st.gen || st.gemma) && rmode != st.rmode) continue;   // v7/v8: RoPE-Stil muss zur Meta passen
        float qeps = 0.0f;
        if (nq) memcpy(&qeps, nq->op_params, 4);
        int lo = INT32_MAX, hi = -1;
        for (ggml_tensor * t : claim) { auto it = idx.find(t); if (it != idx.end()) { lo = std::min(lo, it->second); hi = std::max(hi, it->second); } }
        if (hi < 0) continue;
        bool clean = true;
        for (int j = lo; j <= hi && clean; j++) {
            ggml_tensor * n = cg->nodes[j];
            if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
            if (n->op == GGML_OP_NONE || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE) continue;
            if (!claim.count(n)) clean = false;
        }
        if (!clean) { if (st.log) BALOG("barra-attn: L%d Bereich %d..%d nicht sauber - Vulkan-Pfad\n", L, lo, hi); continue; }
        battn_range r; r.start = lo; r.end = hi; r.L = L; r.T = T;
        r.hnorm = hnorm; r.wo = wo; r.setk = setk; r.setv = setv; r.ropeq = rq; r.scale = fap[0]; r.eps = qeps;
        out.push_back(r);
        i = hi;
    }
}

// ---------- v8: Gemma-4-Ausfuehrung (2 Layer-Typen, KV-Sharing, V-Norm auf der GPU) ----------
static bool battn_exec_gemma(const battn_range & r) {
    battn_state & st = g_battn;
    const int T = r.T, S = st.S, D = st.D;
    const int ty = st.ltype[r.L];
    const bool haskv = ty <= 1, full = (ty & 1) != 0;
    const int hd = full ? st.HD : st.HDS;
    const int qw = st.H * hd;
    const battn_layerq & Q = st.q[r.L];
    double t0 = now_ms_battn();
    const float * h = (const float *) battn_host_ptr(r.hnorm);
    float * odst = (float *) battn_host_ptr(r.wo);
    ggml_fp16_t * kdst = haskv ? (ggml_fp16_t *) battn_host_ptr(r.setk) : nullptr;
    ggml_fp16_t * vdst = haskv ? (ggml_fp16_t *) battn_host_ptr(r.setv) : nullptr;
    const int32_t * pos = (const int32_t *) battn_host_ptr(r.ropeq->src[1]);
    const int64_t * kids = haskv ? (const int64_t *) battn_host_ptr(r.setk->src[1]) : nullptr;
    const int64_t * vids = haskv ? (const int64_t *) battn_host_ptr(r.setv->src[1]) : nullptr;
    if (!h || !odst || !pos || (haskv && (!kdst || !vdst || !kids || !vids))) { if (st.log) BALOG("barra-attn: L%d Hostzeiger fehlt\n", r.L); return false; }
    for (int t = 0; t < T; t++) if (pos[t] != t) { if (st.log) BALOG("barra-attn: L%d pos[%d]=%d != %d - Vulkan-Pfad\n", r.L, t, pos[t], t); return false; }
    barra_zc_cpu_begin(&st.bX); barra_zc_cpu_begin(&st.bSrow);
    float * srow = (float *) st.bSrow.map;
    for (int t = T; t < S; t++) srow[t] = 1.0f;
    { int16_t * zx = (int16_t *) st.bX.map;
      memset(zx + (size_t) T * D, 0, (size_t) (S - T) * D * 2);
      battn_par(T, [&](int t) {
          const float * hr = h + (size_t) t * D; int16_t * zr = zx + (size_t) t * D;
          float mx = 0; for (int d = 0; d < D; d++) { float a = fabsf(hr[d]); if (a > mx) mx = a; }
          const float sr = std::max(1.0f, mx / (32767.0f * Q.q.isc));
          srow[t] = sr;
          const float k = 1.0f / (sr * Q.q.isc);
          for (int d = 0; d < D; d++) { int v = (int) lrintf(hr[d] * k); zr[d] = (int16_t) (v > 32767 ? 32767 : v < -32768 ? -32768 : v); }
      }); }
    barra_zc_cpu_end(&st.bX); barra_zc_cpu_end(&st.bSrow);
    double t1 = now_ms_battn();
    if (barra_tpu_infer(&st.tpu, st.midq[r.L], &st.bX, &st.bQ, 0) ||
        (full && barra_tpu_infer(&st.tpu, st.midq[r.L] + 1, &st.bX, &st.bQ2, 0)) ||
        (haskv && barra_tpu_infer(&st.tpu, st.midkv[r.L], &st.bX, &st.bK, 0))) {
        BALOG("barra-attn: L%d TPU q/kv fehlgeschlagen\n", r.L); return false; }
    double t2 = now_ms_battn();
    barra_zc_cpu_end(&st.bQ); if (full) barra_zc_cpu_end(&st.bQ2);
    if (haskv) barra_zc_cpu_end(&st.bK);
    barra_zc_cpu_end(&st.bQK); barra_zc_cpu_end(&st.bSc); barra_zc_cpu_end(&st.bAoF);
    if (haskv) { barra_zc_cpu_end(full ? &st.bKf : &st.bKs); barra_zc_cpu_end(full ? &st.bVf : &st.bVs); }
    VkCommandBufferBeginInfo bg = {}; bg.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(st.cb, &bg);
    VkMemoryBarrier mb = {}; mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    struct { float q1osc, q1ozp, q2osc, q2ozp, kosc, kozp, eps; uint32_t gqo, gko, tabo; } pr8 = {
        Q.q.osc, (float) Q.q.ozp, Q.q2.osc, (float) Q.q2.ozp, Q.k.osc, (float) Q.k.ozp, r.eps,
        (uint32_t) (r.L * st.HD), (uint32_t) (st.NL * st.HD + r.L * st.HD),
        full ? (uint32_t) (2 * st.NL * st.HD + 2 * st.S * (st.HDS / 2)) : (uint32_t) (2 * st.NL * st.HD) };
    const battn_kern & krp = full ? st.krp8f : st.krp8s;
    const battn_kern & kqk = full ? st.kqk8f : st.kqk8s;
    const battn_kern & kav = full ? st.kav8f : st.kav8s;
    vkCmdBindPipeline(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, krp.pipe);
    vkCmdBindDescriptorSets(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, krp.pll, 0, 1, &krp.ds, 0, 0);
    vkCmdPushConstants(st.cb, krp.pll, VK_SHADER_STAGE_COMPUTE_BIT, 0, 40, &pr8);
    vkCmdDispatch(st.cb, (uint32_t) (st.H + (haskv ? 2 : 0)), (uint32_t) S, 1);
    vkCmdPipelineBarrier(st.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
    float pq = 256.0f * r.scale;
    vkCmdBindPipeline(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, kqk.pipe);
    vkCmdBindDescriptorSets(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, kqk.pll, 0, 1, &kqk.ds, 0, 0);
    vkCmdPushConstants(st.cb, kqk.pll, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &pq);
    vkCmdDispatch(st.cb, (uint32_t) (S / 128), (uint32_t) (S / 128), (uint32_t) st.H);
    vkCmdPipelineBarrier(st.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
    uint32_t ps = (uint32_t) S;
    vkCmdBindPipeline(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.ksm.pipe);
    vkCmdBindDescriptorSets(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.ksm.pll, 0, 1, &st.ksm.ds, 0, 0);
    vkCmdPushConstants(st.cb, st.ksm.pll, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &ps);
    vkCmdDispatch(st.cb, (uint32_t) (st.H * S), 1, 1);
    vkCmdPipelineBarrier(st.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
    vkCmdBindPipeline(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, kav.pipe);
    vkCmdBindDescriptorSets(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, kav.pll, 0, 1, &kav.ds, 0, 0);
    vkCmdDispatch(st.cb, (uint32_t) (hd / 128), (uint32_t) (S / 128), (uint32_t) st.H);
    vkCmdPipelineBarrier(st.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
    vkEndCommandBuffer(st.cb);
    vkResetFences(st.dev, 1, &st.fen);
    VkSubmitInfo si = {}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &st.cb;
    if (vkQueueSubmit(st.queue, 1, &si, st.fen) != VK_SUCCESS) { BALOG("barra-attn: L%d GPU-Submit fehlgeschlagen\n", r.L); return false; }
    vkWaitForFences(st.dev, 1, &st.fen, VK_TRUE, UINT64_MAX);
    double t3 = now_ms_battn();
    barra_zc_cpu_begin(&st.bAoF); barra_zc_cpu_begin(&st.bAo);
    if (st.hasgate) { barra_zc_cpu_begin(&st.bG); barra_zc_cpu_begin(&st.bG2); }
#if defined(__aarch64__)
    { const __fp16 * af = (const __fp16 *) st.bAoF.map;
      int16_t * zao = (int16_t *) st.bAo.map;
      memset(zao + (size_t) T * qw, 0, (size_t) (S - T) * qw * 2);
      // Gegatete Attention (qwen3.5): a * sigmoid(gate) MUSS vor Maximumsuche und
      // Zeilenskala passieren - das o-Paket ist auf den GEGATETEN Tensor kalibriert
      // (der Kalibrier-Dump nimmt genau diesen ab). Die beiden Gate-Haelften liegen
      // wie q1/q2 als Kopf-Haelften vor, je Zeile HH = qw/2 Werte.
      const int HH = qw / 2;
      const int16_t * g1 = st.hasgate ? (const int16_t *) st.bG.map : nullptr;
      const int16_t * g2 = st.hasgate ? (const int16_t *) st.bG2.map : nullptr;
      const float gs1 = Q.g.osc, gs2 = Q.g2.osc;
      const int gz1 = Q.g.ozp, gz2 = Q.g2.ozp;
      battn_par(T, [&](int t) {
          const __fp16 * ar = af + (size_t) t * qw; int16_t * zr = zao + (size_t) t * qw;
          if (g1) {
              static thread_local std::vector<float> gt;
              if ((int) gt.size() < qw) gt.resize(qw);
              const int16_t * gr1 = g1 + (size_t) t * HH;
              const int16_t * gr2 = g2 + (size_t) t * HH;
              float mxg = 0;
              for (int d = 0; d < qw; d++) {
                  const float g = d < HH ? ((float) gr1[d] - gz1) * gs1 : ((float) gr2[d - HH] - gz2) * gs2;
                  const float a = (float) ar[d] / (1.0f + expf(-g));
                  gt[d] = a; const float aa = fabsf(a); if (aa > mxg) mxg = aa;
              }
              const float srg = std::max(1.0f, mxg / (32767.0f * Q.o.isc));
              st.srow_c[t] = srg;
              const float kg = 1.0f / (srg * Q.o.isc);
              for (int d = 0; d < qw; d++) { int v = (int) lrintf(gt[d] * kg); zr[d] = (int16_t) (v > 32767 ? 32767 : v < -32768 ? -32768 : v); }
              return;
          }
          float mx = 0; for (int d = 0; d < qw; d++) { float a = fabsf((float) ar[d]); if (a > mx) mx = a; }
          const float sr = std::max(1.0f, mx / (32767.0f * Q.o.isc));
          st.srow_c[t] = sr;
          const float k = 1.0f / (sr * Q.o.isc);
          for (int d = 0; d < qw; d++) { int v = (int) lrintf((float) ar[d] * k); zr[d] = (int16_t) (v > 32767 ? 32767 : v < -32768 ? -32768 : v); }
      }); }
#endif
    barra_zc_cpu_end(&st.bAoF); barra_zc_cpu_end(&st.bAo);
    if (barra_tpu_infer(&st.tpu, st.mido[r.L], &st.bAo, &st.bO, 0)) { BALOG("barra-attn: L%d TPU o fehlgeschlagen\n", r.L); return false; }
    double t4 = now_ms_battn();
    barra_zc_cpu_begin(&st.bO);
    if (haskv) { barra_zc_cpu_begin(full ? &st.bKf : &st.bKs); barra_zc_cpu_begin(full ? &st.bVf : &st.bVs); }
    const size_t onb1 = r.wo->nb[1];
    const ggml_fp16_t * ksrc = haskv ? (const ggml_fp16_t *) (full ? st.bKf.map : st.bKs.map) : nullptr;
    const ggml_fp16_t * vsrc = haskv ? (const ggml_fp16_t *) (full ? st.bVf.map : st.bVs.map) : nullptr;
    const size_t knb1 = haskv ? r.setk->nb[1] : 0, vnb1 = haskv ? r.setv->nb[1] : 0;
    if (haskv) {   // ids VOR dem ersten memcpy pruefen -> kein Teilschreiben bei Fallback
        const int64_t krows = r.setk->ne[1], vrows = r.setv->ne[1];
        for (int t = 0; t < T; t++) if (kids[t] < 0 || kids[t] >= krows || vids[t] < 0 || vids[t] >= vrows) {
            if (st.log) BALOG("barra-attn: L%d SET_ROWS id ausser Bereich (t=%d) — Vulkan-Fallback\n", r.L, t);
            barra_zc_cpu_end(full ? &st.bKf : &st.bKs); barra_zc_cpu_end(full ? &st.bVf : &st.bVs);
            return false;
        }
    }
    battn_par(T, [&](int t) {
        float * orow = (float *) ((char *) odst + (size_t) t * onb1);
        const float os = Q.o.osc * st.srow_c[t];
        const int16_t * zr = (const int16_t *) st.bO.map + (size_t) t * D;
        for (int d = 0; d < D; d++) orow[d] = zr[d] * os;
        if (haskv) {
            memcpy((char *) kdst + (size_t) kids[t] * knb1, ksrc + (size_t) t * hd, (size_t) hd * 2);
            memcpy((char *) vdst + (size_t) vids[t] * vnb1, vsrc + (size_t) t * hd, (size_t) hd * 2);
        }
    });
    if (haskv) { barra_zc_cpu_end(full ? &st.bKf : &st.bKs); barra_zc_cpu_end(full ? &st.bVf : &st.bVs); }
    st.layers++; double te = now_ms_battn(); st.ms_total += te - t0;
    if (st.log) BALOG("barra-attn: L%d ty=%d T=%d ok %.2f ms [quant %.2f tpu-qkv %.2f gpu %.2f ctx+tpu-o %.2f rueck %.2f]\n",
                      r.L, ty, T, te - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3, te - t4);
    return true;
}

// ---------- Ausfuehrung eines Attention-Blocks (v2.1: Split-Packages + Zeilennorm) ----------
static bool battn_exec(const battn_range & r) {
    battn_state & st = g_battn;
    if (st.gemma) return battn_exec_gemma(r);
    const int T = r.T, S = st.S, D = st.D, HHD = st.H * st.HD;
    const battn_layerq & Q = st.q[r.L];
    double t0 = now_ms_battn();
    const float * h = (const float *) battn_host_ptr(r.hnorm);
    float * odst = (float *) battn_host_ptr(r.wo);
    ggml_fp16_t * kdst = (ggml_fp16_t *) battn_host_ptr(r.setk);
    ggml_fp16_t * vdst = (ggml_fp16_t *) battn_host_ptr(r.setv);
    const int32_t * pos = (const int32_t *) battn_host_ptr(r.ropeq->src[1]);
    const int64_t * kids = (const int64_t *) battn_host_ptr(r.setk->src[1]);
    const int64_t * vids = (const int64_t *) battn_host_ptr(r.setv->src[1]);
    if (!h || !odst || !kdst || !vdst || !pos || !kids || !vids) { if (st.log) BALOG("barra-attn: L%d Hostzeiger fehlt\n", r.L); return false; }
    for (int t = 0; t < T; t++) if (pos[t] != t) { if (st.log) BALOG("barra-attn: L%d pos[%d]=%d != %d (n_past>0?) - Vulkan-Pfad\n", r.L, t, pos[t], t); return false; }
    // Dump-Vorbereitung: h VOR dem odst-Schreiben sichern (Allocator kann dst auf h legen)
    static const char * ddir = getenv("BARRA_ATTN_DUMP");
    static int dL = getenv("BARRA_ATTN_DUMP_L") ? atoi(getenv("BARRA_ATTN_DUMP_L")) : 0;
    static bool dumped = false;
    std::vector<float> hdump;
    const bool dump_now = ddir && !dumped && r.L == dL;
    if (dump_now) { hdump.resize((size_t) T * D);
        for (int t = 0; t < T; t++) memcpy(&hdump[(size_t) t * D], (const char *) h + (size_t) t * r.hnorm->nb[1], D * sizeof(float)); }
    // h -> geklemmte Zeilennorm + int8/int16 (q/k/v teilen den Eingang; Skalen aus identischer Kalibrierung)
    const int QMAXA = st.bits == 16 ? 32767 : 127;
    barra_zc_cpu_begin(&st.bX); barra_zc_cpu_begin(&st.bSrow);
    float * srow = (float *) st.bSrow.map;
    for (int t = T; t < S; t++) srow[t] = 1.0f;
    if (st.bits == 16) {
        int16_t * zx = (int16_t *) st.bX.map;
        memset(zx + (size_t) T * D, 0, (size_t) (S - T) * D * 2);   // izp==0 (geprueft)
        battn_par(T, [&](int t) {
            const float * hr = h + (size_t) t * D; int16_t * zr = zx + (size_t) t * D;
            float mx = 0; for (int d = 0; d < D; d++) { float a = fabsf(hr[d]); if (a > mx) mx = a; }
            const float sr = std::max(1.0f, mx / (32767.0f * Q.q.isc));
            srow[t] = sr;
            const float k = 1.0f / (sr * Q.q.isc);
            for (int d = 0; d < D; d++) { int v = (int) lrintf(hr[d] * k); zr[d] = (int16_t) (v > 32767 ? 32767 : v < -32768 ? -32768 : v); }
        });
    } else {
        int8_t * zx = (int8_t *) st.bX.map;
        memset(zx, (int8_t) Q.q.izp, (size_t) S * D);
        battn_par(T, [&](int t) {
            const float * hr = h + (size_t) t * D; int8_t * zr = zx + (size_t) t * D;
            float mx = 0; for (int d = 0; d < D; d++) { float a = fabsf(hr[d]); if (a > mx) mx = a; }
            // geklemmte Zeilennorm (v2.2): normale Zeilen srow=1, Ausreisserzeilen komprimiert
            const float sr = std::max(1.0f, mx / (127.0f * Q.q.isc));
            srow[t] = sr;
            const float k = 1.0f / (sr * Q.q.isc);
            for (int d = 0; d < D; d++) { int v = (int) lrintf(hr[d] * k) + Q.q.izp; zr[d] = (int8_t) (v > 127 ? 127 : v < -128 ? -128 : v); }
        });
    }
    barra_zc_cpu_end(&st.bX); barra_zc_cpu_end(&st.bSrow);
    (void) QMAXA;
    double t1 = now_ms_battn();
    // Paket-IDs: aus pkglist.txt, wenn vorhanden (Teilmengen-Kits), sonst die alte Arithmetik.
    const uint32_t PNONE = 0xffffffffu;
    auto PID = [&](const std::vector<uint32_t> & tab, uint32_t fb) -> uint32_t {
        if (!st.haspkgmap) return fb;
        const uint32_t v = tab[r.L];
        return v;
    };
    // TPU q(/q2)/k/v. k/v-Pakete erwarten dieselbe Eingangsquantisierung (gleiche Kalibrierdaten).
    if (st.gen && st.qchunk) {
        if (barra_tpu_infer(&st.tpu, PID(st.pq, (uint32_t) (4 * r.L)), &st.bX, &st.bQ, 0) ||
            barra_tpu_infer(&st.tpu, PID(st.pq2, (uint32_t) (4 * r.L + 1)), &st.bX, &st.bQ2, 0) ||
            barra_tpu_infer(&st.tpu, PID(st.pkv, (uint32_t) (4 * r.L + 2)), &st.bX, &st.bK, 0)) {
            BALOG("barra-attn: L%d TPU q1/q2/kv fehlgeschlagen\n", r.L); return false; }
        if (st.hasgate) {   // Gate-Projektion aus demselben Eingang (gleiche Kalibrierdaten wie q)
            if (st.pg[r.L] == PNONE || st.pg2[r.L] == PNONE) {
                BALOG("barra-attn: L%d Gate-Pakete fehlen\n", r.L); return false; }
            if (barra_tpu_infer(&st.tpu, st.pg[r.L], &st.bX, &st.bG, 0) ||
                barra_tpu_infer(&st.tpu, st.pg2[r.L], &st.bX, &st.bG2, 0)) {
                BALOG("barra-attn: L%d TPU gate fehlgeschlagen\n", r.L); return false; }
        }
    } else if (st.glm || st.gen) {
        if (barra_tpu_infer(&st.tpu, PID(st.pq, (uint32_t) (3 * r.L)), &st.bX, &st.bQ, 0) ||
            barra_tpu_infer(&st.tpu, PID(st.pkv, (uint32_t) (3 * r.L + 1)), &st.bX, &st.bK, 0)) {
            BALOG("barra-attn: L%d TPU q/kv fehlgeschlagen\n", r.L); return false; }
    } else if (st.kvmerge) {
        if (barra_tpu_infer(&st.tpu, PID(st.pq, (uint32_t) (4 * r.L)), &st.bX, &st.bQ, 0) ||
            barra_tpu_infer(&st.tpu, PID(st.pq2, (uint32_t) (4 * r.L + 1)), &st.bX, &st.bQ2, 0) ||
            barra_tpu_infer(&st.tpu, PID(st.pkv, (uint32_t) (4 * r.L + 2)), &st.bX, &st.bK, 0)) {
            BALOG("barra-attn: L%d TPU q1/q2/kv fehlgeschlagen\n", r.L); return false; }
    } else if (st.qchunk) {
        if (barra_tpu_infer(&st.tpu, PID(st.pq, (uint32_t) (5 * r.L)), &st.bX, &st.bQ, 0) ||
            barra_tpu_infer(&st.tpu, PID(st.pq2, (uint32_t) (5 * r.L + 1)), &st.bX, &st.bQ2, 0) ||
            barra_tpu_infer(&st.tpu, PID(st.pkv, (uint32_t) (5 * r.L + 2)), &st.bX, &st.bK, 0) ||
            barra_tpu_infer(&st.tpu, PID(st.pv, (uint32_t) (5 * r.L + 3)), &st.bX, &st.bV, 0)) {
            BALOG("barra-attn: L%d TPU q1/q2/k/v fehlgeschlagen\n", r.L); return false; }
    } else {
        if (barra_tpu_infer(&st.tpu, PID(st.pq, (uint32_t) (4 * r.L)), &st.bX, &st.bQ, 0) ||
            barra_tpu_infer(&st.tpu, PID(st.pkv, (uint32_t) (4 * r.L + 1)), &st.bX, &st.bK, 0) ||
            barra_tpu_infer(&st.tpu, PID(st.pv, (uint32_t) (4 * r.L + 2)), &st.bX, &st.bV, 0)) {
            BALOG("barra-attn: L%d TPU q/k/v fehlgeschlagen\n", r.L); return false; }
    }
    double t2 = now_ms_battn();
    // GPU: rope -> qkt -> softmax -> av
    barra_zc_cpu_end(&st.bQ); barra_zc_cpu_end(&st.bK);
    if (!st.kvmerge) barra_zc_cpu_end(&st.bV);
    if (st.qchunk) barra_zc_cpu_end(&st.bQ2);
    barra_zc_cpu_end(&st.bQK); barra_zc_cpu_end(&st.bSc); barra_zc_cpu_end(&st.bAoF);
    VkCommandBufferBeginInfo bg = {}; bg.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(st.cb, &bg);
    VkMemoryBarrier mb = {}; mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    struct { float qosc, qozp, kosc, kozp, eps; uint32_t gqo, gko, tabo; } pr = {
        Q.q.osc, (float) Q.q.ozp, Q.k.osc, (float) Q.k.ozp, r.eps,
        (uint32_t) (r.L * st.HD), (uint32_t) (st.NL * st.HD + r.L * st.HD), (uint32_t) (2 * st.NL * st.HD) };
    struct { float q1osc, q1ozp, q2osc, q2ozp, kosc, kozp, eps; uint32_t gqo, gko, tabo; } pr4 = {
        Q.q.osc, (float) Q.q.ozp, Q.q2.osc, (float) Q.q2.ozp, Q.k.osc, (float) Q.k.ozp, r.eps,
        (uint32_t) (r.L * st.HD), (uint32_t) (st.NL * st.HD + r.L * st.HD), (uint32_t) (2 * st.NL * st.HD) };
    vkCmdBindPipeline(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.krp.pipe);
    vkCmdBindDescriptorSets(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.krp.pll, 0, 1, &st.krp.ds, 0, 0);
    if (st.gen || st.qchunk) vkCmdPushConstants(st.cb, st.krp.pll, VK_SHADER_STAGE_COMPUTE_BIT, 0, 40, &pr4);
    else                     vkCmdPushConstants(st.cb, st.krp.pll, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, &pr);
    vkCmdDispatch(st.cb, (uint32_t) (st.H + st.NKV), (uint32_t) S, 1);
    vkCmdPipelineBarrier(st.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
    float pq = 256.0f * r.scale;
    vkCmdBindPipeline(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.kqk.pipe);
    vkCmdBindDescriptorSets(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.kqk.pll, 0, 1, &st.kqk.ds, 0, 0);
    vkCmdPushConstants(st.cb, st.kqk.pll, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &pq);
    vkCmdDispatch(st.cb, (uint32_t) (S / 128), (uint32_t) (S / 128), (uint32_t) st.H);
    vkCmdPipelineBarrier(st.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
    uint32_t ps = (uint32_t) S;
    vkCmdBindPipeline(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.ksm.pipe);
    vkCmdBindDescriptorSets(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.ksm.pll, 0, 1, &st.ksm.ds, 0, 0);
    vkCmdPushConstants(st.cb, st.ksm.pll, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &ps);
    vkCmdDispatch(st.cb, (uint32_t) (st.H * S), 1, 1);
    vkCmdPipelineBarrier(st.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
    struct { float vosc, vozp; } pa = { Q.v.osc, (float) Q.v.ozp };
    vkCmdBindPipeline(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.kav.pipe);
    vkCmdBindDescriptorSets(st.cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.kav.pll, 0, 1, &st.kav.ds, 0, 0);
    vkCmdPushConstants(st.cb, st.kav.pll, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &pa);
    vkCmdDispatch(st.cb, 1, (uint32_t) (S / 128), (uint32_t) st.H);
    vkCmdPipelineBarrier(st.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
    vkEndCommandBuffer(st.cb);
    vkResetFences(st.dev, 1, &st.fen);
    VkSubmitInfo si = {}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &st.cb;
    if (vkQueueSubmit(st.queue, 1, &si, st.fen) != VK_SUCCESS) { BALOG("barra-attn: L%d GPU-Submit fehlgeschlagen\n", r.L); return false; }
    vkWaitForFences(st.dev, 1, &st.fen, VK_TRUE, UINT64_MAX);
    double t3 = now_ms_battn();
    // ctx (f16) -> geklemmte Zeilennorm + int8/int16 fuers o-Package
    barra_zc_cpu_begin(&st.bAoF); barra_zc_cpu_begin(&st.bAo);
#if defined(__aarch64__)
    const __fp16 * af = (const __fp16 *) st.bAoF.map;
    if (st.bits == 16) {
        int16_t * zao = (int16_t *) st.bAo.map;
        memset(zao + (size_t) T * HHD, 0, (size_t) (S - T) * HHD * 2);
        battn_par(T, [&](int t) {
            const __fp16 * ar = af + (size_t) t * HHD; int16_t * zr = zao + (size_t) t * HHD;
            float mx = 0; for (int d = 0; d < HHD; d++) { float a = fabsf((float) ar[d]); if (a > mx) mx = a; }
            const float sr = std::max(1.0f, mx / (32767.0f * Q.o.isc));
            st.srow_c[t] = sr;
            const float k = 1.0f / (sr * Q.o.isc);
            for (int d = 0; d < HHD; d++) { int v = (int) lrintf((float) ar[d] * k); zr[d] = (int16_t) (v > 32767 ? 32767 : v < -32768 ? -32768 : v); }
        });
    } else {
        int8_t * zao = (int8_t *) st.bAo.map;
        memset(zao + (size_t) T * HHD, (int8_t) Q.o.izp, (size_t) (S - T) * HHD);
        battn_par(T, [&](int t) {
            const __fp16 * ar = af + (size_t) t * HHD; int8_t * zr = zao + (size_t) t * HHD;
            float mx = 0; for (int d = 0; d < HHD; d++) { float a = fabsf((float) ar[d]); if (a > mx) mx = a; }
            const float sr = std::max(1.0f, mx / (127.0f * Q.o.isc));
            st.srow_c[t] = sr;
            const float k = 1.0f / (sr * Q.o.isc);
            for (int d = 0; d < HHD; d++) { int v = (int) lrintf((float) ar[d] * k) + Q.o.izp; zr[d] = (int8_t) (v > 127 ? 127 : v < -128 ? -128 : v); }
        });
    }
#else
#error "battn: aarch64 only"
#endif
    barra_zc_cpu_end(&st.bAoF); barra_zc_cpu_end(&st.bAo);
    if (st.hasgate) { barra_zc_cpu_end(&st.bG); barra_zc_cpu_end(&st.bG2); }
    // TPU o (letztes Modell des Layers)
    const uint32_t omid = st.haspkgmap ? st.po[r.L] : (st.gen && st.qchunk) ? (uint32_t) (4 * r.L + 3) : ((st.glm || st.gen) ? (uint32_t) (3 * r.L + 2) : (st.kvmerge ? (uint32_t) (4 * r.L + 3) : (st.qchunk ? (uint32_t) (5 * r.L + 4) : (uint32_t) (4 * r.L + 3))));
    if (barra_tpu_infer(&st.tpu, omid, &st.bAo, &st.bO, 0)) { BALOG("barra-attn: L%d TPU o fehlgeschlagen\n", r.L); return false; }
    double t4 = now_ms_battn();
    // o zurueckschreiben (per-Zeilen-Skala), KV-Handoff
    barra_zc_cpu_begin(&st.bO); barra_zc_cpu_begin(&st.bQK);
    barra_zc_cpu_begin(st.kvmerge ? &st.bK : &st.bV);
    const size_t onb1 = r.wo->nb[1];
    const int rowe = st.NKV * st.HD;
    const ggml_fp16_t * ksrc = (const ggml_fp16_t *) st.bQK.map + (size_t) S * st.H * st.HD;
    const size_t knb1 = r.setk->nb[1], vnb1 = r.setv->nb[1];
    { const int64_t krows = r.setk->ne[1], vrows = r.setv->ne[1];   // ids vor dem ersten memcpy pruefen
      for (int t = 0; t < T; t++) if (kids[t] < 0 || kids[t] >= krows || vids[t] < 0 || vids[t] >= vrows) {
          if (st.log) BALOG("barra-attn: L%d SET_ROWS id ausser Bereich (t=%d) — Vulkan-Fallback\n", r.L, t);
          barra_zc_cpu_end(&st.bO); barra_zc_cpu_end(&st.bQK); barra_zc_cpu_end(st.kvmerge ? &st.bK : &st.bV);
          return false;
      } }
    battn_par(T, [&](int t) {
        float * orow = (float *) ((char *) odst + (size_t) t * onb1);
        const float os = Q.o.osc * st.srow_c[t];
        if (st.bits == 16) {
            const int16_t * zr = (const int16_t *) st.bO.map + (size_t) t * D;
            for (int d = 0; d < D; d++) orow[d] = (zr[d] - Q.o.ozp) * os;
        } else {
            const int8_t * zr = (const int8_t *) st.bO.map + (size_t) t * D;
            for (int d = 0; d < D; d++) orow[d] = (zr[d] - Q.o.ozp) * os;
        }
        memcpy((char *) kdst + (size_t) kids[t] * knb1, ksrc + (size_t) t * rowe, (size_t) rowe * sizeof(ggml_fp16_t));
        ggml_fp16_t * vrow = (ggml_fp16_t *) ((char *) vdst + (size_t) vids[t] * vnb1);
        const float vs = Q.v.osc * srow[t];
        if (st.bits == 16) {
            const int16_t * v16 = st.kvmerge ? (const int16_t *) st.bK.map + (size_t) t * 2 * rowe + rowe
                                             : (const int16_t *) st.bV.map + (size_t) t * rowe;
            for (int d = 0; d < rowe; d++) { __fp16 hv = (__fp16) ((v16[d] - Q.v.ozp) * vs); memcpy(&vrow[d], &hv, 2); }
        } else {
            const int8_t * v8 = (const int8_t *) st.bV.map + (size_t) t * rowe;
            for (int d = 0; d < rowe; d++) { __fp16 hv = (__fp16) ((v8[d] - Q.v.ozp) * vs); memcpy(&vrow[d], &hv, 2); }
        }
    });
    st.layers++; double te = now_ms_battn(); st.ms_total += te - t0;
    if (st.log) BALOG("barra-attn: L%d T=%d ok %.2f ms [quant %.2f tpu-qkv %.2f gpu %.2f ctx+tpu-o %.2f rueck %.2f] (kum %ld Layer)\n",
                      r.L, T, te - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3, te - t4, st.layers);
    // Stufen-Dump fuer die Fehlersuche: BARRA_ATTN_DUMP=<dir> (+ optional BARRA_ATTN_DUMP_L, Default 0)
    if (dump_now) {
        dumped = true;
        auto wr = [&](const char * name, const void * p, size_t n) {
            char pa[512]; snprintf(pa, 512, "%s/dump_%s", ddir, name);
            FILE * fo = fopen(pa, "wb"); if (fo) { fwrite(p, 1, n, fo); fclose(fo); }
        };
        wr("h_f32.bin", hdump.data(), hdump.size() * 4);
        wr("srow_f32.bin", srow, (size_t) S * 4);
        wr("srowc_f32.bin", st.srow_c.data(), (size_t) S * 4);
        const size_t dbpe = st.bits / 8;
        wr("x_i8.bin", st.bX.map, (size_t) S * D * dbpe);
        barra_zc_cpu_begin(&st.bQ); barra_zc_cpu_begin(&st.bK);
        const size_t qelem = st.qchunk ? (size_t) HHD / 2 : (size_t) HHD;   // bQ ist bei qchunk nur die halbe q-Breite
        wr("q_i8.bin", st.bQ.map, (size_t) S * qelem * dbpe);
        if (st.qchunk) { barra_zc_cpu_begin(&st.bQ2); wr("q2_i8.bin", st.bQ2.map, (size_t) S * qelem * dbpe); }
        if (st.kvmerge) {   // bK haelt [k|v]-Zeilen -> fuer die Dumps de-interleaven
            const size_t re = (size_t) st.NKV * st.HD;
            std::vector<int16_t> tk((size_t) S * re), tv((size_t) S * re);
            const int16_t * kvm = (const int16_t *) st.bK.map;
            for (int t = 0; t < S; t++) {
                memcpy(&tk[(size_t) t * re], kvm + (size_t) t * 2 * re, re * 2);
                memcpy(&tv[(size_t) t * re], kvm + (size_t) t * 2 * re + re, re * 2);
            }
            wr("k_i8.bin", tk.data(), tk.size() * 2);
            wr("v_i8.bin", tv.data(), tv.size() * 2);
        } else {
            wr("k_i8.bin", st.bK.map, (size_t) S * st.NKV * st.HD * dbpe);
            wr("v_i8.bin", st.bV.map, (size_t) S * st.NKV * st.HD * dbpe);
        }
        wr("qk_f16.bin", st.bQK.map, (size_t) (S * HHD + S * st.NKV * st.HD) * 2);
        wr("aof_f16.bin", st.bAoF.map, (size_t) S * HHD * 2);
        wr("ao_i8.bin", st.bAo.map, (size_t) S * HHD * dbpe);
        wr("o_i8.bin", st.bO.map, (size_t) S * D * dbpe);
        std::vector<float> oc((size_t) T * D);
        for (int t = 0; t < T; t++) memcpy(&oc[(size_t) t * D], (const char *) odst + (size_t) t * r.wo->nb[1], D * sizeof(float));
        wr("odst_f32.bin", oc.data(), oc.size() * 4);
        BALOG("barra-attn: DUMP L%d -> %s\n", r.L, ddir);
    }
    return true;
}

// ---------- Kalibrier-Dump (BARRA_ATTN_CALDUMP=<dir>) ----------
// Vulkan rechnet normal; wir dumpen je Attention-Layer h_norm [T,D] und ctx (FLASH_ATTN-dst,
// [T,H*HD]) als f32 fuer die Package-Kalibrierung auf dem PC. Braucht weder Meta noch tpud —
// toleranter Scan (attn_output-MUL_MAT -> FLASH_ATTN -> rueckwaerts bis attn_q-MUL_MAT,
// beliebige ROPE/ADD/MUL/RMS_NORM dazwischen). Nur der erste Batch mit T>=32 je Layer.
struct battn_calrange { int start, end, L; ggml_tensor *hnorm, *fa; };
static void battn_cal_scan(ggml_cgraph * cg, std::vector<battn_calrange> & out) {
    std::map<ggml_tensor *, int> idx;
    for (int i = 0; i < cg->n_nodes; i++) idx[cg->nodes[i]] = i;
    for (int i = 0; i < cg->n_nodes; i++) {
        ggml_tensor * wo = cg->nodes[i];
        if (wo->op != GGML_OP_MUL_MAT) continue;
        int L = battn_wname(wo->src[0], "output");
        if (L < 0) continue;
        const bool caldbg = getenv("BARRA_CAL_DBG") != nullptr;
        ggml_tensor * fa = battn_skipv(wo->src[1], nullptr);
        if (caldbg) BALOG("cal-scan: L%d o-proj, src1 op=%d name=%s\n", L, fa ? (int) fa->op : -1, fa ? fa->name : "(null)");
        // Gegatete Attention (qwen3.5/arch qwen35, src/models/qwen35.cpp): zwischen
        // FLASH_ATTN_EXT und der o-Projektion steht ggml_mul(attn, sigmoid(gate)).
        // Ohne diesen Schritt bricht die Kette und der Scanner findet NIE einen
        // FA-Knoten - am 29.8.2026 an qwen38-4b-distill gemessen: 0 Dump-Dateien
        // statt 8, waehrend qwen3-4b (ungegatet) 72 lieferte.
        // ctx_t bleibt der GEGATETE Tensor: genau der ist der Eingang der
        // o-Projektion, also das, worauf das o-Package kalibriert werden muss.
        ggml_tensor * ctx_t = fa;
        if (fa && fa->op == GGML_OP_MUL) {
            ggml_tensor * a0 = battn_skipv(fa->src[0], nullptr);
            if (a0 && a0->op == GGML_OP_FLASH_ATTN_EXT) fa = a0;
        }
        if (!fa || fa->op != GGML_OP_FLASH_ATTN_EXT || fa->type != GGML_TYPE_F32) {
            if (caldbg) BALOG("cal-scan: L%d KEIN FA (op=%d typ=%d)\n", L, fa ? (int) fa->op : -1, fa ? (int) fa->type : -1);
            continue;
        }
        ggml_tensor * t = battn_skipv(fa->src[0], nullptr);
        int guard = 0;
        while (t && t->op != GGML_OP_MUL_MAT && guard++ < 8 &&
               (t->op == GGML_OP_ROPE || t->op == GGML_OP_ADD || t->op == GGML_OP_MUL || t->op == GGML_OP_RMS_NORM))
            t = battn_skipv(t->src[0], nullptr);
        if (!t || t->op != GGML_OP_MUL_MAT || battn_wname(t->src[0], "q") != L) {
            if (caldbg) BALOG("cal-scan: L%d q-Weg bricht (op=%d name=%s)\n", L, t ? (int) t->op : -1, t ? t->name : "(null)");
            continue;
        }
        if (t->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(t->src[1])) continue;
        auto iw = idx.find(t); auto ifa = idx.find(ctx_t);   // Ende = GEGATETER Tensor
        if (iw == idx.end() || ifa == idx.end() || ifa->second < iw->second) continue;
        battn_calrange r; r.start = iw->second; r.end = ifa->second; r.L = L; r.hnorm = t->src[1]; r.fa = ctx_t;
        out.push_back(r);
        // KEIN Index-Ruecksetzer: fa liegt VOR wo — ein Ruecksprung war die Endlosschleife,
        // die am 23.8. ZWEI OOM-Kernel-Panics ausgeloest hat. Die wo-Schleife laeuft einfach weiter.
        if ((int) out.size() > 512) { BALOG("barra-attn: CALDUMP-Scan ausser Kontrolle (>512 Ranges) - abgebrochen\n"); out.clear(); return; }
    }
}
static ggml_status battn_caldump_run(const char * cdir, ggml_backend_t backend, ggml_cgraph * cgraph) {
    std::vector<battn_calrange> rs;
    battn_cal_scan(cgraph, rs);
    if (rs.empty()) return ggml_backend_vk_graph_compute_impl(backend, cgraph);
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *) backend->context;
    static std::set<int> done;
    int cur = 0;
    for (const battn_calrange & r : rs) {
        if (r.start > cur) {
            ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + cur; sub.n_nodes = r.start - cur;
            ggml_status stt = ggml_backend_vk_graph_compute_impl(backend, &sub);
            if (stt != GGML_STATUS_SUCCESS) return stt;
        }
        ggml_vk_synchronize(ctx);
        const int T = (int) r.hnorm->ne[1], D = (int) r.hnorm->ne[0];
        const bool want = T >= 32 && !done.count(r.L);
        std::vector<float> h;
        if (want) {
            const char * hp = (const char *) battn_host_ptr(r.hnorm);
            if (hp) { h.resize((size_t) T * D);
                for (int t = 0; t < T; t++) memcpy(&h[(size_t) t * D], hp + (size_t) t * r.hnorm->nb[1], (size_t) D * 4); }
        }
        { ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + r.start; sub.n_nodes = r.end - r.start + 1;
          ggml_status stt = ggml_backend_vk_graph_compute_impl(backend, &sub);
          if (stt != GGML_STATUS_SUCCESS) return stt; }
        ggml_vk_synchronize(ctx);
        if (want && !h.empty()) {
            // Zwei Geometrien: ohne Gate zeigt r.fa auf den FLASH_ATTN_EXT-Knoten [HD,H,T];
            // bei gegateter Attention (qwen3.5) auf den gegateten Tensor, den build_attn
            // schon als 2D [D,T] zurueckgibt. Beide liefern dieselben D Werte je Token -
            // genau das, worauf das o-Package kalibriert wird. Die 2D-Form fehlte, deshalb
            // wurde am 29.8.2026 trotz gefundener Ranges nichts geschrieben.
            const bool three_d = (int) r.fa->ne[2] == T;
            const int HD_ = three_d ? (int) r.fa->ne[0] : 0;
            const int H_  = three_d ? (int) r.fa->ne[1] : 0;
            const int D2  = three_d ? HD_ * H_ : (int) r.fa->ne[0];
            const char * fp = (const char *) battn_host_ptr(r.fa);
            if (fp && (three_d || ((int) r.fa->ne[1] == T && r.fa->ne[2] == 1))) {
                std::vector<float> cx((size_t) T * D2);
                if (three_d) {
                    for (int t = 0; t < T; t++) for (int hh = 0; hh < H_; hh++)
                        memcpy(&cx[((size_t) t * H_ + hh) * HD_], fp + t * r.fa->nb[2] + hh * r.fa->nb[1], (size_t) HD_ * 4);
                } else {
                    for (int t = 0; t < T; t++)
                        memcpy(&cx[(size_t) t * D2], fp + (size_t) t * r.fa->nb[1], (size_t) D2 * 4);
                }
                char pa[512];
                snprintf(pa, 512, "%s/cal_h_L%d.bin", cdir, r.L);
                FILE * f = fopen(pa, "wb"); if (f) { fwrite(h.data(), 4, h.size(), f); fclose(f); }
                snprintf(pa, 512, "%s/cal_ctx_L%d.bin", cdir, r.L);
                f = fopen(pa, "wb"); if (f) { fwrite(cx.data(), 4, cx.size(), f); fclose(f); }
                done.insert(r.L);
                BALOG("barra-attn: CALDUMP L%d T=%d D=%d H=%d HD=%d D2=%d -> %s\n", r.L, T, D, H_, HD_, D2, cdir);
            }
        }
        cur = r.end + 1;
    }
    if (cur < cgraph->n_nodes) {
        ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + cur; sub.n_nodes = cgraph->n_nodes - cur;
        return ggml_backend_vk_graph_compute_impl(backend, &sub);
    }
    return GGML_STATUS_SUCCESS;
}

// ---------- FFN-Offload: Bereichs-Scanner ----------
// Der Ablaufplaner kann uns die FFN nicht geben, wenn die Gewichte in Vulkan-Puffern liegen
// (29.8.2026 gemessen: separates ggml-Backend bekommt 0 Aufrufe bei -ngl 99). Also derselbe
// Weg wie bei der Attention: den Bereich im Graphen erkennen und selbst rechnen.
// Form je Layer:  up = MUL_MAT(ffn_up, h) ; gate = MUL_MAT(ffn_gate, h)
//                 act = GLU(gate, up) oder SILU(gate)*up ; down = MUL_MAT(ffn_down, act)
struct bffn_range { int start, end, L, T; ggml_tensor * h; ggml_tensor * down; };
static int bffn_kind(const ggml_tensor * w, int * layer) {   // 0=up 1=gate 2=down
    if (!w) return -1;
    int L = -1; char k[24] = { 0 };
    if (sscanf(w->name, "blk.%d.ffn_%23[a-z_].weight", &L, k) != 2) return -1;
    if (layer) *layer = L;
    if (!strcmp(k, "up")) return 0;
    if (!strcmp(k, "gate")) return 1;
    if (!strcmp(k, "down")) return 2;
    return -1;
}
static void bffn_scan(ggml_cgraph * cg, std::vector<bffn_range> & out) {
    const bool dbg = getenv("BARRA_FFN_SCAN_DBG") != nullptr;
    std::map<ggml_tensor *, int> idx;
    for (int i = 0; i < cg->n_nodes; i++) idx[cg->nodes[i]] = i;
    for (int i = 0; i < cg->n_nodes; i++) {
        ggml_tensor * dn = cg->nodes[i];
        if (dn->op != GGML_OP_MUL_MAT) continue;
        int L = -1;
        if (bffn_kind(dn->src[0], &L) != 2) continue;   // an der down-Projektion aufhaengen
        std::set<ggml_tensor *> claim; claim.insert(dn);
        ggml_tensor * act = battn_skipv(dn->src[1], &claim);
        if (!act) continue;
        ggml_tensor * g = nullptr, * u = nullptr;
        if (act->op == GGML_OP_GLU) { claim.insert(act);
            g = battn_skipv(act->src[0], &claim); u = act->src[1] ? battn_skipv(act->src[1], &claim) : nullptr; }
        else if (act->op == GGML_OP_MUL) { claim.insert(act);
            ggml_tensor * a0 = battn_skipv(act->src[0], &claim);
            ggml_tensor * a1 = battn_skipv(act->src[1], &claim);
            if (a0 && a0->op == GGML_OP_UNARY) { claim.insert(a0); g = battn_skipv(a0->src[0], &claim); u = a1; }
            else if (a1 && a1->op == GGML_OP_UNARY) { claim.insert(a1); g = battn_skipv(a1->src[0], &claim); u = a0; }
        }
        if (!g || !u) { if (dbg) BALOG("bffn: L%d act-Form unbekannt (op=%d)\n", L, (int) act->op); continue; }
        int Lg = -1, Lu = -1;
        if (g->op != GGML_OP_MUL_MAT || bffn_kind(g->src[0], &Lg) != 1 || Lg != L) { if (dbg) BALOG("bffn: L%d gate passt nicht\n", L); continue; }
        if (u->op != GGML_OP_MUL_MAT || bffn_kind(u->src[0], &Lu) != 0 || Lu != L) { if (dbg) BALOG("bffn: L%d up passt nicht\n", L); continue; }
        claim.insert(g); claim.insert(u);
        if (g->src[1] != u->src[1]) { if (dbg) BALOG("bffn: L%d gate/up haben verschiedene Eingaenge\n", L); continue; }
        ggml_tensor * h = g->src[1];
        if (h->type != GGML_TYPE_F32 || !ggml_is_contiguous(h) || h->ne[2] != 1) continue;
        const int T = (int) h->ne[1];
        int lo = INT32_MAX, hi = -1;
        for (ggml_tensor * t : claim) { auto it = idx.find(t); if (it != idx.end()) { lo = std::min(lo, it->second); hi = std::max(hi, it->second); } }
        if (hi < 0) continue;
        bool clean = true;
        for (int j = lo; j <= hi && clean; j++) {
            ggml_tensor * n = cg->nodes[j];
            if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
            if (n->op == GGML_OP_NONE || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE) continue;
            if (!claim.count(n)) clean = false;
        }
        if (!clean) { if (dbg) BALOG("bffn: L%d Bereich %d..%d nicht sauber\n", L, lo, hi); continue; }
        bffn_range r; r.start = lo; r.end = hi; r.L = L; r.T = T; r.h = h; r.down = dn;
        out.push_back(r);
        if (dbg) BALOG("bffn: L%d Bereich %d..%d T=%d OK\n", L, lo, hi, T);
        i = hi;
    }
}

// FFN-Eingaenge abnehmen (BARRA_FFN_CALDUMP=<dir>): je Layer der Tensor, der in
// blk.N.ffn_gate.weight hineingeht - also das, worauf ein FFN-Package kalibriert werden
// muss. Der alte Weg (numpy-Nachbau des ganzen Modells in gen_ffn_pkgs.py) scheitert bei
// hybriden Modellen, weil er die DeltaNet-Schichten nachbauen muesste. (29.8.2026)
static void battn_ffn_caldump(const char * cdir, ggml_backend_t backend, ggml_cgraph * cgraph) {
    static std::set<int> done;
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *) backend->context;
    ggml_vk_synchronize(ctx);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_MUL_MAT || !n->src[0] || !n->src[1]) continue;
        int L = -1; char kind[24] = { 0 };
        if (sscanf(n->src[0]->name, "blk.%d.ffn_%23[a-z_].weight", &L, kind) != 2) continue;
        if (strcmp(kind, "gate") != 0 || done.count(L)) continue;
        ggml_tensor * h = n->src[1];
        if (h->type != GGML_TYPE_F32 || h->ne[2] != 1) continue;
        const int T = (int) h->ne[1], D = (int) h->ne[0];
        if (T < 32) continue;
        const char * hp = (const char *) battn_host_ptr(h);
        if (!hp) continue;
        std::vector<float> buf((size_t) T * D);
        for (int t = 0; t < T; t++) memcpy(&buf[(size_t) t * D], hp + (size_t) t * h->nb[1], (size_t) D * 4);
        char pa[512]; snprintf(pa, sizeof(pa), "%s/cal_ffn_L%d.bin", cdir, L);
        FILE * f = fopen(pa, "wb"); if (f) { fwrite(buf.data(), 4, buf.size(), f); fclose(f); done.insert(L);
            BALOG("barra-attn: FFN-CALDUMP L%d T=%d D=%d -> %s\n", L, T, D, cdir); }
    }
}

// ---------- FFN-Offload: Zustand und Ausfuehrung ----------
// Ein Paket je Layer rechnet f(h) = Wd(silu(Wg h) * (Wu h)) fuer B Zeilen. Der Prefill wird
// in Kacheln zu B Zeilen zerlegt. Quantisierung mit der festen Kalibrierskala aus ffn.meta.
struct bffn_state {
    int env = -1; bool ready = false, open = false;
    int D = 0, FF = 0, B = 0, bits = 0, NL = 0;
    std::vector<battn_qgrp> q;
    barra_tpu tpu{}; barra_zbuf bH{}, bY{}, bH2{}, bY2{};   // zwei Saetze: waehrend die TPU
                                                            // Kachel i rechnet, bereitet die CPU i+1 vor
    long long tiles = 0; double ms_tpu = 0, ms_glue = 0;
};
static bffn_state g_bffn;
static bool bffn_init() {
    bffn_state & s = g_bffn;
    if (s.env >= 0) return s.ready;
    s.env = 0;
    const char * mp = getenv("BARRA_FFN_META"); if (!mp) return false;
    FILE * f = fopen(mp, "r"); if (!f) { BALOG("bffn: meta %s nicht lesbar\n", mp); return false; }
    // ZEILENWEISE lesen: der Parser im alten Backend liest den Kopfrest mit fgets und frisst
    // dabei die erste Datenzeile, wenn die optionalen Felder fehlen (29.8.2026 getreten).
    char line[512];
    if (!fgets(line, sizeof line, f) || sscanf(line, "%d %d %d %d %d", &s.D, &s.FF, &s.B, &s.bits, &s.NL) != 5) {
        fclose(f); BALOG("bffn: meta-Kopf ungueltig\n"); return false; }
    if (s.D < 1 || s.D > 65536 || s.FF < 1 || s.FF > 262144 || s.B < 1 || s.B > 4096 || s.NL < 1 || s.NL > 512 || s.bits != 16) {
        fclose(f); BALOG("bffn: meta-Kopf unplausibel (D=%d FF=%d B=%d bits=%d NL=%d)\n", s.D, s.FF, s.B, s.bits, s.NL); return false; }
    s.q.resize(s.NL);
    for (int i = 0; i < s.NL; i++) {
        if (!fgets(line, sizeof line, f) || sscanf(line, "%f %d %f %d", &s.q[i].isc, &s.q[i].izp, &s.q[i].osc, &s.q[i].ozp) != 4) {
            fclose(f); BALOG("bffn: meta Layer %d ungueltig\n", i); return false; } }
    fclose(f);
    if (barra_tpu_open(&s.tpu)) { BALOG("bffn: tpud nicht erreichbar (BARRA_SOCK_DIR?)\n"); return false; }
    s.open = true;
    const uint32_t sz = (uint32_t) s.B * (uint32_t) s.D * 2;
    if (barra_zc_alloc(&s.bH, sz) || barra_zc_alloc(&s.bY, sz) ||
        barra_zc_alloc(&s.bH2, sz) || barra_zc_alloc(&s.bY2, sz)) { BALOG("bffn: Puffer fehlgeschlagen\n"); return false; }
    s.ready = true;
    BALOG("bffn: AKTIV D=%d FF=%d B=%d bits=%d NL=%d\n", s.D, s.FF, s.B, s.bits, s.NL);
    return true;
}
// Fliessband: submit(i) -> waehrenddessen Kachel i+1 quantisieren -> wait(i) -> dequantisieren.
// barra erlaubt genau EINE offene Abgabe je Verbindung, und zwischen submit und wait duerfen
// die uebergebenen Puffer nicht angefasst werden - daher zwei Puffersaetze im Wechsel.
// hsrc/ydst: private Puffer (Zeilen dicht, Schrittweite D). Werden sie uebergeben, fasst
// diese Funktion die Vulkan-Tensoren NICHT an - noetig bei Zeilenaufteilung, weil der
// Allocator Ein- und Ausgang versetzt uebereinanderlegen darf (29.8.2026 gemessen).
static bool bffn_exec(const bffn_range & r, int row0 = 0, int row1 = -1,
                      const float * hsrc = nullptr, float * ydst = nullptr) {
    bffn_state & s = g_bffn;
    if (!s.ready || r.L < 0 || r.L >= s.NL) return false;
    const int D = s.D, B = s.B;
    if (row1 < 0) row1 = r.T;
    const int T = row1 - row0;   // nur dieser Zeilenausschnitt geht ueber die TPU
    if (T <= 0) return true;
    if ((int) r.h->ne[0] != D || (int) r.down->ne[0] != D) return false;
    const char * hp = hsrc ? nullptr : (const char *) battn_host_ptr(r.h);
    char * yp = ydst ? nullptr : (char *) battn_host_ptr(r.down);
    if ((!hsrc && !hp) || (!ydst && !yp)) return false;
    const size_t hstride = hsrc ? (size_t) D * 4 : r.h->nb[1];
    const size_t ystride = ydst ? (size_t) D * 4 : r.down->nb[1];
    const char * hbase = hsrc ? (const char *) hsrc : hp;
    char * ybase = ydst ? (char *) ydst : yp;
    const int hoff = hsrc ? 0 : row0;   // private Puffer beginnen bei Zeile 0
    const int yoff = ydst ? 0 : row0;
    const battn_qgrp & Qz = s.q[r.L];
    const float ki = 1.0f / Qz.isc;
    barra_zbuf * IN[2] = { &s.bH, &s.bH2 };
    barra_zbuf * OUT[2] = { &s.bY, &s.bY2 };
    auto fill = [&](int slot, int t0, int nrow) {
        barra_zc_cpu_begin(IN[slot]);
        int16_t * zi = (int16_t *) IN[slot]->map;
        battn_par(nrow, [&](int t) {
            const float * hr = (const float *) (hbase + (size_t) (hoff + t0 + t) * hstride);
            int16_t * zr = zi + (size_t) t * D;
            for (int d = 0; d < D; d++) { int v = (int) lrintf(hr[d] * ki) + Qz.izp;
                zr[d] = (int16_t) (v > 32767 ? 32767 : v < -32768 ? -32768 : v); } });
        if (nrow < B) memset(zi + (size_t) nrow * D, 0, (size_t) (B - nrow) * D * 2);
        barra_zc_cpu_end(IN[slot]);
    };
    auto drain = [&](int slot, int t0, int nrow) {
        barra_zc_cpu_begin(OUT[slot]);
        const int16_t * zo = (const int16_t *) OUT[slot]->map;
        battn_par(nrow, [&](int t) {
            const int16_t * zr = zo + (size_t) t * D;
            float * yr = (float *) (ybase + (size_t) (yoff + t0 + t) * ystride);
            for (int d = 0; d < D; d++) yr[d] = ((float) zr[d] - Qz.ozp) * Qz.osc; });
        barra_zc_cpu_end(OUT[slot]);
    };
    const int ntile = (T + B - 1) / B;
    int slot = 0;
    double t_start = now_ms_battn();
    fill(slot, 0, (B <= T) ? B : T);
    if (barra_tpu_submit(&s.tpu, (uint32_t) r.L, IN[slot], OUT[slot])) {
        BALOG("bffn: L%d submit fehlgeschlagen\n", r.L); return false; }
    for (int i = 0; i < ntile; i++) {
        const int t0 = i * B, nrow = (t0 + B <= T) ? B : (T - t0);
        const int nxt = (slot + 1) & 1;
        const int t0n = (i + 1) * B, nrown = (t0n + B <= T) ? B : (T - t0n);
        if (i + 1 < ntile) fill(nxt, t0n, nrown);   // CPU-Arbeit WAEHREND die TPU rechnet
        uint32_t us = 0;
        if (barra_tpu_wait(&s.tpu, IN[slot], OUT[slot], &us)) {
            BALOG("bffn: L%d wait fehlgeschlagen\n", r.L); return false; }
        if (i + 1 < ntile && barra_tpu_submit(&s.tpu, (uint32_t) r.L, IN[nxt], OUT[nxt])) {
            BALOG("bffn: L%d submit(%d) fehlgeschlagen\n", r.L, i + 1); return false; }
        drain(slot, t0, nrow);
        s.tiles++; s.ms_tpu += us / 1000.0;
        slot = nxt;
    }
    s.ms_glue += (now_ms_battn() - t_start) - 0.0;
    return true;
}

// ---------- Wrapper um graph_compute ----------
static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // Diagnoselauf des FFN-Scanners: nur schauen, nichts rechnen (BARRA_FFN_SCAN_DBG=1).
    if (getenv("BARRA_FFN_SCAN_DBG")) {
        static int shown = 0;
        if (shown < 2) { shown++;
            std::vector<bffn_range> fr; bffn_scan(cgraph, fr);
            BALOG("bffn: Graph mit %d Knoten -> %d FFN-Bereiche\n", cgraph->n_nodes, (int) fr.size());
        }
    }
    static const char * fdir = getenv("BARRA_FFN_CALDUMP");
    if (fdir) {   // erst rechnen lassen, dann die FFN-Eingaenge abnehmen
        ggml_status st_ = ggml_backend_vk_graph_compute_impl(backend, cgraph);
        if (st_ == GGML_STATUS_SUCCESS) battn_ffn_caldump(fdir, backend, cgraph);
        return st_;
    }
    // FFN-Offload: Bereiche selbst rechnen, den Rest des Graphen wie gehabt. Rueckfall
    // je Bereich, falls etwas nicht passt - dann rechnet Vulkan ihn normal.
    if (bffn_init()) {
        static int fmin = -1;
        if (fmin < 0) { const char * e = getenv("BARRA_FFN_MIN_BATCH"); fmin = e ? atoi(e) : g_bffn.B; }
        std::vector<bffn_range> fr; bffn_scan(cgraph, fr);
        std::vector<bffn_range> use;
        for (const bffn_range & r : fr) if (r.T >= fmin && r.L < g_bffn.NL) use.push_back(r);
        if (!use.empty()) {
            ggml_backend_vk_context * vctx = (ggml_backend_vk_context *) backend->context;
            int cur = 0;
            for (const bffn_range & r : use) {
                if (r.start > cur) {
                    ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + cur; sub.n_nodes = r.start - cur;
                    ggml_status st_ = ggml_backend_vk_graph_compute_impl(backend, &sub);
                    if (st_ != GGML_STATUS_SUCCESS) return st_;
                }
                ggml_vk_synchronize(vctx);
                // Zeilenaufteilung: die FFN ist zeilenweise unabhaengig, also koennen GPU und TPU
                // DIESELBE Schicht gleichzeitig rechnen - verschiedene Token. Innerhalb eines
                // Graphen ist das die einzige echte Ueberlappung: die naechste Schicht haengt an
                // dieser FFN, es gibt sonst nichts Unabhaengiges. Vulkans graph_compute blockiert,
                // deshalb laeuft die TPU-Haelfte in einem eigenen Thread.
                // BARRA_FFN_SPLIT = Prozent der Zeilen fuer die TPU (Default 100 = alles TPU).
                static int spct = -1;
                if (spct < 0) { const char * e = getenv("BARRA_FFN_SPLIT"); spct = e ? atoi(e) : 100;
                    if (spct < 0) spct = 0; if (spct > 100) spct = 100; }
                bool done_ok = false;
                if (spct >= 100) {
                    done_ok = bffn_exec(r);
                } else {
                    int rows_tpu = (r.T * spct) / 100;
                    rows_tpu = (rows_tpu / g_bffn.B) * g_bffn.B;   // ganze Kacheln
                    const int rows_gpu = r.T - rows_tpu;
                    // Der Allocator darf den Ausgang auf den Eingang legen (dieselbe Warnung
                    // steht im CHECK-Pfad der Attention). Bei Zeilenaufteilung waere das fatal:
                    // die GPU wuerde beim Rechnen ihrer Haelfte die Eingaenge ueberschreiben,
                    // die der TPU-Thread parallel noch liest. Dann KEIN Split.
                    const char * hb = (const char *) battn_host_ptr(r.h);
                    const char * yb = (const char *) battn_host_ptr(r.down);
                    const size_t hsz = (size_t) r.T * r.h->nb[1], ysz = (size_t) r.T * r.down->nb[1];
                    const bool alias = hb && yb && (hb < yb + ysz) && (yb < hb + hsz);
                    if (alias && g_battn.log) BALOG("bffn: L%d Ein- und Ausgang ueberlappen - kein Split\n", r.L);
                    // Seit die TPU-Haelfte auf PRIVATEN Puffern rechnet, ist eine Ueberlappung
                    // von Ein- und Ausgang harmlos - die Pruefung von vorhin ist damit hinfaellig.
                    (void) alias;
                    if (rows_tpu <= 0 || rows_gpu <= 0) { done_ok = bffn_exec(r); }
                    else {
                        // Diagnose (BARRA_FFN_SPLIT_NOTPU=1): Zeilen kuerzen, TPU-Haelfte NICHT
                        // rechnen. Greift die Kuerzung, bleibt die zweite Haelfte stehen und die
                        // Perplexitaet muss kaputt sein. Kommt ein normales Ergebnis heraus, hat
                        // die GPU trotz gekuerztem ne[1] ALLE Zeilen gerechnet.
                        static int notpu = -1;
                        if (notpu < 0) notpu = getenv("BARRA_FFN_SPLIT_NOTPU") ? 1 : 0;
                        // Diagnose (BARRA_FFN_SPLIT_SERIAL=1): dieselbe Aufteilung, aber die
                        // TPU-Haelfte ERST NACH der GPU. Ist das Ergebnis dann richtig, war es
                        // ein Rennen um den Puffer, kein Rechenfehler.
                        // Eingangszeilen der TPU-Haelfte VOR jeder GPU-Arbeit wegkopieren und
                        // das Ergebnis erst NACH der GPU zurueckschreiben. Der Allocator darf
                        // Ein- und Ausgang versetzt uebereinanderlegen; gemessen am 29.8.2026:
                        // ohne diese Kopien verwarf die GPU die TPU-Zeilen komplett (PPL 533
                        // = derselbe Wert wie ganz ohne TPU) bzw. verfaelschte deren Eingang.
                        const int nrow_tpu = r.T - rows_gpu;
                        std::vector<float> hcopy((size_t) nrow_tpu * g_bffn.D);
                        std::vector<float> ycopy((size_t) nrow_tpu * g_bffn.D);
                        { const char * hb2 = (const char *) battn_host_ptr(r.h);
                          if (!hb2) { done_ok = false; }
                          else { for (int t = 0; t < nrow_tpu; t++)
                                   memcpy(&hcopy[(size_t) t * g_bffn.D], hb2 + (size_t) (rows_gpu + t) * r.h->nb[1], (size_t) g_bffn.D * 4); } }
                        std::atomic<int> tpu_ok{0};
                        std::thread th([&] { tpu_ok = bffn_exec(r, 0, nrow_tpu, hcopy.data(), ycopy.data()) ? 1 : -1; });
                        std::vector<std::pair<ggml_tensor *, int64_t>> saved;
                        for (int j2 = r.start; j2 <= r.end; j2++) {
                            ggml_tensor * n = cgraph->nodes[j2];
                            if (n->ne[1] == r.T) { saved.push_back({ n, n->ne[1] }); n->ne[1] = rows_gpu; }
                            for (int sx = 0; sx < GGML_MAX_SRC; sx++) {
                                ggml_tensor * sv = n->src[sx];
                                if (sv && sv->ne[1] == r.T && sv->op != GGML_OP_NONE) {
                                    bool have = false; for (auto & pr : saved) if (pr.first == sv) have = true;
                                    if (!have) { saved.push_back({ sv, sv->ne[1] }); sv->ne[1] = rows_gpu; } } }
                        }
                        ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + r.start; sub.n_nodes = r.end - r.start + 1;
                        ggml_status st_ = ggml_backend_vk_graph_compute_impl(backend, &sub);
                        ggml_vk_synchronize(vctx);
                        for (auto & pr : saved) pr.first->ne[1] = pr.second;
                        th.join();
                        if (st_ != GGML_STATUS_SUCCESS) return st_;
                        done_ok = tpu_ok.load() == 1;
                        if (done_ok) {   // TPU-Zeilen jetzt an ihren Platz
                            char * yb2 = (char *) battn_host_ptr(r.down);
                            if (!yb2) done_ok = false;
                            else for (int t = 0; t < nrow_tpu; t++)
                                memcpy(yb2 + (size_t) (rows_gpu + t) * r.down->nb[1], &ycopy[(size_t) t * g_bffn.D], (size_t) g_bffn.D * 4);
                        }
                        if (done_ok) { cur = r.end + 1; continue; }   // beide Haelften fertig
                    }
                }
                if (!done_ok) {
                    ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + r.start; sub.n_nodes = r.end - r.start + 1;
                    ggml_status st_ = ggml_backend_vk_graph_compute_impl(backend, &sub);
                    if (st_ != GGML_STATUS_SUCCESS) return st_;
                }
                cur = r.end + 1;
            }
            if (cur < cgraph->n_nodes) {
                ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + cur; sub.n_nodes = cgraph->n_nodes - cur;
                return ggml_backend_vk_graph_compute_impl(backend, &sub);
            }
            return GGML_STATUS_SUCCESS;
        }
    }
    static const char * cdir = getenv("BARRA_ATTN_CALDUMP");
    if (cdir) return battn_caldump_run(cdir, backend, cgraph);
    if (!ggml_vk_battn_env_on() || !battn_init()) return ggml_backend_vk_graph_compute_impl(backend, cgraph);
    std::vector<battn_range> ranges;
    battn_scan(cgraph, ranges);
    if (ranges.empty()) return ggml_backend_vk_graph_compute_impl(backend, cgraph);
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *) backend->context;
    static int check = -1; if (check < 0) check = getenv("BARRA_ATTN_CHECK") ? 1 : 0;
    g_battn.calls++;
    int cur = 0;
    for (const battn_range & r : ranges) {
        if (r.start > cur) {
            ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + cur; sub.n_nodes = r.start - cur;
            ggml_status stt = ggml_backend_vk_graph_compute_impl(backend, &sub);
            if (stt != GGML_STATUS_SUCCESS) return stt;
        }
        ggml_vk_synchronize(ctx);
        if (check) {
            // CHECK: h_norm sichern (der Allocator darf dst auf h_norm legen!), erst Vulkan-Referenz rechnen,
            // h_norm zurueckschreiben, dann unser Block; cos loggen. Referenz bleibt im dst (PPL = Baseline).
            const int T = r.T, D = g_battn.D;
            std::vector<float> hsave((size_t) T * D), ref((size_t) T * D);
            { const char * hp = (const char *) battn_host_ptr(r.hnorm);
              for (int t = 0; t < T; t++) memcpy(&hsave[(size_t) t * D], hp + (size_t) t * r.hnorm->nb[1], D * sizeof(float)); }
            ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + r.start; sub.n_nodes = r.end - r.start + 1;
            ggml_status stt = ggml_backend_vk_graph_compute_impl(backend, &sub);
            if (stt != GGML_STATUS_SUCCESS) return stt;
            ggml_vk_synchronize(ctx);
            { const char * rp = (const char *) battn_host_ptr(r.wo);
              for (int t = 0; t < T; t++) memcpy(&ref[(size_t) t * D], rp + (size_t) t * r.wo->nb[1], D * sizeof(float)); }
            { char * hp = (char *) battn_host_ptr(r.hnorm);
              for (int t = 0; t < T; t++) memcpy(hp + (size_t) t * r.hnorm->nb[1], &hsave[(size_t) t * D], D * sizeof(float)); }
            if (const char * dd = getenv("BARRA_ATTN_DUMP")) {
                char pa[512]; snprintf(pa, 512, "%s/dump_ref_L%d_f32.bin", dd, r.L);
                FILE * fo = fopen(pa, "wb"); if (fo) { fwrite(ref.data(), 4, ref.size(), fo); fclose(fo); }
                snprintf(pa, 512, "%s/dump_href_L%d_f32.bin", dd, r.L);
                fo = fopen(pa, "wb"); if (fo) { fwrite(hsave.data(), 4, hsave.size(), fo); fclose(fo); }
            }
            if (battn_exec(r)) {
                char * op = (char *) battn_host_ptr(r.wo);
                double ab = 0, aa = 0, bb = 0;
                for (int t = 0; t < T; t++) {
                    const float * ar = (const float *) (op + (size_t) t * r.wo->nb[1]);
                    const float * br = &ref[(size_t) t * D];
                    for (int d = 0; d < D; d++) { ab += (double) ar[d] * br[d]; aa += (double) ar[d] * ar[d]; bb += (double) br[d] * br[d]; }
                }
                BALOG("barra-attn: CHECK L%d T=%d cos=%.6f\n", r.L, T, ab / sqrt(aa * bb + 1e-30));
            } else BALOG("barra-attn: CHECK L%d exec fehlgeschlagen\n", r.L);
            { char * op = (char *) battn_host_ptr(r.wo);
              for (int t = 0; t < T; t++) memcpy(op + (size_t) t * r.wo->nb[1], &ref[(size_t) t * D], D * sizeof(float)); }
            cur = r.end + 1;
            continue;
        }
        if (!battn_exec(r)) {
            g_battn.fb++;
            ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + r.start; sub.n_nodes = r.end - r.start + 1;
            ggml_status stt = ggml_backend_vk_graph_compute_impl(backend, &sub);
            if (stt != GGML_STATUS_SUCCESS) return stt;
        }
        cur = r.end + 1;
    }
    if (cur < cgraph->n_nodes) {
        ggml_cgraph sub = *cgraph; sub.nodes = cgraph->nodes + cur; sub.n_nodes = cgraph->n_nodes - cur;
        return ggml_backend_vk_graph_compute_impl(backend, &sub);
    }
    return GGML_STATUS_SUCCESS;
}
