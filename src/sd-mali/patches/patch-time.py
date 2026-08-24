import sys
# --- ggml-vulkan: graph_compute Timer
c='/home/kevin/stable-diffusion.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp'; u=open(c).read()
if 'barra_t0' not in u:
    a='static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {\n    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");\n'
    assert u.count(a)==1
    u=u.replace(a,a+'    static const bool barra_time = getenv("BARRA_TIME") != nullptr;\n    const auto barra_t0 = std::chrono::steady_clock::now();\n    auto barra_t1 = barra_t0;\n')
    # loop start: erste Fundstelle nach flops_per_submit
    idx=u.find('uint64_t flops_per_submit = std::min(flops_cap, ctx->last_total_flops / 40u);')
    assert idx>0
    loop='    for (int i = 0; i < cgraph->n_nodes; i++) {\n'
    li=u.find(loop, idx); assert li>0
    u=u[:li]+'    barra_t1 = std::chrono::steady_clock::now();\n'+u[li:]
    b='    ctx->last_total_flops = total_flops;\n\n    if (vk_perf_logger_enabled) {\n        // End the command buffer and submit/wait\n'
    assert u.count(b)==1
    u=u.replace(b,'''    ctx->last_total_flops = total_flops;

    if (barra_time) {
        const auto t2 = std::chrono::steady_clock::now();
        ggml_vk_synchronize(ctx);
        const auto t3 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        fprintf(stderr, "barra-vk: nodes=%d submits=%u pre=%.1f ms record=%.1f ms wait=%.1f ms total=%.1f ms\n",
                cgraph->n_nodes, submit_count, ms(barra_t0, barra_t1), ms(barra_t1, t2), ms(t2, t3), ms(barra_t0, t3));
    }

    if (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
''')
    if '#include <chrono>' not in u: u=u.replace('#include <set>\n','#include <set>\n#include <chrono>\n',1)
    open(c,'w').write(u); print('vk patched')
# --- sd.cpp: compute()/execute_graph Timer
h='/home/kevin/stable-diffusion.cpp/src/core/ggml_extend.hpp'; s=open(h).read()
if 'barra_tp0' not in s:
    a='        ggml_cgraph* gf = nullptr;\n        if (!prepare_compute_graph(get_graph, &gf)) {\n            return std::nullopt;\n        }\n        GGML_ASSERT(gf != nullptr);\n        rebuild_params_tensor_set();\n'
    assert s.count(a)==1
    s=s.replace(a,'''        ggml_cgraph* gf = nullptr;
        const auto barra_tp0 = std::chrono::steady_clock::now();
        if (!prepare_compute_graph(get_graph, &gf)) {
            return std::nullopt;
        }
        GGML_ASSERT(gf != nullptr);
        rebuild_params_tensor_set();
        if (getenv("BARRA_TIME") != nullptr) fprintf(stderr, "barra-sd %s: prepare_graph=%.1f ms nodes=%d\n", get_desc().c_str(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - barra_tp0).count(), gf->n_nodes);
''')
    b='        GraphWeightDoneGuard graph_weight_done_guard(this, &params_to_prepare);\n\n        if (!alloc_compute_buffer(gf)) {\n'
    assert s.count(b)==1
    s=s.replace(b,'        GraphWeightDoneGuard graph_weight_done_guard(this, &params_to_prepare);\n        static const bool barra_time = getenv("BARRA_TIME") != nullptr;\n        const auto barra_te0 = std::chrono::steady_clock::now();\n        auto barra_te1 = barra_te0, barra_te2 = barra_te0, barra_te3 = barra_te0, barra_te4 = barra_te0;\n\n        if (!alloc_compute_buffer(gf)) {\n')
    c2='        if (sched != nullptr) {\n            ggml_backend_sched_reset(sched);\n'
    assert s.count(c2)==1
    s=s.replace(c2,'        barra_te1 = std::chrono::steady_clock::now();\n'+c2)
    d='        copy_data_to_backend_tensor(gf, !preserve_backend_tensor_data_map);\n'
    assert s.count(d)==1
    s=s.replace(d,'        barra_te2 = std::chrono::steady_clock::now();\n'+d+'        barra_te3 = std::chrono::steady_clock::now();\n')
    e='        if (status != GGML_STATUS_SUCCESS) {\n            LOG_ERROR("%s compute failed: %s", get_desc().c_str(), ggml_status_to_string(status));\n            return std::nullopt;\n        }\n'
    assert s.count(e)==1
    s=s.replace(e,e+'''        barra_te4 = std::chrono::steady_clock::now();
        if (barra_time) {
            auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            fprintf(stderr, "barra-sd %s: alloc_buf=%.1f ms sched_alloc=%.1f ms copy_in=%.1f ms compute+sync=%.1f ms\n",
                    get_desc().c_str(), ms(barra_te0, barra_te1), ms(barra_te1, barra_te2), ms(barra_te2, barra_te3), ms(barra_te3, barra_te4));
        }
''')
    if '#include <chrono>' not in s: s=s.replace('#include <', '#include <chrono>\n#include <',1)
    open(h,'w').write(s); print('sd patched')
