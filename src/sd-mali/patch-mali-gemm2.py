#!/usr/bin/env python3
# Fix fuer den Mali-GEMM-Patch: Pipelines muessen auf dem DEVICE persistieren
# (ggml_vk_load_shaders ist ein wiederholter Walk; lokale shared_ptr je Walk = neue
# Objekte = angeforderte Pipeline wird nie kompiliert -> Segfault).
import io, os
P = os.path.expanduser("~/stable-diffusion.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp")
s = io.open(P, encoding="utf-8").read()

# 1) Device-Felder ergaenzen
a = "    vk_matmul_pipeline2 pipeline_matmul_f16_f32;\n"
if "pipeline_mul_mm_mali_f16" not in s:
    assert s.count(a) == 1
    s = s.replace(a, a + "    // barra: Mali-GEMM (gemm_best-Port) — persistente Pipelines fuer den load_shaders-Walk\n    vk_pipeline pipeline_mul_mm_mali_f16;\n    vk_pipeline pipeline_mul_mm_mali_f16_f32;\n", 1)
    print("Device-Felder ergaenzt")

# 2) Block auf Device-Felder umstellen
old = """    // barra: Mali-GEMM (gemm_best-Port) — ersetzt die l-Pipelines der f16-Matmuls (nur per Env).
    if (getenv("GGML_VK_MALI_GEMM") != nullptr) {
        vk_pipeline mali_f16, mali_f16_f32;
        ggml_vk_create_pipeline(device, mali_f16,     "mul_mm_mali_f16",     mul_mm_mali_f16_len,     mul_mm_mali_f16_data,     "main", 3, sizeof(vk_mat_mat_push_constants), {128, 128, 1}, std::vector<uint32_t>{}, 1);
        ggml_vk_create_pipeline(device, mali_f16_f32, "mul_mm_mali_f16_f32", mul_mm_mali_f16_f32_len, mul_mm_mali_f16_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), {128, 128, 1}, std::vector<uint32_t>{}, 1);
        if (device->pipeline_matmul_f16.f16acc)     { device->pipeline_matmul_f16.f16acc->l = mali_f16;         device->pipeline_matmul_f16.f16acc->a_l = mali_f16; }
        if (device->pipeline_matmul_f16_f32.f16acc) { device->pipeline_matmul_f16_f32.f16acc->l = mali_f16_f32; device->pipeline_matmul_f16_f32.f16acc->a_l = mali_f16_f32; }
        std::cerr << "barra: Mali-GEMM aktiv (f16-l-Pipelines ersetzt)" << std::endl;
    }
"""
new = """    // barra: Mali-GEMM (gemm_best-Port) — ersetzt die l-Pipelines der f16-Matmuls (nur per Env).
    if (getenv("GGML_VK_MALI_GEMM") != nullptr) {
        ggml_vk_create_pipeline(device, device->pipeline_mul_mm_mali_f16,     "mul_mm_mali_f16",     mul_mm_mali_f16_len,     mul_mm_mali_f16_data,     "main", 3, sizeof(vk_mat_mat_push_constants), {128, 128, 1}, std::vector<uint32_t>{}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_mul_mm_mali_f16_f32, "mul_mm_mali_f16_f32", mul_mm_mali_f16_f32_len, mul_mm_mali_f16_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), {128, 128, 1}, std::vector<uint32_t>{}, 1);
        if (device->pipeline_matmul_f16.f16acc)     { device->pipeline_matmul_f16.f16acc->l = device->pipeline_mul_mm_mali_f16;         device->pipeline_matmul_f16.f16acc->a_l = device->pipeline_mul_mm_mali_f16; }
        if (device->pipeline_matmul_f16_f32.f16acc) { device->pipeline_matmul_f16_f32.f16acc->l = device->pipeline_mul_mm_mali_f16_f32; device->pipeline_matmul_f16_f32.f16acc->a_l = device->pipeline_mul_mm_mali_f16_f32; }
    }
"""
if old in s:
    s = s.replace(old, new, 1)
    print("Block umgestellt")
else:
    assert new in s, "weder alter noch neuer Block gefunden"
    print("Block schon umgestellt")
io.open(P, "w", encoding="utf-8").write(s)
print("fertig")
