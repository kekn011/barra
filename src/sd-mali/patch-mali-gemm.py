#!/usr/bin/env python3
# Integriert mul_mm_mali.comp (gemm_best-Port) in ggml-vulkan:
# 1) Shader in den vulkan-shaders-Ordner kopieren
# 2) vulkan-shaders-gen.cpp: zwei Varianten erzeugen (B=f16, B=f32)
# 3) ggml-vulkan.cpp: unter GGML_VK_MALI_GEMM=1 die l-Pipelines der f16-Matmuls ersetzen
import io, shutil, sys, os
ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/stable-diffusion.cpp/ggml/src/ggml-vulkan")
SRC  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mul_mm_mali.comp")  # repo-relativ

# 1) Shader kopieren
shutil.copy(SRC, os.path.join(ROOT, "vulkan-shaders/mul_mm_mali.comp"))
print("Shader kopiert")

# 2) shaders-gen
P = os.path.join(ROOT, "vulkan-shaders/vulkan-shaders-gen.cpp")
s = io.open(P, encoding="utf-8").read()
if "mul_mm_mali" not in s:
    a = "void process_shaders() {\n"
    assert s.count(a) == 1
    s = s.replace(a, a + """    // barra: Mali-GEMM (gemm_best-Port)
    string_to_spv("mul_mm_mali_f16",     "mul_mm_mali.comp", {{"B_F16", "1"}});
    string_to_spv("mul_mm_mali_f16_f32", "mul_mm_mali.comp", {{"B_F32", "1"}});
""", 1)
    io.open(P, "w", encoding="utf-8").write(s)
    print("shaders-gen gepatcht")
else:
    print("shaders-gen schon gepatcht")

# 3) ggml-vulkan.cpp
P = os.path.join(ROOT, "ggml-vulkan.cpp")
s = io.open(P, encoding="utf-8").read()
if "GGML_VK_MALI_GEMM" not in s:
    a = """                                get_fa_spec_constants(fa.first), aligned ? Bc : 1, true,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }
"""
    assert s.count(a) == 1, "FA-Anker nicht eindeutig: %d" % s.count(a)
    s = s.replace(a, a + """
    // barra: Mali-GEMM (gemm_best-Port) — ersetzt die l-Pipelines der f16-Matmuls (nur per Env).
    if (getenv("GGML_VK_MALI_GEMM") != nullptr) {
        vk_pipeline mali_f16, mali_f16_f32;
        ggml_vk_create_pipeline(device, mali_f16,     "mul_mm_mali_f16",     mul_mm_mali_f16_len,     mul_mm_mali_f16_data,     "main", 3, sizeof(vk_mat_mat_push_constants), {128, 128, 1}, std::vector<uint32_t>{}, 1);
        ggml_vk_create_pipeline(device, mali_f16_f32, "mul_mm_mali_f16_f32", mul_mm_mali_f16_f32_len, mul_mm_mali_f16_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), {128, 128, 1}, std::vector<uint32_t>{}, 1);
        if (device->pipeline_matmul_f16.f16acc)     { device->pipeline_matmul_f16.f16acc->l = mali_f16;         device->pipeline_matmul_f16.f16acc->a_l = mali_f16; }
        if (device->pipeline_matmul_f16_f32.f16acc) { device->pipeline_matmul_f16_f32.f16acc->l = mali_f16_f32; device->pipeline_matmul_f16_f32.f16acc->a_l = mali_f16_f32; }
        std::cerr << "barra: Mali-GEMM aktiv (f16-l-Pipelines ersetzt)" << std::endl;
    }
""", 1)
    io.open(P, "w", encoding="utf-8").write(s)
    print("ggml-vulkan gepatcht")
else:
    print("ggml-vulkan schon gepatcht")
