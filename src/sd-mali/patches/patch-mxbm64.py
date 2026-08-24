import sys
root='/home/kevin/stable-diffusion.cpp/ggml/src/ggml-vulkan/'
g=root+'vulkan-shaders/vulkan-shaders-gen.cpp'; t=open(g).read()
if 'mxbm64' in t: print('already'); sys.exit(0)
a='    string_to_spv("mul_mm_mali_mx_f16_f32",  "mul_mm_mali.comp", {{"B_F32", "1"}, {"ACC_MIXED", "1"}});\n'
assert t.count(a)==1
t=t.replace(a,a+'    string_to_spv("mul_mm_mali_mxbm64_f16_f32",  "mul_mm_mali.comp", {{"B_F32", "1"}, {"ACC_MIXED", "1"}, {"BM", "64"}});\n')
open(g,'w').write(t)
c=root+'ggml-vulkan.cpp'; u=open(c).read()
f='    vk_pipeline pipeline_mul_mm_mali_mx_f16_f32;\n'; assert u.count(f)==1
u=u.replace(f,f+'    vk_pipeline pipeline_mul_mm_mali_mxbm64_f16_f32;\n')
old='''        {
            const char * fa = getenv("GGML_VK_MALI_F32ACC");
            vk_pipeline f32l = device->pipeline_mul_mm_mali_f32acc_f16_f32;
            if (fa != nullptr && strcmp(fa, "sc") == 0) f32l = device->pipeline_mul_mm_mali_sc_f16_f32;
            if (fa != nullptr && strcmp(fa, "mx") == 0) f32l = device->pipeline_mul_mm_mali_mx_f16_f32;
            if (device->pipeline_matmul_f16_f32.f32acc) { auto &pp = device->pipeline_matmul_f16_f32.f32acc; pp->l = pp->a_l = f32l; }
        }
'''
assert u.count(old)==1
new='''        ggml_vk_create_pipeline(device, device->pipeline_mul_mm_mali_mxbm64_f16_f32, "mul_mm_mali_mxbm64_f16_f32", mul_mm_mali_mxbm64_f16_f32_len, mul_mm_mali_mxbm64_f16_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), {64, 128, 1}, std::vector<uint32_t>{}, 1);
        {
            // Default: mx (f16-Teilsummen je BK-Tile, f32-Gesamtakkumulator) + mxbm64 auf dem s-Slot (Tails/kleine Grids).
            // GGML_VK_MALI_F32ACC=f32|sc|mx|mxl (mxl = mx ohne bm64-Routing)
            const char * fa = getenv("GGML_VK_MALI_F32ACC");
            vk_pipeline f32l = device->pipeline_mul_mm_mali_mx_f16_f32;
            bool use_bm64 = true;
            if (fa != nullptr && strcmp(fa, "f32") == 0) { f32l = device->pipeline_mul_mm_mali_f32acc_f16_f32; use_bm64 = false; }
            if (fa != nullptr && strcmp(fa, "sc") == 0)  { f32l = device->pipeline_mul_mm_mali_sc_f16_f32; use_bm64 = false; }
            if (fa != nullptr && strcmp(fa, "mxl") == 0) { use_bm64 = false; }
            if (device->pipeline_matmul_f16_f32.f32acc) {
                auto &pp = device->pipeline_matmul_f16_f32.f32acc;
                pp->l = pp->a_l = f32l;
                if (use_bm64) pp->s = pp->a_s = device->pipeline_mul_mm_mali_mxbm64_f16_f32;
            }
        }
'''
u=u.replace(old,new)
gate='    if (mm_s && (mmp->s == ctx->device->pipeline_mul_mm_mali_bm64_f16 || mmp->s == ctx->device->pipeline_mul_mm_mali_bm64_f16_f32)) {'
assert u.count(gate)==1
u=u.replace(gate,'    if (mm_s && (mmp->s == ctx->device->pipeline_mul_mm_mali_bm64_f16 || mmp->s == ctx->device->pipeline_mul_mm_mali_bm64_f16_f32 || mmp->s == ctx->device->pipeline_mul_mm_mali_mxbm64_f16_f32)) {')
open(c,'w').write(u); print('patched')
