# Shader-Varianten: B_SCALE (B*2^-7 beim Laden, D*2^7 beim Store, f16acc bleibt) und
# ACC_MIXED (f16-Teilsummen je BK-Tile, f32-Gesamtakkumulator). Plus Pipelines + Env-Routing.
import sys
root='/home/kevin/stable-diffusion.cpp/ggml/src/ggml-vulkan/'
sh=root+'vulkan-shaders/mul_mm_mali.comp'
s=open(sh).read()
if 'B_SCALE' in s: print('shader already'); sys.exit(0)
# Defines
s=s.replace('#ifdef ACC_F32\n#define ACCV vec4\n#else\n#define ACCV f16vec4\n#endif',
'''#ifdef ACC_F32
#define ACCV vec4
#else
#define ACCV f16vec4
#endif
#ifdef B_SCALE
#define BSC float16_t(0.0078125)
#define DSC 128.0
#else
#define BSC float16_t(1.0)
#define DSC 1.0
#endif''')
# B-Load skalieren (vec4-Pfad + Tail)
old_b='''                if (kk + 4u <= end_k) {
                    vb = b_v4 ? f16vec4(data_b4[base >> 2u])
                              : f16vec4(data_b[base], data_b[base+1u], data_b[base+2u], data_b[base+3u]);
                } else {
                    if (kk      < end_k) vb.x = float16_t(data_b[base]);
                    if (kk + 1u < end_k) vb.y = float16_t(data_b[base+1u]);
                    if (kk + 2u < end_k) vb.z = float16_t(data_b[base+2u]);
                }'''
new_b='''                if (kk + 4u <= end_k) {
                    vb = b_v4 ? f16vec4(data_b4[base >> 2u])
                              : f16vec4(data_b[base], data_b[base+1u], data_b[base+2u], data_b[base+3u]);
                } else {
                    if (kk      < end_k) vb.x = float16_t(data_b[base]);
                    if (kk + 1u < end_k) vb.y = float16_t(data_b[base+1u]);
                    if (kk + 2u < end_k) vb.z = float16_t(data_b[base+2u]);
                }
#ifdef B_SCALE
                vb *= BSC;
#endif'''
assert s.count(old_b)==1; s=s.replace(old_b,new_b)
# Akkumulatoren: ACC_MIXED
old_acc='''    ACCV acc[MM][MNV];
    [[unroll]] for (int i = 0; i < MM; i++) [[unroll]] for (int v = 0; v < MNV; v++) acc[i][v] = ACCV(0.0);
'''
new_acc='''#ifdef ACC_MIXED
    vec4 acc32[MM][MNV];
    [[unroll]] for (int i = 0; i < MM; i++) [[unroll]] for (int v = 0; v < MNV; v++) acc32[i][v] = vec4(0.0);
    ACCV acc[MM][MNV];
#else
    ACCV acc[MM][MNV];
    [[unroll]] for (int i = 0; i < MM; i++) [[unroll]] for (int v = 0; v < MNV; v++) acc[i][v] = ACCV(0.0);
#endif
'''
assert s.count(old_acc)==1; s=s.replace(old_acc,new_acc)
old_loop='''        barrier();
        [[unroll]] for (uint k4 = 0u; k4 < BK/4u; k4++) {'''
new_loop='''        barrier();
#ifdef ACC_MIXED
        [[unroll]] for (int i = 0; i < MM; i++) [[unroll]] for (int v = 0; v < MNV; v++) acc[i][v] = ACCV(0.0);
#endif
        [[unroll]] for (uint k4 = 0u; k4 < BK/4u; k4++) {'''
assert s.count(old_loop)==1; s=s.replace(old_loop,new_loop)
old_end='''        }
        barrier();
    }

    const uint dbase'''
new_end='''        }
#ifdef ACC_MIXED
        [[unroll]] for (int i = 0; i < MM; i++) [[unroll]] for (int v = 0; v < MNV; v++) acc32[i][v] += vec4(acc[i][v]);
#endif
        barrier();
    }

    const uint dbase'''
assert s.count(old_end)==1; s=s.replace(old_end,new_end)
old_st='if (col + j < p.N) data_d[dbase + (col + j) * p.stride_d + row] = float(acc[i][v][j]);'
new_st='''if (col + j < p.N) data_d[dbase + (col + j) * p.stride_d + row] =
#ifdef ACC_MIXED
                    acc32[i][v][j];
#else
                    float(acc[i][v][j]) * DSC;
#endif'''
assert s.count(old_st)==1; s=s.replace(old_st,new_st)
open(sh,'w').write(s)
# shaders-gen
g=root+'vulkan-shaders/vulkan-shaders-gen.cpp'
t=open(g).read()
anchor='    string_to_spv("mul_mm_mali_a32b16",'
assert t.count(anchor)==1
t=t.replace(anchor,'''    string_to_spv("mul_mm_mali_sc_f16_f32",  "mul_mm_mali.comp", {{"B_F32", "1"}, {"B_SCALE", "1"}});
    string_to_spv("mul_mm_mali_mx_f16_f32",  "mul_mm_mali.comp", {{"B_F32", "1"}, {"ACC_MIXED", "1"}});
'''+anchor)
open(g,'w').write(t)
# ggml-vulkan.cpp: Felder + Pipelines + Routing per Env GGML_VK_MALI_F32ACC=sc|mx (default: f32acc-Variante)
c=root+'ggml-vulkan.cpp'
u=open(c).read()
if 'pipeline_mul_mm_mali_sc' in u: print('cpp already'); sys.exit(0)
fld='    vk_pipeline pipeline_mul_mm_mali_a32b16;\n'
assert u.count(fld)==1
u=u.replace(fld, fld+'    vk_pipeline pipeline_mul_mm_mali_sc_f16_f32;\n    vk_pipeline pipeline_mul_mm_mali_mx_f16_f32;\n')
old_r='        if (device->pipeline_matmul_f16_f32.f32acc) { auto &pp = device->pipeline_matmul_f16_f32.f32acc; pp->l = pp->a_l = device->pipeline_mul_mm_mali_f32acc_f16_f32; }\n'
assert u.count(old_r)==1
new_r='''        ggml_vk_create_pipeline(device, device->pipeline_mul_mm_mali_sc_f16_f32, "mul_mm_mali_sc_f16_f32", mul_mm_mali_sc_f16_f32_len, mul_mm_mali_sc_f16_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), {128, 128, 1}, std::vector<uint32_t>{}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_mul_mm_mali_mx_f16_f32, "mul_mm_mali_mx_f16_f32", mul_mm_mali_mx_f16_f32_len, mul_mm_mali_mx_f16_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), {128, 128, 1}, std::vector<uint32_t>{}, 1);
        {
            const char * fa = getenv("GGML_VK_MALI_F32ACC");
            vk_pipeline f32l = device->pipeline_mul_mm_mali_f32acc_f16_f32;
            if (fa != nullptr && strcmp(fa, "sc") == 0) f32l = device->pipeline_mul_mm_mali_sc_f16_f32;
            if (fa != nullptr && strcmp(fa, "mx") == 0) f32l = device->pipeline_mul_mm_mali_mx_f16_f32;
            if (device->pipeline_matmul_f16_f32.f32acc) { auto &pp = device->pipeline_matmul_f16_f32.f32acc; pp->l = pp->a_l = f32l; }
        }
'''
u=u.replace(old_r,new_r)
open(c,'w').write(u)
print('patched all')
