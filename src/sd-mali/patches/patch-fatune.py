import sys
c='/home/kevin/stable-diffusion.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp'; u=open(c).read()
if 'GGML_VK_FA_TUNE' in u: print('already'); sys.exit(0)
i=u.find('static vk_fa_tuning_params get_fa_tuning_params_scalar(')
assert i>0
r=u.find('\n    return result;\n}', i); assert r>0
ins='''
    // barra: FA-Scalar-Override per Env (nur fuer grosse Query-Bloecke): GGML_VK_FA_TUNE=wg,sg,Br,Bc,dsplit,rowsplit,staging
    if (n_rows >= 64) {
        static int have = -1; static uint32_t t[7];
        if (have < 0) {
            have = 0;
            const char * v = getenv("GGML_VK_FA_TUNE");
            if (v != nullptr) {
                int n = 0; uint32_t x = 0; bool d = false;
                for (const char * p = v;; ++p) {
                    if (*p >= '0' && *p <= '9') { x = x * 10u + uint32_t(*p - '0'); d = true; }
                    else { if (d && n < 7) t[n++] = x; x = 0; d = false; if (*p == 0) break; }
                }
                have = (n == 7) ? 1 : 0;
                if (!have) std::cerr << "barra: GGML_VK_FA_TUNE braucht 7 Zahlen" << std::endl;
            }
        }
        if (have == 1) {
            vk_fa_tuning_params o = result;
            o.workgroup_size = t[0]; o.subgroup_size = t[1]; o.block_rows = t[2]; o.block_cols = t[3];
            o.d_split = t[4]; o.row_split = t[5]; o.shmem_staging = t[6] != 0;
            const uint32_t cpi = (o.d_split * o.row_split) ? o.workgroup_size / (o.d_split * o.row_split) : 0;
            const bool ok = o.d_split > 0 && o.row_split > 0 && cpi > 0 &&
                            (o.workgroup_size % (o.d_split * o.row_split)) == 0 &&
                            (o.block_cols % cpi) == 0 && (o.block_rows % o.row_split) == 0 &&
                            (hsk % (o.d_split * 4)) == 0 && (hsv % (o.d_split * 4)) == 0 &&
                            (o.subgroup_size == 0 || (o.workgroup_size % o.subgroup_size) == 0);
            if (ok && ggml_vk_flash_attn_scalar_shmem_support(device, o, hsk, hsv, f32acc, k_type, v_type)) {
                static bool said = false;
                if (!said) { said = true; std::cerr << "barra: FA_TUNE aktiv: "; o.print(); }
                result = o;
                if (o.subgroup_size == 0) result.disable_subgroups = true;
            } else {
                static bool warned = false;
                if (!warned) { warned = true; std::cerr << "barra: FA_TUNE UNGUELTIG (Constraints/Shmem), Default bleibt" << std::endl; }
            }
        }
    }
'''
u=u[:r]+ins+u[r:]
old='    const bool f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;\n'
assert u.count(old)==1
u=u.replace(old,'    static const bool barra_fa_f16acc = getenv("GGML_VK_FA_F16ACC") != nullptr;\n    const bool f32acc = !ctx->device->fp16 || (dst->op_params[3] == GGML_PREC_F32 && !barra_fa_f16acc) || k->type == GGML_TYPE_BF16;\n')
open(c,'w').write(u); print('patched')
