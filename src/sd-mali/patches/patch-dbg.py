import re,sys
p='/home/kevin/stable-diffusion.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp'
s=open(p).read()
if 'barra-dbg' in s: print('already patched'); sys.exit(0)
# 1) in ggml_vk_mul_mat_q_f16 nach split_k
anchor='    const uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);\n'
assert s.count(anchor)==1, s.count(anchor)
dbg=anchor+r'''
    // barra: Pfad-Diagnose (env GGML_VK_BARRA_DBG) — einmal je Form/Pfad
    if (getenv("GGML_VK_BARRA_DBG") != nullptr) {
        static std::set<std::string> barra_seen;
        char key[640];
        snprintf(key, sizeof key, "q_f16 m=%llu n=%llu k=%llu b=%llu src0=%s src1=%s prec=%d xnc=%d ync=%d qy=%d qxdq=%d qydq=%d al=%d pipe=%s splitk=%u wg=%ux%u name=%s",
            (unsigned long long)ne01, (unsigned long long)ne11, (unsigned long long)ne10, (unsigned long long)(ne12*ne13),
            ggml_type_name(src0->type), ggml_type_name(src1->type), (int)dst->op_params[0],
            (int)x_non_contig, (int)y_non_contig, (int)quantize_y, (int)qx_needs_dequant, (int)qy_needs_dequant, (int)aligned,
            pipeline->name.c_str(), split_k, pipeline->wg_denoms[0], pipeline->wg_denoms[1], src0->name);
        if (barra_seen.insert(key).second) std::cerr << "barra-dbg: " << key << std::endl;
    }
'''
s=s.replace(anchor,dbg)
# 2) Dispatcher-Zweige
def br(tag):
    return ('        if (getenv("GGML_VK_BARRA_DBG") != nullptr) { static std::set<std::string> sn; char key[256]; snprintf(key, sizeof key, "%s m=%lld n=%lld k=%lld b=%lld src0=%s src1=%s name=%s", "'+tag+'", (long long)src0->ne[1], (long long)src1->ne[1], (long long)src0->ne[0], (long long)(src1->ne[2]*src1->ne[3]), ggml_type_name(src0->type), ggml_type_name(src1->type), src0->name); if (sn.insert(key).second) std::cerr << "barra-dbg: " << key << std::endl; }\n')
for call,tag in [('        ggml_vk_fwht(ctx, subctx, src1, dst);\n','fwht'),
                 ('        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, cgraph, node_idx);\n','p021'),
                 ('        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, cgraph, node_idx);\n','nc'),
                 ('        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, node_idx);\n','mmv')]:
    assert s.count(call)==1,(call,s.count(call))
    s=s.replace(call,br(tag)+call)
ms='    if (needs_split) {\n        // Choose the number of rows'
assert s.count(ms)==1
s=s.replace(ms,'    if (needs_split) {\n'+br('msplit')+'        // Choose the number of rows')
open(p,'w').write(s)
print('patched')
