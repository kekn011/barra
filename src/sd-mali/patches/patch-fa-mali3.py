import sys, shutil, re
root='/home/kevin/stable-diffusion.cpp/ggml/src/ggml-vulkan/'
shutil.copy('/mnt/c/Users/kevin/AppData/Local/Temp/claude/C--Users-kevin-projects-pixel-cluster-base/ec4a4720-077e-41d3-9cd9-c71c1f80d54c/scratchpad/fa_mali.comp', root+'vulkan-shaders/fa_mali.comp')
# (name, HD, RT, CT, extra defines)
VARS=[('r4c4ps',40,4,4,', {"PV_PS", "1"}'),
      ('r4c4ps32',40,4,4,', {"PV_PS", "1"}, {"ACC32", "1"}'),
      ('r4c4psmx',40,4,4,', {"PV_PS", "1"}, {"ACC_MIXED", "1"}'),
      ('r2c4psmx',40,2,4,', {"PV_PS", "1"}, {"ACC_MIXED", "1"}'),
      ('h80r2psmx',80,2,4,', {"PV_PS", "1"}, {"ACC_MIXED", "1"}'),
      ('h80r4psmx',80,4,4,', {"PV_PS", "1"}, {"ACC_MIXED", "1"}'),
      ('h80r1psmx',80,1,4,', {"PV_PS", "1"}, {"ACC_MIXED", "1"}')]
g=root+'vulkan-shaders/vulkan-shaders-gen.cpp'; t=open(g).read()
t=re.sub(r'    string_to_spv\("fa_mali_[^\n]*\n','',t)
a='    string_to_spv("mul_mm_mali_a32b16",'
assert t.count(a)==1
gen=''.join('    string_to_spv("fa_mali_%s", "fa_mali.comp", {{"HD", "%d"}, {"RT", "%d"}, {"CT", "%d"}%s});\n'%(n,hd,rt,ct,ex) for n,hd,rt,ct,ex in VARS)
t=t.replace(a,gen+a); open(g,'w').write(t); print('gen patched')
c=root+'ggml-vulkan.cpp'; u=open(c).read()
if 'pipeline_fa_mali_var' not in u:
    f='    vk_pipeline pipeline_fa_mali_hd40_r4;\n    vk_pipeline pipeline_fa_mali_hd40_r2;\n'; assert u.count(f)==1
    u=u.replace(f,'    vk_pipeline pipeline_fa_mali_var[8];\n')
u=u.replace('    vk_pipeline pipeline_fa_mali_var[8];\n','    vk_pipeline pipeline_fa_mali_var[16];\n')
s=u.find('    // barra: Mali-FlashAttention (hd=40)'); e=u.find('    // barra: Mali-GEMM (gemm_best-Port)', s)
assert s>0 and e>s
create='''    // barra: Mali-FlashAttention (hd=40), Env GGML_VK_MALI_FA=<variante>|off (Default r4c4psmx), GGML_VK_MALI_FA80=<variante>|off (Default h80r2psmx)
    {
        const char * fa_env = getenv("GGML_VK_MALI_FA");
        if (fa_env == nullptr || strcmp(fa_env, "off") != 0) {
'''+''.join('            ggml_vk_create_pipeline(device, device->pipeline_fa_mali_var[%d], "fa_mali_%s", fa_mali_%s_len, fa_mali_%s_data, "main", 7, sizeof(vk_flash_attn_push_constants), {%d, 1, 1}, std::vector<uint32_t>{}, 1, true, true, 16);\n'%(i,n,n,n,16*rt) for i,(n,hd,rt,ct,ex) in enumerate(VARS))+'''        }
    }
'''
u=u[:s]+create+u[e:]
# Hook ersetzen: gesamten Block von "static const char * barra_fa" bis zum schliessenden "    }\n" vor "if (split_k > 1)"
hs=u.find('    {\n        static const char * barra_fa = getenv("GGML_VK_MALI_FA");')
he=u.find('    if (split_k > 1) {\n        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_split_k_reduce, 1);\n')
assert hs>0 and he>hs
names=', '.join('"%s"'%n for n,_,_,_,_ in VARS)
hds=', '.join(str(hd) for _,hd,_,_,_ in VARS)
hook='''    {
        // barra: Mali-FlashAttention-Hook (hd 40/80, K/V f16, ohne Maske/Sinks/GQA)
        static const char * names[] = {%s};
        static const uint32_t hds[] = {%s};
        static const int nvar = %d;
        static int sel40 = -1, sel80 = -1;
        if (sel40 == -1) {
            const char * e40 = getenv("GGML_VK_MALI_FA");
            const char * e80 = getenv("GGML_VK_MALI_FA80");
            sel40 = (e40 != nullptr && strcmp(e40, "off") == 0) ? -2 : -3;
            sel80 = (e80 != nullptr && strcmp(e80, "off") == 0) ? -2 : -3;
            if (e40 != nullptr && strcmp(e40, "off") == 0) { sel80 = -2; }
            for (int vi = 0; vi < nvar; vi++) {
                if (sel40 == -3 && hds[vi] == 40 && ((e40 == nullptr && strcmp(names[vi], "r4c4psmx") == 0) || (e40 != nullptr && strcmp(e40, names[vi]) == 0))) sel40 = vi;
                if (sel80 == -3 && hds[vi] == 80 && ((e80 == nullptr && strcmp(names[vi], "h80r2psmx") == 0) || (e80 != nullptr && strcmp(e80, names[vi]) == 0))) sel80 = vi;
            }
            if (sel40 >= 0) std::cerr << "barra: Mali-FA hd40 = " << names[sel40] << std::endl;
            if (sel80 >= 0) std::cerr << "barra: Mali-FA hd80 = " << names[sel80] << std::endl;
        }
        vk_pipeline mfa = nullptr;
        if (HSK == 40 && sel40 >= 0) mfa = ctx->device->pipeline_fa_mali_var[sel40];
        if (HSK == 80 && sel80 >= 0) mfa = ctx->device->pipeline_fa_mali_var[sel80];
        if (mfa && HSK == HSV && mask == nullptr && sinks == nullptr &&
            k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            gqa_ratio == 1 && neq2 == nek2 && neq2 == nev2 && neq3 == nek3 && neq3 == nev3 &&
            logit_softcap == 0.0f && max_bias == 0.0f && (q_stride %% 4) == 0 && (k_stride %% 4) == 0 && (v_stride %% 4) == 0) {
            vk_flash_attn_push_constants pcm = pc;
            pcm.split_kv = KV; pcm.k_num = 1;
            ggml_pipeline_request_descriptor_sets(ctx, mfa, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, mfa, {q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf}, pcm, { N, (uint32_t)neq2, (uint32_t)neq3 });
            return;
        }
    }
'''%(names,hds,len(VARS))
u=u[:hs]+hook+u[he:]
open(c,'w').write(u); print('cpp patched')
