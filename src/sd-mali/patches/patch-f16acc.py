import sys
p='/home/kevin/stable-diffusion.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp'
s=open(p).read()
if 'barra_f16acc_all' in s: print('already'); sys.exit(0)
old='    if (prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support)) {\n        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {\n            return ctx->device->pipeline_matmul_f16_f32.f16acc;'
assert s.count(old)==1
new='''    // barra: GGML_VK_MALI_F16ACC_ALL=1 → PREC_F32-Anforderung ignorieren, f16acc-Mali-Kernel auch fuer ff.net.2 & Co.
    static const bool barra_f16acc_all = getenv("GGML_VK_MALI_F16ACC_ALL") != nullptr;
    if ((prec == GGML_PREC_DEFAULT || barra_f16acc_all) && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support)) {
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
            return ctx->device->pipeline_matmul_f16_f32.f16acc;'''
s=s.replace(old,new); open(p,'w').write(s); print('patched')
