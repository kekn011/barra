import sys
c='/home/kevin/stable-diffusion.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp'; u=open(c).read()
if 'is_mx' in u: print('already'); sys.exit(0)
old='''    if (mm_s && (mmp->s == ctx->device->pipeline_mul_mm_mali_bm64_f16 || mmp->s == ctx->device->pipeline_mul_mm_mali_bm64_f16_f32 || mmp->s == ctx->device->pipeline_mul_mm_mali_mxbm64_f16_f32)) {
        const uint32_t tiles128 = CEIL_DIV(m, 128u) * CEIL_DIV(n, 128u);
        if (tiles128 <= 48 || ((m % 128u) != 0u && m <= 640u)) {'''
assert u.count(old)==1
new='''    const bool is_mx = mmp->s == ctx->device->pipeline_mul_mm_mali_mxbm64_f16_f32;
    if (mm_s && (mmp->s == ctx->device->pipeline_mul_mm_mali_bm64_f16 || mmp->s == ctx->device->pipeline_mul_mm_mali_bm64_f16_f32 || is_mx)) {
        const uint32_t tiles128 = CEIL_DIV(m, 128u) * CEIL_DIV(n, 128u);
        // mx-Familie (prec=F32): nur m-Tails auf BM64 (kleine Grids laufen dort mit 128er-Tile schneller)
        if ((!is_mx && tiles128 <= 48) || ((m % 128u) != 0u && m <= 640u)) {'''
u=u.replace(old,new); open(c,'w').write(u); print('patched')
