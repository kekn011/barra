#!/usr/bin/env python3
# Stufe 3: Block-Schleife der Mali-main 4x entrollt (4 unabhaengige Loads in Flight wie im Original). Idempotent.
import os
p = os.path.expanduser('~/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp'); s = open(p).read()
if 'mali_iter(' not in s:
    old_start = s.index('void mali_compute(const uint row, const uint lane, const bool valid) {')
    old_end = s.index('    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) { temp[j] = subgroupClusteredAdd(temp[j], MALI_LPR); }')
    new = r'''void mali_iter(inout FLOAT_TYPE temp[NUM_COLS], const uint row, const uint blk) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        const uint b_block_idx = (j*p.batch_stride_b) / QUANT_K_Q8_1 + blk + b_offset;
        const uint b_block_idx_outer = b_block_idx / 4;
        const uint b_block_idx_inner = b_block_idx % 4;
        cache_b_ds = vec2(data_b[b_block_idx_outer].ds[b_block_idx_inner]);
        const ivec4 b0 = data_b_v4[b_block_idx_outer].qs[b_block_idx_inner * 2];
        const ivec4 b1 = data_b_v4[b_block_idx_outer].qs[b_block_idx_inner * 2 + 1];
        cache_b_qs[0] = b0.x; cache_b_qs[1] = b0.y; cache_b_qs[2] = b0.z; cache_b_qs[3] = b0.w;
        cache_b_qs[4] = b1.x; cache_b_qs[5] = b1.y; cache_b_qs[6] = b1.z; cache_b_qs[7] = b1.w;
        const uint a_block_idx = (row * p.ncols) / QUANT_K_Q8_1 + blk + a_offset;
        temp[j] += mmvq_dot_product(a_block_idx, 0);
    }
}

void mali_compute(const uint row, const uint lane, const bool valid) {
    get_offsets(a_offset, b_offset, d_offset);
    a_offset *= QUANT_K / QUANT_K_Q8_1;
    b_offset /= QUANT_K_Q8_1;

    FLOAT_TYPE temp[NUM_COLS];
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) { temp[j] = FLOAT_TYPE(0.0f); }

    const uint nblk = p.ncols / QUANT_K_Q8_1;
    uint blk = lane;
    // 4 Bloecke je Durchlauf (Loads unabhaengig -> mehr in Flight)
    while (blk + 3u*MALI_LPR < nblk) {
        mali_iter(temp, row, blk);
        mali_iter(temp, row, blk + MALI_LPR);
        mali_iter(temp, row, blk + 2u*MALI_LPR);
        mali_iter(temp, row, blk + 3u*MALI_LPR);
        blk += 4u*MALI_LPR;
    }
    if (blk + MALI_LPR < nblk) {
        mali_iter(temp, row, blk);
        mali_iter(temp, row, blk + MALI_LPR);
        blk += 2u*MALI_LPR;
    }
    if (blk < nblk) {
        mali_iter(temp, row, blk);
    }
'''
    s = s[:old_start] + new + s[old_end:]
    open(p, 'w').write(s); print('mul_mat_vecq.comp: Schleife entrollt')
else:
    print('schon entrollt')
