#!/usr/bin/env python3
# Mali-Q6_K-MMVQ v2: 16-Byte-Loads auf dem gepolsterten 224-B-Layout (14 uvec4 je Block). Ersetzt den Funnel-Pfad. Idempotent.
import os
p = os.path.expanduser('~/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl'); s = open(p).read()
start = s.index('#if defined(MALI_Q6K_WIDE)')
end = s.index('#if defined(DATA_A_Q6_K) && !defined(MALI_Q6K_WIDE)')
new = r'''#if defined(MALI_Q6K_WIDE)
// Mali-Pfad Q6_K v2 auf 224-B-Geraete-Layout (Host-Repack): Block = 14 uvec4: ql 0..7, qh 8..11, scales 12, d in 13 (.x low half).
// Thread = 32er-Subblock j (h=j/4 Haelfte, r=j%4). Wert 4(q-32) = ((q<<2)^0x80) als signed Byte; 2 Skalen (je 16 Elemente).
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    const uint ib_k = ib_a / 8;
    const uint j = ib_a % 8;
    const uint h = j >> 2, r = j & 3u;
    const uint base = ib_k * 14u;
    const uvec4 ql0 = data_a_q4k_v4[base + h * 4u + (r & 1u) * 2u];
    const uvec4 ql1 = data_a_q4k_v4[base + h * 4u + (r & 1u) * 2u + 1u];
    const uvec4 qh0 = data_a_q4k_v4[base + 8u + h * 2u];
    const uvec4 qh1 = data_a_q4k_v4[base + 8u + h * 2u + 1u];
    const uvec4 scv = data_a_q4k_v4[base + 12u];
    const uint  dw  = data_a_q4k_v4[base + 13u].x;
    const uint ql_shift = (r >> 1) * 4u;
    const uint qh_shift = r * 2u;
    // scales[h*8 + 2r], [h*8 + 2r + 1]: Wort (h*8+2r)/4 = 2h + (r>>1), Byte-Offset (2r)&3 = (r&1)*2
    const uint scw = scv[2u * h + (r >> 1)] >> ((r & 1u) * 16u);
    const float sc_a = float(int(scw << 24) >> 24);
    const float sc_b = float(int((scw << 16) & 0xFF000000u) >> 24);
    const float d = unpackHalf2x16(dw & 0xFFFFu).x;
    int32_t qa = 0, qb = 0;
    uint v;
    v = ((((ql0.x >> ql_shift) & 0x0F0F0F0Fu) << 2) | (((qh0.x >> qh_shift) & 0x03030303u) << 6)) ^ 0x80808080u; qa = dotPacked4x8AccSatEXT(int(v), cache_b_qs[0], qa);
    v = ((((ql0.y >> ql_shift) & 0x0F0F0F0Fu) << 2) | (((qh0.y >> qh_shift) & 0x03030303u) << 6)) ^ 0x80808080u; qa = dotPacked4x8AccSatEXT(int(v), cache_b_qs[1], qa);
    v = ((((ql0.z >> ql_shift) & 0x0F0F0F0Fu) << 2) | (((qh0.z >> qh_shift) & 0x03030303u) << 6)) ^ 0x80808080u; qa = dotPacked4x8AccSatEXT(int(v), cache_b_qs[2], qa);
    v = ((((ql0.w >> ql_shift) & 0x0F0F0F0Fu) << 2) | (((qh0.w >> qh_shift) & 0x03030303u) << 6)) ^ 0x80808080u; qa = dotPacked4x8AccSatEXT(int(v), cache_b_qs[3], qa);
    v = ((((ql1.x >> ql_shift) & 0x0F0F0F0Fu) << 2) | (((qh1.x >> qh_shift) & 0x03030303u) << 6)) ^ 0x80808080u; qb = dotPacked4x8AccSatEXT(int(v), cache_b_qs[4], qb);
    v = ((((ql1.y >> ql_shift) & 0x0F0F0F0Fu) << 2) | (((qh1.y >> qh_shift) & 0x03030303u) << 6)) ^ 0x80808080u; qb = dotPacked4x8AccSatEXT(int(v), cache_b_qs[5], qb);
    v = ((((ql1.z >> ql_shift) & 0x0F0F0F0Fu) << 2) | (((qh1.z >> qh_shift) & 0x03030303u) << 6)) ^ 0x80808080u; qb = dotPacked4x8AccSatEXT(int(v), cache_b_qs[6], qb);
    v = ((((ql1.w >> ql_shift) & 0x0F0F0F0Fu) << 2) | (((qh1.w >> qh_shift) & 0x03030303u) << 6)) ^ 0x80808080u; qb = dotPacked4x8AccSatEXT(int(v), cache_b_qs[7], qb);
    return FLOAT_TYPE(float(cache_b_ds.x) * d * 0.25 * (sc_a * float(qa) + sc_b * float(qb)));
}
#endif

'''
if 'Q6_K v2' in s:
    print('schon v2')
else:
    s = s[:start] + new + s[end:]
    open(p, 'w').write(s); print('q6_K MMVQ v2 (224-B uvec4) eingebaut')
