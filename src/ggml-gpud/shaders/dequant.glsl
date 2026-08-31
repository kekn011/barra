// Byte-Zugriff + Dequantisierung fuer Gewichte im Binding 0 (uint src0[]).
// Layouts exakt nach ggml (block_q4_K 144 B, block_q6_K 210 B, block_q8_0 34 B, f16, f32).
// Korrektheit vor Tempo (M1); die schnellen Pfade (mmvq/GEMM) kommen in M2.
uint u8(uint boff)  { return (src0[boff >> 2] >> ((boff & 3u) * 8u)) & 0xFFu; }
uint u16at(uint boff){ return u8(boff) | (u8(boff + 1u) << 8); }
float f16at(uint boff){ return unpackHalf2x16(u16at(boff)).x; }
float f32at(uint boff){ return uintBitsToFloat(src0[boff >> 2]); }

// q4_K: d(f16) dmin(f16) scales[12] qs[128]; 256 Elemente
void q4k_sm(uint blk, uint j, out float sc, out float mn) {
    uint q = blk + 4u;
    if (j < 4u) { sc = float(u8(q + j) & 63u); mn = float(u8(q + j + 4u) & 63u); }
    else {
        sc = float((u8(q + j + 4u) & 0xFu) | ((u8(q + j - 4u) >> 6) << 4));
        mn = float((u8(q + j + 4u) >> 4)   | ((u8(q + j)      >> 6) << 4));
    }
}
float deq_q4k(uint rowb, uint k) {
    uint blk = rowb + (k >> 8) * 144u; uint i = k & 255u;
    float d = f16at(blk), dmin = f16at(blk + 2u);
    uint is = (i >> 6) * 2u + ((i & 63u) >> 5);
    float sc, mn; q4k_sm(blk, is, sc, mn);
    uint qb = blk + 16u + (i >> 6) * 32u + (i & 31u);
    uint q = ((i & 63u) < 32u) ? (u8(qb) & 0xFu) : (u8(qb) >> 4);
    return d * sc * float(q) - dmin * mn;
}
// q6_K: ql[128] qh[64] scales[16](int8) d(f16 @208); 256 Elemente
float deq_q6k(uint rowb, uint k) {
    uint blk = rowb + (k >> 8) * 210u; uint i = k & 255u;
    uint h = i >> 7; uint r = i & 127u; uint l = r & 31u; uint seg = r >> 5; uint is = l >> 4;
    uint qlb = blk + h * 64u; uint qhb = blk + 128u + h * 32u; uint scb = blk + 192u + h * 8u;
    uint qlv = (seg == 0u || seg == 2u) ? u8(qlb + l) : u8(qlb + l + 32u);
    uint low = (seg < 2u) ? (qlv & 0xFu) : (qlv >> 4);
    uint hi  = (u8(qhb + l) >> (seg * 2u)) & 3u;
    int q = int(low | (hi << 4)) - 32;
    int sc = int(u8(scb + is + seg * 2u)); if (sc > 127) sc -= 256;
    return f16at(blk + 208u) * float(sc) * float(q);
}
// q8_0: d(f16) qs[32](int8); 32 Elemente
float deq_q8_0(uint rowb, uint k) {
    uint blk = rowb + (k >> 5) * 34u; uint i = k & 31u;
    int q = int(u8(blk + 2u + i)); if (q > 127) q -= 256;
    return f16at(blk) * float(q);
}
// type: 0 f32, 1 f16, 2 q4_K, 3 q6_K, 4 q8_0   (GGML-Typnummern werden im Backend abgebildet)
float deq(uint type, uint rowb, uint k) {
    if (type == 0u) return f32at(rowb + k * 4u);
    if (type == 1u) return f16at(rowb + k * 2u);
    if (type == 2u) return deq_q4k(rowb, k);
    if (type == 3u) return deq_q6k(rowb, k);
    return deq_q8_0(rowb, k);
}
