#!/usr/bin/env python3
# q6_K-Repack: Geraete-Layout 224 B je Block (16-B-aligned) statt 210. Host: Alloc-Groesse, Offsets, set/get/cpy/memset,
# Subbuffer-Groessen; Shader: Struct-Padding in types.glsl (alle q6_K-Shader erben den Stride automatisch). Idempotent.
import os, re
root = os.path.expanduser('~/llama.cpp/ggml/src/ggml-vulkan/')

# ---------- types.glsl ----------
p = root + 'vulkan-shaders/types.glsl'; s = open(p).read()
if 'Q6K_DEV_PAD' not in s:
    s = s.replace('''struct block_q6_K
{
    uint8_t ql[QUANT_K_Q6_K/2];
    uint8_t qh[QUANT_K_Q6_K/4];
    int8_t scales[QUANT_K_Q6_K/16];
    float16_t d;
};

struct block_q6_K_packed16
{
    uint16_t ql[QUANT_K_Q6_K/2/2];
    uint16_t qh[QUANT_K_Q6_K/4/2];
    int16_t scales[QUANT_K_Q6_K/16/2];
    float16_t d;
};
''', '''// Q6K_DEV_PAD: Geraete-Layout 224 B je Block (210 + 14 Padding) -> 16-Byte-aligned; Host repackt bei set/get_tensor (Mali-Fork)
struct block_q6_K
{
    uint8_t ql[QUANT_K_Q6_K/2];
    uint8_t qh[QUANT_K_Q6_K/4];
    int8_t scales[QUANT_K_Q6_K/16];
    float16_t d;
    uint8_t pad[14];
};

struct block_q6_K_packed16
{
    uint16_t ql[QUANT_K_Q6_K/2/2];
    uint16_t qh[QUANT_K_Q6_K/4/2];
    int16_t scales[QUANT_K_Q6_K/16/2];
    float16_t d;
    uint16_t pad[7];
};
''')
    assert 'Q6K_DEV_PAD' in s
    open(p, 'w').write(s); print('types.glsl: q6_K auf 224 B gepolstert')
else:
    print('types.glsl: schon')

# ---------- ggml-vulkan.cpp ----------
p = root + 'ggml-vulkan.cpp'; s = open(p).read()
if 'VK_Q6K_DEV_BLK' not in s:
    # 1) Helfer hinter vk_tensor_offset
    old = '''static uint32_t get_misalign_bytes(const ggml_backend_vk_context * ctx, const ggml_tensor * t)
{
    return ((vk_tensor_offset(t) + t->view_offs) & (ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1));;
}'''
    new = '''// ---- q6_K-Repack (Mali-Fork): Geraete-Layout 224 B je Block statt 210 (16-Byte-aligned) ----
static constexpr uint64_t VK_Q6K_BLK = 210, VK_Q6K_DEV_BLK = 224;
static inline bool vk_is_q6k(const ggml_tensor * t) { return t->type == GGML_TYPE_Q6_K; }
// ggml-Byteposition (210er-Stride) -> Geraete-Byteposition (224er-Stride); Layout innerhalb des Blocks identisch
static inline uint64_t vk_q6k_dev_pos(uint64_t o) { return (o / VK_Q6K_BLK) * VK_Q6K_DEV_BLK + (o % VK_Q6K_BLK); }
static inline uint64_t vk_q6k_dev_size(uint64_t off, uint64_t size) { return size == 0 ? 0 : vk_q6k_dev_pos(off + size - 1) + 1 - vk_q6k_dev_pos(off); }
static inline uint64_t vk_q6k_dev_bytes_for(uint64_t ggml_bytes) { return (ggml_bytes / VK_Q6K_BLK) * VK_Q6K_DEV_BLK + (ggml_bytes % VK_Q6K_BLK ? VK_Q6K_DEV_BLK : 0); }
static inline uint64_t vk_dev_nbytes(const ggml_tensor * t) { return vk_is_q6k(t) ? vk_q6k_dev_bytes_for(ggml_nbytes(t)) : ggml_nbytes(t); }
static uint64_t vk_tensor_dev_offset(const ggml_tensor * t) {
    uint64_t vo = t->view_offs;
    if (vo != 0 && vk_is_q6k(t)) {
        GGML_ASSERT(vo % VK_Q6K_BLK == 0);   // Views auf q6_K nur an Blockgrenzen
        vo = vk_q6k_dev_pos(vo);
    }
    return vk_tensor_offset(t) + vo;
}

static uint32_t get_misalign_bytes(const ggml_backend_vk_context * ctx, const ggml_tensor * t)
{
    return ((vk_tensor_dev_offset(t)) & (ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1));;
}'''
    assert old in s; s = s.replace(old, new)
    # 2) alle "vk_tensor_offset(X) + X->view_offs" -> vk_tensor_dev_offset(X)
    s, n = re.subn(r'vk_tensor_offset\(([A-Za-z_][A-Za-z0-9_\[\]]*)\) \+ \1->view_offs', r'vk_tensor_dev_offset(\1)', s)
    print('view_offs-Ersetzungen:', n)
    # 3) get_alloc_size
    old = '''static size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_nbytes(tensor);
'''
    new = '''static size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return vk_dev_nbytes(tensor);   // q6_K: 224 B je Block auf dem Geraet
'''
    assert old in s; s = s.replace(old, new)
    # 4) tensor_subbuffer Groesse + UMA-Host-Schutz
    old = '''    GGML_ASSERT(buffer != nullptr);

    size_t size = ggml_nbytes(tensor);
'''
    new = '''    GGML_ASSERT(buffer != nullptr);
    // q6_K liegt auf dem Geraet gepolstert (224 B/Block); ein q6_K-Tensor im Host-Speicher (UMA-Pfad) haette ggml-Layout -> nicht zulaessig
    GGML_ASSERT(!(vk_is_q6k(tensor) && ctx->device->uma && ggml_backend_buffer_is_host(tensor->buffer)));

    size_t size = vk_dev_nbytes(tensor);
'''
    assert old in s; s = s.replace(old, new)
    # 5) qx_sz-Berechnungen (4 Stellen)
    a = 'const uint64_t qx_sz = ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type);'
    b = 'const uint64_t qx_sz = vk_is_q6k(src0) ? vk_q6k_dev_bytes_for(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type)) : ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type);'
    n1 = s.count(a); s = s.replace(a, b)
    a2 = 'const uint64_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);'
    b2 = 'const uint64_t qx_sz = ggml_vk_align_size(vk_is_q6k(src0) ? vk_q6k_dev_bytes_for(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type)) : ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);'
    n2 = s.count(a2); s = s.replace(a2, b2)
    print('qx_sz-Stellen:', n1, n2)
    # 6) set/get/cpy/memset/2d
    old = '''static void ggml_backend_vk_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {'''
    new = '''// q6_K: Host-Daten (210er-Stride) blockweise in das 224er-Geraete-Layout schreiben / zurueckholen (chunked)
static void vk_q6k_write(vk_buffer & buf, uint64_t dev_base, const uint8_t * data, size_t offset, size_t size) {
    const size_t CHUNK_BLOCKS = 65536;   // ~14 MB Geraete-Bytes je Chunk
    size_t pos = offset, end = offset + size;
    std::vector<uint8_t> tmp;
    while (pos < end) {
        size_t chunk_end = std::min(end, (pos / VK_Q6K_BLK + CHUNK_BLOCKS) * VK_Q6K_BLK);
        uint64_t d0 = vk_q6k_dev_pos(pos), d1 = vk_q6k_dev_pos(chunk_end - 1) + 1;
        tmp.assign(d1 - d0, 0);
        for (size_t q = pos; q < chunk_end; ) {
            size_t within = q % VK_Q6K_BLK;
            size_t n = std::min(chunk_end - q, (size_t)VK_Q6K_BLK - within);
            memcpy(tmp.data() + (vk_q6k_dev_pos(q) - d0), data + (q - offset), n);
            q += n;
        }
        ggml_vk_buffer_write(buf, dev_base + d0, tmp.data(), d1 - d0);
        pos = chunk_end;
    }
}
static void vk_q6k_read(vk_buffer & buf, uint64_t dev_base, uint8_t * data, size_t offset, size_t size) {
    const size_t CHUNK_BLOCKS = 65536;
    size_t pos = offset, end = offset + size;
    std::vector<uint8_t> tmp;
    while (pos < end) {
        size_t chunk_end = std::min(end, (pos / VK_Q6K_BLK + CHUNK_BLOCKS) * VK_Q6K_BLK);
        uint64_t d0 = vk_q6k_dev_pos(pos), d1 = vk_q6k_dev_pos(chunk_end - 1) + 1;
        tmp.resize(d1 - d0);
        ggml_vk_buffer_read(buf, dev_base + d0, tmp.data(), d1 - d0);
        for (size_t q = pos; q < chunk_end; ) {
            size_t within = q % VK_Q6K_BLK;
            size_t n = std::min(chunk_end - q, (size_t)VK_Q6K_BLK - within);
            memcpy(data + (q - offset), tmp.data() + (vk_q6k_dev_pos(q) - d0), n);
            q += n;
        }
        pos = chunk_end;
    }
}

static void ggml_backend_vk_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {'''
    assert old in s; s = s.replace(old, new)
    # memset: Geraete-Bereich
    old = '''    ggml_vk_buffer_memset(buf, vk_tensor_dev_offset(tensor) + offset, val32, size);'''
    new = '''    if (vk_is_q6k(tensor)) {
        ggml_vk_buffer_memset(buf, vk_tensor_dev_offset(tensor) + vk_q6k_dev_pos(offset), val32, vk_q6k_dev_size(offset, size));
    } else {
        ggml_vk_buffer_memset(buf, vk_tensor_dev_offset(tensor) + offset, val32, size);
    }'''
    assert old in s; s = s.replace(old, new)
    # set_tensor
    old = '''    ggml_vk_buffer_write(buf, vk_tensor_dev_offset(tensor) + offset, data, size);
}'''
    new = '''    if (vk_is_q6k(tensor)) {
        vk_q6k_write(buf, vk_tensor_dev_offset(tensor), (const uint8_t *) data, offset, size);
        return;
    }
    ggml_vk_buffer_write(buf, vk_tensor_dev_offset(tensor) + offset, data, size);
}'''
    assert s.count(old) == 1; s = s.replace(old, new)
    # set_tensor_2d
    old = '''    ggml_vk_buffer_write_2d(buf, vk_tensor_dev_offset(tensor) + offset, data, stride_data, stride_tensor, size, n_copies);
}'''
    new = '''    if (vk_is_q6k(tensor)) {
        for (size_t i = 0; i < n_copies; i++) {
            vk_q6k_write(buf, vk_tensor_dev_offset(tensor), (const uint8_t *) data + i * stride_data, offset + i * stride_tensor, size);
        }
        return;
    }
    ggml_vk_buffer_write_2d(buf, vk_tensor_dev_offset(tensor) + offset, data, stride_data, stride_tensor, size, n_copies);
}'''
    assert s.count(old) == 1; s = s.replace(old, new)
    # get_tensor
    old = '''    ggml_vk_buffer_read(buf, vk_tensor_dev_offset(tensor) + offset, data, size);
}'''
    new = '''    if (vk_is_q6k(tensor)) {
        vk_q6k_read(buf, vk_tensor_dev_offset(tensor), (uint8_t *) data, offset, size);
        return;
    }
    ggml_vk_buffer_read(buf, vk_tensor_dev_offset(tensor) + offset, data, size);
}'''
    assert s.count(old) == 1; s = s.replace(old, new)
    # get_tensor_2d
    old = '''    ggml_vk_buffer_read_2d(buf, vk_tensor_dev_offset(tensor) + offset, data, stride_tensor, stride_data, size, n_copies);
}'''
    new = '''    if (vk_is_q6k(tensor)) {
        for (size_t i = 0; i < n_copies; i++) {
            vk_q6k_read(buf, vk_tensor_dev_offset(tensor), (uint8_t *) data + i * stride_data, offset + i * stride_tensor, size);
        }
        return;
    }
    ggml_vk_buffer_read_2d(buf, vk_tensor_dev_offset(tensor) + offset, data, stride_tensor, stride_data, size, n_copies);
}'''
    assert s.count(old) == 1; s = s.replace(old, new)
    # cpy_tensor
    old = '''        ggml_vk_buffer_copy(dst_buf, vk_tensor_dev_offset(dst), src_buf, vk_tensor_dev_offset(src), ggml_nbytes(src));'''
    new = '''        GGML_ASSERT(vk_is_q6k(src) == vk_is_q6k(dst));
        ggml_vk_buffer_copy(dst_buf, vk_tensor_dev_offset(dst), src_buf, vk_tensor_dev_offset(src), vk_dev_nbytes(src));'''
    assert old in s; s = s.replace(old, new)
    open(p, 'w').write(s); print('ggml-vulkan.cpp: Repack eingebaut')
else:
    print('ggml-vulkan.cpp: schon')
