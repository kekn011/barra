#!/usr/bin/env python3
# q6_K-Repack, Teil 2: die ASYNC-Pfade (set/get_tensor_async, cpy_tensor_async) schrieben/lasen ohne Repack (dev_offset + offset).
# Genau die nutzt der Model-Loader (Upload-Backend, 1-MB-Chunks) und der Scheduler (Weight-Kopien) -> NaN-PPL / '8888'-Generation
# in ALLEN Varianten, waehrend test-backend-ops (synchroner set_tensor) bestand. Fix: q6_K in den async-Pfaden auf den
# synchronen Repack-Pfad umleiten; cpy_tensor_async fuer q6_K ablehnen (-> generischer sync Fallback). Idempotent.
import os
p = os.path.expanduser('~/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp'); s = open(p).read()
if 'Q6K_ASYNC_FIX' in s:
    print('already patched'); raise SystemExit
n0 = len(s)

# 1) set_tensor_2d_async: q6_K -> synchroner Repack-Write je Kopie
old = '''    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    if (size == 0) {
        return;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context cpy_ctx;
'''
new = '''    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    if (size == 0) {
        return;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    // Q6K_ASYNC_FIX: q6_K muss repackt werden (224-B-Geraetelayout) -> synchroner Repack-Pfad (Loader-Chunks duerfen Bloecke teilen)
    if (vk_is_q6k(tensor)) {
        vk_buffer qbuf = buf_ctx->dev_buffer;
        for (size_t i = 0; i < n_copies; i++) {
            vk_q6k_write(qbuf, vk_tensor_dev_offset(tensor), (const uint8_t *) data + i * stride_data, offset + i * stride_tensor, size);
        }
        return;
    }

    vk_context cpy_ctx;
'''
assert s.count(old) == 1; s = s.replace(old, new)

# 2) get_tensor_2d_async: q6_K -> erst synchronisieren, dann synchroner Repack-Read
old = '''    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    if (size == 0) {
        return;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
'''
new = '''    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    if (size == 0) {
        return;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    // Q6K_ASYNC_FIX: q6_K liegt repackt -> ausstehende Arbeit abschliessen, dann synchroner Repack-Read
    if (vk_is_q6k(tensor)) {
        ggml_vk_synchronize(ctx);
        vk_buffer qbuf = buf_ctx->dev_buffer;
        for (size_t i = 0; i < n_copies; i++) {
            vk_q6k_read(qbuf, vk_tensor_dev_offset(tensor), (uint8_t *) data + i * stride_data, offset + i * stride_tensor, size);
        }
        return;
    }

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
'''
assert s.count(old) == 1; s = s.replace(old, new)

# 3) cpy_tensor_async: vk->vk mit Geraete-Bytes (beide q6_K) ; host->vk fuer q6_K ablehnen (Fallback = sync set_tensor mit Repack)
old = '''    if (dst->buffer->buft != ggml_backend_vk_get_default_buffer_type(backend_dst)) {
        return false;
    }

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

    if (ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;

        // Async copy only works within the same device
        if (src_buf_ctx->dev_buffer->device != dst_buf->device) {
            return false;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_buffer_copy_async(compute_ctx, dst_buf, vk_tensor_dev_offset(dst),
                                   src_buf_ctx->dev_buffer, vk_tensor_dev_offset(src),
                                   ggml_nbytes(src));
        return true;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
'''
new = '''    if (dst->buffer->buft != ggml_backend_vk_get_default_buffer_type(backend_dst)) {
        return false;
    }

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

    if (ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;

        // Async copy only works within the same device
        if (src_buf_ctx->dev_buffer->device != dst_buf->device) {
            return false;
        }

        // Q6K_ASYNC_FIX: vk->vk nur gleichartig (beide repackt), Geraete-Bytes kopieren
        if (vk_is_q6k(src) != vk_is_q6k(dst)) {
            return false;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_buffer_copy_async(compute_ctx, dst_buf, vk_tensor_dev_offset(dst),
                                   src_buf_ctx->dev_buffer, vk_tensor_dev_offset(src),
                                   vk_dev_nbytes(src));
        return true;
    }

    // Q6K_ASYNC_FIX: host->vk fuer q6_K nicht async (kein Repack dort) -> Aufrufer faellt auf synchronen set_tensor zurueck
    if (vk_is_q6k(dst)) {
        return false;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
'''
assert s.count(old) == 1; s = s.replace(old, new)

# 4) Ueberlappungs-Check mit Geraete-Bytes
old = '''        auto a_base = vk_tensor_dev_offset(a);
        auto a_size = ggml_nbytes(a);
        auto b_base = vk_tensor_dev_offset(b);
        auto b_size = ggml_nbytes(b);
'''
new = '''        auto a_base = vk_tensor_dev_offset(a);
        auto a_size = vk_dev_nbytes(a);   // Q6K_ASYNC_FIX
        auto b_base = vk_tensor_dev_offset(b);
        auto b_size = vk_dev_nbytes(b);
'''
assert s.count(old) == 1; s = s.replace(old, new)

open(p, 'w').write(s); print('patched', n0, '->', len(s))
