import re, shutil, sys, os
home = os.path.expanduser('~')
wc = home + '/whisper.cpp'
srcdir = '/mnt/c/Users/kevin/projects/pixel-cluster-base/src'
# 1) Dateien einkopieren
shutil.copy(srcdir + '/experiments/gpu-attn/barra.c', wc + '/src/barra.c')
shutil.copy(srcdir + '/experiments/gpu-attn/barra.h', wc + '/src/barra.h')
shutil.copy(srcdir + '/whisper-mali/tpu/whisper-barra.cpp', wc + '/src/whisper-barra.cpp')

f = wc + '/src/whisper.cpp'
s = open(f, encoding='utf-8').read()
if 'wsp_barra_encode' in s:
    print('ALREADY_PATCHED'); sys.exit(0)

# 2a) Prototyp nach den Includes
anchor = 'static bool whisper_encode_external(const whisper_state & wstate) {'
assert anchor in s
s = s.replace(anchor, 'extern "C" int wsp_barra_encode(const float*, float*, int, int);\n\n' + anchor, 1)

# 2b) build_graph_encoder: barra-Zweig direkt nach Graph-Anlage
anchor2 = '''    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, WHISPER_MAX_NODES, false);

    struct ggml_tensor * cur = ggml_view_tensor(ctx0, wstate.embd_conv);'''
assert anchor2 in s
s = s.replace(anchor2, '''    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, WHISPER_MAX_NODES, false);

    if (getenv("WHISPER_BARRA")) {
        // barra-TPU-Encoder: embd_enc als Input-Tensor, Inhalt kommt aus wsp_barra_encode()
        struct ggml_tensor * cur_b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx);
        ggml_set_input(cur_b);
        ggml_set_name(cur_b, "embd_enc");
        wstate.embd_enc = cur_b;
        ggml_set_output(cur_b);
        ggml_build_forward_expand(gf, cur_b);
        ggml_free(ctx0);
        return gf;
    }

    struct ggml_tensor * cur = ggml_view_tensor(ctx0, wstate.embd_conv);''', 1)

# 2c) encode_internal: statt Encoder-Graph rechnen -> unsere Kette
anchor3 = '''        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
    }'''
# kommt 2x vor (conv+encoder Abschnitte aehneln sich) -> den ENCODER-Block anhand des Kontexts treffen
enc_block = '''        ggml_cgraph * gf = whisper_build_graph_encoder(wctx, wstate);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
    }'''
assert enc_block in s
s = s.replace(enc_block, '''        ggml_cgraph * gf = whisper_build_graph_encoder(wctx, wstate);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        if (getenv("WHISPER_BARRA")) {
            const auto & model_b = wctx.model;
            const int T  = (int) wstate.embd_conv->ne[0];
            const int NS = (int) wstate.embd_conv->ne[1];
            std::vector<float> conv_b((size_t)T*NS), x0_b((size_t)T*NS), enc_b((size_t)T*NS), pe_b((size_t)T*NS), lw_b(NS), lb_b(NS);
            ggml_backend_tensor_get(wstate.embd_conv, conv_b.data(), 0, (size_t)T*NS*sizeof(float));
            ggml_backend_tensor_get(model_b.e_pe, pe_b.data(), 0, (size_t)T*NS*sizeof(float));
            for (int t = 0; t < T; ++t) for (int c = 0; c < NS; ++c) x0_b[(size_t)t*NS + c] = conv_b[(size_t)c*T + t] + pe_b[(size_t)t*NS + c];
            if (wsp_barra_encode(x0_b.data(), enc_b.data(), T, NS) != 0) return false;
            ggml_backend_tensor_get(model_b.e_ln_w, lw_b.data(), 0, NS*sizeof(float));
            ggml_backend_tensor_get(model_b.e_ln_b, lb_b.data(), 0, NS*sizeof(float));
            for (int t = 0; t < T; ++t) {
                float * r = enc_b.data() + (size_t)t*NS;
                double m = 0; for (int c = 0; c < NS; ++c) m += r[c]; m /= NS;
                double v = 0; for (int c = 0; c < NS; ++c) { double e = r[c]-m; v += e*e; } v /= NS;
                float inv = (float)(1.0/sqrt(v + 1e-5));
                for (int c = 0; c < NS; ++c) r[c] = (float)((r[c]-m)*inv)*lw_b[c] + lb_b[c];
            }
            ggml_backend_tensor_set(wstate.embd_enc, enc_b.data(), 0, (size_t)T*NS*sizeof(float));
        } else
        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
    }''', 1)
open(f, 'w', encoding='utf-8').write(s)

# 3) CMake: barra.c + whisper-barra.cpp in die whisper-Lib
cm = wc + '/src/CMakeLists.txt'
c = open(cm, encoding='utf-8').read()
if 'whisper-barra.cpp' not in c:
    assert 'whisper.cpp' in c
    c = c.replace('whisper.cpp', 'whisper.cpp\n            barra.c\n            whisper-barra.cpp', 1)
    open(cm, 'w', encoding='utf-8').write(c)
print('PATCH_OK')
