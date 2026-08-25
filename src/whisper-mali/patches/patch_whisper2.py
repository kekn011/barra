import os
f = os.path.expanduser('~/whisper.cpp/src/whisper.cpp')
s = open(f, encoding='utf-8').read()
old = '''            ggml_backend_tensor_set(wstate.embd_enc, enc_b.data(), 0, (size_t)T*NS*sizeof(float));
        } else'''
new = '''            ggml_backend_tensor_set(wstate.embd_enc, enc_b.data(), 0, (size_t)T*NS*sizeof(float));
            ggml_backend_sched_reset(sched);   // wie ggml_graph_compute_helper: sonst assert !is_alloc beim 2. Fenster
        } else'''
assert old in s
s = s.replace(old, new, 1)
open(f, 'w', encoding='utf-8').write(s)
print('PATCH2_OK')
