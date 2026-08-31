#pragma once
// ggml-gpud: ggml-Backend fuer glibc-Programme (Container/Pod) auf der Mali-G715 ueber den Android-seitigen
// gpud-zc-Daemon (v3): Tensoren liegen in dmabufs (Zero-Copy, UMA), der Graph geht als Stufenliste in EINEM
// Roundtrip je Batch. Konzept: ggml-gpud-Konzept (30.8.26), Etappe M1.
#include "ggml.h"
#include "ggml-backend.h"
#ifdef __cplusplus
extern "C" {
#endif
GGML_BACKEND_API ggml_backend_t            ggml_backend_gpud_init(void);
GGML_BACKEND_API bool                      ggml_backend_is_gpud(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_gpud_buffer_type(void);
GGML_BACKEND_API ggml_backend_reg_t         ggml_backend_gpud_reg(void);
#ifdef __cplusplus
}
#endif
