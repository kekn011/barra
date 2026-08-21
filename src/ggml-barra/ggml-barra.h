#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#ifdef  __cplusplus
extern "C" {
#endif
GGML_BACKEND_API ggml_backend_t     ggml_backend_barra_init(void);
GGML_BACKEND_API bool               ggml_backend_is_barra(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_barra_reg(void);
#ifdef  __cplusplus
}
#endif
