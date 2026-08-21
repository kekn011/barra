# ggml-barra: TPU-FFN-Backend (Tensor G3) ueber libbarra/tpud. Quelle: pixel-cluster-base/src/ggml-barra (+ src/barra/barra.c|h)
ggml_add_backend_library(ggml-barra ggml-barra.cpp barra.c)
target_include_directories(ggml-barra PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ggml-barra PRIVATE ggml-cpu)
