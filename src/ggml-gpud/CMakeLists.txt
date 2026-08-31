ggml_add_backend_library(ggml-gpud
                         ggml-gpud.cpp
                         barra.c
                        )
target_include_directories(ggml-gpud PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
