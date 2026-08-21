#!/usr/bin/env python3
"""battn_install.py — ggml-barra v2 (Attention-Offload) in den llama.cpp-Baum einbauen. Idempotent.
Nutzung: python3 battn_install.py [~/llama.cpp]"""
import os, sys, shutil

LL=os.path.expanduser(sys.argv[1] if len(sys.argv)>1 else "~/llama.cpp")
SP=os.path.dirname(os.path.abspath(__file__))
SRC="/mnt/c/Users/kevin/projects/pixel-cluster-base/src"
VK=os.path.join(LL,"ggml/src/ggml-vulkan")

# 1. Dateien kopieren
shutil.copy(os.path.join(SP,"ggml-vulkan-barra.inc"), os.path.join(VK,"ggml-vulkan-barra.inc"))
shutil.copy(os.path.join(SRC,"barra/barra.c"), os.path.join(VK,"barra.c"))
shutil.copy(os.path.join(SRC,"barra/barra.h"), os.path.join(VK,"barra.h"))
print("kopiert: ggml-vulkan-barra.inc, barra.c, barra.h")

# 2. ggml-vulkan.cpp patchen
p=os.path.join(VK,"ggml-vulkan.cpp"); s=open(p).read(); orig=s
OLD_DEF="static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {"
NEW_DEF="static ggml_status ggml_backend_vk_graph_compute_impl(ggml_backend_t backend, ggml_cgraph * cgraph) {"
if NEW_DEF not in s:
    assert OLD_DEF in s, "graph_compute-Definition nicht gefunden"
    s=s.replace(OLD_DEF,NEW_DEF,1)
DECL="static bool ggml_vk_battn_env_on(void);\n"
ANCH_OPT="static void ggml_vk_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * graph)"
if DECL not in s:
    assert ANCH_OPT in s, "graph_optimize nicht gefunden"
    s=s.replace(ANCH_OPT, DECL+ANCH_OPT,1)
GUARD='    if (ggml_vk_battn_env_on()) { return; }\n'
ANCH_LOG='VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");\n'
if GUARD not in s:
    assert ANCH_LOG in s, "graph_optimize-Log-Anker nicht gefunden"
    s=s.replace(ANCH_LOG, ANCH_LOG+GUARD,1)
INC='#include "ggml-vulkan-barra.inc"\n'
ANCH_IF="static ggml_backend_i ggml_backend_vk_interface = {"
if INC not in s:
    assert ANCH_IF in s, "Interface-Struct nicht gefunden"
    s=s.replace(ANCH_IF, INC+"\n"+ANCH_IF,1)
if s!=orig:
    open(p,"w").write(s); print("ggml-vulkan.cpp gepatcht")
else:
    print("ggml-vulkan.cpp schon gepatcht")

# 3. CMakeLists: barra.c mitcompilieren
c=os.path.join(VK,"CMakeLists.txt"); s=open(c).read()
ADD="\n# ggml-barra v2 (Attention-Offload)\ntarget_sources(ggml-vulkan PRIVATE barra.c)\n"
if "target_sources(ggml-vulkan PRIVATE barra.c)" not in s:
    open(c,"a").write(ADD); print("CMakeLists erweitert")
else:
    print("CMakeLists schon erweitert")
print("OK")
