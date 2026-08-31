#!/usr/bin/env python3
# ggml-rpc entschlacken (Messreihe 30.8.26, barra-13c0, Qwen2.5-1.5B Q4_K_M, llama.cpp 8d274dd):
#   (1) Client: Sockets leben so lange wie der Prozess (vorher weak_ptr-Cache -> 35..265 HELLO-Roundtrips
#       je Lauf, jeder ein Reconnect gegen den Ein-Verbindungs-rpc-server)
#   (2) Client: GET_ALLOC_SIZE-Antworten lokal cachen (vorher 3 Roundtrips je Token fuer FLASH_ATTN_EXT)
#   (3) Server: graph_optimize des echten Backends aufrufen (der Client-Scheduler optimiert nur sein
#       RPC-Backend, das keine Optimierung hat)
#   (4) Server: Graph per ggml_build_forward_expand aufbauen, damit visited_hash_set/use_counts gefuellt
#       sind - ohne sie liefert ggml_node_get_use_count() 0 und ggml_can_fuse() lehnt jede Fusion ab
# Wirkung lokal (Pod auf dem GPU-Node): Decode 14,6 -> 16,5..16,8 t/s (nativ 22,6); Rechnen 51,5 -> 47..48 ms.
# Anwendung:  python3 ggml-rpc-slim.py <llama.cpp>/ggml/src/ggml-rpc/ggml-rpc.cpp
import re, sys

p = sys.argv[1]; s = open(p).read(); NL = chr(92) + 'n'
if 'barra: rpc-slim' in s:
    print("schon gepatcht"); sys.exit(0)

def rep(old, new, tag):
    global s
    assert old in s, "Anker fehlt: " + tag
    s = s.replace(old, new, 1)

# (1) persistente Sockets
rep("    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;\n\n"
    "    auto it = sockets.find(endpoint);\n    if (it != sockets.end()) {\n"
    "        if (auto sock = it->second.lock()) {\n            return sock;\n        }\n    }\n",
    "    // barra: rpc-slim (1) - Verbindungen leben so lange wie der Prozess (der rpc-server bedient genau\n"
    "    // EINE Verbindung; jede Neuverbindung kostet einen HELLO-Roundtrip).\n"
    "    static std::unordered_map<std::string, std::shared_ptr<socket_t>> sockets;\n\n"
    "    auto it = sockets.find(endpoint);\n    if (it != sockets.end()) {\n        return it->second;\n    }\n",
    "socket-cache")

# (2) alloc_size-Cache
rep("    if (rpc_get) {\n"
    "        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;\n"
    "        auto sock = get_socket(buft_ctx->endpoint);\n",
    "    if (rpc_get) {\n"
    "        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;\n"
    "        // barra: rpc-slim (2) - die Antwort haengt nur von Op/Typ/Formen ab: einmal fragen, dann lokal.\n"
    "        static std::mutex cache_mutex;\n"
    "        static std::unordered_map<std::string, size_t> cache;\n"
    "        char key[512]; int kl = snprintf(key, sizeof(key), \"%s|%d|%d|%lld,%lld,%lld,%lld\", buft_ctx->endpoint.c_str(), (int)tensor->op, (int)tensor->type,\n"
    "            (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2], (long long)tensor->ne[3]);\n"
    "        for (int i = 0; i < GGML_MAX_SRC && tensor->src[i] && kl < (int)sizeof(key) - 64; i++)\n"
    "            kl += snprintf(key + kl, sizeof(key) - kl, \"|%d:%lld,%lld,%lld,%lld\", (int)tensor->src[i]->type,\n"
    "                (long long)tensor->src[i]->ne[0], (long long)tensor->src[i]->ne[1], (long long)tensor->src[i]->ne[2], (long long)tensor->src[i]->ne[3]);\n"
    "        {\n            std::lock_guard<std::mutex> lock(cache_mutex);\n"
    "            auto cit = cache.find(key);\n            if (cit != cache.end()) return cit->second;\n        }\n"
    "        auto sock = get_socket(buft_ctx->endpoint);\n",
    "alloc_size")
m = re.search(r"(RPC_CMD_GET_ALLOC_SIZE[^\n]*\n(?:.*\n){0,6}?)(\s*)return response\.alloc_size;", s)
assert m, "Anker fehlt: alloc_size return"
s = s[:m.start(2)] + m.group(2) + "{ std::lock_guard<std::mutex> lock(cache_mutex); cache[key] = response.alloc_size; }" \
    + m.group(2) + "return response.alloc_size;" + s[m.end():]

# (4) Graph mit Hash-Set/use_counts aufbauen
rep("    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);",
    "    // barra: rpc-slim (4) - Graph gross genug fuer Knoten UND Blaetter (ggml_build_forward_expand fuellt\n"
    "    // visited_hash_set und use_counts; ohne sie lehnt ggml_can_fuse() jede Backend-Fusion ab).\n"
    "    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes + n_tensors, false);",
    "buf_size")
rep("    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);\n    graph->n_nodes = n_nodes;",
    "    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes + n_tensors, false);",
    "new_graph")
start = s.index("        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);")
endmark = "            return false;\n        }\n    }\n"
end = s.index(endmark, start) + len(endmark)
s = s[:start] + (
    "        ggml_tensor * node = create_node(id, ctx, tensor_ptrs, tensor_map);\n"
    "        // id == 0 -> nullptr ist erwartet; sonst Deserialisierungsfehler\n"
    "        if (node == nullptr && id != 0) {\n"
    "            GGML_LOG_ERROR(\"[%s] failed to create graph node %d (id=%\" PRId64 \")" + NL + "\", __func__, i, id);\n"
    "            return false;\n"
    "        }\n"
    "        if (node != nullptr) {\n"
    "            ggml_build_forward_expand(graph, node);   // barra: rpc-slim (4)\n"
    "        }\n"
    "    }\n"
    "    if ((uint32_t)graph->n_nodes != n_nodes) {\n"
    "        GGML_LOG_WARN(\"[%s] Graph nach build_forward_expand: %d Knoten statt %u" + NL + "\", __func__, graph->n_nodes, n_nodes);\n"
    "    }\n") + s[end:]

# (3) graph_optimize des echten Backends (nur im graph_compute-Pfad, dort wo der Graph gespeichert wird)
old = ("    ggml_status status = ggml_backend_graph_compute(backends[device], graph);\n"
       "    GGML_ASSERT(status == GGML_STATUS_SUCCESS && \"Unsuccessful graph computations are not supported with RPC\");\n"
       "    stored_graphs[device].graph = graph;\n")
rep(old,
    "    // barra: rpc-slim (3) - das echte Backend sieht den Graphen hier zum ersten Mal: optimieren (Vulkan:\n"
    "    // Umordnung fuer Fusionen), der Client-Scheduler kennt nur das RPC-Backend ohne Optimierung.\n"
    "    if (backends[device]->iface.graph_optimize) {\n"
    "        backends[device]->iface.graph_optimize(backends[device], graph);\n"
    "    }\n" + old,
    "graph_compute")

open(p, 'w').write(s); print("rpc-slim gepatcht:", p)
