#!/usr/bin/env python3
"""LOCAL ONLY: apply the JOIN4 patch set to a pin checkout (mirror of the
kernel's patch function; the kernel embeds this logic + join4_phase6.h).

Usage: python3 probes/edge0_port/join4_apply.py /tmp/lp2
Verifies every anchor, then build with cmake to prove the C compiles.
"""
import sys
from pathlib import Path

EDGE0 = Path(__file__).resolve().parent
J4 = (EDGE0 / "join4_phase6.h").read_text()


def replace_once(path, old, new):
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{path.name}: anchor count {n} != 1 :: {old[:70]!r}")
    path.write_text(text.replace(old, new))
    print(f"  ok: {path.name} :: {old[:50]!r}")


def main():
    llama = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/llamapin")
    cpu = llama / "ggml/src/ggml-cpu/ggml-cpu.c"
    replace_once(cpu, "static struct ggml_state g_state = {0};\n",
                 J4 + "\nstatic struct ggml_state g_state = {0};\n")
    replace_once(cpu,
                 "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
                 "        return;\n"
                 "    }\n"
                 "\n"
                 "    // extra_buffer op?\n",
                 "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
                 "        return;\n"
                 "    }\n"
                 "\n"
                 "    ggml_phase6_route_trace(params, tensor);\n"
                 "\n"
                 "    // extra_buffer op?\n")
    replace_once(cpu,
                 "    const int ith = params->ith;\n"
                 "    const int nth = params->nth;\n\n"
                 "    const enum ggml_type type = src0->type;\n\n"
                 "    const bool src1_cont = ggml_is_contiguous(src1);\n",
                 "    const int ith = params->ith;\n"
                 "    const int nth = params->nth;\n\n"
                 "    const enum ggml_type type = src0->type;\n\n"
                 "    const bool src1_cont = ggml_is_contiguous(src1);\n"
                 "    const bool phase6 = phase6_enabled();\n")
    replace_once(cpu,
                 "    // reset current_chunk\n"
                 "    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {\n"
                 "        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);\n"
                 "        *current_chunk_ctr = nth;\n"
                 "    }\n\n"
                 "    ggml_barrier(params->threadpool);\n\n"
                 "    for (int cur_a = 0; cur_a < n_as; ++cur_a) {",
                 "    // reset current_chunk\n"
                 "    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {\n"
                 "        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);\n"
                 "        *current_chunk_ctr = nth;\n"
                 "    }\n\n"
                 "    ggml_barrier(params->threadpool);\n\n"
                 "    if (phase6 && ith == 0) phase6_prepare(src0, ids);\n"
                 "    ggml_barrier(params->threadpool);\n\n"
                 "    for (int cur_a = 0; cur_a < n_as; ++cur_a) {")
    replace_once(cpu,
                 "        const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n",
                 "        const char * src0_cur = phase6 ? phase6_tensor_ptr(src0, cur_a)\n"
                 "            : (const char *) src0->data + cur_a * nb02;\n")
    replace_once(cpu,
                 "        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time\n"
                 "        // Try fused ops, fall back to normal compute\n",
                 "        if (state->ith == 0 && join4_prof_on()) join4_node_start(node, node_n);\n"
                 "        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time\n"
                 "        // Try fused ops, fall back to normal compute\n")
    replace_once(cpu,
                 "        if (node_n + 1 < cgraph->n_nodes) {\n"
                 "            ggml_barrier(state->threadpool);\n"
                 "        }\n"
                 "    }\n"
                 "\n"
                 "#ifdef GGML_USE_OPENMP",
                 "        if (node_n + 1 < cgraph->n_nodes) {\n"
                 "            ggml_barrier(state->threadpool);\n"
                 "        }\n"
                 "    }\n"
                 "\n"
                 "    if (state->ith == 0 && join4_prof_on()) join4_graph_end();\n"
                 "\n"
                 "#ifdef GGML_USE_OPENMP")
    iqp = llama / "ggml/src/ggml-cpu/iqp.cpp"
    replace_once(iqp, '#include "iqp.h"\n',
                 '#include "iqp.h"\n\nextern "C" const char * ggml_cpu_phase6_iqp_source(const struct ggml_tensor *, int64_t);\n')
    replace_once(iqp,
                 "    const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n",
                 "    const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n"
                 "    const char * phase6_src0_cur = ggml_cpu_phase6_iqp_source(src0, cur_a);\n"
                 "    if (phase6_src0_cur != NULL) src0_cur = phase6_src0_cur;\n")
    loader = llama / "src/llama-model-loader.cpp"
    replace_once(loader, "#include <cstring>\n", "#include <cstring>\n#include <cstdlib>\n")
    replace_once(loader, "#include <regex>\n",
                 '#include <regex>\n\nextern "C" void ggml_cpu_phase6_register_lazy_tensor(const struct ggml_tensor *, int, size_t, size_t, size_t);\n')
    replace_once(loader,
                 "        size_t n_size = ggml_nbytes(cur);\n\n        const bool from_mapping = use_mmap || lazy.has(cur);",
                 "        size_t n_size = ggml_nbytes(cur);\n\n"
                 '        if (lazy.has(cur) && std::getenv("GGML_PHASE6_BOUNDED_CACHE") != nullptr) {\n'
                 "            const auto & file = files.at(weight->idx);\n"
                 "            ggml_cpu_phase6_register_lazy_tensor(cur, file->file_id(), weight->offs, n_size, cur->nb[2]);\n"
                 "        }\n\n"
                 "        const bool from_mapping = use_mmap || lazy.has(cur);")
    qwen = llama / "src/models/qwen35moe.cpp"
    replace_once(qwen,
                 '        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);\n'
                 '        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);',
                 '        const int expert_flags = flags | TENSOR_READ_LAZY;\n'
                 '        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, expert_flags);\n'
                 '        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, expert_flags);')
    g = llama / "src/llama-graph.cpp"
    replace_once(g, "#include <cstring>\n#include <numeric>",
                 "#include <cstdlib>\n#include <cstring>\n#include <numeric>")
    replace_once(g,
                 "    ggml_tensor * logits = nullptr;\n\n    if (probs_in == nullptr) {",
                 "\n    // ---- edge0 k1/k2 (env; default native) ----\n"
                 "    int64_t edge0_k1 = n_expert_used;\n"
                 "    int64_t edge0_k2 = n_expert_used;\n"
                 '    if (const char * e1 = getenv("GGML_MOE_K1")) { int v = atoi(e1); if (v > 0) edge0_k1 = v; }\n'
                 '    if (const char * e2 = getenv("GGML_MOE_K2")) { int v = atoi(e2); if (v > 0) edge0_k2 = v; }\n'
                 "    if (edge0_k2 < edge0_k1) edge0_k2 = edge0_k1;\n"
                 "    if (edge0_k2 > n_expert) edge0_k2 = n_expert;\n"
                 "    n_expert_used = edge0_k1;\n"
                 "\n    ggml_tensor * logits = nullptr;\n\n    if (probs_in == nullptr) {")
    replace_once(g,
                 "    const uint32_t n_expert_used_il = hparams.n_expert_used(il);",
                 "    const uint32_t n_expert_used_il = (uint32_t) n_expert_used; "
                 "// edge0: follows k1 (uniform arch; native-identical unset)")
    replace_once(g,
                 '        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]\n'
                 '        cb(weights_sum, "ffn_moe_weights_sum", il);',
                 '        ggml_tensor * weights_sum = nullptr;\n'
                 '        if (edge0_k2 == edge0_k1) {\n'
                 '            weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]\n'
                 '        } else {\n'
                 '            // reference mass over top-k2 (paper 2609.04575 Eq.2 denominator)\n'
                 '            ggml_tensor * sel_k2 = ggml_argsort_top_k(ctx0, selection_probs, (int) edge0_k2); // [k2, T]\n'
                 '            ggml_tensor * rk2 = ggml_get_rows(ctx0, probs, sel_k2); // [1, k2, T]\n'
                 '            rk2 = ggml_reshape_2d(ctx0, rk2, edge0_k2, n_tokens); // [k2, T]\n'
                 '            weights_sum = ggml_sum_rows(ctx0, rk2); // [k2, T]\n'
                 '            weights_sum = ggml_sum_rows(ctx0, rk2); // [1, T]\n'
                 '        }\n'
                 '        cb(weights_sum, "ffn_moe_weights_sum", il);')
    cli = llama / "tools/cli/cli-context.cpp"
    replace_once(cli,
                 "[ Prompt: %.1f t/s | Generation: %.1f t/s ]",
                 "[ Prompt: %.6f t/s | Generation: %.6f t/s ]")
    print("ALL ANCHORS OK")


if __name__ == "__main__":
    main()
