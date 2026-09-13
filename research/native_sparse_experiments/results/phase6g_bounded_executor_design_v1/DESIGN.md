# Phase 6g: bounded expert executor design

- Design ID: `phase6g_bounded_executor_design_v1`
- Scope: source-level design only; no existing repository files were edited
- Research repository HEAD inspected: `6f363b3` (`research/experimental-massive`)
- llama.cpp source inspected: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Model: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`
- Model size: `10,656,955,008` bytes
- Model SHA-256: `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Model source: `unsloth/Qwen3.5-35B-A3B-GGUF`, commit `bc014a17be43adabd7066b7a86075ff935c6a4e2`
- Target: Linux/x86 CPU-only, exact native top-8 routing, no model or weight changes

## Executive conclusion

A real bounded expert executor can be integrated without replacing the GGML
graph or the llama.cpp runtime. The smallest viable path is:

1. keep the existing model-file `mmap` only as a virtual metadata/backing
   mapping and keep routed expert tensors out of the compute path;
2. copy the existing loader `(file index, absolute tensor offset, shape,
   strides, type)` into model-owned expert descriptors;
3. keep model-owned file handles alive after `llama_model_loader` is destroyed;
4. add a Linux `pread`-at-offset primitive;
5. allocate a fixed byte-budget cache of exact packed expert planes; and
6. make the CPU `MUL_MAT_ID` executor resolve each selected expert to a pinned
   cache slot before dispatching the existing generic or IQP kernel.

This preserves the current GGML graph, native router, `ids` tensor, K=8,
quantized bytes, and output layout. It is a bounded application-storage path,
not an `mmap`/page-cache heuristic. The mapped expert address must never be
dereferenced in explicit-cache mode; a missing descriptor or cache miss that
falls back to `src0->data` must be a hard failure in that mode.

An initial real `<=4 GiB` run is plausible without replacing the whole runtime,
but it is not proven by the current 5.16 GiB lazy result. The lazy result still
allows uncontrolled routed expert pages to become resident. Phase 0 records
three routed tensors per layer with total packed bytes
`(69,206,016 * 2 + 85,983,232) * 40 = 8,975,810,560` bytes, leaving roughly
`1,681,144,448` bytes of the recorded model file outside that routed pool,
before runtime buffers, KV/recurrent state, and allocator overhead. A small
explicit cache could therefore fit under 4 GiB if the non-routed floor is
measured and remains below the budget. The first bounded run must measure this;
the current 5.16 GiB RSS is an upper bound for the true non-routed floor, not a
proof that the floor itself exceeds 4 GiB.

The `<=2--2.5 GiB` objective is uncertain. It requires a measured decomposition
of non-routed tensors, recurrent/KV state, scratch, and runtime metadata. Expert
cache engineering alone cannot guarantee it.

## Exact current ownership and tensor layout

### Loader identity and offsets

In `src/llama-model-loader.h:34-50`,
`llama_model_loader::llama_tensor_weight` stores:

```text
idx   = source GGUF file index
offs  = gguf_get_data_offset(metadata) + gguf_get_tensor_offset(...)
tensor = loader metadata tensor
```

`offs` is already an absolute byte offset in the source file. The constructor
checks `offs + ggml_nbytes(tensor)` against the file size. The loader retains
these records in `weights_map` (`src/llama-model-loader.h:124-127`) and keeps
source handles in `files` (`:120`) while loading.

This is the authoritative offset source. Do not recompute offsets from tensor
names or assume tensors are adjacent in the GGUF file. Split GGUFs are handled
by `idx`; each expert descriptor must retain that index.

### Qwen3.5 routed tensors

`src/models/qwen35moe.cpp:94-103` creates, for each of 40 repeating layers:

```text
blk.N.ffn_gate_inp.weight       [n_embd, n_expert]
blk.N.ffn_gate_exps.weight      [n_embd, n_ff_exp, n_expert]
blk.N.ffn_up_exps.weight        [n_embd, n_ff_exp, n_expert]
blk.N.ffn_down_exps.weight      [n_ff_exp, n_embd, n_expert]
```

The exact Phase 0 route trace confirms this model uses the separate gate/up
branch rather than the optional merged `ffn_gate_up_exps` branch. The recorded
whole-tensor sizes are:

| tensor | whole tensor | per expert plane (`/256`) |
|---|---:|---:|
| `ffn_gate_exps` | 69,206,016 B | 270,336 B = 264 KiB |
| `ffn_up_exps` | 69,206,016 B | 270,336 B = 264 KiB |
| `ffn_down_exps` | 85,983,232 B | 335,872 B = 328 KiB |

The per-expert calculation is valid only after checking the actual tensor
metadata. In GGML, `struct ggml_tensor` (`ggml/include/ggml.h:685-717`) stores
`ne[]`, `nb[]`, `data`, `buffer`, and `extra`. For a contiguous 3-D expert
tensor, the exact address formula is:

```text
expert_base = tensor_file_off + expert_id * tensor->nb[2]
row_base    = expert_base + output_row * tensor->nb[1]
```

For the Phase 0 model, the cache descriptor should assert:

```text
tensor->ne[2] == 256
tensor->ne[3] == 1
tensor->nb[2] == tensor->nb[1] * tensor->ne[1]
ggml_nbytes(tensor) == tensor->nb[2] * tensor->ne[2]
tensor->type == GGML_TYPE_IQ2_XXS
```

The full-plane read is `[expert_base, expert_base + tensor->nb[2])`.
`tensor->nb[1]` is the packed row size, so this preserves the exact IQ2_XXS
block boundaries and row padding expected by the CPU kernels.

### Routing and `MUL_MAT_ID`

`src/llama-graph.cpp:1993-2250` builds the MoE graph. It computes router
logits, softmax probabilities, top-k IDs, and weights, then calls
`build_lora_mm_id()` for each routed expert matrix. That helper calls
`ggml_mul_mat_id()` at `src/llama-graph.cpp:1545-1559`.

For the separate Qwen gate/up branch, `MUL_MAT_ID` is created for `up_exps`
and `gate_exps` (`src/llama-graph.cpp:2188-2205`), and `down_exps` is used by
the later expert FFN path. The selected IDs and expert weights remain ordinary
GGML tensors; the storage layer must not inspect or regenerate router results.

In `ggml/src/ggml-cpu/ggml-cpu.c:1572-1677`, the CPU implementation groups
`ids` into `matrix_rows`. Each entry is an `{ i1, i2 }` pair: selected expert
ID and input-token row. `matrix_row_counts[cur_a]` is the number of selected
rows for expert `cur_a`; `cne1` is that count. For one-token native top-8
decode, selected IDs are distinct, so eight experts normally have `cne1 == 1`.

The generic path currently forms:

```text
src0_cur = (char *) src0->data + cur_a * nb02
```

and feeds rows from `src0_cur` to `type_traits_cpu[type].vec_dot`. The IQP
path is selected by `ggml_cpu_iqp_supports_mul_mat_id()` and called at
`:1672-1674`; it currently computes the same `src0_cur` internally.

`ggml/src/ggml-cpu/iqp.cpp:1130-1187` decodes eight consecutive source rows
into a per-thread `block_iqp_x8` panel, then performs the exact selected
expert operation. The storage hook must therefore resolve the expert base
pointer before both paths, and the IQP function should accept that resolved
pointer (or a storage callback) rather than reconstructing it from the mmap.

## Proposed ownership boundary

The current loader has an important lifetime split:

- `llama_model_loader::files`, `weights_map`, and loader metadata are temporary
  loader-owned state;
- `llama_model_base::load_tensors()` calls `ml.init_mappings()` and then moves
  `ml.mappings` into `llama_model::impl::mappings` at
  `src/llama-model.cpp:1842-1853`;
- `llama_model::impl` (`src/llama-model.cpp:1144-1181`) owns mappings and GGML
  backend buffers after loading, but does not currently own `ml.files`.

An explicit cache must not store a pointer to `llama_model_loader`, a pointer
to loader metadata, or a `llama_tensor_weight::tensor` pointer. Those objects
can die after model construction. Add a model-owned object, conceptually:

```text
llama_bounded_moe_storage
  vector<moe_file> files                 // owns open source handles
  vector<moe_tensor_desc> tensors        // copied by tensor name/pointer
  vector<aligned_cache_slot> slots       // fixed byte budget
```

`moe_tensor_desc` should contain the copied `idx`, absolute `offs`, type,
`ne[4]`, `nb[4]`, `nbytes`, tensor name, and a stable owner key. Register it
after `load_arch_tensors()` has created the actual tensors and before the
temporary loader is destroyed. The natural integration point is
`llama_model_base::load_tensors()` after the `tensors_by_name` population at
`src/llama-model.cpp:1700-1707`, using
`ml.get_weight(ggml_get_name(tensor))` to copy the offset record.

At the end of loading, move `ml.files` into the model implementation alongside
the existing move of `ml.mappings`. This is necessary even if mappings remain
enabled: the explicit path needs a live file descriptor for `pread`. A separate
re-open by filename is less safe because split-file naming, direct-I/O mode,
and file lifetime would diverge from the loader.

The first implementation can use `ggml_tensor::extra` (`ggml/include/ggml.h:714`)
for a tagged pointer to `moe_tensor_desc`, provided it asserts `extra == NULL`
and is CPU-only. A later general implementation should use a dedicated
backend/CPU registry or field because `extra` is backend-private territory.
The descriptor must be unregistered before its model-owned storage is freed.

## Smallest viable explicit `pread` cache

### Read primitive

`src/llama-mmap.h:17-42` exposes `llama_file::file_id()`, `seek()`, and
`read_raw()`. `src/llama-io.cpp:265-375` shows that Linux direct I/O already
uses an owned file descriptor, but the current seek/read API has a shared file
cursor and is not safe as a concurrent cache primitive.

Add a narrow `llama_file::read_raw_at(size_t offset, void * dst, size_t len)`:

- Linux/x86 first implementation: loop on `::pread(file_id(), ...)`, retry
  `EINTR`, and fail on short/zero reads;
- regular buffered descriptors are sufficient for the correctness prototype;
- optionally use aligned buffers and the existing direct-I/O mode only after
  correctness is established;
- do not call `seek()` from worker threads.

The cache calls this with `desc.offs + expert_id * desc.nb[2]` and length
`desc.nb[2]`. Validate overflow, file index, expert range, and `[offset, end)`
against the source file size before the read.

### Cache unit and synchronization

Use a full packed expert plane as the first cache unit. It supports both the
generic vec-dot fallback and IQP without changing dequantization. Let
`S = max(desc.nb[2])` over registered routed tensors (328 KiB for this model)
and allocate a fixed `B`-byte pool with aligned slots. Each slot records:

```text
key = (descriptor identity, expert_id)
valid, loading, pin_count, last_use, byte_size
```

The cache is shared by the model, but lookup/load metadata is protected by one
short mutex or a small shard lock. The file read itself must not hold a global
lock if avoidable. A loading state plus condition variable prevents duplicate
reads for the same key. Eviction may select only `pin_count == 0` slots.

The smallest safe `MUL_MAT_ID` integration is operation-scoped pinning:

1. After the existing `matrix_rows` barrier, thread 0 enumerates all nonempty
   `cur_a` values for the current node and deduplicates expert IDs.
2. It acquires/loads those keys into cache slots and publishes a small
   per-node table `cur_a -> slot pointer` in `params->wdata` or an executor
   sidecar.
3. All worker threads synchronize once.
4. The existing `cur_a` loop uses the published slot pointer for both generic
   and IQP computation. No worker dereferences the routed `src0->data`.
5. All workers synchronize once; thread 0 unpins the operation keys.

For one-token K=8 decode, a node requires at most eight planes and therefore
fits in eight slots. If a prompt/batch node requires more keys than the byte
budget, process deterministic groups of keys with the same prepare/compute/
release barrier, or fail closed in the first decode-only prototype. Do not
evict an in-flight slot: the IQP path partitions output panels across threads,
so one selected expert can be read once and consumed concurrently by all
workers.

This cache is explicitly bounded in application memory. It does not claim to
bound kernel page cache when regular `pread` is used; for total-system memory
accounting, add direct I/O or issue carefully scoped `posix_fadvise(...,
POSIX_FADV_DONTNEED)` after reads. The initial application-RSS experiment
should use regular `pread` for simplicity and report both process RSS and read
bytes.

### CPU integration points

The narrow source change set would be:

1. `src/llama-mmap.h/.cpp` or `src/llama-io.*`: add the offset read method.
2. `src/llama-model-loader.h/.cpp`: expose/copy exact weight descriptors only
   as needed during model construction; do not change GGUF offsets.
3. `src/llama-model.cpp`: add model-owned storage and move/retain source file
   ownership; register only Qwen routed expert tensors.
4. `ggml/src/ggml-cpu/ggml-cpu.c`: before the existing generic/IQP branch,
   resolve `cur_a` through the cache and pass the resulting base pointer to
   the selected path. Preserve the current path when storage is disabled.
5. `ggml/src/ggml-cpu/iqp.h/.cpp`: add a resolved-source argument to
   `ggml_compute_forward_mul_mat_id_iqp()` or a small storage resolve helper;
   leave IQ2_XXS decode and arithmetic unchanged initially.

The dispatch must be opt-in, for example `GGML_EXPLICIT_MOE_CACHE=1`, and
must hard-fail if an eligible routed tensor lacks a descriptor, cannot be read,
or would fall back to mmap. The generic control can continue to run with the
existing `src0->data` path. This gives a clean exact A/B without changing the
router or model representation.

## Memory and feasibility gates

The existing Phase 0 routed tensor accounting gives a useful bound:

```text
routed expert pool across 40 layers = 8,975,810,560 B = 8.359375 GiB
recorded model file                    = 10,656,955,008 B
non-routed file remainder               = 1,681,144,448 B = 1.5657 GiB
```

The remainder is not the application floor: GGML buffers, non-expert tensor
padding, tokenizer/metadata, recurrent state, KV state, and scratch add to it.
Conversely, the observed 5,155.3 MiB routed-lazy RSS is not the floor because
the current `mmap` pages are still reclaimable only at OS discretion and can be
faulted by execution.

Therefore:

- `<=4 GiB`: feasible as an initial exact bounded-storage run without
  replacing the runtime, provided the experiment uses mmap only as untouched
  backing, explicit cache memory stays small (for example 16--64 MiB), and the
  measured non-routed/runtime floor plus state stays under roughly 4 GiB. It is
  a hypothesis to verify, not a result claimed by this note.
- `<=2--2.5 GiB`: not established. If the measured non-routed/runtime floor
  is already near or above that range, expert streaming alone cannot reach it;
  a representation or execution change would then be required.

The first real run should record: process peak RSS, `/proc/<pid>/smaps_rollup`,
cache allocated bytes, cache resident bytes, mapped expert pages, non-routed
tensor bytes, exact `pread` bytes, cache hits/misses, and output/route hashes.

## Risks and fail-closed rules

- **Descriptor lifetime:** never retain loader metadata or a loader pointer;
  copy fields and own file handles in the model.
- **Split files:** key by `(file_idx, absolute_offset)`; a name-only offset is
  incorrect for split GGUF models.
- **Stride/reshape:** reject non-contiguous or reshaped routed tensors until a
  byte-accurate descriptor handles them. The formula above assumes
  `nb[2] == nb[1] * ne[1]`.
- **Concurrent eviction:** pin every plane used by a node until all worker
  threads leave it. A pointer returned without a lifetime token is unsafe.
- **Mapped fallback:** in explicit mode, abort rather than silently touching
  `src0->data`; otherwise RSS and traffic are not bounded.
- **Direct I/O alignment:** do not enable O_DIRECT in the first correctness
  patch unless offset, length, and destination alignment are all handled.
- **Prompt/batch behavior:** decode has at most eight selected planes per
  routed node; prompt evaluation can have a larger working set. Start with a
  deterministic grouped path or leave prompt on the existing path while
  clearly labeling the result decode-only.
- **`ggml_tensor::extra`:** it is a convenient CPU-only prototype hook but may
  conflict with a backend-specific owner. Assert and document ownership, then
  replace it with a dedicated registry/field before multi-backend support.
- **Cache key aliasing:** two layers can select the same expert ID; the layer/
  tensor descriptor must be part of the key.
- **Read accounting:** logical selected bytes and physical bytes read differ
  under cache hits. Report both; the cache must not change native selected IDs.
- **Numerical behavior:** reading packed bytes into a slot must be byte exact.
  Compare route equality and the extracted generated payload against Phase 0;
  reject any result that required K reduction, expert dropping, or fallback.

## Recommended next implementation experiment

Implement only the full-plane, operation-pinned cache on Linux with regular
`pread`, retaining the existing generic CPU math first. Run:

1. resident mmap control;
2. existing routed-expert lazy mmap control;
3. explicit cache with 16 MiB, 32 MiB, and 64 MiB budgets;
4. one-token 64-decode-token throughput with exact route/output checks; and
5. a separate prompt-plus-decode run that reports whether prompt pages break
   the RAM budget.

If the control and cache outputs are byte-identical and the cache RSS is
bounded, then add IQP/single-row integration to the same resolved-source API.
If `<=4 GiB` fails despite the expert pool being bounded, decompose the
non-routed floor before considering any model representation change.
