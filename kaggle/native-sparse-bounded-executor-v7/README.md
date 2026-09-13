# Phase 7A contiguous exact sidecar v2 — cold-cache A/B

This is the exact-IQP bounded executor from v3 at the next measured memory
budget: 2,281 full routed bundles, or 2,000,000,000 cache bytes. It retains
owned source descriptors, global byte-bounded LRU, direct IQP cache-slot
source resolution, and the exact native Qwen top-8 path.

The experiment adds an exact storage-side transformation: all gate/up/down
planes for each `(layer, expert)` are copied into one contiguous fixed-size
sidecar record. Cache misses then use one aligned `pread` rather than three
independent plane reads. The sidecar is byte-copied from the original GGUF;
no values are requantized or changed. It uses the same 64-token,
three-repeat protocol. Unlike v1, it explicitly evicts sidecar pages before
each bounded process and after initial construction, so read timing is not
confounded by sidecar page-cache warming.
