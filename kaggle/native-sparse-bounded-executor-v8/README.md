# Phase 7B asynchronous bounded expert reads — parallel pread

This is the exact-IQP bounded executor at the established 4-GiB cache design
point: 2,281 full routed bundles, or 2,000,000,000 cache bytes. It retains
owned source descriptors, explicit three-plane `pread`, global byte-bounded
LRU, direct IQP cache-slot source resolution, and the exact native Qwen
top-8 path.

On a cache miss, the candidate reserves all missing selected experts for the
current routed node and reads each expert's gate/up/down planes concurrently
with one pthread per missing expert. It joins before exact compute, so the
candidate changes I/O scheduling only; no weight bytes or arithmetic change.
The same 64-token, three-repeat protocol is used.
