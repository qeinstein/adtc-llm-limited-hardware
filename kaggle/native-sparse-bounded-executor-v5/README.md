# Phase 6G bounded executor v5 — 2.5-GiB design point

This is the exact-IQP bounded executor from v3 at the next measured memory
budget: 826 full routed bundles, or 724,025,344 cache bytes. It retains
owned source descriptors, explicit `pread`, global byte-bounded LRU, direct
IQP cache-slot source resolution, and the exact native Qwen top-8 path.

The purpose is to measure the stretch RAM frontier below 2.5 GiB and the
resulting actual fresh read traffic. It uses the same 64-token, three-repeat
protocol. This is a storage-boundary characterization, not a quality result
or a proposed throughput solution.
