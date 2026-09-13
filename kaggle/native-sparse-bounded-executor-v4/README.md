# Phase 6G bounded executor v4 — 3-GiB design point

This is the exact-IQP bounded executor from v3 at the next measured memory
budget: 1,439 full routed bundles, or 1,261,346,816 cache bytes. It retains
owned source descriptors, explicit `pread`, global byte-bounded LRU, direct
IQP cache-slot source resolution, and the exact native Qwen top-8 path.

The purpose is to test whether the measured non-MoE floor plus this realistic
cache remains below 3 GiB RSS and to measure the resulting actual fresh read
traffic. It uses the same 64-token, three-repeat protocol.
