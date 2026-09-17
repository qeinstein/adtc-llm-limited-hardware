# Phase 4 cache budget design

The measured non-MoE operational floor is **1,612.914 MiB RSS** from the
Phase 3 skip-MoE arm. This budget treats that measurement as unavoidable and
reserves another **256 MiB** for explicit I/O staging, allocator slack,
temporary buffers, and process headroom. The remaining bytes are the maximum
resident routed-expert cache budget.

The exact packed routed bundle is **876,544 bytes** per `(layer, expert)` key.
The fresh-byte estimates below replay the committed 23-token Phase 0 route
trace with a global LRU and exact bundle sizes. They are logical bytes supplied
to the kernel; they are not a claim about physical NVMe traffic.

| total RSS budget | non-MoE floor | safety/staging reserve | cache bytes | whole bundles | replay hit rate | fresh logical bytes/token |
|---:|---:|---:|---:|---:|---:|---:|
| 4 GiB (4096 MiB) | 1612.914 MiB | 256 MiB | 2226.938 MiB | 2664 | 65.23% | 97.53 MB |
| 3 GiB (3072 MiB) | 1612.914 MiB | 256 MiB | 1202.914 MiB | 1439 | 60.60% | 110.52 MB |
| 2.5 GiB (2560 MiB) | 1612.914 MiB | 256 MiB | 690.484 MiB | 826 | 49.50% | 141.66 MB |

The 256 MiB reserve is a design allowance, not a measured allocation yet. A
bounded executor should enforce the cache limit in bytes, use fixed aligned
slots, and keep the staging arena inside that reserve. It should not depend on
the kernel's mmap page reclamation. Layer-aware eviction remains a follow-up
comparison; the table is deliberately the reproducible global-LRU baseline.

At 4 GiB, the exact route trace leaves enough budget for a cache larger than
the previously measured 2,048-bundle point. At 3 GiB, the 1,439-bundle point
is between the existing 1,024 and 2,048 measurements. At 2.5 GiB, the 826
bundle point is between 512 and 1,024. These estimates therefore define the
next bounded-cache test points rather than extrapolating a throughput claim.
