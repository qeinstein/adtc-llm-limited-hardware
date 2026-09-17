# Phase 5G — explicit byte-bounded expert-cache prototype

## Hypothesis

An explicit fixed-slot cache can enforce the expert-residency budget in bytes
without relying on mmap page residency.  This is a model-free correctness and
accounting prototype; it intentionally does not claim model throughput.

## Contract

`PreadExpertStore` indexes packed records by byte offset and performs one
explicit `pread` per cache miss.  `BoundedExpertCache` accepts exact
`(layer, expert)` keys, copies the returned record into one fixed-size slot,
and evicts the oldest resident key deterministically.  The slot budget is
`floor(capacity_bytes / slot_bytes) * slot_bytes`; remainder bytes are not
silently used.  The prototype is single-threaded for batch-1 decode and keeps
I/O scheduling and the expert kernel separate.

The Phase 0 artifact measures one routed expert bundle at **876,544 bytes**
for all 40 layers.  The initial executor slot is therefore 876,544 bytes,
already a 4-KiB multiple.  Two slots are sufficient for a future read/compute
double buffer; a 256-MiB non-model reserve leaves ample room for that staging,
runtime scratch, cache metadata, and measurement headroom.

## Deployment-budget arithmetic

The measured non-MoE floor is 1,612.9 MiB.  The following conservative design
sets aside 256 MiB above that floor and gives the remainder to fixed expert
slots.  Fresh-byte rates are from the original 23-token exact trace replay;
the pending multi-prompt corpus will replace them for policy selection.

| Total RSS target | Bytes available after floor+reserve | Slots | Fixed cache RSS | Reserve | Trace global-LRU fresh bytes/token |
|---:|---:|---:|---:|---:|---:|
| 4 GiB | 2,335,283,609.6 | 2,664 | 2,226.94 MiB | 256 MiB | 97.53 MB |
| 3 GiB | 1,261,541,785.6 | 1,439 | 1,202.91 MiB | 256 MiB | 110.52 MB |
| 2.5 GiB | 724,670,873.6 | 826 | 690.48 MiB | 256 MiB | 141.66 MB |

The exact integer slot/cache values are recorded in `budget.json`.
The cache is not yet connected to model inference because the multi-prompt
trace and compute-representation decision are still pending.

## Correctness and limitation

The unit tests cover hit/miss behavior, deterministic LRU order, fixed-slot
accounting, zero-capacity behavior, oversized-record rejection, and an actual
small-file `pread` into a cache slot.  No model weights or routes are changed.
This prototype is evidence that bounded storage can be enforced cleanly, not
evidence of end-to-end speed or physical NVMe traffic; those require a Kaggle
executor experiment after representation and cache policy selection.
