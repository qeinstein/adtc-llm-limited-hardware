# Phase 5C — exact IQ2-derived representation candidates

The first custom-layout benchmark showed that resolving grid/sign metadata
offline is mathematically exact but does not automatically improve the hot
loop.  This note records the candidate space used to choose the next narrow
test; it does not change the control model.

## Candidate definitions

- `IQ2_XXS`: the 66-byte upstream block, 2.0625 bpw, control.
- `packed_resolved`: the same grid/sign codeword and scale information in a
  70-byte packed block (2.1875 bpw).  It removes C-struct padding from the
  first 74-byte prototype while retaining the same lookup schedule.
- `natural_resolved`: the 74-byte naturally aligned block (2.3125 bpw) used
  by the completed Phase 5B benchmark.  It is exact but slower than control.
- `cacheline_resolved`: an 80-byte block (2.5 bpw) aligned for a regular
  record schedule.  This is a layout hypothesis only and is not yet worth a
  model benchmark unless the packed candidate changes the result.

The first and third entries are measured; the packed candidate is the focused
follow-up.  The 80-byte candidate is retained as a storage/compute tradeoff,
not presented as a result.

## Storage and cache arithmetic

The control routed pool is 8,975,810,560 bytes and each exact routed
`(layer, expert)` bundle is 876,544 bytes.  The estimates below scale the
measured block representation by `candidate_bytes / 66` and round each cache
record up to 4 KiB.  The pool estimate is therefore a planning number; a
production converter must account for every tensor's actual row padding.

| Representation | Block bytes | bpw | Pool estimate | 4-GiB slots | 3-GiB slots | 2.5-GiB slots |
|---|---:|---:|---:|---:|---:|---:|
| IQ2_XXS control | 66 | 2.0625 | 8.36 GiB | 2,664 | 1,439 | 826 |
| packed_resolved | 70 | 2.1875 | 8.87 GiB | 2,511 | 1,356 | 779 |
| natural_resolved | 74 | 2.3125 | 9.37 GiB | 2,375 | 1,283 | 737 |
| cacheline_resolved | 80 | 2.5000 | 10.13 GiB | 2,192 | 1,184 | 680 |

All slot counts reserve the measured 1,612.9 MiB non-MoE floor plus 256 MiB
for staging, scratch, metadata, and headroom.  The exact machine-readable
arithmetic is in `representation_candidates.json`.

## Decision

The 74-byte result rejects padding-only resolution as an optimization.  The
70-byte candidate is the smallest meaningful follow-up: it tests whether the
negative result came from record padding/cache footprint while preserving the
same exact values.  If it remains slower, the evidence points to the table
lookup and accumulation schedule rather than GGUF byte packing alone; the
next work should move to a genuinely different executor or a measured
non-MoE kernel target, not accumulate more metadata fields.
