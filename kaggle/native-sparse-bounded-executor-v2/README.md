# Phase 6G bounded executor v2

Version 1 reached the real model and correctly allocated the explicit cache,
but aborted on `pread failed`. Source inspection and the v1 artifact show the
cause: the prototype retained the loader's file descriptor after the temporary
model loader had released it.

Version 2 duplicates each registered source descriptor and closes those owned
duplicates at process exit. Everything else is unchanged: exact native Qwen
top-8 routing, the existing generic exact expert kernel, explicit `pread`, a
fixed 2,000,000,000-byte global LRU cache, and the same 64-token/3-repeat
workload. This is a storage ownership fix, not a model or arithmetic change.
