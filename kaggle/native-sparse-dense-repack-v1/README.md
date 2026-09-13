# Native-sparse dense Q5_K repack v1

This is a controlled dense-projection donor experiment.  It compares the
mainline llama.cpp CPU backend with its standard weight-repacking path enabled
and disabled on the exact Qwen3.5-35B-A3B IQ2_XXS resident workload.

The Qwen3.5 attention and Gated-DeltaNet projection tensors in the committed
floor inventory are Q5_K.  This arm therefore tests whether the existing CPU
repack/layout machinery materially changes the real batch-1 decode path before
we write another bespoke dense kernel.

The model, prompt, 64-token decode, 3 repetitions, 4 threads, CPU-only
configuration, and exact native top-8 route are held constant.  The two arms
are `dense_no_repack` (`--no-repack`) and `dense_repack` (`--repack`).  Output
hashes are checked against the established exact control hash.  The experiment
does not alter routing, K, weights, or quantization semantics.

This is a performance/donor measurement, not a claim that the generic
repacking implementation is the final dense executor.  If the arms are
identical, the result means that this backend/build does not expose a useful
Qwen dense repack path; it does not rule out a custom Q5_K layout.
