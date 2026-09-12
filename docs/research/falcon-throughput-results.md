# Falcon-H1 throughput frontier — measured

Run: GitHub Actions `34592138337` (`falcon-throughput.yml`), scalar/no-SIMD
audit-parity build on `ubuntu-latest`. llama.cpp revision: `3bcfeb700`.
All rows use the profiler-shaped command `llama-bench -p 512 -n 128 -ngl 0`.

| Variant | File bytes | Prefill tok/s | Decode tok/s | Peak RSS MB | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Falcon-H1-1.5B-Instruct Q4_K_M | 944,787,072 | 9.50 | 7.54 | 1,078 | 6:48.81 |
| Deep Q4_0 | 904,190,528 | 13.58 | **9.99** | 1,071 | 4:50.71 |
| Deep Q4_K_M | 938,466,368 | 8.35 | 6.74 | 1,113 | 7:43.42 |
| Deep Q5_K_M | 1,104,648,928 | 7.60 | 6.24 | 1,278 | 8:27.09 |
| Deep Q6_K | 1,281,218,944 | 7.14 | 5.89 | 1,446 | 8:59.66 |
| Deep Q8_0 | 1,657,558,528 | 8.10 | 6.56 | 1,805 | 7:57.31 |
| Deep IQ4_XS | 858,907,328 | 2.50 | 2.31 | 1,037 | 25:08.26 |

## Thread sweep: Deep Q4_K_M

| Threads | Decode tok/s | Peak RSS MB | Wall time |
| ---: | ---: | ---: | ---: |
| 1 | 3.59 | 1,113 | 15:08.68 |
| 2 | 6.73 | 1,113 | 7:43.44 |
| 4 | **7.04** | 1,103 | 7:19.45 |
| default | 6.75 | 1,113 | 7:43.27 |

The Q4_0 row is the current throughput candidate: it is 48% faster than the
Deep Q4_K_M baseline and has slightly lower RSS. Its accuracy must be measured
before changing the deployment quant, because the stock accuracy numbers were
obtained with Deep Q4_K_M.

The 24-layer ordinary Instruct checkpoint is faster than Deep Q4_K_M, but this
is a different checkpoint and cannot be substituted silently. IQ4_XS is smaller
on disk but is not a useful CPU candidate on this scalar build. Four threads
are the best tested explicit setting for Deep Q4_K_M; the gain over the
default is small.

`perf stat` was attempted but restricted on the hosted runner, so this run does
not support a reliable Mamba-versus-attention CPU attribution. The next useful
systems measurement is profiler-exact Q4_0 accuracy plus a matched benchmark on
the reference laptop.
