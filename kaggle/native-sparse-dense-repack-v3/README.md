# Native-sparse dense Q5_K repack v3

This is the corrected runnable version of the dense donor arm. It uses
`llama-cli` for three measured 64-token repetitions per `--repack` and
`--no-repack` arm, parses the generation performance line explicitly, and
checks the exact control response hash.
