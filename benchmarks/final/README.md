# benchmarks/final — ADTC profiler release snapshot

- `result.json`: schema-valid ADTC report with the release metadata and the
  benchmark values requested by the team: 16.0 tok/s rounded from observed
  16.46 and 15.5 tok/s, with 2502.49 MB peak RSS and 0.72 ARC-Easy accuracy.
  Its reproducibility SHA remains the actual profiler evidence commit rather
  than being hand-edited to impersonate a fresh run.
- Evidence source: the last full profiler run was 35514252643 on commit
  `c454f1a6342b5426c943c2096029c26f993e694d`, profiler pin `12be4f3`.
  Rerun the full profiler on the current release commit before treating the
  snapshot as the final Gate-2 audit artifact.
- Harness: `.github/workflows/official-profiler.yml` (manual dispatch).
- Human report: `REPORT.md` BENCHMARKS; method notes: `docs/profiler.md`,
  `ARCHITECTURE.md` §13.
