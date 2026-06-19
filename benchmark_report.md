# Standardized Serving Telemetry & Performance Profile

This report provides standardized latency, memory, and throughput metrics collected during local verification.

## 1. Environment & Config Summary
- **Target Model:** Qwen 2.5 14B Q4_0 (Custom serving configuration)
- **Engine Strategy:** 2.82 GB Attention Cache in RAM + 50% Active FFN Columns Swapping
- **Engine Baseline RSS:** 118.9 MB
- **Engine Loaded RSS:** 125.2 MB
- **Initialization Latency:** 0.08 seconds

## 2. Quantitative Performance Summary Table

| Prompt Scenario | Length (Tokens) | Prefill Time (s) | Prefill Throughput (tps) | Decode Speed (tps) | Avg Token Latency (ms) | Peak RAM RSS (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Short Query | 16 | 1.371s | 11.7 tps | 27.48 tps | 36.4 ms | 194.2 MB |
| Medium Context | 128 | 5.177s | 24.7 tps | 21.86 tps | 45.7 ms | 194.8 MB |
| Long Context | 512 | 33.248s | 15.4 tps | 13.22 tps | 75.6 ms | 165.5 MB |

## 3. Detailed Percentile Decoding Latency (ms)

| Prompt Scenario | P50 (Median) | P90 | P95 | Peak Memory RSS (MB) |
| :--- | :---: | :---: | :---: | :---: |
| Short Query | 35.9 ms | 40.6 ms | 41.4 ms | 194.2 MB |
| Medium Context | 44.1 ms | 50.7 ms | 51.0 ms | 194.8 MB |
| Long Context | 75.1 ms | 77.5 ms | 78.6 ms | 165.5 MB |

## 4. Resident Memory RSS Profile Over Time (ASCII Line Charts)

### Memory Profile: Short Query
```text
 194.2 MB │                             ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
 184.4 MB │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●││││││││││││││││││││││││││││││││
 174.5 MB │  │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
 164.7 MB │  │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
 154.8 MB │  │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
 145.0 MB │  │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
 135.1 MB │  │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
 125.2 MB │ ●│││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
       └─────────────────────────────────────────────────────────────
        Start Prefill ──────────> Decode ──────────> End Generation
```

### Memory Profile: Medium Context
```text
 194.8 MB │                                 ●●●●●●●●●●●●●●●●●●●●●●●●●●●●
 194.7 MB │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●││││││││││││││││││││││││││││
 194.6 MB │  │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
 194.6 MB │  │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
 194.5 MB │  │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
 194.4 MB │  │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
 194.4 MB │  │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
 194.3 MB │ ●│││││││││││││││││││││││││││││││││││││││││││││││││││││││││││
       └─────────────────────────────────────────────────────────────
        Start Prefill ──────────> Decode ──────────> End Generation
```

### Memory Profile: Long Context
```text
 194.8 MB │ ●                                                           
 190.5 MB │ │                                                           
 186.3 MB │ │                                                           
 182.1 MB │ │                                                           
 177.9 MB │ │                                                           
 173.7 MB │ │                                                           
 169.5 MB │ │                                                           
 165.3 MB │ │●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
       └─────────────────────────────────────────────────────────────
        Start Prefill ──────────> Decode ──────────> End Generation
```
