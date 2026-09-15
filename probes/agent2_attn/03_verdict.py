"""Agent2 probe 03: ms model + per-candidate verdicts from results.json.

MAC model (decode GEMV-bound, assumption A3), validated against phase6b TSC:
  GDN/layer : qkv 8192x2048=16.78M + gate 4096x2048=8.39M  -> x30 = 755.0M
  FULL/layer: QG 8192x2048=16.78M + K/V 2x1.05M + O 8.39M  -> x10 = 272.6M
  total attn-proj MACs = 1027.6M; ms/MMAC = 47.5/1027.6 (A3).
  check: qkv 48.9% vs 48.3% TSC; GDN-gate 24.5% vs 25.3%; full-QG 16.3% vs 16.0%.
Baseline: 215.2 ms/token (4.647 tok/s). Goal: <66.7 ms/token.
RAM bytes at A4: dense Q5_K ~= 0.69 B/param (phase7c: attn/GDN Q5_K families).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights")
BASE_MS = 215.2
ATTN_MS = 47.5
GDN_M = 30 * (8192 * 2048 + 4096 * 2048) / 1e6
FULL_M = 10 * (8192 * 2048 + 2 * 512 * 2048 + 2048 * 4096) / 1e6
TOT_M = GDN_M + FULL_M
MS_PER_MMAC = ATTN_MS / TOT_M
PER_HEAD_M = 10 * (512 * 2048 + 2048 * 256) / 1e6  # QG rows + O cols per head idx
B_P = 0.69  # A4


def toks(saved):
    return 1000.0 / (BASE_MS - saved)


def main():
    res = json.load(open(os.path.join(W, "results.json")))
    agg = res["agg"]
    out = {"model": {"gdn_mmac": GDN_M, "full_mmac": FULL_M, "total_mmac": TOT_M,
                     "ms_per_mmac": MS_PER_MMAC, "per_head_saved_mmac": PER_HEAD_M,
                     "full_attn_proj_ms": FULL_M * MS_PER_MMAC,
                     "base_ms": BASE_MS, "attn_ms": ATTN_MS},
           "candidates": []}
    print(f"MAC model: GDN={GDN_M:.1f} FULL={FULL_M:.1f} TOT={TOT_M:.1f} MMAC; "
          f"ms/MMAC={MS_PER_MMAC:.4f}; full-attn proj={FULL_M*MS_PER_MMAC:.2f} ms; "
          f"per-head={PER_HEAD_M*MS_PER_MMAC:.3f} ms")
    # A. static head pruning
    for keep in (12, 8, 4):
        d = 16 - keep
        saved = d * PER_HEAD_M * MS_PER_MMAC
        params_rm = 10 * d * (512 * 2048 + 2048 * 256)
        out["candidates"].append({
            "id": f"static-16to{keep}", "kind": "query-head pruning",
            "err_out": agg["static_topN_mean"][str(keep)], "ms_saved": saved,
            "tok_s": toks(saved), "ram_mb": -params_rm * B_P / 1e6})
    # B. gate low-rank
    for r in (4, 8, 16):
        new_m = 10 * 16 * r * (2048 + 256) / 1e6
        old_m = 10 * 16 * 256 * 2048 / 1e6
        saved = (old_m - new_m) * MS_PER_MMAC
        params_new = 10 * 16 * r * (2048 + 256)
        params_old = 10 * 16 * 256 * 2048
        out["candidates"].append({
            "id": f"gate-rank{r}", "kind": "gate compression",
            "err_out": agg["lowrank_gate_mean"][str(r)], "ms_saved": saved,
            "tok_s": toks(saved), "ram_mb": (params_new - params_old) * B_P / 1e6})
    # C. dynamic oracle (optimistic: oracle err, static-equivalent ms, overhead=0)
    for keep in (8, 4):
        d = 16 - keep
        saved = d * PER_HEAD_M * MS_PER_MMAC
        out["candidates"].append({
            "id": f"dyn-oracle-top{keep}", "kind": "token-dynamic head execution",
            "err_out": agg["dynamic_topN_mean"][str(keep)], "ms_saved": saved,
            "tok_s": toks(saved), "ram_mb": 0.0})
    for c in out["candidates"]:
        print(f"{c['id']:>16}: err={c['err_out']:.4f} saved={c['ms_saved']:.2f}ms "
              f"tok/s={c['tok_s']:.3f} ram={c['ram_mb']:+.1f}MB")
    json.dump(out, open(os.path.join(W, "verdict_input.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
