"""07: aggregate all results -> tables + compute projections.

FLOP model (mults per token, ONE MoE layer; router 256x2048 kept exact in
all variants and excluded from the reduction ratio):
  teacher routed : 8 exps x (gate 512x2048 + up 512x2048 + down 2048x512)
                 = 25,165,824
  rule 'beta'    : compose 8K + eval M x (2x2048 gate/up + 2048 down)
                 = 8K + 6144 M          (nonlinear evals = M only)
  rule 'betas'   : 8K + K x 4096 (linear pre-score) + M x 2048
  + private      : + P_routed x 6144
(SiLU exp/div not counted; same family for teacher and dict.)

Time projections (assumptions stated in REPORT.md):
  expert-block baseline ~110 ms/token (all 40 MoE layers, decode);
  per-layer = 2.75 ms; projected layer ms = 2.75 / R.
  end-to-end frame 5.196 tok/s = 192.5 ms/token = 82.5 non-MoE + 110 MoE.

Storage (per layer): teacher IQ2 = 256 x (2x270336 + 335872) B = 224 MB.
  dict at same ~2.5 bits/param: K x 6144 params + P x 6144 + codes 256 x K f16.

Outputs: aggregate.json + printed tables.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")

TEACHER_MULTS = 8 * (512 * 2048 * 2 + 512 * 2048)
ROUTER_MULTS = 256 * 2048
TEACHER_BYTES = 256 * (2 * 270336 + 335872)


def flops_beta(K, M, Pr=0):
    return 8 * K + 6144 * M + 6144 * Pr


def flops_betas(K, M, Pr=0):
    return 8 * K + 4096 * K + 2048 * M + 6144 * Pr


def storage(K, P):
    return K * 6144 * 2.5 / 8 + P * 6144 * 2.5 / 8 + 256 * K * 2


def proj(R):
    layer_ms = 2.75 / R
    moe_ms = 110.0 / R
    toks = 1000.0 / (82.5 + moe_ms)
    return layer_ms, moe_ms, toks


def main():
    agg = {"flops_model": {"teacher_mults": TEACHER_MULTS,
                           "router_mults_excluded": ROUTER_MULTS,
                           "teacher_bytes_per_layer": TEACHER_BYTES},
           "tables": {}}
    oc = json.load(open(os.path.join(OUT, "oracle_summary.json")))
    print("== ORACLE (best-subset, true coeffs) held-out rel err ==")
    print("M      mean     median   p95      p99      max      cos")
    for M in ("128", "256", "512", "1024", "2048"):
        g = oc["grid"][M]
        print(f"{int(M):5d}  {g['rel_mean']:.4f}   {g['rel_median']:.4f}   "
              f"{g['rel_p95']:.4f}   {g['rel_p99']:.4f}   {g['rel_max']:.4f}   "
              f"{g['cos_mean']:.4f}")
    print(f"kill gate: {oc['kill_gate']}")
    agg["tables"]["oracle"] = oc["grid"]

    gs = json.load(open(os.path.join(OUT, "dict_global_summary.json")))
    print("\n== GLOBAL rule=beta (pure composition) median/mean @M ==")
    for k in sorted(gs["K"], key=int):
        row = []
        for M in ("128", "256", "512", "1024", "2048"):
            g = gs["K"][k]["beta"][M]
            row.append(f"{g['rel_median']:.3f}/{g['rel_mean']:.3f}")
        print(f"K={k:>5s} " + "  ".join(row))
    print("\n== GLOBAL rule=betas median/mean @M ==")
    for k in sorted(gs["K"], key=int):
        row = []
        for M in ("128", "256", "512", "1024", "2048"):
            g = gs["K"][k]["betas"][M]
            row.append(f"{g['rel_median']:.3f}/{g['rel_mean']:.3f}")
        print(f"K={k:>5s} " + "  ".join(row))
    agg["tables"]["global"] = gs["K"]

    cs = json.load(open(os.path.join(OUT, "dict_comm_summary.json")))
    print("\n== COMMUNITY rule=beta median/mean @M ==")
    for key in sorted(cs["configs"]):
        row = []
        for M in ("128", "256", "512", "1024", "2048"):
            g = cs["configs"][key]["beta"][M]
            row.append(f"{g['rel_median']:.3f}/{g['rel_mean']:.3f}")
        print(f"{key:>10s} " + "  ".join(row))
    agg["tables"]["comm"] = cs["configs"]

    pv = json.load(open(os.path.join(OUT, "dict_private_summary.json")))
    print("\n== PRIVATE (rule=beta) ==")
    for tag, cfg in pv["cfgs"].items():
        for P in sorted(cfg, key=lambda x: int(x)):
            r = cfg[P]
            row = []
            for M in ("256", "512", "1024"):
                g = r["M"][M]
                row.append(f"{g['rel_median']:.3f}/{g['rel_p99']:.3f}/"
                           f"{g['rel_max']:.3f}")
            print(f"{tag} P={P} priv_routed={r['priv_routed_mean']:.1f} "
                  + "  ".join(row) + "  [med/p99/max]")
    agg["tables"]["private"] = pv["cfgs"]

    print("\n== PROJECTIONS (mults/token/layer, 110ms MoE baseline) ==")
    print("config              mults        R     layer_ms  moe_ms   tok/s")
    rows = [("teacher", TEACHER_MULTS)]
    for k in (512, 1024, 2048, 4096, 8192):
        for M in (256, 512, 1024, 2048):
            rows.append((f"beta K{k} M{M}", flops_beta(k, M)))
    for k in (1024, 4096):
        for M in (512, 1024):
            rows.append((f"betas K{k} M{M}", flops_betas(k, M)))
    for name, f in rows:
        R = TEACHER_MULTS / f
        lms, mms, tps = proj(R)
        print(f"{name:>18s}  {f:>10d}  {R:6.2f}x  {lms:7.3f}  {mms:6.2f}  "
              f"{tps:5.2f}")
        agg.setdefault("projections", {})[name] = {
            "mults": f, "R": R, "layer_ms": lms, "moe_ms": mms,
            "toks": tps}
    print("\n== STORAGE per layer ==")
    print(f"teacher: {TEACHER_BYTES/1e6:.1f} MB")
    for K, P in ((1024, 0), (4096, 0), (8192, 0), (4096, 2048),
                 (4096, 8192)):
        s = storage(K, P)
        print(f"K={K} P={P}: {s/1e6:.1f} MB  ({TEACHER_BYTES/s:.1f}x)")
    json.dump(agg, open(os.path.join(OUT, "aggregate.json"), "w"), indent=1)
    print("OK 07", flush=True)


if __name__ == "__main__":
    sys.exit(main())
