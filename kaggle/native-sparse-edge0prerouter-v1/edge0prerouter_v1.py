"""Edge0 prerouter pilot: can a cheap probe predict MoE routes early?

Phase 3 core ML question (decision-relevant, pre-registered below).
Input: trace v3 trainset (trace_p*.npz via kernel_sources): per-eval
gate-input hiddens [T,40,2048] fp16 + exact offline top-8 ids [T,40,8].

Three input variants (per layer L, eval t):
  (a) same-layer  H[t,L]   -> topk[t,L]  (distillation/recoverability bound;
      ~zero timing gain — same input availability as the true router)
  (b) prev-layer  H[t,L-1] -> topk[t,L]  (ONE-LAYER-EARLY: input available a
      full layer earlier; fetch overlaps layer L-1 MoE + layer L attention.
      THIS is the Edge0-style prefetch mechanism under test. L=0 uses H[t,0]
      of... nothing earlier exists -> L0 excluded from (b), reported as such)
  (c) prev-token  H[t-1,L] -> topk[t,L]  (ONE-TOKEN-EARLY: maximum overlap;
      smooth generalization of persistence)

Model: multinomial logistic regression (same function class as the true
router: linear-softmax) on Gaussian random-projected hiddens (seed-fixed,
reproducible). Dim sweep {64, 256} + full-2048 reference for (a).

Split: BY PROMPT, category-stratified (no leakage; evals within a prompt
correlate). Train p00-p04,p06-p10,p12-p15,p17-p19,p21 (18 prompts);
test p05,p11,p16,p20,p22 (5 prompts, one per category).
Eval 0 DROPPED everywhere (v3 quirk: byte-identical across prompts).

Baselines: per-layer popularity top-8 (train) and prev-token persistence.

PRE-REGISTERED DECISION (before seeing any probe output):
  GO (build prefetch engine on prerouter): (b) or (c) mean recall@8 >= 0.85
      AND probe cost <= 5 ms/token (4-CPU numpy, all 40 layers).
  CONDITIONAL (prefetch predicted + on-demand fill): recall@8 >= 0.70.
  NO-GO (static pins + overlap pipe only): recall@8 < 0.70.
K4-recall (recall@4 of true top-4) REPORTED for Phase 4, not gated.

Probe cost reference: 18 tok/s <=> 55.6 ms/token total budget.
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get(
    "EDGE0_OUT", "/kaggle/working/native-sparse-edge0prerouter-v1-results"))
OUT.mkdir(parents=True, exist_ok=True)

IN_GLOB = os.environ.get("EDGE0_INPUT_GLOB", "/kaggle/input/**/trace_p*.npz")
SMOKE = os.environ.get("EDGE0_SMOKE", "") == "1"  # local only: 2 layers, few iters
RP_SEED = 7
N_LAYERS = 40
N_EXP = 256
K = 8

TEST_PIDS = {5, 11, 16, 20, 22}
DIMS = [64, 256]
FULL_DIM = 2048

RECALL_GO = 0.85
RECALL_COND = 0.70
MS_PER_TOK_GO = 5.0


def load_all():
    files = sorted(glob.glob(IN_GLOB, recursive=True))
    print(f"input files: {len(files)}", flush=True)
    need = 2 if SMOKE else 20
    assert len(files) >= need, f"expected 23 trainset npz, found {len(files)}"
    data = {}
    for f in files:
        pid = int(Path(f).stem.split("_p")[1])
        z = np.load(f)
        H = z["hidden_fp16"].astype(np.float32)
        data[pid] = {"H": H[1:], "ids": z["topk_ids"][1:].astype(np.int32)}
    return data


def rp_matrix(d_in, d_out):
    rng = np.random.RandomState(RP_SEED + d_out)
    return (rng.randn(d_in, d_out) / np.sqrt(d_out)).astype(np.float32)


def recall_at(pred8, true8, k_list=(1, 4, 8)):
    s_true = set(int(x) for x in true8[:8])
    s4 = set(int(x) for x in true8[:4])
    out = {}
    for k in k_list:
        s_pred = set(int(x) for x in pred8[:k])
        if k <= 4:
            out[f"r{k}"] = len(s_pred & s4) / 4  # K4: cover of true top-4
        else:
            out[f"r{k}"] = len(s_pred & s_true) / 8
    return out


def fit_probe(Xtr, Ytr):
    from sklearn.linear_model import LogisticRegression
    it = 20 if SMOKE else 300
    clf = LogisticRegression(solver="saga",
                             max_iter=it, tol=0.05, random_state=RP_SEED,
                             verbose=0)
    # single-label fit on top-1 (ranking recovered from predict_proba)
    clf.fit(Xtr, Ytr)
    return clf


def eval_probe(clf, Xte, Y8te):
    P = clf.predict_proba(Xte)
    top8 = np.argsort(-P, axis=1)[:, :8]
    agg = {"r1": 0.0, "r4": 0.0, "r8": 0.0}
    for i in range(len(Xte)):
        r = recall_at(top8[i], Y8te[i])
        for k in agg:
            agg[k] += r[k]
    for k in agg:
        agg[k] /= len(Xte)
    return agg


def main():
    t0 = time.time()
    data = load_all()
    pids = sorted(data)
    layers = [0, 1] if SMOKE else list(range(N_LAYERS))
    results = {"schema": "native-sparse-edge0prerouter/v1",
               "rp_seed": RP_SEED,
               "test_pids": sorted(TEST_PIDS),
               "train_pids": [p for p in pids if p not in TEST_PIDS],
               "eval0_dropped": True, "variants": {}, "baselines": {},
               "timings_ms_per_tok": {}, "fit_sec": {}}
    # ---- assemble per-layer datasets per variant ----
    # X shapes: (a) H[t,L]; (b) H[t,L-1] (L>=1); (c) H[t-1,L] (t>=1)
    for variant in ("a", "b", "c"):
        results["variants"][variant] = {}
        for dim in DIMS + ([FULL_DIM] if variant == "a" else []):
            RP = None if dim == FULL_DIM else rp_matrix(FULL_DIM, dim)
            per_layer, t_fit = {}, 0.0
            for L in layers:
                if variant == "b" and L == 0:
                    continue
                Xtr, Ytr, Xte, Y8te = [], [], [], []
                for pid in pids:
                    H, I = data[pid]["H"], data[pid]["ids"]
                    T = H.shape[0]
                    if variant == "a":
                        X, Y8 = H[:, L, :], I[:, L, :]
                    elif variant == "b":
                        X, Y8 = H[:, L - 1, :], I[:, L, :]
                    else:
                        X, Y8 = H[:-1, L, :], I[1:, L, :]
                    if RP is not None:
                        X = X @ RP
                    if pid in TEST_PIDS:
                        Xte.append(X)
                        Y8te.append(Y8)
                    else:
                        Xtr.append(X)
                        Ytr.append(Y8[:, 0])  # top-1 single label
                Xtr = np.concatenate(Xtr)
                Ytr = np.concatenate(Ytr)
                Xte = np.concatenate(Xte)
                Y8te = np.concatenate(Y8te)
                f0 = time.time()
                clf = fit_probe(Xtr, Ytr)
                t_fit += time.time() - f0
                agg = eval_probe(clf, Xte, Y8te)
                agg["n_train"] = len(Xtr)
                agg["n_test"] = len(Xte)
                per_layer[str(L)] = agg
            key = f"dim{dim}"
            results["variants"][variant][key] = per_layer
            results["fit_sec"][f"{variant}_{key}"] = round(t_fit, 1)
            r8 = np.mean([v["r8"] for v in per_layer.values()])
            r4 = np.mean([v["r4"] for v in per_layer.values()])
            print(f"variant {variant} {key}: layers={len(per_layer)} "
                  f"mean r8={r8:.4f} r4={r4:.4f} fit={t_fit:.0f}s", flush=True)
    # ---- baselines (test prompts only) ----
    pop_r = {"r1": [], "r4": [], "r8": []}
    per_r = {"r1": [], "r4": [], "r8": []}
    for L in layers:
        Itr = np.concatenate([data[p]["ids"][:, L, :] for p in pids
                              if p not in TEST_PIDS])
        cnt = np.bincount(Itr.ravel(), minlength=N_EXP)
        pop8 = np.argsort(-cnt)[:8]
        for pid in pids:
            if pid not in TEST_PIDS:
                continue
            I = data[pid]["ids"][:, L, :]
            for t in range(len(I)):
                for k, v in recall_at(pop8, I[t]).items():
                    pop_r[k].append(v)
                if t > 0:
                    for k, v in recall_at(I[t - 1], I[t]).items():
                        per_r[k].append(v)
    results["baselines"]["popularity"] = {k: float(np.mean(v))
                                          for k, v in pop_r.items()}
    results["baselines"]["persistence"] = {k: float(np.mean(v))
                                           for k, v in per_r.items()}
    print("baselines:", json.dumps(results["baselines"]), flush=True)
    # ---- probe latency: full 40-layer inference per token, numpy ----
    rng = np.random.RandomState(0)
    Htok = rng.randn(N_LAYERS, FULL_DIM).astype(np.float32)
    RP256 = rp_matrix(FULL_DIM, 256)
    W = rng.randn(N_LAYERS, 256, N_EXP).astype(np.float32)
    b = rng.randn(N_LAYERS, N_EXP).astype(np.float32)
    WF = rng.randn(N_LAYERS, FULL_DIM, N_EXP).astype(np.float32)
    nrep = 200
    t_start = time.time()
    for _ in range(nrep):
        for L in range(N_LAYERS):
            z = (Htok[L] @ RP256) @ W[L] + b[L]
            z = z - z.max()
            e = np.exp(z)
            p = e / e.sum()
            p.argpartition(-8)
    results["timings_ms_per_tok"]["rp256_probe"] = (
        (time.time() - t_start) / nrep * 1000)
    # batched-across-layers (deployment-faithful for variant (c), whose 40
    # probes can all run after the previous token completes; lower bound
    # for (a)/(b), whose hiddens arrive layer-by-layer during the pass)
    t_start = time.time()
    for _ in range(nrep):
        Z = np.einsum("lj,lij->li", Htok @ RP256, W) + b
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        P = E / E.sum(1, keepdims=True)
        np.argpartition(-P, 8, axis=1)
    results["timings_ms_per_tok"]["rp256_probe_batched"] = (
        (time.time() - t_start) / nrep * 1000)
    t_start = time.time()
    for _ in range(nrep):
        for L in range(N_LAYERS):
            z = Htok[L] @ WF[L]
            z = z - z.max()
            e = np.exp(z)
            p = e / e.sum()
            p.argpartition(-8)
    results["timings_ms_per_tok"]["true_router_cost"] = (
        (time.time() - t_start) / nrep * 1000)
    print("timings:", json.dumps(results["timings_ms_per_tok"]), flush=True)
    # ---- pre-registered decision ----
    for v in ("b", "c"):
        r8 = np.mean([x["r8"]
                      for x in results["variants"][v]["dim256"].values()])
        results[f"decision_recall8_{v}"] = round(float(r8), 4)
    best = max(results["decision_recall8_b"], results["decision_recall8_c"])
    ms = results["timings_ms_per_tok"]["rp256_probe"]
    if best >= RECALL_GO and ms <= MS_PER_TOK_GO:
        dec = "GO"
    elif best >= RECALL_COND:
        dec = "CONDITIONAL"
    else:
        dec = "NO-GO"
    results["decision"] = dec
    results["wall_sec"] = time.time() - t0
    (OUT / "result.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({"decision": dec, "best_r8": best, "ms": ms,
                      "wall_h": results["wall_sec"] / 3600}), flush=True)


if __name__ == "__main__":
    main()
