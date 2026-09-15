"""Agent2 probe 02: gated full-attention analysis on REAL Qwen3.5 weights.

Inputs: probes/agent2_attn/weights/*.pt from 01_fetch_weights.py
Math (matches llama.cpp qwen35moe build_layer_attn + HF qwen3_next pattern):
  x      = RMSNorm(emb) * input_ln           (N=calib, 2048)   [proxy states]
  qg     = x @ Wq.T  -> (N,16,512): q_h=(..., :256), g_h=(..., 256:)
  q_h    = RMSNorm(q_h)*q_norm ; k = RMSNorm(Wk x)*k_norm ; v = Wv x
  o_h(t) = softmax(q_t K^T/16) V              (decode sim, cache=all N; NO RoPE)
  y_h    = sigmoid(g_h) * o_h ; c_h = Wo_h @ y_h ; z = sum_h c_h
Outputs: results.json + stdout summary.
ASSUMPTIONS (stated in verdict): (A1) embedding+norm proxy states stand in for
true deep hidden states; (A2) no RoPE in decode sim (norms/scales preserved);
(A3) decode GEMV-bound: ms propto weight-MACs, calibrated to 47.5 ms attn budget.
"""
import json
import os
import torch

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
LAYERS = [3, 19, 39]
N_HEAD, HD, D = 16, 256, 2048
EPS = 1e-6


def rmsn(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def eff_rank(s):
    p = (s ** 2) / (s ** 2).sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def main():
    torch.manual_seed(0)
    emb = torch.load(os.path.join(W, "calib_embed.pt")).to(torch.float32)
    N = emb.shape[0]
    QTOK = min(32, N)  # last QTOK positions act as decode queries
    res = {"layers": {}, "assumptions": ["A1", "A2", "A3"]}
    agg = {"abl": [], "topN": [], "rank": [], "gate_abs": [], "sig": []}
    for L in LAYERS:
        ln = torch.load(os.path.join(W, f"L{L}_input_ln.pt")).to(torch.float32)
        Wq = torch.load(os.path.join(W, f"L{L}_q_proj.pt")).to(torch.float32)
        Wo = torch.load(os.path.join(W, f"L{L}_o_proj.pt")).to(torch.float32)
        Wk = torch.load(os.path.join(W, f"L{L}_k_proj.pt")).to(torch.float32)
        Wv = torch.load(os.path.join(W, f"L{L}_v_proj.pt")).to(torch.float32)
        qn = torch.load(os.path.join(W, f"L{L}_q_norm.pt")).to(torch.float32)
        kn = torch.load(os.path.join(W, f"L{L}_k_norm.pt")).to(torch.float32)
        assert Wq.shape == (8192, 2048) and Wo.shape == (2048, 4096)
        x = rmsn(emb, ln)  # (N,2048) proxy states
        qg = (x @ Wq.T).view(N, N_HEAD, 2 * HD)
        q, g = qg[..., :HD], qg[..., HD:]          # pre-norm q, pre-sigmoid gate
        s = torch.sigmoid(g)
        # --- gate magnitude stats ---
        ga = g.abs()
        lr = {
            "gate_abs_mean": float(ga.mean()),
            "gate_abs_p50": float(ga.median()),
            "gate_abs_p90": float(ga.quantile(0.90)),
            "gate_abs_p99": float(ga.quantile(0.99)),
            "gate_pre_std": float(g.std()),
            "frac_sig_lt_0.01": float((s < 0.01).float().mean()),
            "frac_sig_lt_0.10": float((s < 0.10).float().mean()),
            "frac_sig_gt_0.90": float((s > 0.90).float().mean()),
            "frac_sig_mid_0.1_0.9": float(((s > 0.10) & (s < 0.90)).float().mean()),
            "within_head_dim_std_mean": float(g.std(-1).mean()),
            "within_head_dim_std_std": float(g.std(-1).std()),
            "token_std_per_dim_mean": float(g.std(0).mean()),
        }
        # per-head gate stats + effective ranks (activation & weight)
        heads = []
        for h in range(N_HEAD):
            Gh = g[:, h, :]                       # (N,256) gate activations
            Wgh = Wq[h * 512 + 256:(h + 1) * 512, :]  # (256,2048) gate rows
            sa = torch.linalg.svdvals(Gh)
            sw = torch.linalg.svdvals(Wgh)
            ev = (sa ** 2) / (sa ** 2).sum()
            var_k = {k: float(ev[:k].sum()) for k in (1, 2, 4, 8, 16)}
            rec = {}
            U, S, Vh = torch.linalg.svd(Gh, full_matrices=False)
            denom = float((Gh ** 2).sum())
            for k in (1, 2, 4, 8, 16):
                Gk = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
                rec[k] = float(((Gh - Gk) ** 2).sum() / denom) ** 0.5
            heads.append({
                "gate_abs_mean": float(Gh.abs().mean()),
                "sig_mean": float(torch.sigmoid(Gh).mean()),
                "eff_rank_act": eff_rank(sa), "eff_rank_w": eff_rank(sw),
                "var_explained": var_k, "rel_recon_err": rec})
        lr["heads"] = heads
        lr["eff_rank_act_mean"] = sum(h["eff_rank_act"] for h in heads) / 16
        lr["eff_rank_w_mean"] = sum(h["eff_rank_w"] for h in heads) / 16
        lr["var_explained_mean"] = {str(k): sum(h["var_explained"][k] for h in heads) / 16
                                    for k in (1, 2, 4, 8, 16)}
        lr["rel_recon_err_mean"] = {str(k): sum(h["rel_recon_err"][k] for h in heads) / 16
                                    for k in (1, 2, 4, 8, 16)}
        # --- decode simulation: K/V cache from real weights, last QTOK queries ---
        K = rmsn((x @ Wk.T).view(N, 2, HD), kn)     # (N,2,256)
        V = (x @ Wv.T).view(N, 2, HD)
        Q = rmsn(q[-QTOK:], qn)                      # (QTOK,16,256)
        G = g[-QTOK:]                                # (QTOK,16,256)
        S = torch.sigmoid(G)
        c = torch.zeros(QTOK, N_HEAD, D)             # per-head output contributions
        ynorm = torch.zeros(QTOK, N_HEAD)
        for h in range(N_HEAD):
            kv = h // 8
            sc = (Q[:, h, :] @ K[:, kv, :].T) / 16.0  # (QTOK,N)
            a = torch.softmax(sc, -1)
            o = a @ V[:, kv, :]                        # (QTOK,256)
            y = S[:, h, :] * o
            ynorm[:, h] = y.norm(dim=-1)
            Woh = Wo[:, h * HD:(h + 1) * HD]           # (2048,256)
            c[:, h, :] = y @ Woh.T
        z = c.sum(1)                                   # (QTOK,2048)
        zn = z.norm(dim=-1).clamp_min(1e-9)
        cn = c.norm(dim=-1)                             # (QTOK,16)
        share = (cn / cn.sum(1, keepdim=True))          # per-token head share
        lr["head_share_mean"] = [float(v) for v in share.mean(0)]
        lr["head_share_std"] = [float(v) for v in share.std(0)]
        lr["ynorm_mean"] = [float(v) for v in ynorm.mean(0)]
        # single-head ablation (exact: output is a linear head sum)
        abl = (cn / zn[:, None]).mean(0)                # rel err of dropping h
        order = torch.argsort(abl)
        lr["ablation_rel_err"] = [float(v) for v in abl]
        lr["ablation_sorted_idx"] = [int(v) for v in order]
        # static top-N (drop lowest-contribution heads globally)
        mean_c = cn.mean(0)
        sorder = torch.argsort(mean_c)                  # ascending: drop first
        topN, dynN = {}, {}
        for keep in (15, 14, 12, 8, 4, 2, 1):
            drop = sorder[:16 - keep]
            err = ((c[:, drop, :].sum(1)).norm(dim=-1) / zn).mean()
            topN[keep] = float(err)
            # dynamic oracle: per-token top-keep by ||c_h||
            tk = torch.topk(cn, keep, dim=1).indices
            keep_mask = torch.zeros_like(cn).scatter_(1, tk, 1.0)
            errd = ((c * (1 - keep_mask)[..., None]).sum(1).norm(dim=-1) / zn).mean()
            dynN[keep] = float(errd)
        lr["static_topN_rel_err"] = {str(k): v for k, v in topN.items()}
        lr["dynamic_topN_rel_err"] = {str(k): v for k, v in dynN.items()}
        # deployable low-rank gate: W_g,h ~= U_r S_r V_r^T; a(x)=V_r^T x; g^=U_r S_r a
        # error measured end-to-end in output domain (sigmoid -> y -> o_proj -> z)
        Xq = x[-QTOK:]                       # (QTOK,2048) query states
        lr["lowrank_gate"] = {}
        for r in (1, 2, 4, 8, 16):
            zz = torch.zeros_like(z)
            for h in range(N_HEAD):
                Wgh = Wq[h * 512 + 256:(h + 1) * 512, :]
                U, S_, Vh = torch.linalg.svd(Wgh, full_matrices=False)
                Wr = U[:, :r] @ torch.diag(S_[:r]) @ Vh[:r, :]
                gh = Xq @ Wr.T                  # (QTOK,256) approx pre-sigmoid gate
                kv = h // 8
                sc = (Q[:, h, :] @ K[:, kv, :].T) / 16.0
                o = torch.softmax(sc, -1) @ V[:, kv, :]
                y = torch.sigmoid(gh) * o
                Woh = Wo[:, h * HD:(h + 1) * HD]
                zz += y @ Woh.T
            err = ((zz - z).norm(dim=-1) / zn).mean()
            lr["lowrank_gate"][str(r)] = float(err)
        res["layers"][L] = lr
        agg["abl"].append(abl)
        print(f"L{L}: gate|.|={lr['gate_abs_mean']:.3f} sig<0.1={lr['frac_sig_lt_0.10']:.4f} "
              f"erank_act={lr['eff_rank_act_mean']:.1f} erank_w={lr['eff_rank_w_mean']:.1f} "
              f"abl_med={float(abl.median()):.4f} top8={topN[8]:.4f} dyn8={dynN[8]:.4f}", flush=True)
    # cross-layer aggregate
    A = torch.stack(agg["abl"])
    res["agg"] = {
        "ablation_mean_across_layers": [float(v) for v in A.mean(0)],
        "static_topN_mean": {k: sum(res["layers"][L]["static_topN_rel_err"][k] for L in LAYERS) / 3
                             for k in ("15", "14", "12", "8", "4", "2", "1")},
        "dynamic_topN_mean": {k: sum(res["layers"][L]["dynamic_topN_rel_err"][k] for L in LAYERS) / 3
                              for k in ("15", "14", "12", "8", "4", "2", "1")},
        "lowrank_gate_mean": {k: sum(res["layers"][L]["lowrank_gate"][k] for L in LAYERS) / 3
                              for k in ("1", "2", "4", "8", "16")},
    }
    json.dump(res, open(os.path.join(W, "results.json"), "w"), indent=1)
    print("OK: results.json written", flush=True)


if __name__ == "__main__":
    main()
