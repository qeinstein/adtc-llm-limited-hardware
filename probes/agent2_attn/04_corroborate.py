"""Agent2 probe 04: corroborate split/scale/robustness of the 02 findings.

C1: row/col norms of Wq halves (Q vs gate?), Wk, Wv, Wo per-head slices.
C2: attention entropy with half0-as-Q vs half1-as-Q (validates q/g split).
C3: Gaussian-x replication of gate stats + lowrank err + ablation (proxy check).
C4:Wo per-head col norms; K-cache norms.
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


def main():
    torch.manual_seed(1)
    emb = torch.load(os.path.join(W, "calib_embed.pt")).to(torch.float32)
    N = emb.shape[0]
    out = {}
    for L in LAYERS:
        ln = torch.load(os.path.join(W, f"L{L}_input_ln.pt")).to(torch.float32)
        Wq = torch.load(os.path.join(W, f"L{L}_q_proj.pt")).to(torch.float32)
        Wo = torch.load(os.path.join(W, f"L{L}_o_proj.pt")).to(torch.float32)
        Wk = torch.load(os.path.join(W, f"L{L}_k_proj.pt")).to(torch.float32)
        Wv = torch.load(os.path.join(W, f"L{L}_v_proj.pt")).to(torch.float32)
        qn = torch.load(os.path.join(W, f"L{L}_q_norm.pt")).to(torch.float32)
        kn = torch.load(os.path.join(W, f"L{L}_k_norm.pt")).to(torch.float32)
        d = {}
        # C1: row norms per half
        qrows = Wq.view(N_HEAD, 2 * HD, D)[:, :HD, :].reshape(-1, D)
        grows = Wq.view(N_HEAD, 2 * HD, D)[:, HD:, :].reshape(-1, D)
        d["qrow_norm_mean"] = float(qrows.norm(dim=-1).mean())
        d["grow_norm_mean"] = float(grows.norm(dim=-1).mean())
        d["qrow_norm_p50"] = float(qrows.norm(dim=-1).median())
        d["grow_norm_p50"] = float(grows.norm(dim=-1).median())
        d["krow_norm_mean"] = float(Wk.norm(dim=-1).mean())
        d["vrow_norm_mean"] = float(Wv.norm(dim=-1).mean())
        d["wo_head_fro_mean"] = float(sum(Wo[:, h*HD:(h+1)*HD].norm() for h in range(16)) / 16)
        d["wo_head_fro_min"] = float(min(Wo[:, h*HD:(h+1)*HD].norm() for h in range(16)))
        d["wo_head_fro_max"] = float(max(Wo[:, h*HD:(h+1)*HD].norm() for h in range(16)))
        x = rmsn(emb, ln)
        qg = (x @ Wq.T).view(N, N_HEAD, 2 * HD)
        d["half0_act_std"] = float(qg[..., :HD].std())
        d["half1_act_std"] = float(qg[..., HD:].std())
        d["x_rms_mean"] = float(x.pow(2).mean(-1).sqrt().mean())
        # C2: entropy with each half as Q
        K = rmsn((x @ Wk.T).view(N, 2, HD), kn)
        ents = {}
        for tag, half in (("half0", qg[..., :HD]), ("half1", qg[..., HD:])):
            Q = rmsn(half[-8:], qn)
            e = []
            for h in range(N_HEAD):
                sc = (Q[:, h, :] @ K[:, h // 8, :].T) / 16.0
                a = torch.softmax(sc, -1)
                e.append(float((-(a * (a + 1e-12).log()).sum(-1)).mean()))
            ents[tag] = sum(e) / 16
        d["attn_entropy_half0"] = ents["half0"]
        d["attn_entropy_half1"] = ents["half1"]
        d["attn_entropy_uniform"] = float(torch.log(torch.tensor(float(N))))
        # C3: Gaussian replication (matched RMS)
        xg = rmsn(torch.randn(N, D), ln)
        gg = (xg @ Wq.T).view(N, N_HEAD, 2 * HD)[..., HD:]
        sg = torch.sigmoid(gg)
        d["gauss_gate_abs_mean"] = float(gg.abs().mean())
        d["gauss_gate_pre_std"] = float(gg.std())
        d["gauss_frac_sig_mid"] = float(((sg > 0.10) & (sg < 0.90)).float().mean())
        # Gaussian lowrank r8 output err (deployable W-SVD) + ablation median
        QTOK = 32
        Kg = rmsn((xg @ Wk.T).view(N, 2, HD), kn)
        Vg = (xg @ Wv.T).view(N, 2, HD)
        Q = rmsn((xg @ Wq.T).view(N, N_HEAD, 2 * HD)[..., :HD][-QTOK:], qn)
        c = torch.zeros(QTOK, N_HEAD, D)
        for h in range(N_HEAD):
            a = torch.softmax((Q[:, h, :] @ Kg[:, h // 8, :].T) / 16.0, -1)
            o = a @ Vg[:, h // 8, :]
            y = torch.sigmoid(gg[-QTOK:, h, :]) * o
            c[:, h, :] = y @ Wo[:, h*HD:(h+1)*HD].T
        z = c.sum(1)
        zn = z.norm(dim=-1).clamp_min(1e-9)
        abl = (c.norm(dim=-1) / zn[:, None]).mean(0)
        d["gauss_abl_median"] = float(abl.median())
        Xq = xg[-QTOK:]
        zz = torch.zeros_like(z)
        for h in range(N_HEAD):
            Wh = Wq[h*512+256:(h+1)*512, :]
            U, S_, Vh = torch.linalg.svd(Wh, full_matrices=False)
            Wr = U[:, :8] @ torch.diag(S_[:8]) @ Vh[:8, :]
            gh = Xq @ Wr.T
            a = torch.softmax((Q[:, h, :] @ Kg[:, h//8, :].T) / 16.0, -1)
            y = torch.sigmoid(gh) * (a @ Vg[:, h//8, :])
            zz += y @ Wo[:, h*HD:(h+1)*HD].T
        d["gauss_lowrank8_err"] = float(((zz - z).norm(dim=-1) / zn).mean())
        # static top8 err on Gaussian
        cn = c.norm(dim=-1)
        sorder = torch.argsort(cn.mean(0))
        d["gauss_static8_err"] = float((c[:, sorder[:8], :].sum(1).norm(dim=-1) / zn).mean())
        out[L] = d
        print(f"L{L}: " + " ".join(f"{k}={v:.4f}" for k, v in d.items()), flush=True)
    json.dump(out, open(os.path.join(W, "corroboration.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
