"""Local analysis for the dynskip-ctx-v1 kernel output.

Reads output/dynskip-ctx-v1/ (result.json, dumps.tar.gz, routes, skiplogs,
control/policy texts), streams one layer at a time from the tarball (local
disk is tight), and writes small committed tables to probes/dynskip_contextual/.

Needs (all local, no model download):
  /tmp/agent1_f32  L00/L20/L39 experts (L00 topped up on demand from HF)
  /tmp/agent1_raw  routers + norms
  /tmp/dynskip/shexp/shexp_L{00,39}.npz + atom shexp_L20.npz (shared, bf16)
  probes/agent2_attn/weights/tokenizer.json (flip positions)
"""
from __future__ import annotations

import glob
import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
KOUT = os.path.join(WS, "output/dynskip-ctx-v1/native-sparse-dynskip-ctx-v1-results")
WORK = "/tmp/ctxwork"
RAW = "/tmp/agent1_raw"
DEQ = "/tmp/agent1_f32"
LAYERS = (0, 20, 39)
BARS = (0.01, 0.02, 0.05)

_meta = json.load(open(os.path.join(
    WS, "probes/functional_basis_oracle/fb_assets/context_meta.json")))
TR_P = set(_meta["train_prompts"])
TE_P = set(_meta["held_prompts"])


def router(L):
    return np.fromfile(f"{RAW}/router_L{L:02d}.f32", dtype=np.float32).reshape(256, 2048)


def probs_full(X, W):
    Lg = X.astype(np.float64) @ W.T.astype(np.float64)
    Lg -= Lg.max(axis=1, keepdims=True)
    P = np.exp(Lg)
    P /= P.sum(axis=1, keepdims=True)
    return P


def load_layer_bins(tar, res, LAY):
    """Extract one layer's bins to WORK, return dict kind -> (N,2048) f32 + n_dec list."""
    os.makedirs(WORK, exist_ok=True)
    n_dec = [r["n_dec"] for r in res["prompt_results"]]
    out = {}
    for kind in ("x", "h0", "h1", "B"):
        parts = []
        for pid, n in enumerate(n_dec):
            name = f"prompt_{pid:02d}.L{LAY:02d}.{kind}.bin"
            f = tar.extractfile(name)
            if f is None:
                raise RuntimeError(f"tarball missing {name}")
            a = np.frombuffer(f.read(), dtype=np.float32).copy()
            if a.size != n * 2048:
                raise RuntimeError(f"{name}: {a.size} != {n}*2048")
            parts.append(a.reshape(n, 2048))
        out[kind] = np.concatenate(parts, axis=0)
    return out, n_dec


def layer_prompt_index(n_dec):
    """Row slices per prompt + train/held masks over concat rows."""
    sl, trm, tem = [], [], []
    o = 0
    for pid, n in enumerate(n_dec):
        sl.append(slice(o, o + n))
        trm += [pid in TR_P] * n
        tem += [pid in TE_P] * n
        o += n
    return sl, np.array(trm), np.array(tem)


def route_hook_ids(pid, L):
    rows = [json.loads(l) for l in open(f"{KOUT}/prompt_{pid:02d}.routes.jsonl")]
    ids = []
    for r in rows:
        if r["shape"][0] == 8 and r["shape"][1] == 1:
            lay = int(r["weight"].split("blk.", 1)[1].split(".", 1)[0])
            if lay == L:
                ids.append([int(x) for x in r["ids"]])
    return np.array(ids, dtype=np.int32)


def shared_S(X, L):
    if L == 20:
        sh = np.load(os.path.join(
            WS, "probes/atom_factorization_one_layer/assets/shexp_L20.npz"))
    else:
        sh = np.load(f"/tmp/dynskip/shexp/shexp_L{L:02d}.npz")
    G, U, D = (sh["G"].astype(np.float64), sh["U"].astype(np.float64),
               sh["D"].astype(np.float64))
    w = sh["w"].astype(np.float64).ravel()
    S = np.zeros((X.shape[0], 2048), np.float64)
    for a in range(0, X.shape[0], 512):
        Xc = X[a:a + 512]
        gv, uv = Xc @ G.T, Xc @ U.T
        Sv = ((gv / (1.0 + np.exp(-gv))) * uv) @ D.T
        sg = 1.0 / (1.0 + np.exp(-(Xc @ w)))
        S[a:a + 512] = Sv * sg[:, None]
    return S


GBASE = ("https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
         "bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf")
_GATE_B, _DOWN_B = 270336, 335872
_INV = json.load(open(os.path.join(
    WS, "probes/agent1_expert/gguf_inventory.json")))
_T = {t["name"]: t for t in _INV["tensors"]}


def fetch_dequant_expert(L, e):
    """Range-fetch one expert's quants, dequant, return (G,U,D) f32; cleans up."""
    tg = _T[f"blk.{L}.ffn_gate_exps.weight"]["file_offset"]
    tu = _T[f"blk.{L}.ffn_up_exps.weight"]["file_offset"]
    td = _T[f"blk.{L}.ffn_down_exps.weight"]["file_offset"]
    tmp = []
    outs = []
    for base, nB, kind, ext in ((tg, _GATE_B, "gate", "iq2xxs"), (tu, _GATE_B, "up", "iq2xxs"),
                                (td, _DOWN_B, "down", "iq2s")):
        p = f"{WORK}/tmpE_{L:02d}_{e:03d}_{kind}.{ext}"
        r = subprocess.run(["curl", "-sL", "--fail", "--retry", "3",
                            "-r", f"{base + e * nB}-{base + (e + 1) * nB - 1}",
                            "-o", p, GBASE], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"expert fetch L{L}E{e}: {r.stderr[:200]}")
        tmp.append(p)
    for src, ty, ne0, nr, kind in (
            (tmp[0], "iq2_xxs", 2048, 512, "gate"),
            (tmp[1], "iq2_xxs", 2048, 512, "up"),
            (tmp[2], "iq2_s", 512, 2048, "down")):
        dst = f"{WORK}/tmpE_{L:02d}_{e:03d}_{kind}.f32"
        r = subprocess.run(["/tmp/dequant", src, dst, ty, str(ne0), str(nr)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"dequant L{L}E{e}{kind}: {r.stderr[:200]}")
        outs.append(dst)
    G = np.fromfile(outs[0], dtype=np.float32).reshape(512, 2048)
    U = np.fromfile(outs[1], dtype=np.float32).reshape(512, 2048)
    D = np.fromfile(outs[2], dtype=np.float32).reshape(2048, 512)
    for p in tmp + outs:
        os.unlink(p)
    return G, U, D


def routed_R8(X, top8, w8, L):
    """Sparse exact R8; tops up missing cached experts on demand (streamed)."""
    from collections import defaultdict
    N = X.shape[0]
    use = defaultdict(list)
    for t in range(N):
        for s in range(8):
            use[int(top8[t, s])].append((t, s))
    R8 = np.zeros((N, 2048), np.float64)
    Xf = X.astype(np.float32)
    w8 = w8.astype(np.float64)
    n_fetch = 0
    for i, e in enumerate(sorted(use)):
        p = f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32"
        if os.path.exists(p):
            G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32",
                            dtype=np.float32).reshape(512, 2048)
            U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32",
                            dtype=np.float32).reshape(512, 2048)
            D = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32",
                            dtype=np.float32).reshape(2048, 512)
        else:
            G, U, D = fetch_dequant_expert(L, e)
            n_fetch += 1
        toks = sorted(set(t for t, _ in use[e]))
        Xt = Xf[toks]
        gv, uv = Xt @ G.T, Xt @ U.T
        Y = (((gv / (1.0 + np.exp(-gv))) * uv) @ D.T).astype(np.float64)
        idx = {t: k for k, t in enumerate(toks)}
        for (t, s) in use[e]:
            R8[t] += w8[t, s] * Y[idx[t]]
        del G, U, D, gv, uv, Y
        if (i + 1) % 60 == 0:
            print(f"  L{L} R8 {i + 1}/{len(use)} (fetched {n_fetch})", flush=True)
    print(f"  L{L} R8 done, union {len(use)}, on-demand fetched {n_fetch}", flush=True)
    return R8

def strict_rates(r, v):
    """Best strict max-error skip count over both threshold directions."""
    best = (0, None, None)
    for direction in ("low", "high"):
        order = np.argsort(v) if direction == "low" else np.argsort(-v)
        cm = np.maximum.accumulate(r[order])
        ok = np.where(cm < BAR)[0]
        k = (ok[-1] + 1) if len(ok) > 0 else 0
        if k > best[0]:
            best = (k, direction, float(v[order[k - 1]]) if k > 0 else None)
    return best


BAR = 0.02


def analyze_layer(tar, res, L):
    global BAR
    print(f"===== layer {L} =====", flush=True)
    Z, n_dec = load_layer_bins(tar, res, L)
    X = Z["x"].astype(np.float64)
    H0 = Z["h0"].astype(np.float64)
    H1 = Z["h1"].astype(np.float64)
    N = X.shape[0]
    sl, trm, tem = layer_prompt_index(n_dec)
    W = router(L)
    P = probs_full(X, W)
    top8 = np.argsort(P, axis=1)[:, ::-1][:, :8]
    pw = np.take_along_axis(P, top8, axis=1)
    pw = pw / pw.sum(axis=1, keepdims=True)
    # cross-check 1: recomputed top8 vs hook ids
    hook = np.concatenate([route_hook_ids(pid, L) for pid in range(len(n_dec))], axis=0)
    match = (top8 == hook).all(axis=1).mean()
    print(f"  route cross-check: {match * 100:.2f}% ({match * N:.0f}/{N})", flush=True)
    # cross-check 2: X == h1/rms(h1)*g
    g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32",
                    dtype=np.float32).astype(np.float64)
    rms = np.sqrt((H1 ** 2).mean(axis=1, keepdims=True))
    Xhat = H1 / np.maximum(rms, 1e-30) * g[None, :]
    nerr = np.linalg.norm(X - Xhat, axis=1) / np.maximum(np.linalg.norm(X, axis=1), 1e-30)
    print(f"  X-vs-h1/g recon: max {nerr.max():.2e} mean {nerr.mean():.2e}", flush=True)
    R8 = routed_R8(X, top8, pw, L)
    S = shared_S(X, L)
    B = R8 + S
    # cross-check 3: local B vs runtime-dumped MoE branch output
    Brt = Z["B"].astype(np.float64)
    berr = (np.linalg.norm(B - Brt, axis=1)
            / np.maximum(np.linalg.norm(Brt, axis=1), 1e-30))
    print(f"  B local-vs-runtime: max {berr.max():.2e} mean {berr.mean():.2e}", flush=True)
    Y = H1 + B
    M = H1 - H0  # mixer output, exact
    nY = np.linalg.norm(Y, axis=1)
    nX = np.linalg.norm(X, axis=1)
    r_full = np.linalg.norm(B, axis=1) / nY
    r_rout = np.linalg.norm(R8, axis=1) / nY
    r_blk = np.linalg.norm(M + B, axis=1) / nY
    c_full = (H1 * Y).sum(axis=1) / np.maximum(
        np.linalg.norm(H1, axis=1) * nY, 1e-30)
    c_blk = (H0 * Y).sum(axis=1) / np.maximum(
        np.linalg.norm(H0, axis=1) * nY, 1e-30)
    ent = -(P * np.log(P + 1e-300)).sum(axis=1)
    Ps = np.sort(P, axis=1)[:, ::-1]
    sigs = {"top1": Ps[:, 0], "ent": ent, "top8mass": Ps[:, :8].sum(axis=1),
            "nx": nX, "mixn": np.linalg.norm(M, axis=1),
            "r_ratio": np.linalg.norm(R8, axis=1) / nX,
            "s_ratio": np.linalg.norm(S, axis=1) / nX,
            "cos_xb": (X * B).sum(axis=1) / np.maximum(
                nX * np.linalg.norm(B, axis=1), 1e-30)}
    tab = {"layer": L, "N": N, "route_match": float(match),
           "x_h1_recon_max": float(nerr.max()),
           "B_rt_max": float(berr.max()), "B_rt_mean": float(berr.mean()),
           "min_full": float(r_full.min()), "min_rout": float(r_rout.min()),
           "min_blk": float(r_blk.min()),
           "med_full": float(np.median(r_full)), "med_rout": float(np.median(r_rout)),
           "med_blk": float(np.median(r_blk)),
           "cos_full_mean": float(c_full.mean()), "cos_full_min": float(c_full.min()),
           "cos_blk_mean": float(c_blk.mean()), "cos_blk_min": float(c_blk.min()),
           "mix_over_Y_med": float(np.median(np.linalg.norm(M, axis=1) / nY))}
    for bar in BARS:
        BAR = bar
        o = {"oracle_full": float((r_full < bar).mean()),
             "oracle_rout": float((r_rout < bar).mean()),
             "oracle_blk": float((r_blk < bar).mean())}
        for tgt, r in (("full", r_full), ("rout", r_rout)):
            best = (0, None, None, None)
            for nm, v in sigs.items():
                k, dr, th = strict_rates(r, v)
                if k > best[0]:
                    best = (k, nm, dr, th)
            o[f"best_{tgt}"] = {"rate": best[0] / N, "signal": best[1],
                                "dir": best[2], "thr": best[3]}
        tab[f"bar_{bar}"] = o
    # held transfer for best 2% routed rule (tune train, eval held)
    BAR = 0.02
    b = tab["bar_0.02"]["best_rout"]
    if b["signal"] is not None:
        v = sigs[b["signal"]]
        tr, te = trm, tem
        if b["dir"] == "low":
            sk_te = v[te] <= b["thr"]
        else:
            sk_te = v[te] >= b["thr"]
        tab["held_rout_2pct"] = {
            "signal": b["signal"], "dir": b["dir"], "thr": b["thr"],
            "skip_rate": float(sk_te.mean()),
            "max_err": float(r_rout[te][sk_te].max()) if sk_te.sum() else 0.0,
            "n_viol": int((r_rout[te][sk_te] >= 0.02).sum()),
            "n_skip": int(sk_te.sum()), "n_held": int(te.sum())}
    # per-prompt-split oracle (train vs held, routed 2%)
    tab["split_rout_2pct"] = {
        "train": float((r_rout[trm] < 0.02).mean()),
        "held": float((r_rout[tem] < 0.02).mean())}
    np.savez_compressed(f"/tmp/ctxwork/L{L:02d}_tab.npz",
                        r_full=r_full, r_rout=r_rout, r_blk=r_blk,
                        **{f"s_{k}": v for k, v in sigs.items()},
                        trm=trm, tem=tem)
    del X, H0, H1, R8, S, B, Y, M, P
    return tab


def policy_analysis(res):
    print("===== frozen L20 policy =====", flush=True)
    n_dec = [r["n_dec"] for r in res["prompt_results"]]
    # runtime top1 (control skiplog) aggregated
    rt1, dec_n = [], []
    for pid, n in enumerate(n_dec):
        fw = {}
        for line in open(f"{KOUT}/prompt_{pid:02d}.skipctl.jsonl"):
            r = json.loads(line)
            if r.get("revisit"):
                continue
            fw.setdefault(int(r["fwd"]), []).append(r)
        ids = sorted(fw)
        if len(ids) % 2 == 0 and len(ids) and all(
                len(fw[a]) == len(fw[b]) and all(
                    x["top1"] == y["top1"] for x, y in zip(fw[a], fw[b]))
                for a, b in zip(ids[0::2], ids[1::2])):
            ids = ids[0::2]
        d = [fw[i][0] for i in ids if fw[i][0]["n_rows"] == 1]
        assert len(d) == n, (pid, len(d), n)
        rt1 += [x["top1"] for x in d]
    rt1 = np.array(rt1)
    # local top1 for L20 decode rows
    d = np.load("/tmp/ctxwork/L20_tab.npz")
    loc = d["s_top1"]
    print(f"  runtime-vs-local top1 maxdiff: {np.abs(rt1 - loc).max():.2e}", flush=True)
    skip = rt1 <= 0.02
    print(f"  runtime skip rate: {skip.mean() * 100:.2f}% ({skip.sum()}/{len(skip)})", flush=True)
    rr = d["r_rout"]
    print(f"  skipped hidden err: max {rr[skip].max() * 100:.3f}% "
          f"med {np.median(rr[skip]) * 100:.3f}% n_viol {(rr[skip] >= 0.02).sum()}", flush=True)
    # train/held split of runtime skip
    sl, trm, tem = [], [], []
    o = 0
    for pid, n in enumerate(n_dec):
        trm += [pid in TR_P] * n
        tem += [pid in TE_P] * n
    trm, tem = np.array(trm), np.array(tem)
    print(f"  held skip rate: {skip[tem].mean() * 100:.2f}%, "
          f"held skipped maxerr {rr[tem][skip[tem]].max() * 100:.3f}%", flush=True)
    # flips from texts
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(
            WS, "probes/agent2_attn/weights/tokenizer.json"))
    except Exception as e:
        tok = None
        print(f"  tokenizer unavailable: {e}", flush=True)
    import re

    def gen_of(raw: bytes) -> bytes:
        m = re.search(rb"\n> .*?\n\n(.*)\n\n\[ Prompt:", raw, re.S)
        assert m, "generation span not found"
        return m.group(1)

    div, agree_b, tot_b, worst = 0, 0, 0, []
    for pid in range(len(n_dec)):
        a = gen_of(open(f"{KOUT}/prompt_{pid:02d}.control.txt", "rb").read())
        b = gen_of(open(f"{KOUT}/prompt_{pid:02d}.policy.txt", "rb").read())
        tot_b += max(len(a), len(b))
        if a == b:
            agree_b += len(a)
            continue
        div += 1
        i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y),
                 min(len(a), len(b)))
        agree_b += i
        ctx = {"pid": pid, "first_diff_byte": i, "len_ctl": len(a), "len_pol": len(b),
               "ctl_tail": a[max(0, i - 60):i + 120].decode("utf-8", "replace"),
               "pol_tail": b[max(0, i - 60):i + 120].decode("utf-8", "replace")}
        if tok is not None:
            try:
                n_full = len(tok.encode(a.decode("utf-8", "replace")).ids)
                n_pre = len(tok.encode(a[:i].decode("utf-8", "replace")).ids)
                ctx["flip_tok_pos"] = n_pre
                ctx["flip_tok_frac"] = n_pre / max(n_full, 1)
            except Exception:
                pass
        worst.append(ctx)
    print(f"  diverged prompts: {div}/{len(n_dec)}, byte agreement {agree_b / tot_b * 100:.2f}%", flush=True)
    return {"runtime_skip_rate": float(skip.mean()),
            "runtime_local_top1_maxdiff": float(np.abs(rt1 - loc).max()),
            "skipped_max_err": float(rr[skip].max()),
            "skipped_n_viol": int((rr[skip] >= 0.02).sum()),
            "held_skip_rate": float(skip[tem].mean()),
            "held_skipped_max_err": float(rr[tem][skip[tem]].max()),
            "diverged_prompts": div, "byte_agreement": float(agree_b / tot_b),
            "worst": worst[:8]}


def main():
    res = json.load(open(f"{KOUT}/result.json"))
    assert res["status"] == "ok", res.get("status")
    print(f"kernel: {res['decode']['decode_tokens']} decode tokens, "
          f"{res['summary']['prompts_diverged']} diverged", flush=True)
    tabs = {}
    with tarfile.open(f"{KOUT}/dumps.tar.gz", "r:gz") as tf:
        for L in LAYERS:
            tabs[str(L)] = analyze_layer(tf, res, L)
    pol = policy_analysis(res)
    out = {"layers": tabs, "policy": pol,
           "kernel_summary": res["summary"],
           "decode_tokens": res["decode"]["decode_tokens"]}
    json.dump(out, open(os.path.join(HERE, "tables.json"), "w"), indent=1)
    print("wrote tables.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
