"""fb_03: ingest Kaggle L20 hidden dump -> contextual oracle inputs.

Reads kernel output dir (result.json + prompt_*.L20.bin/.idx +
prompt_*.routes.jsonl):
  1. parse + verify dumps (idx sums vs bin bytes; fail loud on hook miss)
  2. exact local routing (F32 router, softmax/top8/renorm) for all rows
  3. cross-check recomputed top-8 vs route-hook IDs on decode tokens
     (proves the dumped x is the true MoE input AND routing parity)
  4. split by PROMPT 22 train / 10 held (seed-shuffled; disjoint contexts;
     plus a held-vs-train near-duplicate cosine audit)
  5. exact R8 for all contextual tokens (streamed union experts)

Outputs (fb_assets/): context_inputs.npz (Xtr,Xte,top8tr,w8tr,top8te,w8te,
  prompt split meta), context_teacher.npz (R8tr,R8te), context_meta.json

Usage: python3 fb_03_context.py /path/to/kernel/output
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fb_assets")
RAW = "/tmp/agent1_raw"
DEQ = "/tmp/agent1_f32"
L = 20


def main():
    kd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "kaggle_out")
    res = json.load(open(os.path.join(kd, "result.json")))
    assert res["status"] == "ok" and res["decode"]["layer"] == L
    print(f"kernel wall={res['wall_sec'] / 60:.0f}m rows={res['decode']['dump_rows']}",
          flush=True)
    Xs, Rt, meta = [], [], []
    n_decode_tot = 0
    for pr in res["prompt_results"]:
        pid = pr["prompt_id"]
        pre = os.path.join(kd, f"prompt_{pid:02d}.L20")
        evs = [json.loads(l) for l in open(pre + ".idx")]
        assert all("error" not in e for e in evs), f"p{pid} hook error"
        # hook fires twice per forward call with bit-identical data
        # (executor visits the router node in two passes); verify + dedupe
        assert len(evs) % 2 == 0, f"p{pid}: odd event count"
        raw = np.fromfile(pre + ".bin", dtype=np.float32).reshape(-1, 2048)
        off = 0
        for i in range(0, len(evs), 2):
            a, b = int(evs[i]["n_rows"]), int(evs[i + 1]["n_rows"])
            assert a == b, f"p{pid} ev{i}: {a} vs {b}"
            d = np.abs(raw[off:off + a] - raw[off + a:off + 2 * a]).max()
            assert d == 0.0, f"p{pid} ev{i}: pair diff {d}"
            off += 2 * a
        keep = np.concatenate(
            [np.arange(sum(int(evs[j]["n_rows"]) for j in range(i)),
                       sum(int(evs[j]["n_rows"]) for j in range(i))
                       + int(evs[i]["n_rows"]))
             for i in range(0, len(evs), 2)])
        raw = raw[keep]
        evs = evs[0::2]
        nrs = [int(e["n_rows"]) for e in evs]
        # prefill = leading n>1 events (usually one), decode = trailing n==1
        k = 0
        while k < len(nrs) and nrs[k] > 1:
            k += 1
        assert all(n == 1 for n in nrs[k:]), f"p{pid} bad event pattern {nrs}"
        n_pre, n_dec = sum(nrs[:k]), sum(nrs[k:])
        X = raw
        assert X.shape[0] == n_pre + n_dec == pr["dump_rows"] // 2
        Xs.append(X)
        meta.append({"prompt_id": pid, "category": pr["category"],
                     "n_pre": n_pre, "n_dec": n_dec,
                     "off": sum(m["n_pre"] + m["n_dec"] for m in meta)})
        # route cross-check on decode tokens (shape [8,1] rows, blk.20)
        hook20 = []
        for line in open(os.path.join(kd, f"prompt_{pid:02d}.routes.jsonl")):
            row = json.loads(line)
            if row["shape"][0] != 8 or row["shape"][1] != 1:
                continue
            lay = int(row["weight"].split("blk.", 1)[1].split(".", 1)[0])
            if lay == L:
                hook20.append([int(x) for x in row["ids"]])
        assert len(hook20) == n_dec, f"p{pid}: {len(hook20)} vs {n_dec}"
        Rt.append(np.array(hook20, dtype=np.int32))
        n_decode_tot += n_dec
    X = np.concatenate(Xs, axis=0)
    N = X.shape[0]
    print(f"N={N} contextual rows ({n_decode_tot} decode)", flush=True)

    W = np.fromfile(f"{RAW}/router_L{L:02d}.f32",
                    dtype=np.float32).reshape(256, 2048)
    logits = X.astype(np.float64) @ W.T.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    top8 = np.argsort(P, axis=1)[:, ::-1][:, :8]
    pw = np.take_along_axis(P, top8, axis=1)
    pw = pw / pw.sum(axis=1, keepdims=True)
    del P, logits
    # cross-check: decode rows only
    ok, tot = 0, 0
    for m, h in zip(meta, Rt):
        sl = slice(m["off"] + m["n_pre"], m["off"] + m["n_pre"] + m["n_dec"])
        ok += (top8[sl] == h).all(axis=1).sum()
        tot += m["n_dec"]
    print(f"route cross-check: {ok}/{tot} decode tokens exact match",
          flush=True)
    assert ok / tot >= 0.99, "routing parity failed"

    # prompt split 22/10 (seed shuffle), stratified-ish by interleaving cats
    rng = np.random.default_rng(0)
    order = rng.permutation(len(meta))
    tr_p, te_p = sorted(order[:22].tolist()), sorted(order[22:].tolist())
    tr_idx = np.concatenate([np.arange(meta[p]["off"], meta[p]["off"]
                                       + meta[p]["n_pre"] + meta[p]["n_dec"])
                             for p in tr_p])
    te_idx = np.concatenate([np.arange(meta[p]["off"], meta[p]["off"]
                                       + meta[p]["n_pre"] + meta[p]["n_dec"])
                             for p in te_p])
    print(f"train prompts={tr_p} ({len(tr_idx)} rows)", flush=True)
    print(f"held prompts={te_p} ({len(te_idx)} rows)", flush=True)
    assert len(tr_idx) >= 400 and len(te_idx) >= 200
    # near-duplicate audit: max held-vs-train cosine
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    mx = 0.0
    for a in range(0, len(te_idx), 256):
        C = Xn[te_idx[a:a + 256]].astype(np.float64) @ Xn[tr_idx].astype(np.float64).T
        mx = max(mx, float(C.max()))
    print(f"max held-vs-train cosine: {mx:.4f}", flush=True)

    # exact R8 for all rows (stream union)
    w8 = pw.astype(np.float64)
    use = defaultdict(list)
    for t in range(N):
        for s in range(8):
            use[int(top8[t, s])].append((t, s))
    R8 = np.zeros((N, 2048), np.float64)
    eids = sorted(use)
    print(f"streaming {len(eids)} experts for contextual R8...", flush=True)
    Xf = X.astype(np.float32)
    for i, e in enumerate(eids):
        G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32",
                        dtype=np.float32).reshape(512, 2048)
        U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32",
                        dtype=np.float32).reshape(512, 2048)
        D = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32",
                        dtype=np.float32).reshape(2048, 512)
        gv = Xf @ G.T
        uv = Xf @ U.T
        Y = ((((gv / (1.0 + np.exp(-gv))) * uv) @ D.T)).astype(np.float64)
        for (t, s) in use[e]:
            R8[t] += w8[t, s] * Y[t]
        del G, U, D, gv, uv, Y
        if (i + 1) % 80 == 0:
            print(f"  {i+1}/{len(eids)}", flush=True)
    np.savez_compressed(os.path.join(OUT, "context_inputs.npz"),
                        Xtr=X[tr_idx], Xte=X[te_idx],
                        top8tr=top8[tr_idx], w8tr=pw[tr_idx].astype(np.float32),
                        top8te=top8[te_idx], w8te=pw[te_idx].astype(np.float32))
    np.savez_compressed(os.path.join(OUT, "context_teacher.npz"),
                        R8tr=R8[tr_idx].astype(np.float32),
                        R8te=R8[te_idx].astype(np.float32))
    json.dump({"kernel": "l20hdump-v1", "layer": L, "N": N,
               "n_decode": n_decode_tot, "route_match": f"{ok}/{tot}",
               "train_prompts": tr_p, "held_prompts": te_p,
               "n_train": len(tr_idx), "n_held": len(te_idx),
               "max_held_train_cosine": mx,
               "union": len(eids)},
              open(os.path.join(OUT, "context_meta.json"), "w"), indent=1)
    print("OK fb03", flush=True)


if __name__ == "__main__":
    sys.exit(main())
