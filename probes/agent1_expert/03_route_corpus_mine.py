"""Agent1 probe 03: mine the REAL 2016-token route corpus for expert-selection structure.

Questions:
  Q1 (whole-expert dropping): are any experts never selected? (per layer)
  Q2 (selection concentration): per-layer popularity skew -> prior for mass concentration
  Q3 (per-layer K prior): per-layer routing entropy / effective expert count
  Q4 (retention): same-layer expert retention token->token (cache relevance, already
      partly known; recompute per layer for K-adaptive scheduling context)

Input: research/.../phase5e_route_corpus_v1/route_corpus.jsonl (REAL inference routes)
Output: concentration table per layer + global verdicts.
"""
import json
import math
import os
from collections import Counter

CORPUS = ("/home/fluxx/Workspace/adtc-llm-native-sparse/research/native_sparse_experiments"
          "/results/phase5e_route_corpus_v1/route_corpus.jsonl")

recs = [json.loads(l) for l in open(CORPUS)]
print(f"records={len(recs)}")
assert all(len(r["layers"]) == 40 for r in recs)

# Q1: coverage
per_layer_used = [set() for _ in range(40)]
global_bundles = set()
for r in recs:
    for L, ids in enumerate(r["layers"]):
        assert len(ids) == 8 and len(set(ids)) == 8
        for e in ids:
            per_layer_used[L].add(e)
            global_bundles.add((L, e))
n_distinct = [len(s) for s in per_layer_used]
print(f"unique bundles: {len(global_bundles)}/10240 = {len(global_bundles)/10240:.3f}")
print(f"per-layer distinct experts: min={min(n_distinct)} max={max(n_distinct)} mean={sum(n_distinct)/40:.1f}")
print(f"layers with <256 distinct: {sum(1 for n in n_distinct if n < 256)}; "
      f"total never-selected expert slots: {sum(256-n for n in n_distinct)}")

# Q2/Q3: per-layer popularity concentration
print(f"\n{'L':>3s} {'distinct':>8s} {'top8cov':>7s} {'top16':>6s} {'top32':>6s} {'top64':>6s} {'entropy':>7s} {'effN':>6s} {'gini':>5s}")
eff_all = []
for L in range(40):
    c = Counter()
    for r in recs:
        c.update(r["layers"][L])
    tot = sum(c.values())
    freqs = sorted((v / tot for v in c.values()), reverse=True)
    cum = []
    s = 0.0
    for f in freqs:
        s += f
        cum.append(s)
    ent = -sum(f * math.log2(f) for f in freqs)
    eff = 2 ** ent
    eff_all.append(eff)
    # gini over all 256 (zeros included)
    full = sorted([c.get(e, 0) / tot for e in range(256)])
    n = 256
    gini = (2 * sum((i + 1) * x for i, x in enumerate(full)) / (n * sum(full)) - (n + 1) / n) if sum(full) else 0
    print(f"{L:3d} {len(c):8d} {cum[7]:7.3f} {cum[15]:6.3f} {cum[31]:6.3f} {cum[63]:6.3f} {ent:7.2f} {eff:6.1f} {gini:5.3f}")

print(f"\neffective-expert-count: min={min(eff_all):.1f} max={max(eff_all):.1f} mean={sum(eff_all)/40:.1f} (of 256)")
# Interpretation anchor: uniform over 256 -> effN=256; uniform over 32 -> effN=32.

# Q4: retention per layer (fraction of current top-8 present in previous token's top-8, same layer)
print(f"\nper-layer same-expert retention (token t vs t-1, same layer):")
rets = []
for L in range(40):
    hits = tot8 = 0
    for i in range(1, len(recs)):
        # only within same prompt
        if recs[i]["prompt_id"] != recs[i-1]["prompt_id"]:
            continue
        a = set(recs[i-1]["layers"][L]); b = set(recs[i]["layers"][L])
        hits += len(a & b); tot8 += 8
    r = hits / tot8
    rets.append(r)
    print(f"  L{L:02d}: {r:.3f}")
print(f"mean retention: {sum(rets)/40:.3f}")
