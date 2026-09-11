"""Deterministic generation battery for GGUF eval (pre/post-train comparisons).

Runs every prompt in a battery JSON through llama-cpp-python (temperature 0,
fixed seed) and saves raw responses + an index. Used by the Falcon probe
kernel for held-out safety/clinical sets; also runs anywhere a GGUF exists.

    python scripts/probe_generations.py --model model.gguf \
        --battery docs/research/falcon_probe_heldout.json --out-dir /tmp/gen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Deterministic GGUF generation battery")
    p.add_argument("--model", required=True)
    p.add_argument("--battery", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-ctx", type=int, default=2048)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from llama_cpp import Llama

    battery = json.loads(Path(args.battery).read_text())["prompts"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, verbose=False)
    index = []
    for pr in battery:
        pid = pr["id"]
        r = llm.create_completion(
            prompt=pr["text"], max_tokens=int(pr["max_tokens"]),
            temperature=args.temp, seed=args.seed,
        )
        txt = r["choices"][0]["text"]
        (out_dir / f"{pid}.txt").write_text(txt)
        print(f"{pid}: chars={len(txt)} lines={txt.count(chr(10))}", flush=True)
        assert txt.count("\n") < 5000, f"runaway output in {pid}"
        index.append({"id": pid, "section": pr.get("section", ""),
                      "check": pr.get("check", ""), "chars": len(txt)})
    (out_dir / "_index.json").write_text(json.dumps(index, indent=1))
    print(f"battery complete: {len(index)}/{len(battery)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
