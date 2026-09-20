#!/usr/bin/env python3
"""Gate-2 before/after capture: raw base model vs full Jamii system.

Runs identical prompts through (a) the UNMODIFIED base model (vanilla
llama-server, no system prompt, no guidance, no safety) and (b) the full
submitted system (FastAPI /api/chat: system prompt + guidance + safety +
lint), saving paired JSON for REPORT.md.

Needs weights + builds; runs on the release laptop (NOT in CI):

  python3 scripts/capture_before_after.py \\
      --prompts evals/gate2_before_after/prompts.json \\
      --base-model /path/to/base-IQ2_XXS.gguf \\
      --out evals/gate2_before_after/

The submitted-system leg expects `make webui` (or uvicorn) already running.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def post_json(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_ready(base, wait_s=600):
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            urllib.request.urlopen(base + "/health", timeout=5)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"{base} never ready")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--server-bin", default=None)
    ap.add_argument("--webui", default="http://127.0.0.1:8420")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    server_bin = args.server_bin or str(
        ROOT / "runtime/llama.cpp/build-native/bin/llama-server")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[capture] starting VANILLA base server (no patches, no prompt)...")
    base = subprocess.Popen(
        [server_bin, "-m", args.base_model, "--port", "8422", "-t", "4",
         "-c", "2048", "-ngl", "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        wait_ready("http://127.0.0.1:8422")
        for item in prompts:
            pid, prompt = item["prompt_id"], item["prompt"]
            print(f"[capture] {pid} base...", flush=True)
            b = post_json("http://127.0.0.1:8422/v1/chat/completions",
                          {"messages": [{"role": "user", "content": prompt}],
                           "max_tokens": args.max_tokens, "temperature": 0.3,
                           "stream": False})
            bmsg = b["choices"][0]["message"]
            print(f"[capture] {pid} system...", flush=True)
            s = post_json(args.webui + "/api/chat",
                          {"message": prompt, "history": [],
                           "mode": "medium"})
            (outdir / f"{pid}.json").write_text(json.dumps({
                "prompt_id": pid, "prompt": prompt,
                "base": {"thinking": bmsg.get("reasoning_content", "") or "",
                         "text": bmsg.get("content", "") or ""},
                "system": {"reply": s["reply"],
                           "thinking": s.get("thinking", ""),
                           "urgency": s.get("urgency"),
                           "safety_override": s.get("safety_override"),
                           "guidance_cards": s.get("guidance_cards"),
                           "sources": s.get("sources")},
            }, indent=1, ensure_ascii=False))
            print(f"[capture] wrote {pid}.json", flush=True)
    finally:
        base.terminate()
        try:
            base.wait(timeout=20)
        except Exception:
            base.kill()
    print("[capture] done. Paste the paired outputs into REPORT.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
