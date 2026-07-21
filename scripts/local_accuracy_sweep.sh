#!/usr/bin/env bash
# Accuracy sweep: run lm-eval MCQ (arc_easy + medical) on each GGUF exactly as the
# ADTC profiler does — via a llama.cpp server + lm-eval's `gguf` backend
# (POST {base_url}/v1/completions with echo+logprobs). Accuracy is
# hardware-independent, so these acc_norm/acc numbers transfer to the audit.
#
# Usage:
#   LM_PY=$HOME/adtc_models/venv311/bin/python \
#   BIN=$HOME/adtc_models/llama.cpp/build/bin GGUF_DIR=$HOME/adtc_models/gguf \
#   bash scripts/local_accuracy_sweep.sh --tasks arc_easy --limit 100
set -uo pipefail

BIN="${BIN:-$HOME/adtc_models/llama.cpp/build/bin}"
GDIR="${GGUF_DIR:-$HOME/adtc_models/gguf}"
LM_PY="${LM_PY:-$HOME/adtc_models/venv311/bin/python}"
PORT="${PORT:-8177}"
TASKS="arc_easy"
LIMIT="100"
while [ $# -gt 0 ]; do case "$1" in
  --tasks) TASKS="$2"; shift 2;; --limit) LIMIT="$2"; shift 2;;
  --port) PORT="$2"; shift 2;; *) shift;; esac; done

SERVER="$BIN/llama-server"
[ -x "$SERVER" ] || { echo "llama-server missing at $SERVER"; exit 1; }
"$LM_PY" -c "import lm_eval" 2>/dev/null || { echo "lm-eval not importable via $LM_PY"; exit 1; }

RESULTS="$HOME/adtc_models/acc_results"; mkdir -p "$RESULTS"
echo "tasks=$TASKS limit=$LIMIT"
printf "%-34s | %s\n" "model" "scores"
printf '%.0s-' {1..72}; echo

for f in "$GDIR"/*.gguf; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .gguf)
    # start CPU server (ctx big enough for MCQ prompts + echo)
    "$SERVER" -m "$f" -ngl 0 -c 4096 --host 127.0.0.1 --port "$PORT" >/tmp/srv_$name.log 2>&1 &
    SRV=$!
    # wait for health (up to 120s)
    ok=0
    for _ in $(seq 1 120); do
        if curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then ok=1; break; fi
        sleep 1
    done
    if [ "$ok" != 1 ]; then echo "$name | SERVER FAILED (see /tmp/srv_$name.log)"; kill $SRV 2>/dev/null; continue; fi
    # preflight: does /v1/completions return token_logprobs with echo?
    pf=$(curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
         -d '{"prompt":"The capital of France is","logprobs":5,"max_tokens":1,"echo":true,"temperature":0}' 2>/dev/null)
    if ! printf '%s' "$pf" | grep -q "logprobs"; then
        echo "$name | PREFLIGHT: server /v1/completions returned no logprobs — lm-eval gguf backend won't work"
        kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; continue
    fi
    out="$RESULTS/$name"
    "$LM_PY" -m lm_eval --model gguf \
        --model_args "base_url=http://127.0.0.1:$PORT" \
        --tasks "$TASKS" --limit "$LIMIT" --output_path "$out" >/tmp/lmeval_$name.log 2>&1
    # parse
    scores=$("$LM_PY" - "$out" "$TASKS" <<'PY'
import sys, json, glob, os
out, tasks = sys.argv[1], sys.argv[2].split(",")
files = glob.glob(os.path.join(out, "**", "results*.json"), recursive=True) + glob.glob(out + "*results*.json")
if not files:
    print("no results"); sys.exit()
data = json.load(open(sorted(files)[-1])); res = data.get("results", {})
parts = []
for t in tasks:
    r = res.get(t, {})
    v = r.get("acc_norm,none", r.get("acc,none"))
    m = "acc_norm" if "acc_norm,none" in r else "acc"
    parts.append(f"{t}={v*100:.1f}({m})" if isinstance(v, (int, float)) else f"{t}=NA")
print("  ".join(parts))
PY
)
    printf "%-34s | %s\n" "$name" "$scores"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
done
echo ""; echo "Raw results under $RESULTS ; server/lm-eval logs in /tmp/srv_*.log, /tmp/lmeval_*.log"
