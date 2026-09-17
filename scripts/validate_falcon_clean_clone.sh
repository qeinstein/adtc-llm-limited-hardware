#!/usr/bin/env bash
# Validate the public, clean-clone handoff after hosting the final artifact.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/qeinstein/adtc-llm-limited-hardware.git}"
BRANCH="${BRANCH:-research/edge35-adaptive-streaming}"
MODEL_URL="${MODEL_URL:?set MODEL_URL to the hosted Falcon GGUF URL}"
MODEL_SHA256="${MODEL_SHA256:?set MODEL_SHA256 to the recorded final GGUF hash}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORK_DIR/repo"
cd "$WORK_DIR/repo"
MODEL_URL="$MODEL_URL" MODEL_SHA256="$MODEL_SHA256" bash download_model.sh
python3 -c 'from src.config import load_metadata; from src.manifest import validate_metadata; errors=validate_metadata(load_metadata()); assert not errors, errors'
MODEL="model/Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf"
bash scripts/profile_falcon_submission.sh "$MODEL" "$WORK_DIR/profiler"
python3 -m src.main --demo
echo "clean-clone Falcon download, SHA256 verification, profiler, and inference passed"
