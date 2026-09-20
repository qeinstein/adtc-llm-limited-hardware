#!/usr/bin/env bash
# Compatibility wrapper for Linux/macOS. The implementation lives in the
# Python entrypoint so native Windows can use the exact same build recipe.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$HERE/build_runtime.py" "$@"
