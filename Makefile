# Jamii Afya — ADTC 2026 submission. Common workflows.
# Override the interpreter: make PYTHON=./venv/bin/python test
PYTHON ?= python
export PYTHONPATH := .

.PHONY: help setup setup-dev test lint validate data model run demo bench scalar bench-audit accuracy profiler clean

help:
	@echo "Targets:"
	@echo "  setup       create venv + install runtime deps"
	@echo "  setup-dev   install dev/test/train/eval deps"
	@echo "  test        run the offline test suite (no weights needed)"
	@echo "  lint        ruff check"
	@echo "  validate    validate metadata.json against the profiler schema"
	@echo "  data        build fine-tune splits + imatrix calibration corpus"
	@echo "  model       download the GGUF weights"
	@echo "  run|demo    launch the advisor (interactive | metadata test prompts)"
	@echo "  bench       benchmark the interactive engine"
	@echo "  scalar      build a no-SIMD llama.cpp (audit parity)"
	@echo "  bench-audit run llama-bench exactly like the profiler (needs scalar build)"
	@echo "  accuracy    predict S_acc via lm-eval (needs weights + lm-eval)"
	@echo "  profiler    run the official adtc-profiler (Gate-1 self-check)"

setup:
	$(PYTHON) -m venv venv
	./venv/bin/python -m pip install --upgrade pip
	./venv/bin/python -m pip install -r requirements.txt

setup-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	ruff check src tests scripts

validate:
	$(PYTHON) -c "from src.config import load_metadata; from src.manifest import validate_metadata; e=validate_metadata(load_metadata()); print('metadata.json:', 'VALID' if not e else e)"

data:
	$(PYTHON) scripts/prepare_dataset.py

model:
	bash download_model.sh

run:
	$(PYTHON) -m src.main

demo:
	$(PYTHON) -m src.main --demo

bench:
	$(PYTHON) -m src.benchmark

scalar:
	bash scripts/build_llamacpp_scalar.sh

bench-audit:
	$(PYTHON) -m src.benchmark --profiler-parity --llama-bench llama.cpp/build-scalar/bin/llama-bench

accuracy:
	$(PYTHON) -m src.accuracy --tasks arc_easy

profiler:
	bash scripts/run_profiler.sh

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache .ruff_cache
	rm -f submission.json audit.json verdict.json submission_bench.json
