SHELL := /bin/bash

.PHONY: help phase12-stable-full phase12-stable-hybrid phase12-stable-regress phase12-stable-demo phase12-doc-check

PHASE12_BASELINE_ENV := experiments/phase12/phase12_stable_baseline.env
PHASE12_DEVICE ?= cuda
PYTHONPATH ?= .

help:
	@echo "Stable Phase 12 aliases:"
	@echo "  make phase12-stable-full     # Accepted stable eager run"
	@echo "  make phase12-stable-hybrid   # Accepted practical eager run"
	@echo "  make phase12-stable-demo     # Small accepted eager demo flow"
	@echo "  make phase12-stable-regress  # Stable regression harness"
	@echo "  make phase12-doc-check       # Verify Phase 12 quick links and doc-safe commands"

phase12-stable-full:
	@source $(PHASE12_BASELINE_ENV) && \
	PYTHONPATH=$(PYTHONPATH) python experiments/phase12/scripts/run_phase12_behavior.py \
	  --backend triton_fused_meta_strict \
	  --mode FULL \
	  --device $(PHASE12_DEVICE)

phase12-stable-hybrid:
	@source $(PHASE12_BASELINE_ENV) && \
	PYTHONPATH=$(PYTHONPATH) python experiments/phase12/scripts/run_phase12_behavior.py \
	  --backend triton_fused_meta_strict \
	  --mode FULL_HYBRID \
	  --meta-every-n-outer "$$PHASE12_STABLE_META_EVERY_N_OUTER" \
	  --meta-last-n-inner "$$PHASE12_STABLE_META_LAST_N_INNER" \
	  --device $(PHASE12_DEVICE)

phase12-stable-regress:
	@source $(PHASE12_BASELINE_ENV) && \
	bash experiments/phase12/scripts/run_phase12_stable_frontier_regression.sh

phase12-stable-demo:
	@source $(PHASE12_BASELINE_ENV) && \
	bash experiments/phase12/scripts/run_phase12_stable_demo.sh

phase12-doc-check:
	@PYTHONPATH=$(PYTHONPATH) python scripts/check_phase12_docs.py
