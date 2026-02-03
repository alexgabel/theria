# theria

**theria** is a research library for **higher-order learning with modern neural operators**, with a focus on **scaled dot-product attention (SDPA)** and its interaction with **meta-learning (MAML)**, implicit differentiation, and higher-order optimization.

The core goal is **correctness first**:
> establish mathematically sound differentiation contracts *before* kernel fusion and performance optimization.

Phase 10 marks the transition from kernel construction to **scientific experiments**: using the validated operators to study where and why meta-learning succeeds or fails.

---

## Why theria exists

Many highly optimized attention implementations (e.g. FlashAttention / fused SDPA) break or silently approximate higher-order derivatives.

This matters for:
- MAML / bilevel optimization
- implicit layers
- second-order methods
- operator-learning research

theria explores how to:
- define **explicit operator-level differentiation contracts**
- validate **JVP / VJP / HVP** rigorously
- integrate **modern kernels (Triton)** *without losing higher-order correctness*
- characterize the **autograd boundary** in fused operators

---

## Project status (TL;DR)

- **Phase 9: COMPLETE**
  - Explicit Triton backward (dQ, dK, dV)
  - Frozen-stats JVP
  - HVP sanity via finite differences
  - GPU correctness locked by tests
- **Phase 10: IN PROGRESS**
  - Meta-learning (MAML) experiments
  - Empirical study of higher-order failure modes
  - Comparison: full MAML vs FO-MAML under different attention backends

### FO-MAML semantics (theria definition)

The CLI flag `--fo` implements the classic first-order MAML approximation:
- Inner-loop gradients are **not** part of the graph (`create_graph=False` in the inner updates).
- Outer backprop treats the adapted parameters as constants; no second-order terms flow.
- Full MAML (`--fo` absent) keeps `create_graph=True` in the inner loop, enabling higher-order meta-gradients.

See `theria/maml/loops.py::inner_adapt` and `docs/theory/maml_derivation.md` for the exact code/path.

👉 **Canonical status & exit criteria:** `docs/STATUS.md`

---

## Repository layout (important parts only)

```text
theria/                     # Stable library code (do not experiment here)
  attention/                # SDPA operators, Triton kernels, JVP/HVP logic
  maml/                     # Backend-agnostic MAML inner/outer loops
  autograd/                 # Custom autograd.Function wiring
  models/                   # Tiny attention models for tests/experiments
  tasks/                    # Synthetic tasks (e.g. seq classification)

experiments/phase10/        # ALL Phase 10 work lives here
  configs/                  # Experiment configs
  scripts/                  # Runnable experiment entry points
  notebooks/                # Scratch / analysis (optional)
  runs/                     # Outputs (gitignored)

scripts/
  bench_sdpa.py              # Performance sanity checks
  smoke_sdpa.py              # Minimal forward/backward smoke test

docs/
  STATUS.md                  # Phase tracker (single source of truth)
  phase9_*.md                # Backward / JVP / HVP contracts
  design/                    # Operator contracts
  theory/                    # Autograd + MAML derivations

tests/
  test_maml_*.py             # Phase 10 MAML correctness tests
  test_triton_*              # Kernel & higher-order validation
```
---

## Design philosophy

- **Contracts before kernels**  
- **Correctness before performance**  
- **Operator-level semantics over implementation details**  
- CPU-first development; GPU work only once semantics are fixed  

This repository is intended to be readable, reviewable, and useful to researchers working on meta-learning, implicit differentiation, and operator-based models.

---

## Getting started

```bash
conda env create -f environment.yml
conda activate theria
pip install -e .
pytest -q
```

A manual smoke test for SDPA forward/backward is available under `scripts/`.

## Contributing / workflow

Golden sanity commands (CPU + GPU + perf smoke):

```bash
pytest -q
pytest -m gpu -q
CUDA_VISIBLE_DEVICES=0 python scripts/bench_sdpa.py --preset medium --dtype float16
```

See `docs/STATUS.md` for the active phase, exit criteria, and required constraints.

---

## Disclaimer

theria is a research codebase under active development. APIs and backends may change as phases progress; consult `docs/STATUS.md` before extending or refactoring. 
