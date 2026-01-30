# theria

theria is a research toolkit for **higher-order learning with modern neural operators**. It focuses on making differentiable operators—starting with scaled dot-product attention—*correctly* compatible with meta-learning, implicit differentiation, and higher-order optimization, while progressively reintroducing kernel-level performance.

The project is explicitly **correctness-first**: mathematical contracts and higher-order differentiation semantics are established before performance work, and preserved as kernels evolve.

---

## Motivation

Many modern learning setups (e.g. MAML, implicit layers, bilevel optimization) rely on higher-order derivatives such as Hessian–vector products (HVPs) or Jacobian–vector products (JVPs). However, common fused or highly optimized operators (e.g. FlashAttention) often do **not** support higher-order differentiation correctly or at all.

theria explores how to:
- Define clear operator-level differentiation contracts
- Validate first- and higher-order gradients rigorously
- Enable meta-learning with modern neural operators
- Later, reintroduce performance via Triton/CUDA without breaking correctness

---

## Project status

Current work is tracked in `docs/STATUS.md`. That file is the single source of truth for:
- active phase and exit criteria
- kernel contracts and limitations
- golden CI/benchmark commands
- known limitations and boundary tests

If you are new to the repo, start with:
- `docs/STATUS.md`
- `docs/theory/attention_autograd.md`
- `docs/design/operator_contracts.md`

---

## Phase 2 Results (Closed)

- Full MAML with attention on CPU  
- Explicit HVP validated by finite differences  
- FO-MAML vs full MAML distinction locked by tests  
- Documented SDPA higher-order autograd failure  

---

## Current scope (high level)

Implemented:
- Reference SDPA and operator-level contracts
- Explicit HVP/JVP for attention (analytic, validated)
- Boundary tests that lock known failures
- Triton forward paths (QK scaffold, partial fusion, and fully fused forward)
- Benchmarks and phase-specific tests

Known limitations:
- Fully fused backward is not yet implemented
- JVP/HVP in Triton kernels is not yet implemented

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
