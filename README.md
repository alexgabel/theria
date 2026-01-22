# theria

theria is a research toolkit for **higher-order learning with modern neural operators**. It focuses on making differentiable operators—starting with scaled dot-product attention—*correctly* compatible with meta-learning, implicit differentiation, and higher-order optimization, before addressing kernel-level performance.

The project is explicitly **correctness-first**: mathematical contracts and higher-order differentiation semantics are established on CPU before any GPU/Triton/CUDA work is introduced.

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

- **Phase 0** — Operator contract and reference semantics (CPU) ✅
- **Phase 1** — First-order autograd correctness (`gradcheck`) ✅
- **Phase 2** — HVP/JVP semantics and meta-learning support *(in progress)*
- **Phase 3** — Triton/CUDA kernels and performance work *(future)*

All current tests run on CPU. GPU support is intentionally deferred.

A more detailed and up-to-date breakdown is maintained in `docs/STATUS.md`.

---

## Current scope

Implemented:
- Public SDPA operator contract (`sdpa`)
- Reference SDPA implementation
- Custom `autograd.Function`
- Forward equivalence tests vs reference
- First-order gradient correctness tests
- Clean, reproducible CPU-only environment and packaging

In progress:
- Numerical and autograd-based HVP/JVP validation
- Meta-learning compatibility (MAML-style inner/outer loops)

Out of scope (for now):
- Triton kernels
- CUDA-specific optimizations
- Backward-backward (full gradgrad) kernels
- Performance benchmarking

---

## Design philosophy

- **Contracts before kernels**
- **Correctness before performance**
- **Operator-level semantics over implementation details**
- CPU-first development; GPU work only once semantics are fixed

This repository is intended to be readable, reviewable, and useful to researchers working on meta-learning, implicit differentiation, and operator-based models.

---

## Getting started (CPU)

```bash
conda env create -f environment.yml
conda activate theria
pip install -e .
pytest -q
```

A manual smoke test for SDPA forward/backward is available under `scripts/`.

---

## Disclaimer

theria is a research codebase under active development. APIs may change until Phase 2 is complete and merged into `main`.