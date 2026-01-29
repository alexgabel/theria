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
- **Phase 2** — HVP/JVP semantics and meta-learning support ✅  
- **Phase 3** — Attention Autograd Boundary Analysis ✅  
- **Phase 4** — Custom attention operator exposing explicit JVP/HVP (systems + math; performance later) *(in progress)*  
- **Phase 5** – Attention-Specific JVP Mathematics *(future)*
- **Phase 6** – Performance-Aware Reintroduction *(future)*  

All current tests run on CPU. GPU support is intentionally deferred.

A more detailed and up-to-date breakdown is maintained in `docs/STATUS.md`.

Phase 4 introduces a custom attention operator that satisfies the higher-order differentiation contract established in Phase 3.

---

## Phase 2 Results (Closed)

- Full MAML with attention on CPU  
- Explicit HVP validated by finite differences  
- FO-MAML vs full MAML distinction locked by tests  
- Documented SDPA higher-order autograd failure  

---

## Current scope

Implemented:  
- Public SDPA operator contract (`sdpa`)  
- Reference SDPA implementation  
- Custom `autograd.Function`  
- Forward equivalence tests vs reference  
- First-order gradient correctness tests  
- Numerical and autograd-based HVP/JVP validation  
- Meta-learning compatibility (MAML-style inner/outer loops)  
- Clean, reproducible CPU-only environment and packaging  

Known limitations:  
- SDPA math/fused paths are known to fail HVP  

Out of scope (for now):  
- Triton kernels  
- CUDA-specific optimizations  
- Backward-backward (full gradgrad) kernels  
- Performance benchmarking  

---

## Phase 3 Goal

Where exactly does optimized SDPA break higher-order differentiation?  

This phase focuses on diagnosing the autograd boundary failures of optimized attention implementations, without yet attempting fixes.  

### Phase 3 — Attention Autograd Boundary Analysis (Planned)

Objective  
Identify the precise autograd boundary where optimized attention implementations (e.g. FlashAttention / fused SDPA) break higher-order differentiation.

Key questions  
- Which derivative is missing: JVP, VJP, or saved intermediate?  
- Is the failure silent or explicit?  
- Does backward succeed while grad-of-grad fails?  
- Is the issue backend-specific or operator-intrinsic?  

Method  
- Compare explicit attention vs SDPA math vs fused SDPA  
- Inspect autograd graphs and saved tensors  
- Validate HVP existence via:  
  - double backward (when possible)  
  - finite differences  
  - reproduce failures deterministically  

Outcome  
A precise operator-level failure contract:  
“Given inputs Q, K, V, the optimized attention path does not expose X, therefore Y (HVP/JVP) cannot be computed.”  

This contract defines the requirements for Phase 4.

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

theria is a research codebase under active development. APIs may change until Phase 4 is complete and merged into `main`.
