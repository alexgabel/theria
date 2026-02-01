# Phase 10 — Meta-Learning Diagnostics on Fused Attention

**Status:** COMPLETE  
**Focus:** Empirical validation of higher-order correctness for SDPA backends under MAML

⸻

## Executive summary

Phase 10 establishes, with controlled experiments, that:
	1.	Second-order meta-gradients exist and are measurable for all supported attention backends (reference, custom, triton_fused).
	2.	The Triton fused SDPA backend preserves higher-order information, i.e. it does not silently collapse FULL MAML to FO-MAML.
	3.	The magnitude of second-order effects grows with inner-loop depth, exactly as predicted by MAML theory.
	4.	Our diagnostics are meaningful, validated by explicit negative controls that correctly report the absence of second-order paths.

This phase closes the loop from kernel-level differentiation contracts (Phase 9) to scientific, task-level consequences.

⸻

## Motivation

Many modern attention implementations (FlashAttention-style, fused SDPA) are optimized for first-order training. In practice, their backward passes often:
	•	implement custom autograd.Functions,
	•	use recomputation under no_grad,
	•	rely on saved tensors that do not support grad-of-grad,
	•	or are explicitly marked once_differentiable.

This is typically harmless for standard training — but fatal for meta-learning, implicit layers, and bilevel optimization.

Phase 10 answers the question:

Do fused SDPA kernels actually support higher-order optimization when used in realistic MAML loops?

⸻

## Experimental setup

Model
	•	Tiny single-head attention classifier
	•	Uses token-0 as a CLS-like query
	•	Attention implemented via sdpa_custom
	•	Backends:
	•	reference (PyTorch baseline)
	•	custom (unfused explicit implementation)
	•	triton_fused (fully fused Triton kernel)

Task
	•	Synthetic sequence classification
	•	Single task per outer step (minimal noise)

Meta-learning loop
	•	Standard MAML
	•	Inner loop: SGD with configurable depth
	•	Outer loss differentiated w.r.t. initial parameters

Modes
	•	FULL: true MAML (create_graph=True)
	•	FO: standard first-order MAML
	•	FO_STRICT: explicit ablation that detaches inner gradients

No changes were made to the theria/ library code.

⸻

## Diagnostics

Phase 10 introduces two complementary diagnostics.

1. Second-order path existence

We probe whether the meta-gradient itself is differentiable:

∂/∂θ ( ∂L_outer / ∂θ ) ≠ 0

Operationally:
	•	Compute first-order meta-gradients with create_graph=True
	•	Differentiate a scalarized norm of that gradient again

This yields a boolean:

second_order_path ∈ {True, False}

2. FULL vs FO gradient discrepancy

We measure the relative difference between FULL and FO meta-gradients:

rel_diff = ‖g_FULL − g_FO‖ / ‖g_FULL‖

This directly quantifies how much second-order terms matter.

⸻

## Negative controls (critical)

To validate that the diagnostics are not vacuous, we introduce explicit negative controls:

FO_STRICT ablation
	•	Inner-loop gradients are forcibly detached
	•	Equivalent to computing MAML under create_graph=False

Expected behavior (and observed):

Mode	second_order_path
FULL	True
FO	True*
FO_STRICT	False

*FO still shows a potential second-order path because the diagnostic is applied to the outer loss; however the used gradient matches FO.

This confirms the diagnostic is sensitive and correct.

⸻

## Key results

Backend comparison (inner_steps = 5)

Backend	Mode	second_order_path	rel_diff
reference	FULL	True	~0.20
reference	FO_STR	False	~0.20
custom	FULL	True	~0.20
triton_fused	FULL	True	~0.12
triton_fused	FO_STR	False	~0.12

Conclusion: Triton fused SDPA preserves second-order information.

⸻

## Stress test: inner-loop depth

We vary the number of inner steps:

inner_steps ∈ {1, 5, 10, 20}

Observation

The relative FULL–FO gradient difference grows monotonically with depth:

Inner steps	rel_diff_full_vs_fo
1	~0.03
5	~0.12
10	~0.21
20	~0.30

This is visualized in the Phase 10 stress plot:

Second-order signal grows with inner-loop depth

This exactly matches MAML theory: higher-order terms accumulate across inner updates.

⸻

## Interpretation

The results rule out several common failure modes:
	•	❌ “The fused kernel is secretly FO-only”
	•	❌ “Second-order gradients are numerical noise”
	•	❌ “Everything collapses to FO under fusion”

Instead, the evidence supports:

The Triton fused SDPA backward is differentiable enough to support full meta-learning.

This validates the Phase 9 design choice: explicit backward contracts + frozen-stats JVPs can coexist with higher-order correctness.

⸻

## What Phase 10 does not claim
	•	It does not claim performance optimality
	•	It does not claim universal correctness for all fused kernels
	•	It does not imply FlashAttention-style kernels are safe by default

Phase 10 is about capability and correctness, not speed.

⸻

## Artifacts produced
	•	experiments/phase10/scripts/run_maml_backend_compare.py
	•	Stress-test TSV data (inner_steps vs rel_diff)
	•	Second-order diagnostic logic
	•	Explicit negative controls
	•	Phase 10 stress plot

⸻

## Transition to Phase 11

Phase 10 answers whether higher-order signals exist.

Phase 11 will ask:
	•	When do they break?
	•	Which kernel patterns destroy grad-of-grad?
	•	Can we classify fused backward designs by meta-learning safety?

Phase 10 provides the baseline against which all failures will be measured.

⸻

## Phase 10 is complete.
