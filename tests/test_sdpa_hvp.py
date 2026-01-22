import pytest


def test_sdpa_hvp_placeholder():
    """Phase 2: Validate that SDPA supports correct Hessian–vector products (HVP)
    at the operator contract level, independent of backend implementation."""
    # TODO: Numeric test plan:
    # 1. build small, deterministic q/k/v tensors so sdpa output is reproducible.
    # 2. run the forward pass and compute loss from a scalar reduction of sdpa(q, k, v).
    # 3. call .backward() to collect gradients needed for the HVP.
    # 4. use torch.autograd.functional.hvp (or manual double backward) to evaluate
    #    the Hessian-vector product along a direction coming from another random tensor.
    # 5. compare that result against a finite-difference approximation that perturbs
    #    the input direction in small steps to confirm the HVP computation matches
    #    the numerical derivative within tolerance.
    # 6. Keep the test failing until a working implementation and thresholds are in place.
    pytest.fail("TODO: implement SDPA HVP tests")
