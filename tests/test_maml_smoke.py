# tests/test_maml_smoke.py
import torch

from theria.models.tiny_attention import TinyAttentionConfig, TinyAttentionModel
from theria.tasks.synthetic_seqcls import task_sampler
from theria.maml.loops import meta_loss_on_tasks


def test_maml_meta_grad_exists_k1_cpu():
    """
    Phase-2 Smoke Test for MAML meta-gradient computation.

    This test verifies that, for a small TinyAttentionModel and a few synthetic tasks:
    - Meta-gradients exist for all parameters.
    - All meta-gradients are finite.
    - Meta-gradients are non-trivial (non-zero norm).

    The test uses double precision for numerical stability and runs on CPU with small dimensions
    to keep runtime fast and stable.
    """
    torch.manual_seed(0)

    # Configuration with small model size and double precision.
    cfg = TinyAttentionConfig(d_model=16, num_classes=3)
    model = TinyAttentionModel(cfg)

    # Sample a small number of tasks with double precision on CPU.
    tasks = [
        task_sampler(
            B_s=8,
            B_q=8,
            T=12,
            D=cfg.d_model,
            num_classes=cfg.num_classes,
            num_signal_positions=3,
            noise_std=1.0,
            device=torch.device("cpu"),
            dtype=torch.double,
        )
        for _ in range(2)  # Reduced number of tasks for speed
    ]

    # Compute meta-loss over tasks with 1 inner step and inner_lr=0.5.
    meta_loss = meta_loss_on_tasks(model, tasks, inner_lr=0.5, inner_steps=1)

    # Retrieve named parameters for gradient computation.
    params = dict(model.named_parameters())

    # Compute meta-gradients of meta_loss w.r.t. model parameters.
    grads = torch.autograd.grad(
        meta_loss, tuple(params.values()), create_graph=False, allow_unused=False
    )

    # Validate that all gradients exist and are finite.
    for g in grads:
        assert g is not None, "Gradient is None for some parameter."
        assert torch.isfinite(g).all(), "Gradient contains non-finite values."

    # Compute total gradient norm in a robust and readable way.
    total_norm = torch.norm(torch.stack([g.detach().norm() for g in grads]))

    # Validate that the total gradient norm is positive (non-trivial meta-gradient).
    assert total_norm.item() > 0.0, "Meta-gradient norm is zero, expected non-trivial gradients."