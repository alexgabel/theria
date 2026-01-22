# tests/test_maml_smoke.py
import torch

from theria.models.tiny_attention import TinyAttentionConfig, TinyAttentionModel
from theria.tasks.synthetic_seqcls import task_sampler
from theria.maml.loops import meta_loss_on_tasks, named_params


def test_maml_meta_grad_exists_k1_cpu():
    torch.manual_seed(0)

    # Keep dims small so it's fast on CPU and stable.
    cfg = TinyAttentionConfig(d_model=16, num_classes=3)
    model = TinyAttentionModel(cfg)

    tasks = [
        task_sampler(
            B_s=8, B_q=8,
            T=12, D=cfg.d_model,
            num_classes=cfg.num_classes,
            num_signal_positions=3,
            noise_std=1.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for _ in range(4)
    ]

    meta_loss = meta_loss_on_tasks(model, tasks, inner_lr=0.5, inner_steps=1)

    params = named_params(model)
    grads = torch.autograd.grad(meta_loss, tuple(params.values()), create_graph=False, allow_unused=False)

    # 1) All grads exist and are finite
    for g in grads:
        assert g is not None
        assert torch.isfinite(g).all()

    # 2) Not all grads are zero (meta-gradient is nontrivial)
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    assert total_norm.item() > 0.0