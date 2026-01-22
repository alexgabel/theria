import torch
from dataclasses import dataclass

from theria.maml.loops import meta_loss_on_tasks
from theria.models.tiny_attention import TinyAttentionModel, TinyAttentionConfig


@dataclass
class ToyTask:
    x_s: torch.Tensor
    y_s: torch.Tensor
    x_q: torch.Tensor
    y_q: torch.Tensor


def make_attention_toy_tasks(
    n_tasks=2,
    B_s=4,
    B_q=4,
    T=4,
    d_model=8,
    num_classes=3,
    device="cpu",
    dtype=torch.float64,
):
    torch.manual_seed(0)

    tasks = []
    for _ in range(n_tasks):
        x_s = torch.randn(B_s, T, d_model, device=device, dtype=dtype)
        x_q = torch.randn(B_q, T, d_model, device=device, dtype=dtype)

        # Simple but nontrivial labels
        y_s = (x_s.mean(dim=(1, 2)) > 0).long() % num_classes
        y_q = (x_q.mean(dim=(1, 2)) > 0).long() % num_classes

        tasks.append(ToyTask(x_s, y_s, x_q, y_q))

    return tasks


def test_full_maml_vs_fo_maml_meta_gradient_differs():
    """
    Phase-2 lock:
    Full MAML must produce a different meta-gradient than FO-MAML.
    """

    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    cfg = TinyAttentionConfig(
        d_model=8,
        num_classes=3,
    )

    model = TinyAttentionModel(cfg).to(device)
    model_fo = TinyAttentionModel(cfg).to(device)
    model_fo.load_state_dict(model.state_dict())

    tasks = make_attention_toy_tasks(device=device)

    # Full MAML
    for p in model.parameters():
        p.grad = None
    meta_loss = meta_loss_on_tasks(model, tasks, inner_lr=0.5, inner_steps=1, fo=False)
    meta_loss.backward()
    full_grads = [p.grad.clone() for p in model.parameters()]

    # FO-MAML
    for p in model_fo.parameters():
        p.grad = None
    meta_loss_fo = meta_loss_on_tasks(model_fo, tasks, inner_lr=0.5, inner_steps=1, fo=True)
    meta_loss_fo.backward()
    fo_grads = [p.grad.clone() for p in model_fo.parameters()]

    # Invariant: at least one parameter differs
    assert any(
        not torch.allclose(g1, g2)
        for g1, g2 in zip(full_grads, fo_grads)
    ), "Full MAML collapsed to FO-MAML — second-order gradient missing"