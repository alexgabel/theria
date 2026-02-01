"""
Phase 10 experiment: MAML backend comparison.

Compares:
- Attention backend: reference vs custom vs triton_fused
- Meta-learning mode: full MAML vs FO-MAML

Logs:
- outer loss
- gradient norm

Design rules:
- No changes to theria/ library code
- No kernel tuning
- Minimal logging (stdout / CSV-ready)
"""

import argparse
import math
import torch
import torch.nn as nn

from theria.attention.custom import sdpa_custom
from theria.models.tiny_attention import TinyAttentionConfig
from theria.tasks.synthetic_seqcls import task_sampler
from theria.maml.loops import meta_loss_on_tasks


# -----------------------------
# Utilities
# -----------------------------

def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


class Phase10TinyAttentionModel(nn.Module):
    """
    Minimal single-head attention classifier with a swappable backend.

    Uses token 0 as a CLS-like query. Attention is delegated to sdpa_custom
    with H=1 and Tq=1.
    """
    def __init__(self, cfg: TinyAttentionConfig | None = None, backend: str = "reference"):
        super().__init__()
        self.cfg = cfg or TinyAttentionConfig()
        self.backend = backend

        D = self.cfg.d_model
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.classifier = nn.Linear(D, self.cfg.num_classes)

    def set_attention_backend(self, backend: str) -> None:
        self.backend = backend

    def _resolve_backend(self) -> str:
        if self.backend == "triton_fused":
            return "triton_full_fused"
        return self.backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        q = self.q_proj(x[:, :1, :])  # (B, 1, D)
        k = self.k_proj(x)            # (B, T, D)
        v = self.v_proj(x)            # (B, T, D)

        # Add head dim for sdpa_custom: (B, H=1, Tq/Tk, D)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        h = sdpa_custom(q, k, v, backend=self._resolve_backend())  # (B, 1, 1, D)
        h = h.squeeze(1).squeeze(1)  # (B, D)
        return self.classifier(h)


def set_attention_backend(model: Phase10TinyAttentionModel, backend: str):
    """Switch attention backend in the model."""
    if backend not in {"reference", "custom", "triton_fused"}:
        raise ValueError(f"Unknown backend: {backend}")
    model.set_attention_backend(backend)


# -----------------------------
# Main experiment
# -----------------------------

def run_experiment(
    backend: str,
    fo: bool,
    device: torch.device,
    seed: int,
    steps: int,
    inner_steps: int,
    inner_lr: float | None,
):
    torch.manual_seed(seed)

    model = Phase10TinyAttentionModel().to(device)
    set_attention_backend(model, backend)

    for step in range(steps):
        model.zero_grad(set_to_none=True)

        task = task_sampler(device=device)

        outer_loss = meta_loss_on_tasks(
            model=model,
            tasks=[task],
            fo=fo,
            inner_steps=inner_steps,
            inner_lr=inner_lr if inner_lr is not None else None,
        )

        outer_loss.backward()
        gnorm = grad_norm(model.parameters())

        print(
            f"step={step:03d} "
            f"backend={backend:<12} "
            f"mode={'FO' if fo else 'FULL':<4} "
            f"loss={outer_loss.item():.6f} "
            f"grad_norm={gnorm:.6f}"
        )


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True,
                        choices=["reference", "custom", "triton_fused"])
    parser.add_argument("--fo", action="store_true",
                        help="Use first-order MAML (FO-MAML)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=None,
        help="Inner-loop learning rate (overrides default)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device)
    run_experiment(
        backend=args.backend,
        fo=args.fo,
        device=device,
        seed=args.seed,
        steps=args.steps,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
    )


if __name__ == "__main__":
    main()
