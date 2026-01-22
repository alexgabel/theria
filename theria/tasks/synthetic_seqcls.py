# theria/tasks/synthetic_seqcls.py
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TaskBatch:
    x_s: torch.Tensor  # (B_s, T, D)
    y_s: torch.Tensor  # (B_s,)
    x_q: torch.Tensor  # (B_q, T, D)
    y_q: torch.Tensor  # (B_q,)


def _make_split(
    B: int,
    T: int,
    D: int,
    num_classes: int,
    positions: torch.Tensor,
    prototypes: torch.Tensor,
    noise_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a split (support or query):
      - sample labels y ~ Uniform({0..C-1})
      - x is mostly Gaussian noise
      - at chosen 'positions', inject class prototype + noise
    """
    y = torch.randint(low=0, high=num_classes, size=(B,), device=device)
    x = torch.randn((B, T, D), device=device, dtype=dtype) * noise_std

    # Inject signal at selected positions
    # prototypes: (C, D)
    sig = prototypes[y]  # (B, D)
    for pos in positions.tolist():
        x[:, pos, :] = sig + torch.randn((B, D), device=device, dtype=dtype) * (noise_std * 0.5)

    return x, y


def task_sampler(
    *,
    B_s: int = 16,
    B_q: int = 16,
    T: int = 32,
    D: int = 64,
    num_classes: int = 5,
    num_signal_positions: int = 4,
    noise_std: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> TaskBatch:
    """
    Returns a single meta-learning task with a support and query split.

    Each task samples:
      - a subset of positions that contain class-identifying signal
      - class prototypes in embedding space

    This forces attention to learn: token 0 must attend to informative positions.
    """
    if device is None:
        device = torch.device("cpu")

    # Choose which positions in the sequence contain the signal.
    # Ensure we do not always pick position 0 (since token 0 is our "query/CLS").
    # Also keep positions in-bounds.
    valid_positions = torch.arange(1, T, device=device)
    perm = torch.randperm(valid_positions.numel(), device=device)
    positions = valid_positions[perm[:num_signal_positions]]

    # Per-task prototypes: (C, D)
    prototypes = torch.randn((num_classes, D), device=device, dtype=dtype)

    x_s, y_s = _make_split(
        B=B_s, T=T, D=D, num_classes=num_classes,
        positions=positions, prototypes=prototypes,
        noise_std=noise_std, device=device, dtype=dtype
    )
    x_q, y_q = _make_split(
        B=B_q, T=T, D=D, num_classes=num_classes,
        positions=positions, prototypes=prototypes,
        noise_std=noise_std, device=device, dtype=dtype
    )

    return TaskBatch(x_s=x_s, y_s=y_s, x_q=x_q, y_q=y_q)