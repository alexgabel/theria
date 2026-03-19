from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F


DEFAULT_NUM_SAMPLES = 1000
_DATASET_CACHE: dict[tuple[object, ...], "SyntheticSeqClsDataset"] = {}
_DATASET_SPLITS_CACHE: dict[tuple[object, ...], "SyntheticSeqClsDatasetSplits"] = {}
DEFAULT_TRAIN_FRACTION = 0.7
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DatasetSplit = Literal["train", "val", "test"]


@dataclass(frozen=True)
class TaskBatch:
    x_s: torch.Tensor  # (B_s, T, D)
    y_s: torch.Tensor  # (B_s,)
    x_q: torch.Tensor  # (B_q, T, D)
    y_q: torch.Tensor  # (B_q,)


@dataclass(frozen=True)
class SyntheticSeqClsDataset:
    x: torch.Tensor  # (N, T, D)
    y: torch.Tensor  # (N,)
    signal_positions: torch.Tensor  # (S,)
    prototypes: torch.Tensor  # (C, D)
    cls_token: torch.Tensor  # (D,)
    signal_marker: torch.Tensor  # (D,)
    indices: torch.Tensor  # (N,)

    @property
    def num_samples(self) -> int:
        return int(self.x.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.x.shape[1])

    @property
    def d_model(self) -> int:
        return int(self.x.shape[2])

    @property
    def num_classes(self) -> int:
        return int(self.prototypes.shape[0])


@dataclass(frozen=True)
class SyntheticSeqClsDatasetSplits:
    train: SyntheticSeqClsDataset
    val: SyntheticSeqClsDataset
    test: SyntheticSeqClsDataset

    def get(self, split: DatasetSplit) -> SyntheticSeqClsDataset:
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        if split == "test":
            return self.test
        raise ValueError(f"Unknown split: {split}")


def _validate_config(
    *,
    num_samples: int,
    T: int,
    D: int,
    num_classes: int,
    num_signal_positions: int,
) -> None:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if T < 2:
        raise ValueError("T must be >= 2 because token 0 is reserved for CLS")
    if D < 3:
        raise ValueError("D must be >= 3 so the dataset can reserve CLS and signal-marker dimensions")
    if num_classes <= 1:
        raise ValueError("num_classes must be > 1")
    if num_signal_positions <= 0:
        raise ValueError("num_signal_positions must be > 0")
    if num_signal_positions >= T:
        raise ValueError("num_signal_positions must be < T because token 0 is reserved for CLS")


def _resolve_device(device: torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return device


def _resolve_dataset_seed(dataset_seed: int | None) -> int:
    if dataset_seed is not None:
        return int(dataset_seed)
    return int(torch.initial_seed())


def _resolve_split_fractions(
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[float, float, float]:
    fractions = [float(train_fraction), float(val_fraction), float(test_fraction)]
    if any(frac < 0.0 for frac in fractions):
        raise ValueError("Split fractions must be >= 0")
    total = sum(fractions)
    if total <= 0.0:
        raise ValueError("At least one split fraction must be > 0")
    return tuple(frac / total for frac in fractions)


def _resolve_split_name(split: DatasetSplit) -> DatasetSplit:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    return split


def _signal_positions(T: int, num_signal_positions: int) -> torch.Tensor:
    valid_positions = torch.arange(1, T, dtype=torch.long)
    if num_signal_positions == valid_positions.numel():
        return valid_positions
    grid = torch.linspace(0, valid_positions.numel() - 1, steps=num_signal_positions)
    return valid_positions[grid.round().long()]


def _make_prototypes(
    *,
    D: int,
    num_classes: int,
    prototype_scale: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    prototypes = torch.zeros((num_classes, D), dtype=dtype)
    if D >= (num_classes + 2):
        for class_idx in range(num_classes):
            prototypes[class_idx, 2 + class_idx] = prototype_scale
        return prototypes

    raw = torch.randn((num_classes, D), generator=generator, dtype=dtype)
    return F.normalize(raw, dim=-1) * prototype_scale


def make_dataset(
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    T: int = 32,
    D: int = 64,
    num_classes: int = 5,
    num_signal_positions: int = 4,
    noise_std: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    dataset_seed: int | None = None,
) -> SyntheticSeqClsDataset:
    """
    Build one fixed synthetic sequence-classification dataset.

    Design:
      - token 0 is a stable CLS anchor shared by all sequences
      - a small set of informative positions is fixed once for the whole dataset
      - informative tokens carry a shared "look here" marker plus a class prototype
      - all other tokens are pure noise
    """
    _validate_config(
        num_samples=num_samples,
        T=T,
        D=D,
        num_classes=num_classes,
        num_signal_positions=num_signal_positions,
    )

    resolved_device = _resolve_device(device)
    resolved_seed = _resolve_dataset_seed(dataset_seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved_seed)

    signal_positions = _signal_positions(T=T, num_signal_positions=num_signal_positions)

    cls_token = torch.zeros(D, dtype=dtype)
    cls_token[0] = 3.0 * noise_std

    signal_marker = torch.zeros(D, dtype=dtype)
    signal_marker[1] = 2.5 * noise_std

    prototypes = _make_prototypes(
        D=D,
        num_classes=num_classes,
        prototype_scale=3.5 * noise_std,
        generator=generator,
        dtype=dtype,
    )

    y = torch.arange(num_samples, dtype=torch.long) % num_classes
    y = y[torch.randperm(num_samples, generator=generator)]

    x = torch.randn((num_samples, T, D), generator=generator, dtype=dtype) * noise_std
    x[:, 0, :] = cls_token + torch.randn((num_samples, D), generator=generator, dtype=dtype) * (0.15 * noise_std)

    signal_noise = torch.randn(
        (num_samples, num_signal_positions, D),
        generator=generator,
        dtype=dtype,
    ) * (0.25 * noise_std)
    signal_tokens = prototypes[y][:, None, :] + signal_marker.view(1, 1, D) + signal_noise
    x[:, signal_positions, :] = signal_tokens

    if resolved_device.type != "cpu":
        x = x.to(device=resolved_device)
        y = y.to(device=resolved_device)
        signal_positions = signal_positions.to(device=resolved_device)
        prototypes = prototypes.to(device=resolved_device)
        cls_token = cls_token.to(device=resolved_device)
        signal_marker = signal_marker.to(device=resolved_device)

    return SyntheticSeqClsDataset(
        x=x,
        y=y,
        signal_positions=signal_positions,
        prototypes=prototypes,
        cls_token=cls_token,
        signal_marker=signal_marker,
        indices=torch.arange(num_samples, device=resolved_device, dtype=torch.long),
    )


def clear_task_dataset_cache() -> None:
    _DATASET_CACHE.clear()
    _DATASET_SPLITS_CACHE.clear()


def get_cached_dataset(
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    T: int = 32,
    D: int = 64,
    num_classes: int = 5,
    num_signal_positions: int = 4,
    noise_std: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    dataset_seed: int | None = None,
) -> SyntheticSeqClsDataset:
    resolved_device = _resolve_device(device)
    resolved_seed = _resolve_dataset_seed(dataset_seed)
    cache_key = (
        num_samples,
        T,
        D,
        num_classes,
        num_signal_positions,
        float(noise_std),
        str(resolved_device),
        str(dtype),
        resolved_seed,
    )
    dataset = _DATASET_CACHE.get(cache_key)
    if dataset is None:
        dataset = make_dataset(
            num_samples=num_samples,
            T=T,
            D=D,
            num_classes=num_classes,
            num_signal_positions=num_signal_positions,
            noise_std=noise_std,
            device=resolved_device,
            dtype=dtype,
            dataset_seed=resolved_seed,
        )
        _DATASET_CACHE[cache_key] = dataset
    return dataset


def _make_dataset_subset(
    dataset: SyntheticSeqClsDataset,
    indices: torch.Tensor,
) -> SyntheticSeqClsDataset:
    return SyntheticSeqClsDataset(
        x=dataset.x[indices],
        y=dataset.y[indices],
        signal_positions=dataset.signal_positions,
        prototypes=dataset.prototypes,
        cls_token=dataset.cls_token,
        signal_marker=dataset.signal_marker,
        indices=dataset.indices[indices],
    )


def _split_counts_for_class(
    class_count: int,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if class_count <= 0:
        return 0, 0, 0

    fractions = torch.tensor(
        [train_fraction, val_fraction, test_fraction],
        dtype=torch.float64,
    )
    positive = fractions > 0
    min_required = int(positive.sum().item())
    if class_count < min_required:
        return class_count, 0, 0

    raw = fractions * class_count
    counts = torch.floor(raw).to(torch.long)
    minimums = positive.to(torch.long)
    counts = torch.maximum(counts, minimums)

    while int(counts.sum().item()) > class_count:
        slack = counts - minimums
        drop_idx = int(torch.argmax(slack).item())
        if slack[drop_idx].item() <= 0:
            break
        counts[drop_idx] -= 1

    residual = raw - torch.floor(raw)
    while int(counts.sum().item()) < class_count:
        add_idx = int(torch.argmax(residual).item())
        counts[add_idx] += 1
        residual[add_idx] -= 1.0

    return int(counts[0].item()), int(counts[1].item()), int(counts[2].item())


def make_dataset_splits(
    *,
    dataset: SyntheticSeqClsDataset | None = None,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    T: int = 32,
    D: int = 64,
    num_classes: int = 5,
    num_signal_positions: int = 4,
    noise_std: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    dataset_seed: int | None = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> SyntheticSeqClsDatasetSplits:
    train_fraction, val_fraction, test_fraction = _resolve_split_fractions(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    if dataset is None:
        dataset = make_dataset(
            num_samples=num_samples,
            T=T,
            D=D,
            num_classes=num_classes,
            num_signal_positions=num_signal_positions,
            noise_std=noise_std,
            device=device,
            dtype=dtype,
            dataset_seed=dataset_seed,
        )

    resolved_seed = _resolve_dataset_seed(dataset_seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved_seed + 1)

    split_parts: dict[str, list[torch.Tensor]] = {"train": [], "val": [], "test": []}
    for class_idx in range(dataset.num_classes):
        class_mask = dataset.y == class_idx
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
        if class_indices.numel() == 0:
            continue
        perm_cpu = torch.randperm(class_indices.numel(), generator=generator)
        perm = perm_cpu.to(device=class_indices.device)
        class_indices = class_indices[perm]
        n_train, n_val, n_test = _split_counts_for_class(
            int(class_indices.numel()),
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        offset = 0
        if n_train > 0:
            split_parts["train"].append(class_indices[offset:offset + n_train])
            offset += n_train
        if n_val > 0:
            split_parts["val"].append(class_indices[offset:offset + n_val])
            offset += n_val
        if n_test > 0:
            split_parts["test"].append(class_indices[offset:offset + n_test])

    empty = torch.empty(0, dtype=torch.long, device=dataset.y.device)
    split_indices = {
        name: torch.cat(parts) if parts else empty
        for name, parts in split_parts.items()
    }
    for name, indices in list(split_indices.items()):
        if indices.numel() > 1:
            perm = torch.randperm(indices.numel(), device=indices.device)
            split_indices[name] = indices[perm]

    return SyntheticSeqClsDatasetSplits(
        train=_make_dataset_subset(dataset, split_indices["train"]),
        val=_make_dataset_subset(dataset, split_indices["val"]),
        test=_make_dataset_subset(dataset, split_indices["test"]),
    )


def get_cached_dataset_splits(
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    T: int = 32,
    D: int = 64,
    num_classes: int = 5,
    num_signal_positions: int = 4,
    noise_std: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    dataset_seed: int | None = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> SyntheticSeqClsDatasetSplits:
    resolved_device = _resolve_device(device)
    resolved_seed = _resolve_dataset_seed(dataset_seed)
    train_fraction, val_fraction, test_fraction = _resolve_split_fractions(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    cache_key = (
        num_samples,
        T,
        D,
        num_classes,
        num_signal_positions,
        float(noise_std),
        str(resolved_device),
        str(dtype),
        resolved_seed,
        train_fraction,
        val_fraction,
        test_fraction,
    )
    dataset_splits = _DATASET_SPLITS_CACHE.get(cache_key)
    if dataset_splits is None:
        full_dataset = get_cached_dataset(
            num_samples=num_samples,
            T=T,
            D=D,
            num_classes=num_classes,
            num_signal_positions=num_signal_positions,
            noise_std=noise_std,
            device=resolved_device,
            dtype=dtype,
            dataset_seed=resolved_seed,
        )
        dataset_splits = make_dataset_splits(
            dataset=full_dataset,
            dataset_seed=resolved_seed,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        _DATASET_SPLITS_CACHE[cache_key] = dataset_splits
    return dataset_splits


def get_cached_dataset_split(
    *,
    split: DatasetSplit = "train",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    T: int = 32,
    D: int = 64,
    num_classes: int = 5,
    num_signal_positions: int = 4,
    noise_std: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    dataset_seed: int | None = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> SyntheticSeqClsDataset:
    resolved_split = _resolve_split_name(split)
    return get_cached_dataset_splits(
        num_samples=num_samples,
        T=T,
        D=D,
        num_classes=num_classes,
        num_signal_positions=num_signal_positions,
        noise_std=noise_std,
        device=device,
        dtype=dtype,
        dataset_seed=dataset_seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    ).get(resolved_split)


def _per_class_counts(batch_size: int, num_classes: int, *, device: torch.device) -> list[int]:
    counts = torch.full((num_classes,), batch_size // num_classes, dtype=torch.long, device=device)
    remainder = batch_size % num_classes
    if remainder > 0:
        counts[torch.randperm(num_classes, device=device)[:remainder]] += 1
    return counts.tolist()


def _sample_episode_indices(
    dataset: SyntheticSeqClsDataset,
    *,
    B_s: int,
    B_q: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = dataset.y.device
    support_counts = _per_class_counts(B_s, dataset.num_classes, device=device)
    query_counts = _per_class_counts(B_q, dataset.num_classes, device=device)
    support_parts: list[torch.Tensor] = []
    query_parts: list[torch.Tensor] = []

    for class_idx in range(dataset.num_classes):
        class_indices = torch.nonzero(dataset.y == class_idx, as_tuple=False).squeeze(1)
        n_support = support_counts[class_idx]
        n_query = query_counts[class_idx]
        total_needed = n_support + n_query
        if total_needed == 0:
            continue
        if class_indices.numel() == 0:
            raise RuntimeError(f"Dataset has no examples for class {class_idx}")

        if total_needed <= class_indices.numel():
            chosen = class_indices[torch.randperm(class_indices.numel(), device=device)[:total_needed]]
        else:
            draw = torch.randint(class_indices.numel(), (total_needed,), device=device)
            chosen = class_indices[draw]

        if n_support > 0:
            support_parts.append(chosen[:n_support])
        if n_query > 0:
            query_parts.append(chosen[n_support:])

    support_idx = torch.cat(support_parts) if support_parts else torch.empty(0, dtype=torch.long, device=device)
    query_idx = torch.cat(query_parts) if query_parts else torch.empty(0, dtype=torch.long, device=device)
    if support_idx.numel() > 1:
        support_idx = support_idx[torch.randperm(support_idx.numel(), device=device)]
    if query_idx.numel() > 1:
        query_idx = query_idx[torch.randperm(query_idx.numel(), device=device)]
    return support_idx, query_idx


def sample_task(
    dataset: SyntheticSeqClsDataset,
    *,
    B_s: int = 16,
    B_q: int = 16,
    device: torch.device | None = None,
) -> TaskBatch:
    if B_s <= 0 or B_q <= 0:
        raise ValueError("B_s and B_q must both be > 0")

    support_idx, query_idx = _sample_episode_indices(dataset, B_s=B_s, B_q=B_q)
    x_s = dataset.x[support_idx]
    y_s = dataset.y[support_idx]
    x_q = dataset.x[query_idx]
    y_q = dataset.y[query_idx]

    resolved_device = _resolve_device(device) if device is not None else dataset.x.device
    if resolved_device != dataset.x.device:
        x_s = x_s.to(device=resolved_device)
        y_s = y_s.to(device=resolved_device)
        x_q = x_q.to(device=resolved_device)
        y_q = y_q.to(device=resolved_device)

    return TaskBatch(x_s=x_s, y_s=y_s, x_q=x_q, y_q=y_q)


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
    num_samples: int = DEFAULT_NUM_SAMPLES,
    dataset: SyntheticSeqClsDataset | None = None,
    split: DatasetSplit = "train",
    dataset_seed: int | None = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    cache_dataset: bool = True,
) -> TaskBatch:
    """
    Return one support/query episode sampled from a fixed dataset split.

    If `dataset` is not provided, the split dataset is created once per
    configuration (including the process seed) and then reused for the rest of
    the run.
    """
    resolved_device = (
        _resolve_device(device)
        if device is not None else
        (dataset.x.device if dataset is not None else torch.device("cpu"))
    )
    if dataset is None:
        resolved_split = _resolve_split_name(split)
        if cache_dataset:
            dataset = get_cached_dataset_split(
                split=resolved_split,
                num_samples=num_samples,
                T=T,
                D=D,
                num_classes=num_classes,
                num_signal_positions=num_signal_positions,
                noise_std=noise_std,
                device=resolved_device,
                dtype=dtype,
                dataset_seed=dataset_seed,
                train_fraction=train_fraction,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
            )
        else:
            dataset = make_dataset_splits(
                num_samples=num_samples,
                T=T,
                D=D,
                num_classes=num_classes,
                num_signal_positions=num_signal_positions,
                noise_std=noise_std,
                device=resolved_device,
                dtype=dtype,
                dataset_seed=dataset_seed,
                train_fraction=train_fraction,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
            ).get(resolved_split)

    return sample_task(dataset, B_s=B_s, B_q=B_q, device=resolved_device)


def _project_pca_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().to(torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    rank = min(2, x.shape[0], x.shape[1])
    if rank < 2:
        raise ValueError("Need at least two samples and two features for a 2D PCA plot")
    _, _, v = torch.pca_lowrank(x, q=rank, center=False)
    return x @ v[:, :2]


def _plot_dataset_pca(
    dataset: SyntheticSeqClsDataset,
    *,
    out_path: Path,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import matplotlib.pyplot as plt

    seq_features = dataset.x.reshape(dataset.num_samples, -1)
    projection = _project_pca_2d(seq_features)
    labels = dataset.y.detach().cpu()

    class_means = []
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10", dataset.num_classes)
    for class_idx in range(dataset.num_classes):
        mask = labels == class_idx
        class_points = projection[mask]
        ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            s=18,
            alpha=0.7,
            color=cmap(class_idx),
            label=f"class {class_idx}",
        )
        class_means.append(class_points.mean(dim=0))

    mean_points = torch.stack(class_means)
    ax.scatter(
        mean_points[:, 0],
        mean_points[:, 1],
        s=120,
        marker="x",
        linewidths=2.0,
        color="black",
        label="class mean",
    )
    ax.set_title(
        "Synthetic SeqCls PCA\n"
        f"N={dataset.num_samples}, T={dataset.seq_len}, D={dataset.d_model}, "
        f"signal_positions={dataset.signal_positions.detach().cpu().tolist()}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed synthetic sequence-classification dataset diagnostics")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--num-signal-positions", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("synthetic_seqcls_pca.png"))
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    dataset = make_dataset(
        num_samples=args.num_samples,
        T=args.seq_len,
        D=args.d_model,
        num_classes=args.num_classes,
        num_signal_positions=args.num_signal_positions,
        noise_std=args.noise_std,
        dataset_seed=args.seed,
    )
    dataset_splits = make_dataset_splits(dataset=dataset, dataset_seed=args.seed)
    out_path = _plot_dataset_pca(dataset, out_path=args.out)

    class_counts = torch.bincount(dataset.y.detach().cpu(), minlength=dataset.num_classes).tolist()
    split_sizes = {
        "train": dataset_splits.train.num_samples,
        "val": dataset_splits.val.num_samples,
        "test": dataset_splits.test.num_samples,
    }
    print(
        "Built fixed synthetic dataset "
        f"(seed={args.seed}, N={dataset.num_samples}, T={dataset.seq_len}, D={dataset.d_model}, "
        f"signal_positions={dataset.signal_positions.detach().cpu().tolist()}, "
        f"class_counts={class_counts}, split_sizes={split_sizes})"
    )
    print(f"Saved PCA plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
