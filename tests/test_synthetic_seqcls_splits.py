import torch

from theria.tasks.synthetic_seqcls import get_cached_dataset, get_cached_dataset_splits, task_sampler


def test_synthetic_seqcls_splits_are_disjoint_and_cover_dataset():
    full_dataset = get_cached_dataset(
        num_samples=120,
        T=12,
        D=16,
        num_classes=3,
        num_signal_positions=3,
        dataset_seed=11,
        device=torch.device("cpu"),
    )
    splits = get_cached_dataset_splits(
        num_samples=120,
        T=12,
        D=16,
        num_classes=3,
        num_signal_positions=3,
        dataset_seed=11,
        device=torch.device("cpu"),
    )

    train_ids = set(splits.train.indices.tolist())
    val_ids = set(splits.val.indices.tolist())
    test_ids = set(splits.test.indices.tolist())
    full_ids = set(full_dataset.indices.tolist())

    assert train_ids
    assert val_ids
    assert test_ids
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == full_ids

    for split_dataset in (splits.train, splits.val, splits.test):
        assert set(torch.unique(split_dataset.y).tolist()) == {0, 1, 2}

    val_task = task_sampler(
        B_s=6,
        B_q=6,
        T=12,
        D=16,
        num_classes=3,
        num_signal_positions=3,
        dataset_seed=11,
        split="val",
        device=torch.device("cpu"),
    )
    assert val_task.x_s.shape == (6, 12, 16)
    assert val_task.x_q.shape == (6, 12, 16)
