import torch
from torch.utils.data import Dataset, Subset, TensorDataset, random_split

def make_train_val_indices(
    n_items: int,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Return the exact split produced by torch.random_split()."""
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    n_train = int(train_fraction * n_items)
    generator = torch.Generator().manual_seed(seed)
    index_dataset = TensorDataset(torch.arange(n_items))
    split_subsets = random_split(
        index_dataset,
        [n_train, n_items - n_train],
        generator=generator,
    )
    return list(split_subsets[0].indices), list(split_subsets[1].indices)

def make_train_val_subsets(
    dataset: Dataset,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    train_indices, val_indices = make_train_val_indices(
        len(dataset),  # type: ignore[arg-type]
        train_fraction=train_fraction,
        seed=seed,
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def select_sample_indices(n_items: int, n_use: int, seed: int = 42) -> list[int]:
    """Deterministically choose a sorted subset of sample indices."""
    if n_use >= n_items:
        return list(range(n_items))

    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(n_items, generator=generator)[:n_use].tolist()
    chosen.sort()
    return chosen
