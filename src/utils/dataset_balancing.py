"""
Dataset balancing utility.

BUG-010: Unified training set can be subset-dominated.

Per-class balancing is applied per-subset, but larger subsets (FD002, FD004)
still contribute far more windows to the unified dataset than FD001/FD003
before any cross-subset balancing.  This module provides a
`build_unified_dataset` function that caps every subset's contribution to the
same `max_per_subset` budget, removing the skew entirely.

Usage in notebook 12:
    from src.utils.dataset_balancing import build_unified_dataset

    X_unified, y_unified = build_unified_dataset(
        subset_data,          # list of (X_balanced, y_balanced) per subset
        max_per_subset=None,  # None => use smallest balanced subset size
        seed=42,
    )
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from typing import List, Optional, Tuple


def build_unified_dataset(
    subset_data: List[Tuple[np.ndarray, np.ndarray]],
    max_per_subset: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge balanced per-subset arrays into one unified, subset-balanced dataset.

    BUG-010 fix: caps every subset's contribution to `max_per_subset` samples
    so that larger subsets (FD002, FD004) cannot dominate the unified training
    set.  When `max_per_subset` is None the cap is automatically set to the
    number of samples in the **smallest** balanced subset.

    Args:
        subset_data    : list of (X_balanced, y_balanced) arrays, one per
                         subset, each already class-balanced within that subset.
        max_per_subset : hard cap on samples drawn from each subset.  None
                         means use the smallest balanced-subset size.
        seed           : random seed for reproducible sub-sampling.

    Returns:
        (X_unified, y_unified) — shuffled, subset-balanced arrays.
    """
    rng = np.random.default_rng(seed)

    if max_per_subset is None:
        max_per_subset = min(len(X) for X, _ in subset_data)

    X_parts, y_parts = [], []
    for i, (X_sub, y_sub) in enumerate(subset_data):
        n = len(X_sub)
        if n > max_per_subset:
            idx = rng.choice(n, size=max_per_subset, replace=False)
            X_sub = X_sub[idx]
            y_sub = y_sub[idx]
            print(
                f"  Subset {i}: capped {n} -> {max_per_subset} samples "
                f"  dist={dict(sorted(Counter(y_sub.tolist()).items()))}"
            )
        else:
            print(
                f"  Subset {i}: kept all {n} samples "
                f"  dist={dict(sorted(Counter(y_sub.tolist()).items()))}"
            )
        X_parts.append(X_sub)
        y_parts.append(y_sub)

    X_unified = np.concatenate(X_parts, axis=0)
    y_unified = np.concatenate(y_parts, axis=0)

    # Shuffle
    idx = rng.permutation(len(X_unified))
    X_unified = X_unified[idx]
    y_unified = y_unified[idx]

    print(
        f"\nUnified dataset: {X_unified.shape}  "
        f"dist={dict(sorted(Counter(y_unified.tolist()).items()))}"
    )
    return X_unified, y_unified
