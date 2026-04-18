from __future__ import annotations

import numpy as np


def split_held_out(
    edge_index: np.ndarray, ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split an edge_index of shape (2, E) into (train, held_out) by a fixed seed.

    `ratio` is the fraction of edges moved to the held-out set.
    Returns (train, held_out), both shaped (2, E_*).
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be shape (2, E), got {edge_index.shape}")
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")

    n = edge_index.shape[1]
    n_held = int(round(n * ratio))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    held_idx = perm[:n_held]
    train_idx = perm[n_held:]

    return edge_index[:, train_idx], edge_index[:, held_idx]
