"""Minimal itemid-keyed npz (de)serialization for the distillation pipeline.

Embeddings are ragged (variable window count per clip), so they are stored as an
aligned (itemids, object-array-of-[n_windows, D]) pair. Targets are a flat
(itemids, logits) pair. Both round-trip through plain numpy -- no extra deps.
"""

from pathlib import Path

import numpy as np


def save_embeddings(path: str | Path, embeddings: dict[str, np.ndarray]) -> Path:
    """Write {itemid: [n_windows, D] float32} to a compressed npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    itemids = np.array(list(embeddings.keys()))
    arrays = np.empty(len(embeddings), dtype=object)
    for i, key in enumerate(embeddings):
        arrays[i] = np.asarray(embeddings[key], dtype=np.float32)
    np.savez_compressed(path, itemids=itemids, embeddings=arrays)
    return path


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    """Read the {itemid: [n_windows, D]} table written by save_embeddings."""
    data = np.load(path, allow_pickle=True)
    return {itemid: array for itemid, array in zip(data["itemids"].tolist(), data["embeddings"])}


def save_targets(path: str | Path, logits: dict[str, float]) -> Path:
    """Write {itemid: clip_logit} to an npz consumed by DistillDataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    itemids = np.array(list(logits.keys()))
    values = np.array([float(logits[key]) for key in logits], dtype=np.float32)
    np.savez(path, itemids=itemids, logits=values)
    return path


def load_targets(path: str | Path) -> dict[str, float]:
    """Read the {itemid: clip_logit} soft-target table."""
    data = np.load(path, allow_pickle=False)
    return dict(zip(data["itemids"].tolist(), data["logits"].tolist()))
