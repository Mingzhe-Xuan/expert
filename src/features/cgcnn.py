from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.metadata import version

import numpy as np
import torch

from ..irreps import IrrepLayout, IrrepTerm


CGCNN_FEATURE_DIMENSION = 92
CGCNN_SOURCE_LAYOUT = IrrepLayout(
    (IrrepTerm(CGCNN_FEATURE_DIMENSION, 0, "e", "gmtnet_cgcnn"),)
)


@lru_cache(maxsize=1)
def _feature_table() -> torch.Tensor:
    """Return GMTNet's exact JARVIS CGCNN table indexed by atomic number."""

    try:
        from jarvis.core.specie import chem_data, get_node_attributes
    except ImportError as exc:
        raise ImportError("CGCNN-style features require jarvis-tools") from exc
    by_number: dict[int, np.ndarray] = {}
    for symbol, properties in chem_data.items():
        atomic_number = int(properties["Z"])
        feature = np.asarray(
            get_node_attributes(symbol, atom_features="cgcnn"), dtype=np.float32
        )
        if feature.shape != (CGCNN_FEATURE_DIMENSION,):
            raise ValueError(
                f"JARVIS CGCNN feature for {symbol} has shape {feature.shape}, expected (92,)"
            )
        if atomic_number in by_number:
            raise ValueError(f"duplicate JARVIS atomic number {atomic_number}")
        if not np.isfinite(feature).all():
            raise ValueError(f"non-finite JARVIS CGCNN feature for {symbol}")
        by_number[atomic_number] = feature
    expected = set(range(1, max(by_number) + 1))
    if set(by_number) != expected:
        raise ValueError("JARVIS CGCNN table does not cover a contiguous element range")
    table = np.zeros((max(by_number) + 1, CGCNN_FEATURE_DIMENSION), dtype=np.float32)
    for atomic_number, feature in by_number.items():
        table[atomic_number] = feature
    return torch.from_numpy(table)


def cgcnn_node_features(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Map a one-dimensional atomic-number tensor to exact 92D GMTNet inputs."""

    if atomic_numbers.ndim != 1 or atomic_numbers.numel() == 0:
        raise ValueError("atomic_numbers must be a non-empty one-dimensional tensor")
    if atomic_numbers.dtype == torch.bool or torch.is_floating_point(atomic_numbers):
        raise ValueError("atomic_numbers must use an integer dtype")
    indices = atomic_numbers.to(dtype=torch.long)
    table = _feature_table()
    if int(indices.min()) < 1 or int(indices.max()) >= table.shape[0]:
        raise ValueError("atomic number is outside the JARVIS CGCNN feature table")
    return table.to(device=atomic_numbers.device).index_select(0, indices)


def cgcnn_feature_sha256() -> str:
    """Content identity used to reject stale or non-GMTNet feature caches."""

    values = _feature_table().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def cgcnn_feature_metadata() -> dict[str, object]:
    return {
        "provider": "jarvis.core.specie.get_node_attributes",
        "atom_features": "cgcnn",
        "dimension": CGCNN_FEATURE_DIMENSION,
        "jarvis_tools_version": version("jarvis-tools"),
        "table_sha256": cgcnn_feature_sha256(),
    }
