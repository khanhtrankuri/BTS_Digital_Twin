"""Optional offline dense-prior loading for BTS-GeoGS.

This module deliberately has no VGGT dependency: another process can write
``points.npy``, ``colors.npy`` and optional ``confidence.npy``/``normals.npy``.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def load_dense_initialization(path):
    root = Path(path)
    required = {name: root / f"{name}.npy" for name in ("points", "colors")}
    missing = [str(item) for item in required.values() if not item.exists()]
    if missing:
        raise FileNotFoundError("Dense prior requires: " + ", ".join(missing))
    data = {name: np.load(file) for name, file in required.items()}
    for name in ("confidence", "normals"):
        file = root / f"{name}.npy"
        data[name] = np.load(file) if file.exists() else None
    points, colors = np.asarray(data["points"], np.float32), np.asarray(data["colors"], np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("points.npy and colors.npy must both have shape [N, 3].")
    valid = np.isfinite(points).all(1) & np.isfinite(colors).all(1)
    if data["confidence"] is not None:
        confidence = np.asarray(data["confidence"], np.float32).reshape(-1)
        if confidence.shape[0] != points.shape[0]:
            raise ValueError("confidence.npy must have one value per point.")
        data["confidence"] = np.clip(confidence, 0.0, 1.0)
        valid &= np.isfinite(confidence)
    if data["normals"] is not None:
        normals = np.asarray(data["normals"], np.float32)
        if normals.shape != points.shape:
            raise ValueError("normals.npy must have shape [N, 3].")
        valid &= np.isfinite(normals).all(1)
    for key, value in data.items():
        data[key] = value[valid] if value is not None else None
    return data


def voxel_downsample(data, voxel_size):
    """Keep one deterministic representative per voxel without extra deps."""
    if voxel_size <= 0 or len(data["points"]) == 0:
        return data
    voxels = np.floor(data["points"] / float(voxel_size)).astype(np.int64)
    _, keep = np.unique(voxels, axis=0, return_index=True)
    keep.sort()
    return {key: value[keep] if value is not None else None for key, value in data.items()}
