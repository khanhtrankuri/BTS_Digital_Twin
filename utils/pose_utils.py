"""Pose analysis shared by validation, exposure, and multiview tooling."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from scene.colmap_loader import qvec2rotmat


def center_direction_from_qvec_tvec(qvec: Iterable[float], tvec: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return camera center and world-space +Z direction for COLMAP W2C pose."""

    world_to_camera = qvec2rotmat(np.asarray(qvec, dtype=np.float64))
    translation = np.asarray(tvec, dtype=np.float64)
    center = -world_to_camera.T @ translation
    direction = world_to_camera.T[:, 2]
    direction /= max(np.linalg.norm(direction), 1e-12)
    return center, direction


def center_direction_from_runtime(camera) -> tuple[np.ndarray, np.ndarray]:
    """Return center/direction from this repository's transposed ``R`` format."""

    rotation = np.asarray(camera.R, dtype=np.float64)
    center = -rotation @ np.asarray(camera.T, dtype=np.float64)
    direction = rotation[:, 2]
    direction /= max(np.linalg.norm(direction), 1e-12)
    return center, direction


def scene_radius(centers: np.ndarray) -> float:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ValueError("centers must have shape [N,3] with N > 0")
    return max(float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()) * 1.1, 1e-8)


def nearest_pose(target_center: np.ndarray, target_direction: np.ndarray,
                 train_centers: np.ndarray, train_directions: np.ndarray,
                 radius: float, angle_selection_weight: float = 0.0) -> tuple[int, float, float]:
    """Return nearest index plus normalized position and angular distances.

    By default the source is the nearest camera center, matching the dataset
    difficulty statistics. ``angle_selection_weight`` can be enabled by a
    source-view selector that explicitly trades position for orientation.
    """

    target_center = np.asarray(target_center, dtype=np.float64)
    target_direction = np.asarray(target_direction, dtype=np.float64)
    positions = np.linalg.norm(np.asarray(train_centers) - target_center, axis=1) / max(float(radius), 1e-8)
    dots = np.clip(np.asarray(train_directions) @ target_direction, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    index = int(np.argmin(positions + float(angle_selection_weight) * angles / 180.0))
    return index, float(positions[index]), float(angles[index])


def difficulty_bin(position_distance: float, angle_degrees: float) -> str:
    if position_distance > 0.10 and angle_degrees > 18.0:
        return "extreme"
    if position_distance > 0.10:
        return "hard_position"
    if angle_degrees > 18.0:
        return "hard_angle"
    if position_distance < 0.04 and angle_degrees < 8.0:
        return "easy"
    return "medium"


def view_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return math.degrees(math.acos(float(np.clip(first @ second, -1.0, 1.0))))
