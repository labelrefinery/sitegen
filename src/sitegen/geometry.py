"""Rigid-body helpers. Right-handed, z up, meters and radians throughout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IndexArray = NDArray[np.int32]
"""Per-point source indices: which box a return came from, or -1 for terrain."""


def rot_z(yaw: float) -> Array:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(pitch: float) -> Array:
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def quat_from_matrix(r: Array) -> tuple[float, float, float, float]:
    """(x, y, z, w), the ordering both Foxglove and ROS use."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return (
            float((r[2, 1] - r[1, 2]) / s),
            float((r[0, 2] - r[2, 0]) / s),
            float((r[1, 0] - r[0, 1]) / s),
            float(0.25 * s),
        )
    i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(r[i, i] - r[j, j] - r[k, k] + 1.0, 1e-12)) * 2.0
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    w = (r[k, j] - r[j, k]) / s
    return (float(q[0]), float(q[1]), float(q[2]), float(w))


@dataclass(frozen=True)
class Box:
    """An oriented box in the world frame.

    This is simultaneously the collision geometry the LiDAR sees and the
    ground-truth cuboid the scorer compares against, which is deliberate: a
    label can never be more correct than the geometry that produced the
    returns, so there is exactly one definition of where an object is.
    """

    center: Array
    half_extents: Array
    rotation: Array
    class_name: str
    instance_id: str

    def yaw(self) -> float:
        return float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))


def transform(rotation: Array, translation: Array, local: Array) -> Array:
    """Map a local point (or (N,3) array of them) into the parent frame."""
    return local @ rotation.T + translation


def box_at(
    parent_r: Array,
    parent_t: Array,
    local_center: Array,
    half_extents: Array,
    local_r: Array,
    class_name: str,
    instance_id: str,
) -> Box:
    """Place a box defined in a parent frame into the world frame."""
    return Box(
        center=parent_r @ local_center + parent_t,
        half_extents=half_extents,
        rotation=parent_r @ local_r,
        class_name=class_name,
        instance_id=instance_id,
    )
