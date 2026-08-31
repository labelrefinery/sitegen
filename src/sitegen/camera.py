"""A pinhole camera, rendered with the same raycaster the LiDAR uses.

A camera is a spinning sensor whose rays happen to go through a pixel grid
instead of around an axis, so the intersection code is shared and the geometry
is guaranteed consistent between the two: a return in the cloud and a pixel in
the image describe the same surface, because the same ray produced both.

What comes out is a *shaded geometry* render -- flat albedo per class, one
directional light, a sky gradient. It is not photorealistic and is not trying
to be. It exists so the projection maths, the calibration topic and the
image-to-instance association can be built and tested end to end; whether a
model trained on photographs recognises anything in it is a separate question,
and one worth measuring rather than assuming.

The per-pixel instance ids come out for free, since the raycaster already knows
which box each ray hit. That is segmentation ground truth nobody had to draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Array, Box
from .terrain import Terrain

#: Base albedo per class. Deliberately distinguishable rather than accurate.
ALBEDO: dict[str, tuple[float, float, float]] = {
    "excavator": (0.86, 0.66, 0.10),
    "haul_truck": (0.78, 0.76, 0.72),
    "worker": (0.92, 0.36, 0.16),
    "grade_stake": (0.95, 0.93, 0.55),
}
GROUND = (0.52, 0.47, 0.40)
SKY_TOP = (0.45, 0.58, 0.78)
SKY_HORIZON = (0.78, 0.83, 0.88)
SUN = np.array([0.35, 0.30, 0.89])


@dataclass
class CameraIntrinsics:
    """Pinhole model. Resolution is part of it -- intrinsics are only valid at
    the resolution they were calibrated for, and carrying it lets a consumer
    assert rather than guess."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_fov(width: int, height: int, horizontal_fov_deg: float) -> CameraIntrinsics:
        fx = (width / 2.0) / np.tan(np.radians(horizontal_fov_deg) / 2.0)
        return CameraIntrinsics(width, height, fx, fx, width / 2.0, height / 2.0)

    def ray_directions(self) -> Array:
        """Unit rays in the camera frame: x right, y down, z forward."""
        u, v = np.meshgrid(
            np.arange(self.width, dtype=np.float64) + 0.5,
            np.arange(self.height, dtype=np.float64) + 0.5,
            indexing="xy",
        )
        d = np.stack(
            [(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=-1
        )
        return (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)


def _intersect_box_normal(origin: Array, dirs: Array, box: Box) -> tuple[Array, Array]:
    """Slab test that also reports the face normal, for shading."""
    o = box.rotation.T @ (origin - box.center)
    d = dirs @ box.rotation
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (-box.half_extents - o) * inv
        t2 = (box.half_extents - o) * inv
    lo = np.minimum(t1, t2)
    hi = np.maximum(t1, t2)
    t_near = np.max(lo, axis=1)
    t_far = np.min(hi, axis=1)
    hit = (t_far >= np.maximum(t_near, 0.0)) & (t_near > 0.0)

    axis = np.argmax(lo, axis=1)
    local_n = np.zeros_like(dirs)
    local_n[np.arange(len(dirs)), axis] = -np.sign(d[np.arange(len(dirs)), axis])
    return np.where(hit, t_near, np.inf), local_n @ box.rotation.T


def render(
    intrinsics: CameraIntrinsics,
    cam_r: Array,
    cam_t: Array,
    boxes: list[Box],
    terrain: Terrain,
    max_range: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (H, W, 3) uint8 RGB and (H, W) uint16 instance ids (0 = none)."""
    local = intrinsics.ray_directions()
    world = local @ cam_r.T

    t = terrain.intersect(cam_t, world)
    source = np.full(t.shape, -1, dtype=np.int32)
    normal = np.tile(np.array([0.0, 0.0, 1.0]), (len(world), 1))

    for i, box in enumerate(boxes):
        t_box, n_box = _intersect_box_normal(cam_t, world, box)
        closer = t_box < t
        t = np.where(closer, t_box, t)
        source = np.where(closer, i, source)
        normal = np.where(closer[:, None], n_box, normal)

    lambert = np.clip(np.abs(normal @ SUN), 0.0, 1.0)
    shade = (0.35 + 0.65 * lambert)[:, None]

    rgb = np.zeros((len(world), 3))
    ground = (source < 0) & np.isfinite(t) & (t < max_range)
    rgb[ground] = np.array(GROUND) * shade[ground]

    for i, box in enumerate(boxes):
        mask = source == i
        if not np.any(mask):
            continue
        base = ALBEDO.get(box.class_name.split(".")[0], (0.7, 0.7, 0.7))
        rgb[mask] = np.array(base) * shade[mask]

    # Sky, blended toward the horizon by ray elevation.
    sky = ~np.isfinite(t) | (t >= max_range)
    if np.any(sky):
        elevation = np.clip(world[sky, 2] * 2.0 + 0.5, 0.0, 1.0)[:, None]
        rgb[sky] = np.array(SKY_HORIZON) * (1 - elevation) + np.array(SKY_TOP) * elevation

    # Aerial perspective: distant surfaces wash toward the horizon colour.
    depth = np.clip(np.where(np.isfinite(t), t, max_range) / max_range, 0.0, 1.0)[:, None]
    rgb = rgb * (1 - 0.45 * depth) + np.array(SKY_HORIZON) * (0.45 * depth)

    image = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    instances = np.where(source < 0, 0, source + 1).astype(np.uint16)
    shape = (intrinsics.height, intrinsics.width)
    return image.reshape(*shape, 3), instances.reshape(*shape)


#: A surround-view rig: four cameras at the corners and sides of the house,
#: panned outward. Real machines mount them this way for exactly the reason the
#: measurement below shows -- a centre-forward camera spends a quarter of its
#: frame looking at its own boom.
#:
#: (name, offset in the house frame, pan about house z in degrees)
SURROUND_RIG: list[tuple[str, tuple[float, float, float], float]] = [
    ("front_left", (1.3, 1.15, 2.5), 35.0),
    ("front_right", (1.3, -1.15, 2.5), -35.0),
    ("left", (-0.3, 1.35, 2.5), 90.0),
    ("right", (-0.3, -1.35, 2.5), -90.0),
]


def camera_pose(
    house_r: Array, house_t: Array, offset: Array, pan_deg: float = 0.0
) -> tuple[Array, Array]:
    """Camera on the cab: +z forward, +x right, +y down, panned about house z."""
    optical = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]).T
    a = np.radians(pan_deg)
    pan = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return house_r @ pan @ optical, house_r @ offset + house_t
