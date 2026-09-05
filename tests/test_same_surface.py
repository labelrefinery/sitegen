"""The claim the whole design rests on, checked against a real recording.

sitegen has said since its first commit that "a return in the cloud and a pixel
in the image describe the same surface, because the same ray produced both".
That was true by construction while one raycaster drew both, and it stopped
being true by construction the moment the camera moved to Cycles: now there are
two renderers that have to agree about intrinsics, about extrinsics, and about
two different optical conventions. Agreement of that kind is not a property of
the code structure any more. It is a measurement.

So: take LiDAR returns out of the recording, project each into every camera
using only what the recording publishes, and ask the held-out instance mask
what it thinks is at that pixel. The answers must match.

A disagreement here is never cosmetic. It means one of

  - the camera pose written to `/tf` is not the pose Cycles rendered from,
  - K in `/camera/<name>/calibration` is not the lens Blender used,
  - the optical-frame convention was applied twice, or not at all,

and every one of those quietly poisons any camera-to-3D association a labeler
builds on this data. The threshold is not to be relaxed.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from mcap.reader import make_reader

from sitegen.views import _matrix_from_quat
from sitegen.writer import generate

#: A return that lands within this many pixels of its own instance counts.
#: Cycles antialiases nothing in the id pass, but the LiDAR range noise still
#: moves a point a few centimetres along the ray, and at a silhouette a few
#: centimetres is the difference between the truck and the sky behind it.
EDGE_TOLERANCE_PX = 1

REQUIRED_AGREEMENT = 0.99

#: Two seconds is four camera timesteps at 2 Hz and sixteen renders, which is
#: about a minute of Cycles and tens of thousands of projected returns.
DURATION_S = 2.0


def _assets() -> Path:
    return Path(os.environ.get("SITEGEN_ASSETS", "assets"))


@pytest.fixture(scope="session")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from sitegen.cycles import missing_assets

    absent = missing_assets(_assets())
    if absent:
        pytest.skip(
            f"needs the CC0 render assets ({', '.join(absent)}); point "
            "SITEGEN_ASSETS at a directory holding them -- see docs/RENDERING.md"
        )
    out = tmp_path_factory.mktemp("scene") / "site.mcap"
    generate(
        out=out,
        seed=1,
        duration_s=DURATION_S,
        camera_hz=2.0,
        camera_assets=_assets(),
        camera_samples=8,
        work_dir=out.parent / "render",
    )
    return out


def _read(path: Path) -> dict:
    """Only what a consumer of the recording is given: clouds, masks, K, tf."""
    truth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    masks: dict[tuple[str, int], np.ndarray] = {}
    poses: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    calib: dict[str, CameraCalibration] = {}

    with open(path, "rb") as f:
        for _, channel, message in make_reader(f).iter_messages():
            topic = channel.topic
            if topic == "/ground_truth/points":
                cloud = PointCloud()
                cloud.ParseFromString(message.data)
                raw = np.frombuffer(cloud.data, dtype=np.uint8).reshape(-1, 16)
                xyz = raw[:, :12].copy().view(np.float32).astype(np.float64)
                instance = raw[:, 12:].copy().view(np.uint32).reshape(-1)
                truth[message.log_time] = (xyz, instance)
            elif topic.startswith("/ground_truth/camera_instances/"):
                image = CompressedImage()
                image.ParseFromString(message.data)
                masks[(topic.rsplit("/", 1)[1], message.log_time)] = np.array(
                    Image.open(io.BytesIO(image.data))
                )
            elif topic.endswith("/calibration"):
                c = CameraCalibration()
                c.ParseFromString(message.data)
                calib.setdefault(topic.split("/")[2], c)
            elif topic == "/tf":
                tf = FrameTransform()
                tf.ParseFromString(message.data)
                q, p = tf.rotation, tf.translation
                poses[(tf.child_frame_id, message.log_time)] = (
                    _matrix_from_quat(q.x, q.y, q.z, q.w),
                    np.array([p.x, p.y, p.z]),
                )
    return {"truth": truth, "masks": masks, "poses": poses, "calib": calib}


def _agreement(data: dict) -> tuple[int, int, dict[str, tuple[int, int]]]:
    hit = total = 0
    per_camera: dict[str, tuple[int, int]] = {}
    for (camera, ns), mask in sorted(data["masks"].items()):
        if ns not in data["truth"]:
            continue
        points, instance = data["truth"][ns]
        lidar_r, lidar_t = data["poses"][("lidar", ns)]
        cam_r, cam_t = data["poses"][(f"camera_{camera}", ns)]
        k = data["calib"][camera].K
        fx, cx, fy, cy = k[0], k[2], k[4], k[5]
        height, width = mask.shape

        world = points @ lidar_r.T + lidar_t
        local = (world - cam_t) @ cam_r  # world -> optical: +z forward, +y down
        forward = local[:, 2] > 0.05
        u = fx * local[forward, 0] / local[forward, 2] + cx
        v = fy * local[forward, 1] / local[forward, 2] + cy
        want = instance[forward]

        col, row = np.rint(u).astype(int), np.rint(v).astype(int)
        inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        col, row, want = col[inside], row[inside], want[inside]
        if not len(col):
            continue

        # A 1-pixel neighbourhood, which is the whole tolerance allowed.
        matched = np.zeros(len(col), dtype=bool)
        for dx in range(-EDGE_TOLERANCE_PX, EDGE_TOLERANCE_PX + 1):
            for dy in range(-EDGE_TOLERANCE_PX, EDGE_TOLERANCE_PX + 1):
                c = np.clip(col + dx, 0, width - 1)
                r = np.clip(row + dy, 0, height - 1)
                matched |= mask[r, c] == want
        got, seen = int(matched.sum()), len(col)
        hit += got
        total += seen
        before = per_camera.get(camera, (0, 0))
        per_camera[camera] = (before[0] + got, before[1] + seen)
    return hit, total, per_camera


def test_lidar_returns_and_camera_pixels_name_the_same_actor(recording: Path) -> None:
    data = _read(recording)
    assert data["masks"], "the recording has no instance masks"

    hit, total, per_camera = _agreement(data)
    assert total > 10_000, f"only {total} returns landed in a camera; too few to mean anything"

    rate = hit / total
    report = "  ".join(
        f"{name} {h / max(n, 1):.4f} ({n})" for name, (h, n) in sorted(per_camera.items())
    )
    print(f"\nsame-surface agreement {rate:.4f} over {total} returns\n  {report}")
    assert rate >= REQUIRED_AGREEMENT, (
        f"{rate:.4f} of {total} projected returns landed on their own instance; "
        "that is a pose, intrinsics or frame-convention bug between the two "
        f"sensors, not a threshold to lower. Per camera: {report}"
    )
