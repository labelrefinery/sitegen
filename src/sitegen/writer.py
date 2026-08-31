"""Render a scene to MCAP, with observable and ground-truth topics separated.

The split is the contract of this whole tool:

    /lidar/points        observable   the cloud, and nothing else
    /ego/joint_states    observable   proprioception -- free, exact, and the
                                      seed an unlabeled pipeline bootstraps from
    /tf                  observable   rig transforms
    /gnss                observable   ego global pose
    /terrain/heightmap   observable   published once at the start

    /ground_truth/actors      HELD OUT  per-part cuboids for every actor
    /ground_truth/points      HELD OUT  per-point instance ids

A labeler reads the first group. Only the scorer reads the second. Keeping
them in one file rather than two is deliberate -- the truth cannot drift from
the data it describes if they were written by the same pass.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.JointState_pb2 import JointState
from foxglove_schemas_protobuf.JointStates_pb2 import JointStates
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer

from .actors import HOUSE, SLEW_HEIGHT, sensor_pose
from .camera import CameraIntrinsics, camera_pose, render
from .geometry import Array, Box, quat_from_matrix
from .scene import Scene
from .sensors import Lidar, sweep

SITE_LAT, SITE_LON = 37.4419, -122.1430

#: Camera mast offset in the house frame, forward of and above the cab.
CAMERA_OFFSET = np.array([1.6, 0.0, HOUSE[2] + 0.4])

CLASS_COLORS: dict[str, tuple[float, float, float]] = {
    "excavator": (0.98, 0.75, 0.10),
    "haul_truck": (0.20, 0.60, 0.95),
    "worker": (0.95, 0.25, 0.30),
    "grade_stake": (0.70, 0.70, 0.75),
}


def _ts(ns: int) -> Timestamp:
    t = Timestamp()
    t.FromNanoseconds(ns)
    return t


def _identity_pose() -> Pose:
    return Pose(
        position=Vector3(x=0.0, y=0.0, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _xyzi_fields() -> list[PackedElementField]:
    f32 = PackedElementField.FLOAT32
    return [
        PackedElementField(name="x", offset=0, type=f32),
        PackedElementField(name="y", offset=4, type=f32),
        PackedElementField(name="z", offset=8, type=f32),
        PackedElementField(name="intensity", offset=12, type=f32),
    ]


def _xyzinstance_fields() -> list[PackedElementField]:
    f32 = PackedElementField.FLOAT32
    return [
        PackedElementField(name="x", offset=0, type=f32),
        PackedElementField(name="y", offset=4, type=f32),
        PackedElementField(name="z", offset=8, type=f32),
        PackedElementField(name="instance", offset=12, type=PackedElementField.UINT32),
    ]


def _cloud(ns: int, frame: str, points: Array, fourth: Array, fields: list[PackedElementField]) -> PointCloud:
    buf = np.empty((points.shape[0], 4), dtype=np.float32)
    buf[:, :3] = points.astype(np.float32)
    packed = buf.tobytes()
    if fields[3].type == PackedElementField.UINT32:
        view = np.frombuffer(bytearray(packed), dtype=np.uint32).reshape(-1, 4)
        view = view.copy()
        view[:, 3] = fourth.astype(np.uint32)
        packed = view.tobytes()
    else:
        buf[:, 3] = fourth.astype(np.float32)
        packed = buf.tobytes()
    return PointCloud(
        timestamp=_ts(ns),
        frame_id=frame,
        pose=_identity_pose(),
        point_stride=16,
        fields=fields,
        data=packed,
    )


def _cube(box: Box) -> CubePrimitive:
    qx, qy, qz, qw = quat_from_matrix(box.rotation)
    base = box.class_name.split(".")[0]
    r, g, b = CLASS_COLORS.get(base, (0.8, 0.8, 0.8))
    return CubePrimitive(
        pose=Pose(
            position=Vector3(x=box.center[0], y=box.center[1], z=box.center[2]),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        ),
        size=Vector3(
            x=2 * box.half_extents[0], y=2 * box.half_extents[1], z=2 * box.half_extents[2]
        ),
        color=Color(r=r, g=g, b=b, a=0.45),
    )


def generate(
    out: Path,
    seed: int = 1,
    duration_s: float = 60.0,
    rate_hz: float = 10.0,
    truth_points_hz: float = 2.0,
    difficulty: float = 1.0,
    azimuth_steps: int = 450,
    camera_hz: float = 0.0,
    camera_width: int = 960,
    camera_height: int = 540,
    camera_fov_deg: float = 78.0,
) -> dict[str, int]:
    """Write one scene. Returns per-topic message counts."""
    scene = Scene(seed=seed, duration_s=duration_s, difficulty=difficulty)
    lidar = Lidar(
        azimuth_steps=azimuth_steps,
        range_noise_per_m=0.0015 * difficulty,
        dropout_at_max_range=min(0.9, 0.35 * difficulty),
    )
    rng = np.random.default_rng(seed + 1000)
    counts: dict[str, int] = {}

    def bump(topic: str) -> None:
        counts[topic] = counts.get(topic, 0) + 1

    step_ns = int(1e9 / rate_hz)
    truth_every = max(1, int(round(rate_hz / truth_points_hz)))
    camera_every = max(1, int(round(rate_hz / camera_hz))) if camera_hz > 0 else 0
    intrinsics = CameraIntrinsics.from_fov(camera_width, camera_height, camera_fov_deg)
    frames = int(duration_s * rate_hz)
    base_ns = 1_700_000_000_000_000_000

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f, Writer(f) as w:
        # Terrain is static; publish it once so a consumer has it from frame 0.
        heights, pitch = scene.terrain.heightmap()
        w.write_message(
            topic="/terrain/heightmap",
            message=_cloud(
                base_ns,
                "map",
                np.stack(
                    [
                        *np.meshgrid(
                            np.linspace(-scene.terrain.extent, scene.terrain.extent, heights.shape[0]),
                            np.linspace(-scene.terrain.extent, scene.terrain.extent, heights.shape[0]),
                            indexing="xy",
                        ),
                        heights,
                    ],
                    axis=-1,
                ).reshape(-1, 3),
                heights.reshape(-1),
                _xyzi_fields(),
            ),
            log_time=base_ns,
            publish_time=base_ns,
        )
        bump("/terrain/heightmap")

        for i in range(frames):
            t = i / rate_hz
            ns = base_ns + i * step_ns
            state = scene.state_at(t)
            sensor_r, sensor_t = sensor_pose(state.ego)

            points, source = sweep(
                lidar, sensor_r, sensor_t, state.boxes, scene.terrain, rng, scene.dust, t
            )

            intensity = np.clip(1.0 - np.linalg.norm(points, axis=1) / lidar.max_range, 0.0, 1.0)
            w.write_message(
                topic="/lidar/points",
                message=_cloud(ns, "lidar", points, intensity, _xyzi_fields()),
                log_time=ns,
                publish_time=ns,
            )
            bump("/lidar/points")

            qx, qy, qz, qw = quat_from_matrix(sensor_r)
            w.write_message(
                topic="/tf",
                message=FrameTransform(
                    timestamp=_ts(ns),
                    parent_frame_id="map",
                    child_frame_id="lidar",
                    translation=Vector3(x=sensor_t[0], y=sensor_t[1], z=sensor_t[2]),
                    rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
                ),
                log_time=ns,
                publish_time=ns,
            )
            bump("/tf")

            w.write_message(
                topic="/ego/joint_states",
                message=JointStates(
                    timestamp=_ts(ns),
                    joints=[
                        JointState(name=name, position=value)
                        for name, value in state.ego.joints().items()
                    ],
                ),
                log_time=ns,
                publish_time=ns,
            )
            bump("/ego/joint_states")

            w.write_message(
                topic="/gnss",
                message=LocationFix(
                    timestamp=_ts(ns),
                    frame_id="map",
                    latitude=SITE_LAT + state.ego.y * 9e-6,
                    longitude=SITE_LON + state.ego.x * 1.1e-5,
                    altitude=12.0,
                ),
                log_time=ns,
                publish_time=ns,
            )
            bump("/gnss")

            # ---- held out ------------------------------------------------
            w.write_message(
                topic="/ground_truth/actors",
                message=SceneUpdate(
                    entities=[
                        SceneEntity(
                            timestamp=_ts(ns),
                            frame_id="map",
                            id=f"{box.instance_id}/{box.class_name}",
                            lifetime=Duration(nanos=int(step_ns * 1.5)),
                            metadata=[],
                            cubes=[_cube(box)],
                        )
                        for box in state.boxes
                    ]
                ),
                log_time=ns,
                publish_time=ns,
            )
            bump("/ground_truth/actors")

            if camera_every and i % camera_every == 0:
                house_r = sensor_r
                house_t = np.array([state.ego.x, state.ego.y, SLEW_HEIGHT])
                cam_r, cam_t = camera_pose(house_r, house_t, CAMERA_OFFSET)
                image, instances = render(
                    intrinsics, cam_r, cam_t, state.boxes, scene.terrain
                )
                buf = io.BytesIO()
                Image.fromarray(image).save(buf, format="JPEG", quality=85)
                w.write_message(
                    topic="/camera/front/image",
                    message=CompressedImage(
                        timestamp=_ts(ns), frame_id="camera", format="jpeg",
                        data=buf.getvalue(),
                    ),
                    log_time=ns, publish_time=ns,
                )
                bump("/camera/front/image")

                w.write_message(
                    topic="/camera/front/calibration",
                    message=CameraCalibration(
                        timestamp=_ts(ns), frame_id="camera",
                        width=intrinsics.width, height=intrinsics.height,
                        distortion_model="",
                        K=[intrinsics.fx, 0.0, intrinsics.cx,
                           0.0, intrinsics.fy, intrinsics.cy,
                           0.0, 0.0, 1.0],
                        P=[intrinsics.fx, 0.0, intrinsics.cx, 0.0,
                           0.0, intrinsics.fy, intrinsics.cy, 0.0,
                           0.0, 0.0, 1.0, 0.0],
                    ),
                    log_time=ns, publish_time=ns,
                )
                bump("/camera/front/calibration")

                # Held out: per-pixel instance ids, free from the raycaster.
                mask = io.BytesIO()
                Image.fromarray(instances).save(mask, format="PNG", optimize=True)
                w.write_message(
                    topic="/ground_truth/camera_instances",
                    message=CompressedImage(
                        timestamp=_ts(ns), frame_id="camera", format="png",
                        data=mask.getvalue(),
                    ),
                    log_time=ns, publish_time=ns,
                )
                bump("/ground_truth/camera_instances")

            if i % truth_every == 0 and points.shape[0]:
                instance = np.where(source < 0, 0, source + 1)
                w.write_message(
                    topic="/ground_truth/points",
                    message=_cloud(ns, "lidar", points, instance, _xyzinstance_fields()),
                    log_time=ns,
                    publish_time=ns,
                )
                bump("/ground_truth/points")

    return counts
