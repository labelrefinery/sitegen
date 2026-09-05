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

from .actors import SLEW_HEIGHT, sensor_pose
from .camera import SURROUND_RIG, CameraIntrinsics, camera_pose, render
from .cycles import Shot, missing_assets, placements, run as run_cycles, write_job
from .geometry import Array, Box, quat_from_matrix
from .raycast import caster
from .scene import Scene
from .sensors import Lidar, intensity as lidar_intensity, sweep

SITE_LAT, SITE_LON = 37.4419, -122.1430

#: Cameras ride the house, so the rig sweeps the site as the machine swings.

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


def _albedo_key(box: Box) -> str:
    """Which `sensors.REMISSION` row a return off this box is drawn from.

    The ego machine gets its own, because GOOSE-Ex measures `ego_vehicle` at
    half the remission of the `heavy_machinery` parked beside it -- the sensor
    sees its own house at a grazing angle from half a metre away, which is a
    different measurement from seeing a machine across the site.
    """
    if box.instance_id == "ego":
        return "ego"
    return box.class_name.split(".")[0]


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


def _shots(
    scene: Scene, frames: int, rate_hz: float, camera_every: int
) -> list[Shot]:
    """Every camera frame the run will contain, before any of it is rendered.

    Cycles is a batch: one Blender process builds the site once and walks the
    shots, because starting an interpreter and loading an HDRI per image would
    cost more than the images do. So the scene has to be enumerated up front.
    Re-deriving the states in the main loop rather than holding them is
    deliberate -- `Scene` is a pure function of t, and two callers agreeing
    because they both asked it is better than two callers agreeing because one
    of them cached.
    """
    shots: list[Shot] = []
    for i in range(frames):
        if not camera_every or i % camera_every:
            continue
        state = scene.state_at(i / rate_hz)
        house_r, _ = sensor_pose(state.ego)
        house_t = np.array([state.ego.x, state.ego.y, SLEW_HEIGHT])
        for cam_name, offset, pan in SURROUND_RIG:
            cam_r, cam_t = camera_pose(house_r, house_t, np.array(offset), pan)
            shots.append(
                Shot(
                    name=f"{cam_name}_{i:06d}",
                    rotation=[[float(v) for v in row] for row in cam_r],
                    translation=[float(v) for v in cam_t],
                    objects=placements(
                        state.parts, list(range(1, len(state.parts) + 1))
                    ),
                )
            )
    return shots


def generate(
    out: Path,
    seed: int = 1,
    duration_s: float = 60.0,
    rate_hz: float = 10.0,
    truth_points_hz: float = 2.0,
    difficulty: float = 1.0,
    beams: int = 32,
    azimuth_steps: int = 450,
    sensor: str = "calibrated",
    camera_hz: float = 0.0,
    camera_width: int = 960,
    camera_height: int = 540,
    camera_fov_deg: float = 78.0,
    mesh_actors: bool = True,
    camera_renderer: str = "cycles",
    camera_assets: Path | None = None,
    camera_samples: int = 48,
    work_dir: Path | None = None,
) -> dict[str, int]:
    """Write one scene. Returns per-topic message counts."""
    scene = Scene(
        seed=seed,
        duration_s=duration_s,
        difficulty=difficulty,
        mesh_actors=mesh_actors,
    )
    lidar = Lidar(
        beams=beams,
        azimuth_steps=azimuth_steps,
        range_noise_per_m=0.0015 * difficulty,
        dropout_far=min(0.9, 0.35 * difficulty),
    )
    if sensor == "legacy":
        lidar = lidar.legacy()
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

    # Cycles first, so the MCAP is written in one ordered pass afterwards.
    renders: Path | None = None
    if camera_every and camera_renderer == "cycles":
        if not mesh_actors:
            raise SystemExit(
                "--camera-renderer cycles renders the actor meshes; "
                "pass --camera-renderer raycast to keep --actors boxes"
            )
        assets = Path(camera_assets or "assets")
        absent = missing_assets(assets)
        if absent:
            raise SystemExit(
                f"missing render assets in {assets}: {', '.join(absent)}\n"
                "They are CC0 downloads, listed with their URLs in "
                "docs/RENDERING.md and src/sitegen/assets/CREDITS."
            )
        work = Path(work_dir or out.with_suffix(".render"))
        renders = work / "frames"
        shots = _shots(scene, frames, rate_hz, camera_every)
        print(f"rendering {len(shots)} camera frames in Cycles...", flush=True)
        run_cycles(
            write_job(
                work,
                shots,
                intrinsics=intrinsics,
                assets=assets,
                stockpiles=[
                    (p.x, p.y, p.height) for p in scene.terrain.stockpiles
                ],
                samples=camera_samples,
            ),
            renders,
        )

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

            rays = caster(state.parts, state.boxes, scene.terrain, mesh_actors)
            points, source = sweep(
                lidar, sensor_r, sensor_t, rays, rng, scene.dust, t
            )

            # Intensity is drawn after the sweep and from the same generator,
            # so `--sensor legacy` -- which does not draw at all -- leaves the
            # stream where it was and reproduces the old files byte for byte.
            intensity = lidar_intensity(
                lidar,
                points,
                source,
                [_albedo_key(box) for box in state.boxes],
                rng,
            )
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
                for cam_name, offset, pan in SURROUND_RIG:
                    cam_r, cam_t = camera_pose(
                        house_r, house_t, np.array(offset), pan
                    )
                    if renders is not None:
                        name = f"{cam_name}_{i:06d}"
                        image = np.asarray(
                            Image.open(renders / f"{name}.png").convert("RGB")
                        )
                        instances = np.load(renders / f"{name}.ids.npy")
                    else:
                        image, instances = render(
                            intrinsics,
                            cam_r,
                            cam_t,
                            rays,
                            [b.class_name for b in state.boxes],
                        )
                    buf = io.BytesIO()
                    Image.fromarray(image).save(buf, format="JPEG", quality=85)
                    w.write_message(
                        topic=f"/camera/{cam_name}/image",
                        message=CompressedImage(
                            timestamp=_ts(ns), frame_id=f"camera_{cam_name}",
                            format="jpeg", data=buf.getvalue(),
                        ),
                        log_time=ns, publish_time=ns,
                    )
                    bump(f"/camera/{cam_name}/image")

                    w.write_message(
                        topic=f"/camera/{cam_name}/calibration",
                        message=CameraCalibration(
                            timestamp=_ts(ns), frame_id=f"camera_{cam_name}",
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
                    bump(f"/camera/{cam_name}/calibration")

                    qx, qy, qz, qw = quat_from_matrix(cam_r)
                    w.write_message(
                        topic="/tf",
                        message=FrameTransform(
                            timestamp=_ts(ns), parent_frame_id="map",
                            child_frame_id=f"camera_{cam_name}",
                            translation=Vector3(x=cam_t[0], y=cam_t[1], z=cam_t[2]),
                            rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
                        ),
                        log_time=ns, publish_time=ns,
                    )
                    bump("/tf")

                    mask = io.BytesIO()
                    Image.fromarray(instances).save(mask, format="PNG", optimize=True)
                    w.write_message(
                        topic=f"/ground_truth/camera_instances/{cam_name}",
                        message=CompressedImage(
                            timestamp=_ts(ns), frame_id=f"camera_{cam_name}",
                            format="png", data=mask.getvalue(),
                        ),
                        log_time=ns, publish_time=ns,
                    )
                    bump(f"/ground_truth/camera_instances/{cam_name}")

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
