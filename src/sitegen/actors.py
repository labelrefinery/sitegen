"""Site actors, as posed rigid links.

The excavator is articulated on purpose. A single cuboid around a machine at
full reach is mostly air, and around a curled-up one it clips the bucket, so
the label geometry has to follow the kinematic chain:

    base pose (x, y, yaw) -> swing (yaw) -> boom (pitch) -> stick (pitch)
                                                         -> bucket (pitch)

Four articulated degrees of freedom on top of the body pose. Every part is
emitted as its own link, which is what lets a scorer ask whether a labeler got
the *bucket* right -- the part that will actually hit something -- rather than
only whether it drew a plausible blob around the machine.

Each link is a mesh from `meshes.py` plus the frame it sits in. The cuboid the
oracle publishes is derived from that: the tight box of the posed triangles.
Kinematics did not change when the geometry did -- the same joint angles place
the same frames, and only what hangs off them is different.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Array, Part, part_at, rot_y, rot_z

# A 20-tonne class machine, the most common size on a job site. These are the
# envelopes the box renderer used; the meshes are built to fill them.
UNDERCARRIAGE = np.array([4.5, 2.8, 0.9])
HOUSE = np.array([3.6, 2.7, 2.2])
BOOM_LENGTH = 5.7
STICK_LENGTH = 2.9
BUCKET = np.array([1.2, 1.0, 1.1])
LINK_THICKNESS = np.array([0.0, 0.6, 0.7])

SLEW_HEIGHT = 1.1
BOOM_FOOT = np.array([1.2, 0.0, 0.6])

TRUCK_CAB = np.array([2.4, 2.6, 2.9])
TRUCK_BED = np.array([6.4, 2.9, 2.2])
WORKER = np.array([0.6, 0.45, 1.75])
STAKE = np.array([0.05, 0.05, 1.0])


@dataclass
class ExcavatorState:
    """Body pose plus the four joint angles, in radians."""

    x: float
    y: float
    yaw: float
    swing: float
    boom: float
    stick: float
    bucket: float

    def joints(self) -> dict[str, float]:
        """What proprioception publishes. Free, exact, and available offline."""
        return {
            "swing": self.swing,
            "boom": self.boom,
            "stick": self.stick,
            "bucket": self.bucket,
        }


def _link(
    parent_r: Array,
    parent_t: Array,
    length: float,
    name: str,
    instance_id: str,
) -> tuple[Part, Array]:
    """A link along its own +x, plus the world position of its far pin."""
    half = np.array([length / 2.0, LINK_THICKNESS[1] / 2.0, LINK_THICKNESS[2] / 2.0])
    part = part_at(
        parent_r,
        parent_t,
        np.array([length / 2.0, 0.0, 0.0]),
        half,
        name,
        name,
        instance_id,
    )
    tip = parent_r @ np.array([length, 0.0, 0.0]) + parent_t
    return part, tip


def excavator_parts(state: ExcavatorState, instance_id: str) -> list[Part]:
    """Forward kinematics down the chain, one link per rigid part."""
    base_r = rot_z(state.yaw)
    base_t = np.array([state.x, state.y, 0.0])

    parts = [
        part_at(
            base_r,
            base_t,
            np.array([0.0, 0.0, UNDERCARRIAGE[2] / 2.0]),
            UNDERCARRIAGE / 2.0,
            "excavator.undercarriage",
            "excavator.undercarriage",
            instance_id,
        )
    ]

    # Everything above the slew ring turns with the house.
    house_r = base_r @ rot_z(state.swing)
    house_t = base_t + np.array([0.0, 0.0, SLEW_HEIGHT])
    parts.append(
        part_at(
            house_r,
            house_t,
            np.array([-0.3, 0.0, HOUSE[2] / 2.0]),
            HOUSE / 2.0,
            "excavator.house",
            "excavator.house",
            instance_id,
        )
    )

    boom_r = house_r @ rot_y(state.boom)
    boom_t = house_r @ BOOM_FOOT + house_t
    boom, boom_tip = _link(boom_r, boom_t, BOOM_LENGTH, "excavator.boom", instance_id)
    parts.append(boom)

    stick_r = boom_r @ rot_y(state.stick)
    stick, stick_tip = _link(
        stick_r, boom_tip, STICK_LENGTH, "excavator.stick", instance_id
    )
    parts.append(stick)

    bucket_r = stick_r @ rot_y(state.bucket)
    parts.append(
        part_at(
            bucket_r,
            stick_tip,
            np.array([BUCKET[0] / 2.0, 0.0, 0.0]),
            BUCKET / 2.0,
            "excavator.bucket",
            "excavator.bucket",
            instance_id,
        )
    )
    return parts


def sensor_pose(state: ExcavatorState) -> tuple[Array, Array]:
    """LiDAR pose: cab roof, so it swings with the house.

    Mounting it on the house rather than on a static mast is what makes the
    scene worth simulating -- the sensor's own boom sweeps through the field of
    view, self-occlusion changes with swing angle, and the ego's exact pose is
    knowable from proprioception alone.
    """
    house_r = rot_z(state.yaw) @ rot_z(state.swing)
    house_t = np.array([state.x, state.y, SLEW_HEIGHT])
    # On a short mast at the front edge of the cab roof. Sitting it in the
    # middle of the roof, which is the obvious placement, sends most of the
    # downward beams straight into the ego's own house -- two thirds of the
    # cloud, measured. Real machines mast it forward for the same reason.
    return house_r, house_r @ np.array([1.3, 0.0, HOUSE[2] + 1.0]) + house_t


def truck_parts(x: float, y: float, yaw: float, instance_id: str) -> list[Part]:
    """The two units of an articulated hauler, tractor first."""
    r = rot_z(yaw)
    t = np.array([x, y, 0.0])
    return [
        part_at(
            r,
            t,
            np.array([3.2, 0.0, 0.6 + TRUCK_CAB[2] / 2.0]),
            TRUCK_CAB / 2.0,
            "haul_truck.cab",
            "haul_truck.cab",
            instance_id,
        ),
        part_at(
            r,
            t,
            np.array([-1.2, 0.0, 1.0 + TRUCK_BED[2] / 2.0]),
            TRUCK_BED / 2.0,
            "haul_truck.bed",
            "haul_truck.bed",
            instance_id,
        ),
    ]


def worker_part(x: float, y: float, yaw: float, instance_id: str, mesh: str) -> Part:
    """A person, in one of the two baked poses.

    Which pose is not decoration: the spotter holds station and the other
    worker is walking across the swing radius, and a detector that sees a
    standing figure and a striding one is being asked about two silhouettes
    rather than the same one twice.
    """
    return part_at(
        rot_z(yaw),
        np.array([x, y, 0.0]),
        np.array([0.0, 0.0, WORKER[2] / 2.0]),
        WORKER / 2.0,
        mesh,
        "worker",
        instance_id,
    )


def stake_part(x: float, y: float, index: int) -> Part:
    return part_at(
        np.eye(3),
        np.array([x, y, 0.0]),
        np.array([0.0, 0.0, STAKE[2] / 2.0]),
        STAKE / 2.0,
        "grade_stake",
        "grade_stake",
        f"stake_{index}",
    )
