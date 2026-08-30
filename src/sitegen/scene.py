"""The scripted work cycle.

A trench dig with truck loading, which is the most common thing an excavator
does and the one with the most interesting label failure modes: the machine
occludes itself, the truck is stationary for long stretches and then leaves,
and a worker crosses the swing radius while the house is rotating.

Timings come from a real cycle. A 20-tonne machine loading an articulated
hauler runs a dig-swing-dump-return loop in roughly fifteen seconds, and takes
five or six passes to fill the bed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actors import ExcavatorState, excavator_parts, stake_box, truck_parts, worker_box
from .geometry import Box
from .sensors import DustEvent
from .terrain import Stockpile, Terrain

CYCLE_S = 15.0
DIG_SWING = np.radians(-35.0)
DUMP_SWING = np.radians(75.0)
TRUCK_SPOT = (7.5, 9.0, np.radians(105.0))

BASE_YAW = np.radians(15.0)
TRAVEL_START_S = 26.0
TRAVEL_END_S = 34.0
TRAVEL_DISTANCE_M = 6.5
"""How far the machine tracks between stations.

An excavator digging a trench does not stay put. It cuts what it can reach,
then walks the undercarriage back along the trench line and cuts again --
roughly 0.8 m/s, house squared up over the tracks, boom tucked. Without this
the scene has a fixed sensor origin, which quietly removes three things worth
testing: the map never grows, occlusion behind the stockpile never resolves,
and there is no driven trajectory for a terrain labeler to calibrate against.
"""


def smoothstep(a: float, b: float, x: float) -> float:
    if b <= a:
        return 1.0
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


@dataclass
class SceneState:
    ego: ExcavatorState
    boxes: list[Box]
    ego_part_count: int
    """Boxes [0, ego_part_count) belong to the ego machine."""


@dataclass
class Scene:
    """One site, one work cycle, deterministic in `seed`."""

    seed: int = 1
    duration_s: float = 60.0
    difficulty: float = 1.0
    terrain: Terrain = field(default_factory=lambda: Terrain([]))
    dust: list[DustEvent] = field(default_factory=list)
    _stakes: list[Box] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.terrain = Terrain(
            [
                Stockpile(x=18.0, y=-11.0, height=3.4, material="soil"),
                Stockpile(x=-14.0, y=14.0, height=2.1, material="gravel"),
            ]
        )
        # Dust when the loaded truck pulls away, and again on the second haul.
        self.dust = [
            DustEvent(32.0, 39.0, 12.0, 16.0, 9.0, 0.55 * self.difficulty)
        ]
        self._stakes = [
            stake_box(-6.0 + 3.0 * i, 19.0 + float(rng.normal(0.0, 0.15)), i)
            for i in range(6)
        ]

    # -- actors ------------------------------------------------------------

    def base_at(self, t: float) -> tuple[float, float]:
        """Undercarriage position: two dig stations with a walk between."""
        u = smoothstep(TRAVEL_START_S, TRAVEL_END_S, t)
        d = TRAVEL_DISTANCE_M * u
        return (-d * np.cos(BASE_YAW), -d * np.sin(BASE_YAW))

    def travelling(self, t: float) -> float:
        """1.0 while the machine is walking, 0.0 while it is digging."""
        rise = smoothstep(TRAVEL_START_S - 1.0, TRAVEL_START_S + 1.5, t)
        fall = smoothstep(TRAVEL_END_S - 1.5, TRAVEL_END_S + 1.0, t)
        return rise - fall

    def ego_at(self, t: float) -> ExcavatorState:
        """Dig, swing loaded, dump, swing back -- once per CYCLE_S."""
        p = (t % CYCLE_S) / CYCLE_S
        if p < 0.27:  # dig
            u = smoothstep(0.0, 0.27, p)
            swing, boom, stick, bucket = (
                DIG_SWING,
                lerp(-0.15, -0.55, u),
                lerp(0.9, 0.25, u),
                lerp(0.2, -1.1, u),
            )
        elif p < 0.47:  # swing to the truck, boom rising
            u = smoothstep(0.27, 0.47, p)
            swing = lerp(DIG_SWING, DUMP_SWING, u)
            boom, stick, bucket = lerp(-0.55, -1.0, u), lerp(0.25, 0.55, u), -1.1
        elif p < 0.60:  # dump
            u = smoothstep(0.47, 0.60, p)
            swing, boom, stick = DUMP_SWING, -1.0, 0.55
            bucket = lerp(-1.1, 0.45, u)
        elif p < 0.85:  # swing back
            u = smoothstep(0.60, 0.85, p)
            swing = lerp(DUMP_SWING, DIG_SWING, u)
            boom, stick, bucket = lerp(-1.0, -0.15, u), lerp(0.55, 0.9, u), lerp(0.45, 0.2, u)
        else:  # settle over the cut
            swing, boom, stick, bucket = DIG_SWING, -0.15, 0.9, 0.2

        # Walking overrides the dig cycle: house squared up over the tracks,
        # boom tucked in. Nobody travels with the boom out.
        walk = self.travelling(t)
        if walk > 0.0:
            swing = lerp(swing, 0.0, walk)
            boom = lerp(boom, -0.9, walk)
            stick = lerp(stick, 1.35, walk)
            bucket = lerp(bucket, -0.6, walk)

        bx, by = self.base_at(t)
        return ExcavatorState(bx, by, BASE_YAW, swing, boom, stick, bucket)

    def truck_at(self, t: float) -> tuple[float, float, float] | None:
        """Backs in, is loaded, hauls off; a second truck takes its place."""
        cycles = [(4.0, 33.0, "truck_a"), (46.0, 95.0, "truck_b")]
        for arrive, depart, _ in cycles:
            if t < arrive - 6.0 or t > depart + 8.0:
                continue
            x, y, yaw = TRUCK_SPOT
            if arrive > TRAVEL_END_S:
                # Haulers reposition with the machine; otherwise the second
                # load would be a reach the operator would never accept.
                shift_x, shift_y = self.base_at(t)
                x, y = x + shift_x, y + shift_y
            if t < arrive:  # backing in along the haul road
                u = smoothstep(arrive - 6.0, arrive, t)
                return lerp(x + 22.0, x, u), lerp(y + 14.0, y, u), yaw
            if t > depart:  # pulling away
                u = smoothstep(depart, depart + 8.0, t)
                return lerp(x, x + 30.0, u), lerp(y, y + 20.0, u), yaw
            return x, y, yaw
        return None

    def workers_at(self, t: float) -> list[tuple[float, float, float]]:
        """One spotter holds station; one crosses the swing radius at ~t=25."""
        spotter = (13.5, 4.0 + 0.6 * float(np.sin(t * 0.4)), np.radians(200.0))
        u = smoothstep(18.0, 34.0, t)
        crosser = (lerp(-16.0, 9.0, u), lerp(10.0, -8.0, u), np.radians(-40.0))
        return [spotter, crosser]

    def state_at(self, t: float) -> SceneState:
        ego = self.ego_at(t)
        boxes = excavator_parts(ego, "ego")
        ego_parts = len(boxes)

        truck = self.truck_at(t)
        if truck is not None:
            name = "truck_a" if t < 40.0 else "truck_b"
            boxes.extend(truck_parts(truck[0], truck[1], truck[2], name))
        for i, (wx, wy, wyaw) in enumerate(self.workers_at(t)):
            boxes.append(worker_box(wx, wy, wyaw, f"worker_{i}"))
        boxes.extend(self._stakes)
        return SceneState(ego=ego, boxes=boxes, ego_part_count=ego_parts)
