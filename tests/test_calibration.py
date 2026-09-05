"""The sensor statistics GOOSE-Ex settled, pinned so they cannot drift back.

`docs/GOOSE-EX.md` measured sitegen's LiDAR against 2,164 labelled sweeps from
four Ousters on a Liebherr R924 and moved four things: the elevation band, the
maximum range, the ray budget behind `--density`, and intensity from a
function of range to a per-class albedo. Each of those is one constant, and
one constant is exactly the kind of thing that gets adjusted for an unrelated
reason and quietly un-calibrates the sensor.

So this asserts the *statistics*, not the constants: what fraction of a sweep
is the ego machine, where the median return lands, how many returns a worker
at 14 m collects at `--density real`, and whether a person is still the
brightest class in the cloud. The bands are wide enough that reordering an
operation or changing a mesh will not trip them, and narrow enough that
reverting any of the four calibration changes will.

The scene is walked directly rather than through an MCAP, because the numbers
are properties of the sweep and writing 90 MB to disk to read it back would
only make the test slower.
"""

from __future__ import annotations

import numpy as np
import pytest

from sitegen.actors import sensor_pose
from sitegen.raycast import caster
from sitegen.scene import Scene
from sitegen.sensors import Lidar, intensity, sweep

RATE_HZ = 10.0
DURATION_S = 60.0

#: Frames spread across the whole recording rather than taken from the front
#: of it. Both sweep-level statistics move with the dig cycle -- the boom is
#: overhead for part of every fifteen seconds and out over the cut for the
#: rest, and the ego share swings from 18% to well over 30% between the two --
#: so a sample from one phase of one cycle would pin the wrong number.
SPREAD = range(0, int(DURATION_S * RATE_HZ), 15)
DENSE_SPREAD = range(0, int(DURATION_S * RATE_HZ), 50)


def _walk(lidar: Lidar, frames: range = SPREAD) -> dict[str, list]:
    """Sweep the scene the way `writer.generate` does, keeping the truth."""
    scene = Scene(seed=1, duration_s=DURATION_S)
    rng = np.random.default_rng(1001)
    out: dict[str, list] = {"ego": [], "total": [], "median": [], "worker": [], "class": []}
    for i in frames:
        t = i / RATE_HZ
        state = scene.state_at(t)
        sensor_r, sensor_t = sensor_pose(state.ego)
        rays = caster(state.parts, state.boxes, scene.terrain, True)
        points, source = sweep(lidar, sensor_r, sensor_t, rays, rng, scene.dust, t)
        keys = [
            "ego" if b.instance_id == "ego" else b.class_name.split(".")[0]
            for b in state.boxes
        ]
        values = intensity(lidar, points, source, keys, rng)
        ranges = np.linalg.norm(points, axis=1)
        of_class = np.array(["terrain", *keys])[source + 1]

        out["total"].append(len(points))
        out["ego"].append(int((of_class == "ego").sum()))
        out["median"].append(float(np.median(ranges)))
        out["class"].append((of_class, values))
        # Points on the worker who holds station, counted per object and only
        # while they are in the 10-15 m bin the real data is densest in.
        for j, box in enumerate(state.boxes):
            if box.class_name != "worker":
                continue
            mask = source == j
            if mask.any() and 10.0 <= float(np.median(ranges[mask])) < 15.0:
                out["worker"].append(int(mask.sum()))
    return out


@pytest.fixture(scope="module")
def default() -> dict[str, list]:
    return _walk(Lidar(beams=32, azimuth_steps=450))


@pytest.fixture(scope="module")
def dense() -> dict[str, list]:
    """`--density real`: the ray budget that puts a worker within 3.5x."""
    return _walk(Lidar(beams=64, azimuth_steps=1440), frames=DENSE_SPREAD)


def test_ego_share_matches_the_real_machine(default: dict[str, list]) -> None:
    """Real ALICE sweeps are 28.0% ego. The -45..+15 deg band is what buys it;
    the old -25..+5 one gave 15%, which is outside this band by a factor."""
    share = sum(default["ego"]) / sum(default["total"])
    assert 0.24 <= share <= 0.31, share


def test_median_return_lands_near_the_machine(default: dict[str, list]) -> None:
    """Real is 4.5 m, with 89% of the sweep inside 10 m. sitegen reaches 8.6 m;
    before the elevation change it was 14.4 m, aimed at the horizon."""
    assert 7.0 <= float(np.mean(default["median"])) <= 10.5


def test_worker_at_14_m_is_within_a_factor_of_the_real_rig(
    dense: dict[str, list],
) -> None:
    """GOOSE-Ex's median is 141 returns on a person at 10-15 m. `--density
    real` gets 40 of those; the 32 x 450 default gets 6."""
    assert dense["worker"], "the spotter should stand in the 10-15 m bin"
    assert 25 <= float(np.median(dense["worker"])) <= 60


def test_a_person_is_the_brightest_thing_on_the_site(default: dict[str, list]) -> None:
    """The load-bearing property of per-class intensity: hi-vis PPE returns
    several times what soil does, which is what makes intensity a class signal
    rather than a second copy of range. Real medians are 27 against 8.5."""
    pooled: dict[str, list[np.ndarray]] = {}
    for of_class, values in default["class"]:
        for name in ("worker", "terrain", "ego"):
            mask = of_class == name
            if mask.any():
                pooled.setdefault(name, []).append(values[mask])
    medians = {
        k: float(np.median(np.concatenate(v)) * 255.0) for k, v in pooled.items()
    }
    assert medians["worker"] > 2.0 * medians["terrain"], medians
    assert medians["worker"] > 2.0 * medians["ego"], medians
    assert 15.0 <= medians["worker"] <= 40.0, medians
