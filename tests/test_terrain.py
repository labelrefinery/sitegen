"""Terrain that changes, and the arithmetic that says nothing was invented.

A deforming heightfield is easy to make look right and easy to get wrong in a
way no picture shows: soil that quietly appears, a slump that loses a cubic
metre at a window edge, a pile that stands at 60 degrees because the relaxation
gave up. So the properties asserted here are the ones a rendering cannot check.

  - **Mass.** Cut equals what is in the bucket, plus what is in the body, plus
    what has been driven off site, plus what the stockpiles gained. Separately,
    the ground's own volume moves by exactly what was deposited minus what was
    cut. Both to 1e-9 m3, which is float round-off over a 40,401-node grid and
    not a modelling tolerance: nothing in `heightfield.py` creates or destroys
    material, so any real leak shows up thousands of times over that bar.
  - **Repose.** After every edit no two neighbouring nodes differ by more than
    one cell of run at 34 degrees. That is what makes a cut wall slump and a
    pile stay conical, and a relaxation that ran out of iterations would fail
    it rather than look slightly wrong.
  - **Something actually happened.** A recording in which the terrain never
    moved would pass both of the above.

The simulation is walked directly rather than through an MCAP wherever it can
be, because these are properties of the material balance and writing 90 MB to
disk to read it back would only make the test slower. The two that *are* about
the recording -- the republish cadence and the oracle topic -- do go through
one, at a ray budget small enough that the sensor is not what is being timed.
"""

from __future__ import annotations

import numpy as np
import pytest
from foxglove_schemas_protobuf.JointStates_pb2 import JointStates
from mcap.reader import make_reader

from sitegen.earthworks import (
    BED_CAPACITY_M3,
    BUCKET_CAPACITY_M3,
    TERRAIN_COMMIT_HZ,
)
from sitegen.heightfield import BedLoad, Heightfield, ground_field
from sitegen.scene import Scene
from sitegen.terrain import REPOSE_DEG
from sitegen.writer import generate

#: Round-off, not slack. The measured residual over a 120 s run is 1e-13 m3.
CLOSURE_M3 = 1e-9

#: A recording long enough to contain a haul-off, a second hauler and a bucket
#: tipped when there is nothing standing at the spot -- which is the only way
#: the stockpile branch of the balance is exercised at all.
LONG_S = 120.0


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene(seed=1, duration_s=LONG_S)


@pytest.fixture(scope="module")
def short() -> Scene:
    return Scene(seed=1, duration_s=60.0)


def test_the_sampled_ground_is_already_at_the_angle_of_repose() -> None:
    """The cones were drawn at 34 degrees, so the grid starts relaxed.

    This is the reason the first frame of a recording is not a landslide, and
    it is a property of the sampling rather than a coincidence: a cone of slope
    tan(34) sampled on a square grid has a rise between neighbours of at most
    `pitch * tan(34)` along either axis, which is exactly the limit.
    """
    field = ground_field(Scene(seed=1, duration_s=1.0).terrain)
    assert field.slope_excess() < 1e-12
    before = field.z.copy()
    field.relax()
    assert np.abs(field.z - before).max() < 1e-12


def test_every_cubic_metre_is_accounted_for(scene: Scene) -> None:
    """cut == bucket + bed + hauled + what the stockpiles gained."""
    timeline = scene.earthworks
    assert timeline is not None
    start = timeline.volumes[0]
    worst = 0.0
    for volume in timeline.volumes:
        piles = sum(volume.stockpiles) - sum(start.stockpiles)
        residual = volume.cut - (
            volume.bucket + volume.bed + volume.hauled + piles
        )
        worst = max(worst, abs(residual))
    print(f"\nworst balance residual {worst:.3e} m3 over {len(timeline.volumes)} steps")
    assert worst < CLOSURE_M3


def test_the_ground_moves_by_exactly_what_left_and_arrived(scene: Scene) -> None:
    """The other half of the balance, and the one a survey could check.

    `net` is measured off the grid itself -- sum of elevations times cell area
    -- while `cut` and the stockpile totals are accumulated by the operations.
    They agree only if the relaxation moves soil rather than rounding it away,
    which is the failure mode a slope limiter written the obvious way has.
    """
    timeline = scene.earthworks
    assert timeline is not None
    start = timeline.volumes[0]
    worst = 0.0
    for volume in timeline.volumes:
        piles = sum(volume.stockpiles) - sum(start.stockpiles)
        worst = max(worst, abs(volume.net - (piles - volume.cut)))
    assert worst < CLOSURE_M3


def test_the_stockpile_branch_is_exercised(scene: Scene) -> None:
    """A bucket tipped with no body under it has to end up somewhere.

    Without this the balance above would close trivially, because in a 60 s
    recording every bucket goes into a hauler and the piles never move.
    """
    timeline = scene.earthworks
    assert timeline is not None
    gained = sum(timeline.volumes[-1].stockpiles) - sum(timeline.volumes[0].stockpiles)
    assert gained > 0.01, gained


def test_nothing_ends_up_steeper_than_the_angle_of_repose(scene: Scene) -> None:
    timeline = scene.earthworks
    assert timeline is not None
    limit = np.tan(np.radians(REPOSE_DEG))
    for surface in timeline.surfaces:
        excess = surface.field.slope_excess()
        assert excess < 1e-6, (
            f"a face stands {excess / surface.field.pitch + limit:.3f} in run, "
            f"against a repose limit of {limit:.3f}"
        )


def test_the_ground_actually_changes_over_a_recording(short: Scene) -> None:
    """A default 60 s recording moves about two heaped buckets of soil.

    Not the five or six the cycle description implies, and the reason is worth
    stating rather than tuning away. The scripted trajectory drags the cutting
    edge through the same 1.5 m of face every fifteen seconds without advancing
    the cut, so after the first bite each pass recovers only what slumped back
    in -- 0.36 to 0.45 m3 rather than a heaped 1.2. The one full bite in the
    recording is the first one at the *second* dig station, at t = 42 s, which
    takes 1.03 m3 out of ground nothing had touched. Digging a bucketful every
    cycle would mean a dig stroke that advances the face, which is a change to
    the published joint trajectory and not to the terrain model.
    """
    timeline = short.earthworks
    assert timeline is not None
    first, last = timeline.surfaces[0].field, timeline.surfaces[-1].field
    moved = float(np.abs(last.z - first.z).sum()) * first.pitch**2
    removed = float(np.clip(first.z - last.z, 0.0, None).sum()) * first.pitch**2
    print(f"\n{removed:.2f} m3 out of the ground, {moved:.2f} m3 of surface moved")
    assert BUCKET_CAPACITY_M3 <= removed <= 5.0 * BUCKET_CAPACITY_M3
    assert timeline.volumes[-1].cut == pytest.approx(2.13, abs=0.5)

    # Two dig stations, two bowls: the machine walks 6.5 m at t = 26-34 s and
    # cuts again where it stops, so the change is not one hole getting deeper.
    lowered = (first.z - last.z) > 0.05
    xs = first.axes()[0][np.any(lowered, axis=1)]
    assert xs.max() - xs.min() > 4.0, "only one dig station left a mark"


def test_a_full_body_sends_the_rest_to_the_pile() -> None:
    """The overflow branch, which a 120 s recording never reaches.

    Six passes fill the body; the seventh has nowhere to go. That the surplus
    is *returned* rather than silently absorbed is what keeps the balance
    closed when the operator is loading faster than the trucks arrive.
    """
    bed = BedLoad()
    spilled = [bed.add(BUCKET_CAPACITY_M3, BED_CAPACITY_M3) for _ in range(7)]
    assert spilled[:5] == [0.0] * 5
    assert bed.volume() == pytest.approx(BED_CAPACITY_M3, abs=1e-9)
    assert sum(spilled) == pytest.approx(
        7 * BUCKET_CAPACITY_M3 - BED_CAPACITY_M3, abs=1e-9
    )
    assert bed.empty() == pytest.approx(BED_CAPACITY_M3, abs=1e-9)
    assert bed.volume() == 0.0


def test_a_deposit_relaxes_without_losing_any_of_itself() -> None:
    """A pile dropped on flat ground keeps its volume and finds its angle."""
    field = Heightfield(z=np.zeros((81, 81)), x0=-10.0, y0=-10.0, pitch=0.25)
    field.deposit(0.0, 0.0, 12.0, radius=1.0)
    assert field.volume() == pytest.approx(12.0, abs=1e-9)
    assert field.slope_excess() < 1e-6
    # A cone of 12 m3 at 34 degrees stands 1.85 m high; the grid is coarse
    # enough to round the apex off, so this is a band rather than a value.
    peak = float(field.z.max())
    assert 1.5 < peak < 2.0, peak


def test_static_terrain_moves_nothing() -> None:
    """`--terrain static` is the old scene, with no simulation behind it."""
    scene = Scene(seed=1, duration_s=20.0, deforming=False)
    assert scene.earthworks is None
    state = scene.state_at(19.0)
    assert state.volumes is None
    assert state.ground is scene.terrain
    assert all(part.class_name != "haul_truck.load" for part in state.parts)


def test_the_recording_republishes_the_ground_and_holds_out_the_volumes(
    tmp_path,
) -> None:
    """The contract change, checked on a real file.

    `/terrain/heightmap` used to be one message at t0. It still is one message
    at t0 plus one per change, never faster than the commit rate, so a consumer
    that reads the first and stops is where it always was and one that follows
    the topic sees the site as it stands.
    """
    out = tmp_path / "site.mcap"
    counts = generate(
        out=out, seed=1, duration_s=20.0, rate_hz=2.0, beams=8, azimuth_steps=60
    )
    assert counts["/terrain/heightmap"] > 1
    assert counts["/ground_truth/volumes"] == 20

    heightmaps: list[int] = []
    volumes: list[JointStates] = []
    with open(out, "rb") as f:
        for _, channel, message in make_reader(f).iter_messages():
            if channel.topic == "/terrain/heightmap":
                heightmaps.append(message.log_time)
            elif channel.topic == "/ground_truth/volumes":
                record = JointStates()
                record.ParseFromString(message.data)
                volumes.append(record)

    base = heightmaps[0]
    gaps = np.diff(heightmaps) / 1e9
    assert gaps.min() >= 1.0 / TERRAIN_COMMIT_HZ - 1e-6, gaps
    assert heightmaps[0] == base

    names = [j.name for j in volumes[0].joints]
    assert names == [
        "cut", "bucket", "bed", "hauled", "stockpile_0", "stockpile_1", "net",
    ]
    cut = [j.position for record in volumes for j in record.joints if j.name == "cut"]
    assert cut == sorted(cut), "cumulative cut went backwards"
    assert cut[-1] > cut[0]
