"""sitegen's asserted sensor against a real one, on the same axes.

The README claims "a worker at 14 m gets about 20 points; a truck at 45 m gets
a handful". Those numbers were chosen to make clustering hard at the right
distance, not measured. This puts them next to a Liebherr R924 carrying four
Ouster scanners on a working site, and reports the one curve that decides
whether an auto-labeler has anything to work with: **median points on a target
against range**, for people and for vehicles.

Both sides reduce to the same thing -- a list of `goose.Observation`, one per
instance per frame -- so the comparison is a fair one even though the two
pipelines could hardly be more different. sitegen knows exactly which box each
return came from because it fired the ray; GOOSE-Ex knows because somebody
labelled 5,000 clouds by hand.

Usage:

    uv run python compare.py \\
        --goose ../../../data/goose-ex/gooseEx_3d_val \\
        --scene site.mcap \\
        --plot points-on-target.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import goose
from goose import Bin, Observation, Ontology

# sitegen's class names, mapped onto the two comparison groups. The ego machine
# is deliberately excluded: its returns sit at 2-8 m and would swamp the near
# bins with something no labeler has to *find* -- it is always there, and
# proprioception gives it away for free.
SITEGEN_GROUP = {
    "worker": "person",
    "haul_truck": "vehicle",
    "excavator": "vehicle",
}


@dataclass
class Summary:
    """What one dataset looks like, in the terms the comparison needs."""

    label: str
    frames: int
    points_per_frame: float
    ego_fraction: float
    """Share of returns that are the machine the sensor is bolted to."""
    range_median: float
    range_p99: float
    range_max: float
    observations: list[Observation]


# ---------------------------------------------------------------------------
# The real sensor
# ---------------------------------------------------------------------------


# Buckets for the range histogram. The real sensor turns out to spend almost
# all of itself inside 10 m, so the near bins are the fine ones.
HIST_EDGES: tuple[float, ...] = (0, 2, 5, 10, 15, 20, 30, 40, 60, 120)


@dataclass
class Profile:
    """The single-sensor report: what the sweep is made of."""

    total_points: int
    per_class: dict[str, int]
    range_hist: dict[tuple[float, float], int]
    remission: dict[str, tuple[int, float, float, float]]
    """class -> (n, mean, median, p95) of raw remission."""


def summarise_goose(
    root: Path, platform: str, limit: int | None
) -> tuple[Summary, Profile]:
    onto = Ontology.load(root / "goose_label_mapping.csv")
    paths = goose.find_frames(root, platform)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no {platform} frames under {root}")

    ego_keys = onto.keys_for(goose.SITEGEN_EQUIVALENT["ego"])
    obs: list[Observation] = []
    histogram: dict[str, int] = {}
    hist: dict[tuple[float, float], int] = {}
    # Remission is accumulated as (n, sum) plus a reservoir of values, because
    # a median and a p95 cannot be merged across frames from summaries alone.
    samples: dict[str, list[np.ndarray]] = {}
    total = ego = 0
    medians: list[float] = []
    p99s: list[float] = []
    maxima: list[float] = []

    for frame in goose.iter_frames(paths):
        obs.extend(goose.observations(frame, onto))
        keys, counts = np.unique(frame.semantic, return_counts=True)
        for key, count in zip(keys, counts):
            name = onto.name_of.get(int(key), f"unknown_{key}")
            histogram[name] = histogram.get(name, 0) + int(count)
            # Subsample: 5,000 frames of every return would not fit in memory
            # and the quantiles are stable long before that.
            values = frame.remission[frame.semantic == key]
            if len(values) > 2000:
                values = values[:: len(values) // 2000]
            samples.setdefault(name, []).append(values)
        total += len(frame)
        ego += int(np.isin(frame.semantic, list(ego_keys)).sum())
        r = frame.range_m
        binned = np.histogram(r, bins=HIST_EDGES)[0]
        for (lo, hi), n in zip(zip(HIST_EDGES[:-1], HIST_EDGES[1:]), binned):
            hist[(lo, hi)] = hist.get((lo, hi), 0) + int(n)
        medians.append(float(np.median(r)))
        p99s.append(float(np.percentile(r, 99)))
        maxima.append(float(r.max()))

    remission = {}
    for name, chunks in samples.items():
        values = np.concatenate(chunks)
        remission[name] = (
            histogram[name],
            float(values.mean()),
            float(np.median(values)),
            float(np.percentile(values, 95)),
        )

    return (
        Summary(
            label=f"GOOSE-Ex {platform}",
            frames=len(paths),
            points_per_frame=total / len(paths),
            ego_fraction=ego / total,
            range_median=float(np.mean(medians)),
            range_p99=float(np.mean(p99s)),
            range_max=float(np.max(maxima)),
            observations=obs,
        ),
        Profile(
            total_points=total,
            per_class=histogram,
            range_hist=hist,
            remission=remission,
        ),
    )


# ---------------------------------------------------------------------------
# sitegen
# ---------------------------------------------------------------------------


def summarise_sitegen(scene: Path) -> tuple[Summary, dict[str, int]]:
    """Read the held-out truth topics and reduce them to the same shape.

    `/ground_truth/points` carries a per-point `instance` field which is the
    index into that frame's box list, plus one (zero means terrain). The box
    list is exactly the entity list published on `/ground_truth/actors` at the
    same timestamp, whose ids are `"<instance>/<class>"` -- so the two topics
    together give every point a class *and* an object identity, which is what
    the per-instance count needs.
    """
    from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
    from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
    from mcap.reader import make_reader

    actors: dict[int, list[str]] = {}
    clouds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    sweep_sizes: list[int] = []

    with scene.open("rb") as fh:
        topics = ["/ground_truth/points", "/ground_truth/actors", "/lidar/points"]
        for _, channel, message in make_reader(fh).iter_messages(topics=topics):
            if channel.topic == "/ground_truth/actors":
                update = SceneUpdate()
                update.ParseFromString(message.data)
                actors[message.log_time] = [e.id for e in update.entities]
                continue
            cloud = PointCloud()
            cloud.ParseFromString(message.data)
            raw = np.frombuffer(cloud.data, dtype=np.uint8).reshape(-1, cloud.point_stride)
            if channel.topic == "/lidar/points":
                sweep_sizes.append(len(raw))
                continue
            # x y z float32 then a uint32 instance id, 16 bytes a point.
            xyz = raw[:, 0:12].copy().view(np.float32).reshape(-1, 3)
            instance = raw[:, 12:16].copy().view(np.uint32).ravel()
            clouds[message.log_time] = (xyz, instance)

    obs: list[Observation] = []
    histogram: dict[str, int] = {}
    total = ego = 0
    medians: list[float] = []
    p99s: list[float] = []
    maxima: list[float] = []

    for stamp, (xyz, instance) in sorted(clouds.items()):
        ids = actors.get(stamp)
        if ids is None:
            continue  # truth points are published slower than the boxes
        ranges = np.linalg.norm(xyz, axis=1)
        total += len(xyz)
        medians.append(float(np.median(ranges)))
        p99s.append(float(np.percentile(ranges, 99)))
        maxima.append(float(ranges.max()))

        for value in np.unique(instance):
            mask = instance == value
            if value == 0:
                histogram["terrain"] = histogram.get("terrain", 0) + int(mask.sum())
                continue
            entity = ids[int(value) - 1]
            actor, _, part = entity.partition("/")
            # The ego machine is its own bucket. Counting its boom under
            # `excavator` would claim sitegen has excavators in the scene to
            # find, when the only one present is the one under the sensor.
            class_name = "ego" if actor == "ego" else part.split(".")[0]
            histogram[class_name] = histogram.get(class_name, 0) + int(mask.sum())
            if actor == "ego":
                ego += int(mask.sum())

        # Points on target are counted per *object*, not per articulated part:
        # a labeler has to find the machine, and whether its boom and house are
        # separate boxes is a question that comes later.
        by_actor: dict[tuple[str, str], np.ndarray] = {}
        for value in np.unique(instance[instance > 0]):
            entity = ids[int(value) - 1]
            actor, _, part = entity.partition("/")
            if actor == "ego":
                continue
            group = SITEGEN_GROUP.get(part.split(".")[0])
            if group is None:
                continue
            key = (actor, group)
            mask = instance == value
            by_actor[key] = mask if key not in by_actor else (by_actor[key] | mask)
        for (_, group), mask in by_actor.items():
            obs.append(
                Observation(
                    group=group,
                    range_m=float(np.median(ranges[mask])),
                    points=int(mask.sum()),
                )
            )

    return (
        Summary(
            label="sitegen",
            frames=len(clouds),
            # The truth cloud is a sampled copy of the sweep, so quote the
            # sweep's own size -- that is what a labeler actually receives.
            points_per_frame=float(np.mean(sweep_sizes)) if sweep_sizes else 0.0,
            ego_fraction=ego / total if total else 0.0,
            range_median=float(np.mean(medians)),
            range_p99=float(np.mean(p99s)),
            range_max=float(np.max(maxima)),
            observations=obs,
        ),
        histogram,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_overview(real: Summary, synth: Summary) -> str:
    lines = [
        "## Sweep-level",
        "",
        f"| | {real.label} | {synth.label} | ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows = [
        ("labelled frames", real.frames, synth.frames, "{:,.0f}"),
        ("returns per sweep", real.points_per_frame, synth.points_per_frame, "{:,.0f}"),
        ("ego machine share", real.ego_fraction * 100, synth.ego_fraction * 100, "{:.1f}%"),
        ("median return range (m)", real.range_median, synth.range_median, "{:.1f}"),
        ("p99 return range (m)", real.range_p99, synth.range_p99, "{:.1f}"),
        ("max return range (m)", real.range_max, synth.range_max, "{:.1f}"),
    ]
    for name, a, b, fmt in rows:
        ratio = f"{b / a:.2f}x" if a else "--"
        lines.append(f"| {name} | {fmt.format(a)} | {fmt.format(b)} | {ratio} |")
    return "\n".join(lines)


def render_curve(real: Summary, synth: Summary, group: str) -> str:
    real_bins = {(b.lo, b.hi): b for b in goose.bin_observations(real.observations, group)}
    synth_bins = {(b.lo, b.hi): b for b in goose.bin_observations(synth.observations, group)}
    lines = [
        f"## Points on target -- {group}",
        "",
        "| range | GOOSE-Ex median (IQR) | n | sitegen median (IQR) | n | sitegen / real |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lo, hi in zip(goose.RANGE_BINS[:-1], goose.RANGE_BINS[1:]):
        a: Bin | None = real_bins.get((lo, hi))
        b: Bin | None = synth_bins.get((lo, hi))
        if a is None and b is None:
            continue
        def cell(x: Bin | None) -> tuple[str, str]:
            if x is None:
                return "--", "--"
            return f"{x.median:,.0f} ({x.p25:,.0f}-{x.p75:,.0f})", f"{x.n}"
        am, an = cell(a)
        bm, bn = cell(b)
        ratio = "--"
        if a is not None and b is not None and a.median > 0:
            ratio = f"{b.median / a.median:.2f}x"
        lines.append(f"| {lo:.0f}-{hi:.0f} m | {am} | {an} | {bm} | {bn} | {ratio} |")
    return "\n".join(lines)


def render_classes(real_hist: dict[str, int], synth_hist: dict[str, int]) -> str:
    """Class shares, with GOOSE's finer ontology folded onto sitegen's four."""
    real_total = sum(real_hist.values())
    synth_total = sum(synth_hist.values())
    lines = [
        "## Class share of returns",
        "",
        "| sitegen class | GOOSE-Ex classes | GOOSE-Ex | sitegen |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, equivalents in goose.SITEGEN_EQUIVALENT.items():
        real = sum(real_hist.get(e, 0) for e in equivalents)
        synth = synth_hist.get(name, 0)
        lines.append(
            f"| {name} | {', '.join(equivalents)} "
            f"| {100 * real / real_total:.3f}% | {100 * synth / synth_total:.3f}% |"
        )
    return "\n".join(lines)


def render_profile(profile: Profile, top: int = 18) -> str:
    """The single-sensor report: returns, classes, range, intensity."""
    total = profile.total_points
    lines = [
        "## GOOSE-Ex sensor profile",
        "",
        f"Total returns read: **{total:,}**.",
        "",
        "### Range histogram (all returns, from the machine base frame)",
        "",
        "| bin | returns | share | cumulative |",
        "| --- | ---: | ---: | ---: |",
    ]
    running = 0
    for (lo, hi), n in sorted(profile.range_hist.items()):
        running += n
        lines.append(
            f"| {lo:.0f}-{hi:.0f} m | {n:,} | {100 * n / total:.2f}% "
            f"| {100 * running / total:.2f}% |"
        )
    lines += [
        "",
        f"### Returns and remission by class (top {top})",
        "",
        "| class | returns | share | remission mean | median | p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = sorted(profile.per_class.items(), key=lambda kv: -kv[1])[:top]
    for name, n in ranked:
        _, mean, median, p95 = profile.remission[name]
        lines.append(
            f"| {name} | {n:,} | {100 * n / total:.3f}% "
            f"| {mean:.1f} | {median:.1f} | {p95:.1f} |"
        )
    return "\n".join(lines)


def plot(real: Summary, synth: Summary, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, group in zip(axes, ("person", "vehicle")):
        for summary, colour, marker in (
            (real, "#1b6ca8", "o"),
            (synth, "#d1495b", "s"),
        ):
            bins = goose.bin_observations(summary.observations, group)
            if not bins:
                continue
            x = [(b.lo + b.hi) / 2 for b in bins]
            y = [b.median for b in bins]
            ax.plot(x, y, marker=marker, color=colour, label=summary.label, lw=1.8)
            ax.fill_between(
                x, [b.p25 for b in bins], [b.p75 for b in bins], color=colour, alpha=0.15
            )
        # Ten points is roughly where a Euclidean clusterer stops producing a
        # box you would accept, so it is the line that matters.
        ax.axhline(10, color="#888", ls=":", lw=1)
        ax.text(52, 11, "10 points", color="#888", fontsize=8, ha="right")
        ax.set_yscale("log")
        ax.set_xlabel("range (m)")
        ax.set_title(group)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("median points on target")
    axes[0].legend(frameon=False)
    fig.suptitle("Points on target vs range: GOOSE-Ex (real) vs sitegen (analytic)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goose", type=Path, required=True, help="extracted split dir")
    ap.add_argument("--platform", default="alice", help="alice (excavator) or spot")
    ap.add_argument("--scene", type=Path, required=True, help="sitegen .mcap")
    ap.add_argument("--limit", type=int, default=None, help="cap GOOSE frames")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the report here too")
    args = ap.parse_args()

    real, profile = summarise_goose(args.goose, args.platform, args.limit)
    synth, synth_hist = summarise_sitegen(args.scene)

    report = "\n\n".join(
        [
            render_profile(profile),
            render_overview(real, synth),
            render_classes(profile.per_class, synth_hist),
            render_curve(real, synth, "person"),
            render_curve(real, synth, "vehicle"),
        ]
    )
    print(report)
    if args.out:
        args.out.write_text(report + "\n")
    if args.plot:
        plot(real, synth, args.plot)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
