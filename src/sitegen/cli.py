"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import export
from . import score as scoring
from .writer import generate


def _cmd_generate(args: argparse.Namespace) -> None:
    counts = generate(
        out=args.out,
        seed=args.seed,
        duration_s=args.duration,
        rate_hz=args.rate,
        truth_points_hz=args.truth_points_hz,
        difficulty=args.difficulty,
        azimuth_steps=args.azimuth_steps,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    for topic, n in sorted(counts.items()):
        print(f"  {topic:26s} {n:6d} messages")


def _cmd_truth(args: argparse.Namespace) -> None:
    boxes = export.read_truth(args.mcap)
    if args.level == "object":
        boxes = export.merge_to_objects(boxes)
    rows = export.write_tracks_csv(boxes, args.out)
    print(f"wrote {args.out}: {rows} rows, level={args.level}")


def _cmd_ego(args: argparse.Namespace) -> None:
    print(f"wrote {args.out}: {export.write_ego_csv(args.mcap, args.out)} rows")


def _cmd_tf(args: argparse.Namespace) -> None:
    print(f"wrote {args.out}: {export.write_tf_csv(args.mcap, args.out)} rows")


def _cmd_joints(args: argparse.Namespace) -> None:
    print(f"wrote {args.out}: {export.write_joints_csv(args.mcap, args.out)} rows")


def _cmd_sweeps(args: argparse.Namespace) -> None:
    n = export.write_sweeps(args.mcap, args.out)
    print(f"wrote {n} sweeps to {args.out} (float32 x,y,z,intensity; see index.csv)")


def _cmd_score(args: argparse.Namespace) -> None:
    results = scoring.score(
        pred=scoring.read_csv(args.pred),
        truth=scoring.read_csv(args.truth),
        threshold_m=args.threshold,
        class_agnostic=not args.class_aware,
        exclude=tuple(args.exclude or ()),
    )
    print(scoring.render(results))
    if args.json:
        scoring.write_json(results, args.json)
        print(f"\nwrote {args.json}")


def main() -> None:
    p = argparse.ArgumentParser(prog="sitegen", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write one synthetic scene to an MCAP file")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--duration", type=float, default=60.0, help="seconds")
    g.add_argument("--rate", type=float, default=10.0, help="sensor Hz")
    g.add_argument(
        "--truth-points-hz",
        type=float,
        default=2.0,
        help="rate for per-point ground truth; boxes are always at --rate",
    )
    g.add_argument(
        "--difficulty",
        type=float,
        default=1.0,
        help="scales range noise, dropout and dust severity",
    )
    g.add_argument("--azimuth-steps", type=int, default=450)
    g.set_defaults(func=_cmd_generate)

    t = sub.add_parser("truth", help="export ground truth as a tracker CSV")
    t.add_argument("mcap", type=Path)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument(
        "--level",
        choices=("object", "part"),
        default="object",
        help="object merges an actor's parts into one enclosing box (default); "
        "part keeps them separate, for scoring an articulated labeler",
    )
    t.set_defaults(func=_cmd_truth)

    e = sub.add_parser("ego", help="export ego track as t,x,y for OfflinePoly --ego")
    e.add_argument("mcap", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.set_defaults(func=_cmd_ego)

    tf = sub.add_parser("tf", help="export sensor pose as t,x,y,z,qx,qy,qz,qw")
    tf.add_argument("mcap", type=Path)
    tf.add_argument("--out", type=Path, required=True)
    tf.set_defaults(func=_cmd_tf)

    j = sub.add_parser("joints", help="export ego proprioception as t,swing,boom,stick,bucket")
    j.add_argument("mcap", type=Path)
    j.add_argument("--out", type=Path, required=True)
    j.set_defaults(func=_cmd_joints)

    s = sub.add_parser("sweeps", help="export point clouds as raw float32 per frame")
    s.add_argument("mcap", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=_cmd_sweeps)

    c = sub.add_parser("score", help="score predicted tracks against the oracle")
    c.add_argument("pred", type=Path)
    c.add_argument("--truth", type=Path, required=True)
    c.add_argument("--threshold", type=float, default=scoring.DEFAULT_THRESHOLD_M)
    c.add_argument(
        "--class-aware",
        action="store_true",
        help="require the predicted class to match; off by default because a "
        "cold-start pipeline has no class names to be right or wrong about",
    )
    c.add_argument(
        "--exclude",
        nargs="*",
        metavar="CLASS",
        help="drop these truth classes before scoring. Use it for objects the "
        "sensor cannot resolve at all -- grade stakes are 50mm square and "
        "collect a couple of returns, so leaving them in makes recall a "
        "measure of the LiDAR rather than of the labeler",
    )
    c.add_argument("--json", type=Path, help="also write results here")
    c.set_defaults(func=_cmd_score)

    args = p.parse_args()
    args.func(args)
