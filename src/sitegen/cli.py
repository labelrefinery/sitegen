"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .writer import generate


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

    args = p.parse_args()
    counts = generate(
        out=args.out,
        seed=args.seed,
        duration_s=args.duration,
        rate_hz=args.rate,
        truth_points_hz=args.truth_points_hz,
        difficulty=args.difficulty,
        azimuth_steps=args.azimuth_steps,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB)")
    for topic, n in sorted(counts.items()):
        print(f"  {topic:26s} {n:6d} messages")
