"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import export, overlay, views
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
        camera_hz=args.camera_hz,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        mesh_actors=args.actors == "meshes",
        camera_renderer=args.camera_renderer,
        camera_assets=args.camera_assets,
        camera_samples=args.camera_samples,
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


def _cmd_cameras(args: argparse.Namespace) -> None:
    every = views.read_views(args.mcap)
    chosen = views.select(every, count=args.views)
    path = views.write_views(chosen, args.out)
    print(f"wrote {path}: {len(chosen)} of {len(every)} views")
    for v in chosen:
        seen = ", ".join(
            f"{instance} {n}px"
            for cls in ("worker", "haul_truck")
            for instance, n in v.instances(cls)
        )
        print(f"  {v.camera:12s} t={v.t:5.1f}s  {seen or 'nothing'}")


def _cmd_overlay(args: argparse.Namespace) -> None:
    epoch = (
        overlay.scene_epoch_ns(args.scene) if args.scene else overlay.DEFAULT_EPOCH_NS
    )
    counts = overlay.write_overlay(
        args.predictions, args.out, epoch_ns=epoch, show_ids=not args.no_ids
    )
    size_kb = args.out.stat().st_size / 1e3
    print(f"wrote {args.out} ({size_kb:.0f} KB)")
    for topic, n in sorted(counts.items()):
        print(f"  {topic:32s} {n:5d} messages")
    print("\nOpen it together with the scene -- Foxglove merges local files into")
    print("one timeline:  foxglove site.mcap " + str(args.out))


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
    g.add_argument(
        "--actors",
        choices=("meshes", "boxes"),
        default="meshes",
        help="collision and label geometry. meshes is what both sensors see; "
        "boxes is the oriented-cuboid renderer this started as, kept so the "
        "measurements taken before the meshes existed stay reproducible",
    )
    g.add_argument(
        "--camera-hz",
        type=float,
        default=0.0,
        help="render the four-camera rig at this rate (0 = off)",
    )
    g.add_argument(
        "--camera-renderer",
        choices=("cycles", "raycast"),
        default="cycles",
        help="cycles renders the actor meshes in Blender under an HDRI and "
        "needs --camera-assets; raycast is the flat-shaded path, instant and "
        "dependency-free, and the one every pre-mesh number was measured on",
    )
    g.add_argument(
        "--camera-assets",
        type=Path,
        help="directory holding the CC0 HDRI and gravel textures Cycles needs; "
        "see docs/RENDERING.md for the three URLs",
    )
    g.add_argument(
        "--camera-samples",
        type=int,
        default=48,
        help="Cycles samples per pixel for the image; the instance-id pass is "
        "always one sample, because an averaged object index is not one",
    )
    g.add_argument("--camera-width", type=int, default=960)
    g.add_argument("--camera-height", type=int, default=540)
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

    cam = sub.add_parser(
        "cameras",
        help="export camera views with intrinsics, extrinsics and visible actors",
    )
    cam.add_argument("mcap", type=Path)
    cam.add_argument("--out", type=Path, required=True, help="directory to write")
    cam.add_argument(
        "--views",
        type=int,
        default=10,
        help="how many views to select: every one with a visible worker first, "
        "then the clearest views of the truck as a control",
    )
    cam.set_defaults(func=_cmd_cameras)

    o = sub.add_parser(
        "overlay",
        help="write predicted labels to their own MCAP, to view beside the scene",
    )
    o.add_argument("predictions", type=Path, nargs="+", help="tracker-schema CSVs")
    o.add_argument("--out", type=Path, required=True)
    o.add_argument(
        "--scene",
        type=Path,
        help="read t=0 from this scene file; otherwise the default epoch is assumed",
    )
    o.add_argument("--no-ids", action="store_true", help="omit track-id labels")
    o.set_defaults(func=_cmd_overlay)

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
