#!/usr/bin/env python3
"""Join two `dino.py` runs against the ground-truth boxes and print the table.

A score on its own is not a detection. Grounding DINO will happily return
`worker 0.25` for a patch of gravel, so every score here has to land on the
actor it claims: a detection counts only if its box overlaps the pixels the
instance mask says belong to that actor. That mask comes from the same
raycaster that drew the baseline image, and the re-render puts the actor in the
same place, so one bounding box serves both columns.

    uv run python tools/probe/compare.py --views <dir> \
        --baseline baseline_scores.json --render blender_scores.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPORT_THRESHOLD = 0.35
"""What `reference.py` defaults to, and what the README's numbers were measured
at. Scores below it are reported too, in parentheses, because "never detected"
is a claim about a number and 0.00 and 0.31 are different kinds of miss."""

MIN_IOU = 0.10


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    overlap = (x1 - x0) * (y1 - y0)
    areas = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (areas - overlap)


def best(detections: list[dict], bbox: list[int], labels: tuple[str, ...]):
    """The highest-scoring detection of the right kind that lands on `bbox`."""
    hits = [
        d
        for d in detections
        if any(word in d["label"] for word in labels) and iou(d["box"], bbox) >= MIN_IOU
    ]
    return max(hits, key=lambda d: d["score"], default=None)


def cell(detection: dict | None) -> str:
    if detection is None:
        return "--"
    score = f"{detection['score']:.3f}"
    return score if detection["score"] >= REPORT_THRESHOLD else f"({score})"


def figures(manifest: list[dict], views: Path, renders: Path, out: Path) -> None:
    """One JPEG per view, baseline left and Cycles right, at half size.

    Small enough to live in the repo, which matters -- the numbers are the
    finding but nobody believes a table about images without the images.
    """
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    for view in manifest:
        left = Image.open(views / view["image"]).convert("RGB")
        right = Image.open(renders / f"{view['name']}.png").convert("RGB")
        size = (left.width // 2, left.height // 2)
        pair = Image.new("RGB", (size[0] * 2 + 4, size[1]), (255, 255, 255))
        pair.paste(left.resize(size, Image.LANCZOS), (0, 0))
        pair.paste(right.resize(size, Image.LANCZOS), (size[0] + 4, 0))
        pair.save(out / f"{view['name']}.jpg", quality=82, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--render", type=Path, required=True)
    ap.add_argument("--ablation", type=Path, help="scores for --worker-as-box")
    ap.add_argument("--renders", type=Path, help="dir of Cycles PNGs, for --figures")
    ap.add_argument(
        "--figures", type=Path, help="write side-by-side baseline|Cycles JPEGs here"
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    manifest = json.loads((args.views / "views.json").read_text())
    baseline = json.loads(args.baseline.read_text())["views"]
    rendered = json.loads(args.render.read_text())["views"]

    ablation = (
        json.loads(args.ablation.read_text())["views"] if args.ablation else None
    )
    ablation_head = " box in Cycles |" if ablation else ""
    ablation_rule = " ---: |" if ablation else ""
    rows = [
        "| view | worker px | range | baseline worker |" + ablation_head
        + " Cycles worker | label | baseline truck | Cycles truck |",
        "| --- | ---: | ---: | ---: |" + ablation_rule + " ---: | --- | ---: | ---: |",
    ]
    moved = 0
    workers = 0
    for view in manifest:
        name = view["name"]
        for instance in view["instances"]:
            if instance["class"] != "worker" or not instance["bbox"]:
                continue
            workers += 1
            bbox = instance["bbox"]
            before = best(baseline[name], bbox, ("worker", "person"))
            after = best(rendered[name], bbox, ("worker", "person"))
            if after is not None and after["score"] >= REPORT_THRESHOLD:
                moved += 1

            truck_before = truck_after = None
            trucks = [i for i in view["instances"] if i["class"] == "haul_truck"]
            if trucks and trucks[0]["bbox"]:
                truck_before = best(baseline[name], trucks[0]["bbox"], ("truck",))
                truck_after = best(rendered[name], trucks[0]["bbox"], ("truck",))

            px = f"{instance['pixels']}"
            actor = next(
                a
                for a in view["actors"]
                if a["instance"] == instance["instance"]
            )
            middle = (
                f" {cell(best(ablation[name], bbox, ('worker', 'person')))} |"
                if ablation
                else ""
            )
            rows.append(
                f"| {name} {instance['instance']} | {px} | {actor['range_m']:.1f} m | "
                f"{cell(before)} |{middle} {cell(after)} | "
                f"{after['label'] if after else '--'} | "
                f"{cell(truck_before)} | {cell(truck_after)} |"
            )

    rows.append("")
    rows.append(
        f"{moved} of {workers} workers detected at {REPORT_THRESHOLD:.2f} after "
        f"re-rendering; parenthesised scores are below that threshold."
    )
    if args.figures:
        figures(manifest, args.views, args.renders, args.figures)

    table = "\n".join(rows)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
