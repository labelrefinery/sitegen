"""Score predicted tracks against the held-out oracle.

Matching follows the nuScenes detection protocol: greedy assignment in
descending confidence order, against the nearest unmatched ground-truth box
within a **centre-distance** threshold rather than an IoU threshold. Distance
matching is the right call for sparse returns -- a truck at 45 m carrying eight
points will never reach an IoU threshold no matter how good the tracker is,
and an eval that reports zero there tells you nothing about whether the
tracker found the truck.

Reported per class, or class-agnostically, which is the default: a cold-start
pipeline has no class names to be right or wrong about, so demanding them
would score the wrong thing.

    P / R / F1   did it find the objects
    ATE          centre error over true positives, metres
    ASE          1 - IoU of the boxes once centred and aligned; pure size error
    AOE          heading error, radians, in [0, pi/2] after pi-symmetry folding
    IDS          identity switches: how often a ground-truth object's assigned
                 track id changed. This is the number a smoother improves and a
                 per-frame detector cannot.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLD_M = 2.0


@dataclass
class Row:
    track_id: str
    cls: str
    t: float
    x: float
    y: float
    z: float
    w: float
    l: float
    h: float
    theta: float
    conf: float


def read_csv(path: Path) -> list[Row]:
    rows: list[Row] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                Row(
                    track_id=r["track_id"],
                    cls=r["cls"],
                    t=float(r["t"]),
                    x=float(r["x"]),
                    y=float(r["y"]),
                    z=float(r["z"]),
                    w=float(r["w"]),
                    l=float(r["l"]),
                    h=float(r["h"]),
                    theta=float(r["theta"]),
                    conf=float(r.get("conf", 1.0) or 1.0),
                )
            )
    return rows


def aligned_iou(a: Row, b: Row) -> float:
    """3-D IoU with centres and headings aligned -- size agreement only."""
    inter = min(a.w, b.w) * min(a.l, b.l) * min(a.h, b.h)
    union = a.w * a.l * a.h + b.w * b.l * b.h - inter
    return float(inter / union) if union > 0 else 0.0


def yaw_error(a: float, b: float) -> float:
    """Heading difference, folded by pi: a box reversed end-for-end is the
    same box, and penalising that is measuring the labeller's coin flip."""
    d = abs((a - b + np.pi / 2) % np.pi - np.pi / 2)
    return float(d)


@dataclass
class Accumulator:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ate: list[float] = field(default_factory=list)
    ase: list[float] = field(default_factory=list)
    aoe: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float | int]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "ate_m": round(mean(self.ate), 4),
            "ase": round(mean(self.ase), 4),
            "aoe_rad": round(mean(self.aoe), 4),
        }


def score(
    pred: list[Row],
    truth: list[Row],
    threshold_m: float = DEFAULT_THRESHOLD_M,
    class_agnostic: bool = True,
) -> dict[str, dict[str, float | int]]:
    def key(r: Row) -> str:
        return "all" if class_agnostic else r.cls

    frames = sorted({round(r.t, 4) for r in truth} | {round(r.t, 4) for r in pred})
    pred_by_t: dict[float, list[Row]] = defaultdict(list)
    truth_by_t: dict[float, list[Row]] = defaultdict(list)
    for r in pred:
        pred_by_t[round(r.t, 4)].append(r)
    for r in truth:
        truth_by_t[round(r.t, 4)].append(r)

    acc: dict[str, Accumulator] = defaultdict(Accumulator)
    # instance -> the predicted track id it was matched to, last time we saw it
    last_assignment: dict[str, str] = {}
    switches = 0

    for t in frames:
        ps = sorted(pred_by_t.get(t, []), key=lambda r: -r.conf)
        gs = list(truth_by_t.get(t, []))
        taken: set[int] = set()

        for p in ps:
            best_i, best_d = -1, threshold_m
            for i, g in enumerate(gs):
                if i in taken:
                    continue
                if not class_agnostic and g.cls != p.cls:
                    continue
                d = float(np.hypot(p.x - g.x, p.y - g.y))
                if d < best_d:
                    best_i, best_d = i, d
            if best_i < 0:
                acc[key(p)].fp += 1
                continue
            g = gs[best_i]
            taken.add(best_i)
            a = acc[key(g)]
            a.tp += 1
            a.ate.append(best_d)
            a.ase.append(1.0 - aligned_iou(p, g))
            a.aoe.append(yaw_error(p.theta, g.theta))
            previous = last_assignment.get(g.track_id)
            if previous is not None and previous != p.track_id:
                switches += 1
            last_assignment[g.track_id] = p.track_id

        for i, g in enumerate(gs):
            if i not in taken:
                acc[key(g)].fn += 1

    out = {k: v.summary() for k, v in sorted(acc.items())}
    total = Accumulator()
    for v in acc.values():
        total.tp += v.tp
        total.fp += v.fp
        total.fn += v.fn
        total.ate += v.ate
        total.ase += v.ase
        total.aoe += v.aoe
    summary = total.summary()
    summary["id_switches"] = switches
    out["OVERALL"] = summary
    return out


def render(results: dict[str, dict[str, float | int]]) -> str:
    cols = ["tp", "fp", "fn", "precision", "recall", "f1", "ate_m", "ase", "aoe_rad"]
    lines = [f"{'class':<22}" + "".join(f"{c:>11}" for c in cols)]
    lines.append("-" * len(lines[0]))
    for name, row in results.items():
        lines.append(f"{name:<22}" + "".join(f"{row.get(c, ''):>11}" for c in cols))
    if "id_switches" in results.get("OVERALL", {}):
        lines.append(f"\nid switches: {results['OVERALL']['id_switches']}")
    return "\n".join(lines)


def write_json(results: dict[str, dict[str, float | int]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
