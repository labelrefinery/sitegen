#!/usr/bin/env python3
"""Run grounding-dino-tiny over a directory of views and write the scores as JSON.

The single-image companion in GroundingDino.mojo (`tools/reference.py`) is the
authority on how the model is invoked; this loads the same checkpoint with the
same processor defaults and only adds a loop, so the two agree image for image.

It reports at two thresholds. 0.35 is the one the baseline was measured at and
the one the table quotes. 0.10 is there because "never detected" is a claim
about a number, and a score of 0.31 and a score of 0.00 are very different kinds
of miss -- one says the render is close, the other says nothing registered.

    uv run --with torch --with transformers --with pillow \
        python tools/probe/dino.py --dir <views> --out <scores.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPT = "excavator . haul truck . worker . person ."
REPORT_THRESHOLD = 0.35
FLOOR_THRESHOLD = 0.10
MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True, help="directory of images")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    model.eval()

    results = {}
    for path in sorted(args.dir.glob(args.glob)):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=args.prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Everything above the floor comes back; the caller filters to 0.35.
        found = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=FLOOR_THRESHOLD,
            text_threshold=0.25,
            target_sizes=[(image.height, image.width)],
        )[0]
        detections = [
            {
                "label": label,
                "score": round(float(score), 4),
                "box": [round(float(v), 1) for v in box.tolist()],
            }
            for score, box, label in zip(
                found["scores"], found["boxes"], found["text_labels"]
            )
        ]
        detections.sort(key=lambda d: -d["score"])
        results[path.stem] = detections
        shown = ", ".join(
            f"{d['label']} {d['score']:.3f}" for d in detections
        )
        print(f"{path.stem:26s} {shown or '(nothing above %.2f)' % FLOOR_THRESHOLD}")

    args.out.write_text(
        json.dumps({"prompt": args.prompt, "model": MODEL_ID, "views": results}, indent=2)
        + "\n"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
