"""Experiment: cases of the Qwen Turbo evaluation (outfits or event CG), drawn again with FLUX 3 Image Klein.

Prompt, references, size and seed are each case's own (tmp/expression-exp/turbo-eval/jobs.json),
written for Klein: images by ordinal, and a white background where the case wanted transparency
Klein cannot draw. The standard and Turbo pictures already made are shown beside it; nothing is
written there.

    services/worker/runtimes/flux3_klein/.venv/Scripts/python.exe -X utf8 scripts/flux3_klein/turbo_cases.py --group cg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.flux3_klein import klein

EVALUATION = klein.ROOT / "tmp/expression-exp/turbo-eval"  # or another folder laid out like it
# The outfit cases compared before; every event CG case is used.
OUTFITS = ("601c03da", "8bf180fb", "67dd0b22", "bcd0dc85", "034b89ca")


def outfit_sheet(cases: list[dict], out: Path) -> None:
    """A column per case: its portrait, the outfit picture it was given if any, then the
    standard, Turbo and Klein pictures."""
    rows = [("source", "source"), ("outfit", "outfit reference"), ("standard", "40 steps"),
            ("turbo", "Turbo 8"), ("klein", "Klein")]
    cells = [[(case["sources"][1] if key == "outfit" and len(case["sources"]) > 1 else case.get(key),
               f"{case['name']} {name}" if key != "outfit" or len(case["sources"]) > 1 else f"{case['name']} no outfit image")
              for case in cases] for key, name in rows]
    grid(out, cells, 440, width=330)


def cg_sheet(cases: list[dict], out: Path) -> None:
    """A row per case: the standard, Turbo and Klein pictures from the left, so each model is one
    column (every CG is 3:2), then the references it was given. A variation of an existing CG is
    left out: its first reference is that CG, not a new scene."""
    rows = []
    for case in (case for case in cases if not case["variant"]):
        size = f"{case['size'][0]}x{case['size'][1]}"
        rows.append([(case[key], f"{case['name']} {name} {size}")
                     for key, name in (("standard", "40 steps"), ("turbo", "Turbo 8"), ("klein", "Klein"))]
                    + [(image, f"ref {number}") for number, image in enumerate(case["sources"], 1)])
    grid(out, rows, 480)


def grid(out: Path, rows: list[list[tuple[Image.Image | None, str]]], height: int, width: int | None = None) -> None:
    """Rows of labelled pictures at one height; each is as wide as its shape, or ``width`` for all."""
    label = 18
    font = ImageFont.load_default(14)

    def cell_width(image):
        return width or (round(height * image.width / image.height) if image is not None else height // 2)

    widths = [sum(cell_width(image) for image, _ in cells) for cells in rows]
    canvas = Image.new("RGB", (max(widths), (height + label) * len(rows)), "white")
    pen = ImageDraw.Draw(canvas)
    for row, cells in enumerate(rows):
        x, y = 0, row * (height + label)
        for image, text in cells:
            room = cell_width(image)
            if image is not None:
                image = image.convert("RGBA")
                flat = Image.alpha_composite(Image.new("RGBA", image.size, (200, 200, 200, 255)), image).convert("RGB")
                flat.thumbnail((room, height), Image.Resampling.LANCZOS)
                canvas.paste(flat, (x + (room - flat.width) // 2, y + label))
            pen.text((x + 4, y + 2), text, fill="black", font=font)
            x += room
    canvas.save(out / "sheet.jpg", quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description="FLUX 3 Image Klein on the Qwen Turbo evaluation cases")
    parser.add_argument("--group", choices=("outfit", "cg"), default="outfit")
    parser.add_argument("--evaluation", type=Path, default=EVALUATION)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sheet-only", action="store_true", help="redraw the sheet from the saved pictures")
    args = parser.parse_args()
    out = args.out or klein.ROOT / f"outputs/image-edit-tests/klein-{args.group}s-20261016"
    out.mkdir(parents=True, exist_ok=args.sheet_only)
    jobs = [job for job in json.loads((args.evaluation / "jobs.json").read_text(encoding="utf-8"))
            if job["group"] == args.group and (args.group != "outfit" or job["id"] in OUTFITS)]
    if args.group == "outfit":
        jobs.sort(key=lambda job: OUTFITS.index(job["id"]))
    cases = []
    for job in jobs:
        references = [Image.open(args.evaluation / path) for path in job["references"]]
        cases.append({"name": job["id"], "seed": job["seed"], "size": (job["width"], job["height"]),
                      "references": [klein.on_white(image) for image in references], "sources": references,
                      "source": references[0], "variant": job.get("variant"),
                      "standard": Image.open(args.evaluation / job["baseline"]),
                      "turbo": Image.open(args.evaluation / "out" / job["id"] / "image.png"),
                      "request": klein.from_qwen_prompt(job["prompt"], white=job["transparent"])})
    if args.sheet_only:
        for case in cases:
            case["klein"] = Image.open(out / f"{case['name']}-klein.png")
        (outfit_sheet if args.group == "outfit" else cg_sheet)(cases, out)
        return 0
    klein.write_captions(cases)
    pipe = klein.load_pipeline()
    for case in cases:
        case["klein"] = klein.draw(pipe, case, *case["size"])
        case["klein"].save(out / f"{case['name']}-klein.png")
    (out / "run.json").write_text(json.dumps({"peak_allocated_gib": klein.peak_gib(), "cases": [
        {key: case[key] for key in ("name", "seed", "request", "caption", "upsample_seconds", "draw_seconds")}
        for case in cases]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (outfit_sheet if args.group == "outfit" else cg_sheet)(cases, out)
    print(f"done: {out / 'sheet.jpg'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
