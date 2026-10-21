"""Experiment: expression pictures with FLUX 3 Image Klein, beside the Qwen Turbo ones already made.

The portraits and the Qwen-Image-2.1-Turbo expression pictures come from finished jobs in the
worker cache; nothing is written there. Klein draws no alpha, so the edit keeps the base
portrait's alpha and, as in production, only the face is put back onto the portrait.

    services/worker/runtimes/flux3_klein/.venv/Scripts/python.exe -X utf8 scripts/flux3_klein/expressions.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.flux3_klein import klein

ROOT = klein.ROOT
JOBS = ROOT / "services/worker/cache/m2/jobs"
FACES = ROOT / "services/worker/cache/m2/faces"
EXPRESSIONS = ("smile", "angry", "surprised")
# Copied from services/worker/generation/outfit_image.py (EXPRESSION_VERSION 2).
INSTRUCTIONS = {
    "smile": "Change only the facial expression to a gentle, closed-mouth smile.",
    "angry": "Change only the facial expression to anger: furrowed brows, a sharp glare and a tight mouth.",
    "surprised": "Change only the facial expression to surprise: raised brows, wide open eyes and a slightly "
                 "open mouth.",
}
KEEP = ("Keep everything else exactly as it is in the first image: the same character identity, face, "
        "hairstyle, hair and eye colors, outfit, body proportions, illustration style, standing pose, framing "
        "and position.")
# Another work's portraits from the first Qwen Turbo comparison: two young women, a young man and
# a bearded older man, each edited with one seed. Face boxes from that run's detector (face-detect.json).
VARIED_DIR = ROOT / "tmp/expression-exp"
VARIED = ("oboro", "valerius", "ren", "c-191b47755b09c2cdb920bd5e")
VARIED_FACES = {"oboro": [347, 92, 444, 204], "valerius": [342, 39, 426, 146], "ren": [332, 52, 427, 152],
                "c-191b47755b09c2cdb920bd5e": [324, 72, 440, 185]}
VARIED_SEED = 20261012
# Room around the face, as in services/worker/generation/portrait_face.py.
MARGIN = {"side": 0.12, "top": 0.05, "bottom": 0.08}


def qwen_jobs() -> dict[str, dict[str, Path]]:
    """Per base portrait (by hash), its finished Qwen expression job of each kind."""
    found: dict[str, dict[str, Path]] = {}
    for request in sorted(JOBS.glob("*/*/job-request.json"), key=lambda path: path.stat().st_mtime):
        try:
            payload = json.loads(request.read_text(encoding="utf-8"))["payload"]
        except (OSError, ValueError, KeyError):
            continue
        expression = payload.get("expression") or {}
        if expression.get("id") in EXPRESSIONS and (request.parent / "result.zip").is_file():
            # The first one made is the one shown beside it (later shy re-runs are not compared).
            found.setdefault(payload["reference_image"]["sha256"], {}).setdefault(expression["id"], request.parent)
    return {digest: jobs for digest, jobs in found.items() if set(jobs) == set(EXPRESSIONS)}


def face_box(digest: str) -> list[float]:
    (result,) = json.loads((FACES / f"{digest}-v1.json").read_text(encoding="utf-8"))["results"]
    return result["faces"][0]["box"]


def area(face, size):
    left, top, right, bottom = face
    width, height = right - left, bottom - top
    box = (left - MARGIN["side"] * width, top - MARGIN["top"] * height,
           right + MARGIN["side"] * width, bottom + MARGIN["bottom"] * height)
    return (max(0, round(box[0])), max(0, round(box[1])), min(size[0], round(box[2])), min(size[1], round(box[3])))


def composite(source: Image.Image, edit: Image.Image, box) -> Image.Image:
    """As portrait_face.composite: only the face area is taken from the edit, its border blended."""
    size = box[2] - box[0]
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=round(size * 0.2), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1, round(size * 0.06))))
    mixed = Image.composite(edit.convert("RGBa"), source.convert("RGBa"), mask).convert("RGBA")
    result = source.copy()
    changed = mask.getbbox()
    result.paste(mixed.crop(changed), changed)
    return result


def draw_all(cases: list[dict], out: Path) -> None:
    pipe = klein.load_pipeline()
    for case in cases:
        raw = klein.draw(pipe, case, *case["base"].size).resize(case["base"].size, Image.Resampling.LANCZOS)
        raw.save(out / f"{case['name']}-raw.png")
        # Klein draws no alpha: keep the portrait's own outline.
        edit = raw.convert("RGBA")
        edit.putalpha(case["base"].getchannel("A"))
        case["klein"] = composite(case["base"], edit, case["area"])
        case["klein"].save(out / f"{case['name']}-klein.png")
    write_json(out / "run.json", {"peak_allocated_gib": klein.peak_gib(), "cases": [
        {key: case[key] for key in ("name", "seed", "upsample_seconds", "draw_seconds", "qwen_job")} for case in cases]})


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sheet(cases: list[dict], out: Path, name: str = "sheet.jpg", tight: bool = False) -> None:
    """Rows per portrait: base, then each expression as Qwen Turbo, an earlier Klein run if given and
    Klein, cropped round the head, or round the face alone when ``tight``."""
    tile = 300 if not tight else 260
    font = ImageFont.load_default(14)
    portraits = list(dict.fromkeys(case["portrait"] for case in cases))
    kinds = [("qwen", "Qwen Turbo"), *([("earlier", "Klein before")] if "earlier" in cases[0] else []),
             ("klein", "Klein")]
    columns = 1 + len(kinds) * len(EXPRESSIONS)
    canvas = Image.new("RGB", (tile * columns, (tile + 18) * len(portraits)), "white")
    pen = ImageDraw.Draw(canvas)
    for row, portrait in enumerate(portraits):
        mine = {case["expression"]: case for case in cases if case["portrait"] == portrait}
        first = mine[EXPRESSIONS[0]]
        left, top, right, bottom = first["face"]
        if tight:
            pad = (right - left) * 0.15
            crop = (round(left - pad), round(top - pad), round(right + pad), round(bottom + pad))
        else:
            side = (right - left) * 2.1
            middle = (left + right) / 2
            crop = (round(middle - side / 2), round(top - side * 0.22), round(middle + side / 2),
                    round(top + side * 0.78))
        cells = [("base portrait", first["base"])]
        for expression in EXPRESSIONS:
            cells += [(f"{expression} | {label}", mine[expression][key]) for key, label in kinds]
        for column, (label, image) in enumerate(cells):
            flat = Image.alpha_composite(Image.new("RGBA", image.size, (200, 200, 200, 255)), image.convert("RGBA"))
            y = row * (tile + 18)
            canvas.paste(flat.crop(crop).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS),
                         (column * tile, y + 18))
            pen.text((column * tile + 4, y + 2), label, fill="black", font=font)
    canvas.save(out / name, quality=92)


def production_portraits():
    """The portraits of the finished Qwen Turbo expression jobs: (base, face, {expression: (Qwen picture, seed, job)})."""
    for digest, jobs in qwen_jobs().items():
        pictures = {}
        for expression in EXPRESSIONS:
            payload = json.loads((jobs[expression] / "job-request.json").read_text(encoding="utf-8"))["payload"]
            with zipfile.ZipFile(jobs[expression] / "result.zip") as bundle:
                qwen = Image.open(io.BytesIO(bundle.read("image.png"))).convert("RGBA")
            pictures[expression] = (qwen, payload["seed"], jobs[expression].name)
        yield Image.open(jobs[EXPRESSIONS[0]] / "outfit-reference.png").convert("RGBA"), face_box(digest), pictures


def varied_portraits():
    """Portraits of another work, of other ages and both genders, with their Qwen Turbo edits
    (tmp/expression-exp); the raw edits get their face put back as production does."""
    for name in VARIED:
        base = Image.open(VARIED_DIR / "src" / f"{name}.png").convert("RGBA")
        face = VARIED_FACES[name]
        pictures = {}
        for expression in EXPRESSIONS:
            raw = Image.open(VARIED_DIR / "out-turbo-vram32" / name / expression / "image.png").convert("RGBA")
            pictures[expression] = (composite(base, raw, area(face, base.size)), VARIED_SEED, name)
        yield base, face, pictures


def main() -> int:
    parser = argparse.ArgumentParser(description="FLUX 3 Image Klein expression experiment")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--portraits", choices=("production", "varied"), default="production")
    parser.add_argument("--requests", type=Path,
                        help="requests.json of expression_requests.py: one request per expression, in place of "
                             "the fixed one; the pictures go beside it unless --out is given")
    parser.add_argument("--earlier", type=Path, help="an earlier run's folder, shown beside this one")
    args = parser.parse_args()
    requests = json.loads(args.requests.read_text(encoding="utf-8"))["requests"] if args.requests else None
    out = args.out or (args.requests.parent if args.requests
                       else ROOT / "outputs/image-edit-tests/klein-expressions-20261016")
    out.mkdir(parents=True, exist_ok=requests is not None)
    cases = []
    portraits = production_portraits() if args.portraits == "production" else varied_portraits()
    for number, (base, face, pictures) in enumerate(portraits, 1):
        base.save(out / f"p{number}-base.png")
        for expression in EXPRESSIONS:
            qwen, seed, source = pictures[expression]
            cases.append({"name": f"p{number}-{expression}", "portrait": number, "expression": expression,
                          "base": base, "references": [klein.on_white(base)], "face": face,
                          "area": area(face, base.size), "qwen": qwen, "qwen_job": source, "seed": seed,
                          "request": requests[expression] if requests else f"{INSTRUCTIONS[expression]} {KEEP}"})
            if args.earlier:
                cases[-1]["earlier"] = Image.open(args.earlier / f"p{number}-{expression}-klein.png")
    print(f"{len(cases)} pictures of {len(cases) // len(EXPRESSIONS)} portraits", flush=True)
    klein.write_captions(cases)
    write_json(out / "captions.json", [{key: case[key] for key in ("name", "request", "caption")} for case in cases])
    draw_all(cases, out)
    sheet(cases, out)
    sheet(cases, out, "faces.jpg", tight=True)
    print(f"done: {out / 'sheet.jpg'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
