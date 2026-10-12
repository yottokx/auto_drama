"""Find the face in a portrait, on the CPU, with a pinned anime face detector.

An expression is edited into a whole portrait and only its face is kept: this says where the
face is. It runs in the portrait runtime, which already has ONNX Runtime for background
removal, and never touches the GPU the image models share.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "config/m0-models-image.json"
MODEL_ID = "anime-face-detection"
INPUT_SIZE = 640
CONFIDENCE = 0.25
OVERLAP = 0.7


def model_path() -> Path:
    """The detector's weights, exactly as pinned; nothing is fetched here."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    entry = next(row for row in manifest["models"] if row["id"] == MODEL_ID)
    item = entry["files"][0]
    path = ROOT / entry["local_dir"] / item["path"]
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise FileNotFoundError(f"Download the face detection model first: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("Face detection weights do not match the pinned manifest.")
    return path


def detect(image, session) -> list[dict]:
    """Faces as pixel boxes of the given picture, the most certain first."""
    import numpy as np
    from PIL import Image

    picture = image.convert("RGBA")
    # A portrait is transparent around the figure; the detector was trained on opaque pictures.
    flat = Image.new("RGBA", picture.size, (255, 255, 255, 255))
    flat.alpha_composite(picture)
    resized = flat.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    data = (np.asarray(resized, dtype=np.float32) / 255).transpose(2, 0, 1)[None]
    output = session.run(None, {session.get_inputs()[0].name: data})[0][0]
    # One column per candidate: centre, size, then the score of the only class.
    boxes, scores = output[:4].T, output[4:].max(axis=0)
    keep = scores > CONFIDENCE
    boxes, scores = boxes[keep], scores[keep]
    corners = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                        boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], axis=1)
    order, chosen = scores.argsort()[::-1], []
    while order.size:
        best, rest = order[0], order[1:]
        chosen.append(best)
        left, top = np.maximum(corners[best, :2], corners[rest, :2]).T
        right, bottom = np.minimum(corners[best, 2:], corners[rest, 2:]).T
        shared = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
        area = (corners[:, 2] - corners[:, 0]) * (corners[:, 3] - corners[:, 1])
        order = rest[shared / (area[best] + area[rest] - shared) <= OVERLAP]
    scale = (picture.width / INPUT_SIZE, picture.height / INPUT_SIZE) * 2
    limits = (picture.width, picture.height) * 2
    return [{"box": [int(min(max(float(value) * factor, 0), limit))
                     for value, factor, limit in zip(corners[index], scale, limits, strict=True)],
             "score": round(float(scores[index]), 4)} for index in chosen]


def load_session(path: Path | None = None):
    import onnxruntime

    return onnxruntime.InferenceSession(str(path or model_path()), providers=["CPUExecutionProvider"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from PIL import Image

    session = load_session()
    results = []
    for path in args.image:
        with Image.open(path) as image:
            results.append({"image": str(path), "width": image.width, "height": image.height,
                            "faces": detect(image, session)})
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "model": MODEL_ID, "results": results}),
                         encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
