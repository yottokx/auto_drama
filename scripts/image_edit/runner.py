"""One owned subprocess for one Qwen Image 2.1 experiment batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.image_edit.engine import generate


def run_request(request_path: Path, output_dir: Path) -> int:
    try:
        data = json.loads(Path(request_path).read_text(encoding="utf-8-sig"))
        result = generate(data, Path(output_dir))
        return 0 if result["ok"] else 130 if result["cancelled"] else 1
    except Exception as exc:  # noqa: BLE001 - reused directories must remain untouched
        print(f"画像編集実験を開始できません: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return run_request(args.request, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
