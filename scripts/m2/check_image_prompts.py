"""Run opt-in local-model appearance regression checks without modifying projects.

Use python -m scripts.m2.check_image_prompts --output-dir private/m2/<new-run>.
Add --images to render fixtures marked generate_image. Review the saved prompts
and raw image.png files: structural checks cannot prove visual fidelity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.worker.generation.llm import LocalLLM, write_json
from services.worker.generation.pipeline import ROOT, generate_image, image_prompt, load_config
from services.worker.generation.processes import gpu_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1426408052, 773104])
    parser.add_argument("--baseline-prompts", type=Path,
                        help="Optional JSON mapping fixture IDs to prior description prompts.")
    args = parser.parse_args()
    if any(not 0 <= seed < 2**63 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**63).")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Choose a new output directory so prior evidence remains unchanged.")
    output.mkdir(parents=True)
    cases = json.loads((ROOT / "tests/fixtures/m2-image-semantics.json").read_text("utf-8"))
    baselines = json.loads(args.baseline_prompts.read_text("utf-8")) if args.baseline_prompts else {}
    config = load_config()
    write_json(output / "input.json", {"config": config, "fixtures": cases,
                                     "seeds": args.seeds, "baseline_prompts": baselines})
    results = []
    for case in cases:
        for seed in args.seeds:
            work = output / f"{case['id']}-{seed}"
            work.mkdir()
            payload = {"character_id": case["id"], "seed": seed, "instruction": "",
                       "character_result": {key: case[key] for key in
                                            ("id", "name", "age", "gender", "appearance")}}
            with gpu_lock(ROOT / "services/worker/cache/m2/gpu.lock",
                          config["gpu_lock_timeout_seconds"]):
                with LocalLLM(ROOT, config, payload, work / "llm") as llm:
                    prompt = image_prompt(payload, llm)
                    conversion = llm.trace[-1]
                record = {"id": case["id"], "seed": seed, "review": case["review"],
                          "conversion": conversion}
                write_json(work / "conversion.json", record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if args.images and case["generate_image"]:
                    content, metadata = generate_image(payload, prompt, work, config)
                    (work / "character.png").write_bytes(content)
                    record["image"] = metadata
                    if case["id"] in baselines:
                        baseline_dir = work / "baseline"
                        baseline_dir.mkdir()
                        content, metadata = generate_image(
                            payload, baselines[case["id"]], baseline_dir, config)
                        (baseline_dir / "character.png").write_bytes(content)
                        record["baseline"] = metadata
                write_json(work / "result.json", record)
                results.append(record)
                write_json(output / "results.json", results)
                print(f"Completed {case['id']} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
