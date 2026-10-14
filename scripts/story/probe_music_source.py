"""Replay saved BGM caption requests with only one copy of each scene's text.

Arms:
  current  the saved input: every scene carries raw_text and an utterance list
  reseed   the same input with another seed, as the noise floor
  lines    raw_text removed; the utterance lists remain
  script   utterance lists removed; raw_text remains
  once     the first scene's text only inside continuity_group, not also under scene
Nothing is written back to the source jobs.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.worker.generation import pipeline
from services.worker.generation.llm import write_json
from services.worker.generation.model_routing import RoutedLLM
from services.worker.generation.processes import gpu_lock

JOBS = pipeline.ROOT / "services/worker/cache/m2/jobs"


def cases(limit):
    """The newest saved group caption request of each work."""
    newest = {}
    for path in JOBS.glob("*/*/llm/*-music-group-v2-*-1-*.json"):
        request = json.loads(path.read_text(encoding="utf-8"))["request"]
        if len(request["messages"]) != 2:
            continue
        material = json.loads(request["messages"][1]["content"])
        title = material["scene"].get("project_title", "")
        if title and (title not in newest or path.stat().st_mtime > newest[title][0]):
            newest[title] = (path.stat().st_mtime, path, request, material)
    rows = sorted(newest.items(), key=lambda item: -item[1][0])[:limit]
    return [{"title": title, "path": str(path), "request": request, "material": material}
            for title, (_, path, request, material) in rows]


def without(material, key):
    value = copy.deepcopy(material)
    for scene in (value["scene"], *value.get("continuity_group", [])):
        scene.pop(key, None)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--works", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    chosen = cases(args.works)
    for case in chosen:
        material = case["material"]
        case["arms"] = {"current": material, "reseed": material,
                        "lines": without(material, "raw_text"), "script": without(material, "utterances")}
        if material.get("continuity_group"):
            once = copy.deepcopy(material)
            once["scene"].pop("raw_text", None)
            once["scene"].pop("utterances", None)
            case["arms"]["once"] = once
        if args.dry:
            print(case["title"], len(material.get("continuity_group", [])),
                  {arm: len(json.dumps(value, ensure_ascii=False)) for arm, value in case["arms"].items()})
    if args.dry:
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(pipeline.load_config())
    # One model stays loaded for the whole comparison; the production limit is per job.
    config["llm"]["total_timeout_seconds"] = 6 * 3600
    payload = {"seed": args.seed, "profile": {}, "profiles": {}}
    saved = output / "results.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else []
    done = {(row["title"], row["arm"]) for row in results}
    with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
          RoutedLLM(pipeline.ROOT, config, payload, output / "llm") as llm):
        for number, case in enumerate(chosen):
            for arm, material in case["arms"].items():
                if (case["title"], arm) in done:
                    continue
                messages = [case["request"]["messages"][0],
                            {"role": "user", "content": json.dumps(material, ensure_ascii=False)}]
                extra = {"seed": args.seed + number + (1000 if arm == "reseed" else 0),
                         "response_format": case["request"]["response_format"]}
                row = {"title": case["title"], "source": case["path"], "arm": arm, "status": "pending",
                       "group_size": len(material.get("continuity_group", []))}
                start, trace_start = time.monotonic(), len(llm.trace)
                try:
                    llm.select_purpose("music-group")
                    llm._ensure_runtime()
                    reply = llm.chat("music-group", messages, allow_truncated=True, **extra)
                    row.update(content=reply.get("content") or "", finish_reason=reply.get("_finish_reason"))
                    json.loads(row["content"])["english_prompt"]
                    row["status"] = "valid"
                except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
                row["elapsed_seconds"] = time.monotonic() - start
                usage = [item for item in llm.trace[trace_start:] if item.get("type") == "llm_generation"]
                row["usage"] = usage[-1].get("usage", {}) if usage else {}
                results.append(row)
                write_json(saved, results)
                print(json.dumps({key: row.get(key) for key in ("title", "arm", "status", "usage", "error")},
                                 ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
