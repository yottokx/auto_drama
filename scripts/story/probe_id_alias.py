"""Replay saved script requests with long character IDs replaced by short aliases.

For each work: the latest chapter plan, two scene texts and one scene's staging batches,
each generated with the saved input (current) and with aliases (alias) on the same seed.
Aliases exist only inside the request and its raw reply; results are decoded to real IDs.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.contracts.m3 import ScenePlan
from packages.narrative import parse_scene_text
from scripts.story.probe_staging_context import JOBS, arms, messages_for, pick_scene, scene_batches
from services.worker.generation import narrative, pipeline
from services.worker.generation.llm import write_json
from services.worker.generation.model_routing import RoutedLLM
from services.worker.generation.processes import gpu_lock
from services.worker.generation.script_cast import script_metrics

LONG_ID = re.compile(r"character-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SHORT_ID = re.compile(r"character-(\d+)(?![0-9A-Za-z_-])")
EDGE = r"(?![0-9A-Za-z_-])"
PLAN_MARK, SCOPE_MARK = "今回の場面計画（未実施の予定）:\n", "\n執筆範囲と後続場面の担当"
TEXT_TOKENS = 4096


class Aliases:
    """A per-request bijection between long IDs and unused character-N names."""

    def __init__(self, text):
        used = {int(number) for number in SHORT_ID.findall(text)}
        self.forward, number = {}, 1
        for real in dict.fromkeys(LONG_ID.findall(text)):
            while number in used:
                number += 1
            used.add(number)
            self.forward[real] = f"character-{number}"
        self.backward = {alias: real for real, alias in self.forward.items()}

    def encode(self, text):
        for real, alias in self.forward.items():
            text = text.replace(real, alias)
        return text

    def decode(self, text):
        for alias in sorted(self.backward, key=len, reverse=True):
            text = re.sub(r"(?<![0-9A-Za-z_-])" + re.escape(alias) + EDGE, self.backward[alias], text)
        return text


def generate(llm, purpose, messages, extra, use_alias):
    """One request; returns the decoded reply and what was actually sent."""
    table = Aliases(json.dumps([messages, extra], ensure_ascii=False))
    if use_alias:
        messages = [{**item, "content": table.encode(item["content"])} for item in messages]
        extra = json.loads(table.encode(json.dumps(extra, ensure_ascii=False)))
    result = {"aliases": table.forward if use_alias else {}, "long_ids": len(table.forward),
              "input_characters": sum(len(item["content"]) for item in messages), "status": "pending"}
    start, trace_start = time.monotonic(), len(llm.trace)
    try:
        llm.select_purpose(purpose)
        llm._ensure_runtime()
        reply = llm.chat(purpose, messages, allow_truncated=True, **extra)
        raw = reply.get("content") or ""
        result.update(raw=raw, content=table.decode(raw) if use_alias else raw,
                      finish_reason=reply.get("_finish_reason"), status="completed")
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
    result["elapsed_seconds"] = time.monotonic() - start
    usage = [item for item in llm.trace[trace_start:] if item.get("type") == "llm_generation"]
    result["usage"] = usage[-1].get("usage", {}) if usage else {}
    budget = [item for item in llm.trace[trace_start:] if item.get("type") == "context_budget"]
    if budget:
        result["prompt_tokens"] = budget[-1].get("prompt_tokens")
    return result


def check(result, validate):
    if result["status"] != "completed":
        return
    try:
        result["checked"] = validate(result["content"])
        result["status"] = "valid"
    except (ValueError, TypeError, KeyError) as exc:
        result.update(status="invalid", error=f"{type(exc).__name__}: {exc}"[:500])


def saved(path):
    return json.loads(path.read_text(encoding="utf-8"))["request"]


def plan_case(job):
    path = max((job / "script/requests").glob("plan-0*-1.json"), key=lambda item: item.name)
    request = saved(path)
    prompt = request["messages"][1]["content"]

    def validate(text):
        value = json.loads(text)
        added = {row["id"] for row in value.get("new_characters", [])}
        used = [cid for scene in value["scenes"] for cid in scene["character_ids"]]
        unknown = sorted({cid for cid in used if cid not in added and cid not in prompt})
        if unknown:
            raise ValueError(f"Scene character IDs are not in the material: {unknown}")
        return {"scenes": len(value["scenes"]), "character_refs": len(used), "new_characters": len(added)}

    return {"stage": "plan", "name": path.stem, "purpose": "script-plan", "messages": request["messages"],
            "extra": {"response_format": request["response_format"]}, "validate": validate}


def text_cases(job, limit):
    cases = []
    for path in sorted((job / "script/requests").glob("*-text-1.json")):
        request = saved(path)
        body = request["messages"][1]["content"]
        plan = ScenePlan.model_validate_json(body[body.index(PLAN_MARK) + len(PLAN_MARK):body.index(SCOPE_MARK)])
        if not any(LONG_ID.fullmatch(cid) for cid in plan.character_ids):
            continue

        def validate(text, plan=plan):
            lines = parse_scene_text(text, plan.id, set(plan.character_ids))
            metrics = script_metrics(text)
            return {"utterances": len(lines), "body_characters": metrics["body_characters"],
                    "dialogue_lines": metrics["dialogue_lines"],
                    "speakers": {key: row["lines"] for key, row in metrics["speakers"].items()}}

        cases.append({"stage": "text", "name": path.stem, "purpose": "script-scene",
                      "messages": request["messages"], "validate": validate,
                      "extra": {"grammar": narrative._source_grammar(plan), "max_tokens": TEXT_TOKENS}})
    return cases[:limit]


def staging_cases(job):
    batches = scene_batches(job, pick_scene(job))
    cases = []
    for index, row in enumerate(batches):
        schema, prompts = arms(batches, index)

        def validate(text, row=row):
            staging = narrative.Staging.model_validate_json(text)
            if [item.utterance_id for item in staging.emotions] != [line.id for line in row["batch"]]:
                raise ValueError("Emotion annotations must preserve every source ID in order.")
            narrative._directions(row["plan"], row["batch"], staging)
            return {"directions": len(staging.directions)}

        cases.append({"stage": "staging", "name": f"{pick_scene(job)}-batch{index}", "purpose": "script-staging",
                      "prompt": prompts["current"], "schema": schema, "validate": validate,
                      "seed_offset": 17 * index + sum(map(ord, pick_scene(job))),
                      "lines": [{"id": line.id, "speaker": line.speaker_id, "text": line.display_text}
                                for line in row["batch"]]})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append", required=True, help="job id prefix; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--context-size", type=int, help="load the model with this context instead of the configured one")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    works = []
    for prefix in args.job:
        job = next(path for path in JOBS.glob(f"*/{prefix}*") if (path / "script/requests").is_dir())
        works.append((job, [plan_case(job), *text_cases(job, args.scenes), *staging_cases(job)]))
    if args.dry:
        for job, cases in works:
            for case in cases:
                text = case.get("prompt") or case["messages"][1]["content"]
                print(job.name[:8], case["stage"], case["name"], len(text), len(set(LONG_ID.findall(text))),
                      len(LONG_ID.findall(text)))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(pipeline.load_config())
    # One model stays loaded for the whole comparison; the production limit is per job.
    config["llm"]["total_timeout_seconds"] = 6 * 3600
    if args.context_size:
        config["llm"]["context_size"] = args.context_size
    payload = {"seed": args.seed, "profile": {}, "profiles": {}, "story_workflow_version": 2,
               "workflow_policy": "script_continuation_v1"}
    # A stopped run continues: finished cases keep their measured time and usage.
    saved_path = output / "results.json"
    results = json.loads(saved_path.read_text(encoding="utf-8")) if saved_path.is_file() else []
    done = {(row["job"], row["name"], row["arm"]) for row in results}
    with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
          RoutedLLM(pipeline.ROOT, config, payload, output / "llm") as llm):
        for job, cases in works:
            for number, case in enumerate(cases):
                seed = args.seed + case.get("seed_offset", 101 * number)
                for arm in ("current", "alias"):
                    if (job.name, case["name"], arm) in done:
                        continue
                    if case["stage"] == "staging":
                        extra = {"seed": seed, "response_format": {"type": "json_schema", "json_schema": {
                            "name": "staging", "strict": True, "schema": case["schema"]}}}
                        llm.select_purpose("script-staging")
                        llm._ensure_runtime()
                        messages = messages_for(llm, "current", case["prompt"], extra)
                    else:
                        extra, messages = {"seed": seed, **case["extra"]}, case["messages"]
                    result = generate(llm, case["purpose"], messages, extra, arm == "alias")
                    check(result, case["validate"])
                    result.update(job=job.name, stage=case["stage"], name=case["name"], arm=arm, seed=seed,
                                  lines=case.get("lines"))
                    results.append(result)
                    write_json(output / "results.json", results)
                    print(json.dumps({key: result.get(key) for key in (
                        "job", "stage", "name", "arm", "status", "usage", "elapsed_seconds", "checked", "error")},
                        ensure_ascii=False)[:420], flush=True)
        write_json(output / "runtime.json", {"profile": llm.profile, "provenance": llm.provenance()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
