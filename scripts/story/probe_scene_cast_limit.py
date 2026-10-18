"""Who shares a scene when the chapter scene plan may put more than three people in it?

Each chapter scene plan request saved by a script experiment is replayed with the model that
made it, unchanged except for:

  current   the saved request as it is
  uncapped  without "最大3人" in the instruction and without the schema's three-person limit
            on a scene's character_ids (nothing else is reworded)
  focused   uncapped, and character_ids lists only the people who speak or act in the scene;
            someone merely there is left out and, if needed, described in the narration

Each arm runs once per seed, so the two arms can be compared against the spread between seeds.
Only the chapter scene plan runs; no scene is written. Nothing in the product changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.worker.generation import pipeline
from services.worker.generation.llm import write_json
from services.worker.model_config import select_config

CAP = "character_idsは登場人物を最大3人、"
UNCAPPED = "character_idsは登場人物を、"
FOCUSED = "character_idsは当場面で発言・行動する人物を、"
FOCUSED_AFTER = "短縮・改名しません。"
FOCUSED_RULE = "居合わせるだけで発言も行動もしない人物はcharacter_idsに入れず、必要なら地の文で触れます。"
ARMS = ("current", "uncapped", "focused")
SEED_STEP = 7919
# Saved requests carry the short aliases the model saw (character-2 rather than the long ID).
PERSON = re.compile(r"\bid:\s*\n\s*(\S+)\s*\n\s*name:\s*\n\s*([^\n]+)")


def load_run(argument: str) -> dict:
    label, _, path = argument.rpartition("=")
    run = Path(path)
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    snapshot = experiment["approval_snapshot"]
    names = {row["result"]["id"]: row["result"]["name"] for row in snapshot["characters"]}
    cast = json.loads((run / "cast-plan.json").read_text(encoding="utf-8"))["plan"]
    names.update({row["id"]: row["name"] for row in cast["supporting_characters"]})
    requests, retries = {}, {}
    for path in sorted((run / "requests").glob("plan-00?-1.json")):
        requests[int(path.name[5:8])] = json.loads(path.read_text(encoding="utf-8"))["request"]
        retry = path.with_name(path.name.replace("-1.json", "-2.json"))
        if path.with_name(path.name.replace(".json", ".validation.json")).is_file() and retry.is_file():
            # The first answer failed the product's check; production went on with this retry.
            retries[int(path.name[5:8])] = json.loads(retry.read_text(encoding="utf-8"))["request"]
    return {"run": label or run.name, "profile": experiment["profile"], "names": names,
            "mains": [row["result"]["id"] for row in snapshot["characters"]], "requests": requests,
            "retries": retries}


def variants(request: dict, retry: dict | None = None) -> dict[str, dict]:
    system = request["messages"][0]["content"]
    if system.count(CAP) != 1:
        raise ValueError("The saved plan instruction differs from the expected wording.")
    uncapped = copy.deepcopy(request)
    uncapped["messages"][0]["content"] = system.replace(CAP, UNCAPPED)
    scene = uncapped["response_format"]["json_schema"]["schema"]["$defs"]["WeightedScene"]
    if scene["properties"]["character_ids"].pop("maxItems", None) != 3:
        raise ValueError("The saved plan schema has no three-person scene limit.")
    focused = copy.deepcopy(uncapped)
    head, _, tail = system.partition(CAP)
    if FOCUSED_AFTER not in tail:
        raise ValueError("The saved plan instruction has no end to the character_ids sentence.")
    rest, _, after = tail.partition(FOCUSED_AFTER)
    focused["messages"][0]["content"] = head + FOCUSED + rest + FOCUSED_AFTER + FOCUSED_RULE + after
    return {"current": copy.deepcopy(retry or request), "uncapped": uncapped, "focused": focused}


def summary(answer: dict, row: dict) -> dict:
    seen = dict(PERSON.findall(row["request_text"]))
    names = {**row["names"], **seen}
    main_names = {row["names"][cid] for cid in row["mains"]}
    mains = {cid for cid, name in seen.items() if name in main_names} or set(row["mains"])
    known = set(names) | {person["id"] for person in answer.get("new_characters", [])}
    names = {**names, **{person["id"]: person["name"] for person in answer.get("new_characters", [])}}
    scenes = answer["scenes"]
    sizes = [len(scene["character_ids"]) for scene in scenes]
    appearances = Counter(cid for scene in scenes for cid in scene["character_ids"])
    return {
        "scenes": len(scenes), "sizes": sizes, "four_plus": sum(size >= 4 for size in sizes),
        "pairs": sum(size == 2 for size in sizes),
        "all_mains": sum(mains <= set(scene["character_ids"]) for scene in scenes),
        "unknown_ids": sorted({cid for scene in scenes for cid in scene["character_ids"]} - known),
        "new_characters": [person["name"] for person in answer.get("new_characters", [])],
        "appearances": {names.get(cid, cid): count for cid, count in appearances.most_common()},
        "detail": [{"id": scene["id"], "weight": scene.get("length_weight"),
                    "people": [names.get(cid, cid) for cid in scene["character_ids"]],
                    "events": [event["description"] for event in scene.get("required_events", [])]}
                   for scene in scenes]}


def report(results: list[dict], output: Path) -> None:
    lines = ["# 場面の人数上限を外した場面計画", "",
             ("保存済みの場面計画の依頼を、作ったモデルで再送した結果。current は依頼そのまま、"
              "uncapped は「最大3人」の文言と出力形式の3人上限だけを外したもの。"), "",
             "| 依頼 | 案 | シード | 場面数 | 各場面の人数 | 4人以上 | 2人 | メイン全員 |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for row in results:
        for key, value in row["answers"].items():
            arm, seed = key.split("@")
            if "error" in value:
                lines.append(f"| {row['request']} | {arm} | {seed} | 失敗 | {value['error'][:80]} | | | |")
                continue
            result = value["summary"]
            lines.append(f"| {row['request']} | {arm} | {seed} | {result['scenes']} | "
                         f"{'/'.join(map(str, result['sizes']))} | {result['four_plus']} | {result['pairs']} | "
                         f"{result['all_mains']} |")
    lines += ["", "## 場面ごとの顔ぶれ", ""]
    for row in results:
        for key, value in row["answers"].items():
            if "summary" not in value:
                continue
            result = value["summary"]
            lines.append(f"### {row['request']} {key}")
            lines.append(f"登場回数: {result['appearances']}"
                         + (f"、新しい人物: {result['new_characters']}" if result["new_characters"] else "")
                         + (f"、未知のID: {result['unknown_ids']}" if result["unknown_ids"] else ""))
            for scene in result["detail"]:
                lines.append(f"- {scene['id']}（重み{scene['weight']}）{'・'.join(scene['people'])}: "
                             + " / ".join(event[:70] for event in scene["events"]))
            lines.append("")
    (output / "report.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("runs", nargs="+",
                        help="script directories with saved plan requests, optionally as LABEL=PATH")
    parser.add_argument("--seeds", type=int, default=2, help="seeds per arm, from the saved one upward")
    parser.add_argument("--arms", nargs="+", default=["current", "uncapped"], choices=ARMS)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--current-retry", action="store_true",
                        help="run current from the retry request where the saved first answer failed the check")
    parser.add_argument("--resummarise", action="store_true", help="redo the summaries of saved answers only")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [load_run(path) for path in args.runs]
    for row in rows:
        for number, request in row["requests"].items():
            variants(request)
            print(row["run"], row["profile"]["model_id"], "chapter", number, "seed", request["seed"])
    if args.dry:
        return 0
    if args.resummarise:
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        for record in results:
            row = next(value for value in rows if value["run"] == record["run"])
            row["request_text"] = row["requests"][record["chapter"]]["messages"][-1]["content"]
            for value in record["answers"].values():
                if "answer" in value:
                    value["summary"] = summary(value["answer"], row)
        write_json(output / "results.json", results)
        report(results, output)
        return 0
    from services.worker.generation.model_routing import RoutedLLM
    from services.worker.generation.processes import gpu_lock

    saved = output / "results.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else []
    by_model: dict[tuple, list[dict]] = {}
    for row in rows:
        by_model.setdefault((row["profile"]["model_id"], row["profile"]["context_size"]), []).append(row)
    base = pipeline.load_config()
    for (model, _), group in by_model.items():
        profile = group[0]["profile"]
        config = copy.deepcopy(select_config(base, {"profile": profile}, pipeline.ROOT))
        config["llm"]["total_timeout_seconds"] = 4 * 3600
        payload = {"seed": 20261018, "profile": profile, "profiles": {}, "story_workflow_version": 2,
                   "workflow_policy": "script_continuation_v1"}
        with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
              RoutedLLM(pipeline.ROOT, config, payload, output / "llm" / model) as llm):
            for row in group:
                for number, request in row["requests"].items():
                    name = f"{row['run']} 第{number}章"
                    record = next((value for value in results if value["request"] == name), None)
                    if record is None:
                        record = {"request": name, "run": row["run"], "chapter": number, "model": model, "answers": {}}
                        results.append(record)
                    for offset in range(args.seeds):
                        seed = request["seed"] + offset * SEED_STEP
                        row["request_text"] = request["messages"][-1]["content"]
                        retry = row["retries"].get(number) if args.current_retry else None
                        for arm, value in variants(request, retry).items():
                            if arm not in args.arms:
                                continue
                            key = f"{arm}@{seed}"
                            if "summary" in record["answers"].get(key, {}):
                                continue
                            start = time.monotonic()
                            try:
                                llm.select_purpose("script-plan")
                                reply = llm.chat("script-plan", value["messages"], seed=seed,
                                                 response_format=value["response_format"],
                                                 max_tokens=value["max_tokens"])
                                answer = json.loads(reply.get("content") or "")
                                record["answers"][key] = {"summary": summary(answer, row), "answer": answer}
                            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                                record["answers"][key] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                            record["answers"][key]["seconds"] = round(time.monotonic() - start, 1)
                            write_json(saved, results)
                            result = record["answers"][key].get("summary", {})
                            print(json.dumps({"request": name, "key": key, "sizes": result.get("sizes"),
                                              "error": record["answers"][key].get("error")},
                                             ensure_ascii=False), flush=True)
    report(results, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
