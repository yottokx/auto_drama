"""Does a chapter plan add the people its scenes need, if it decides on them after the scenes?

A chapter plan is generated field by field in its schema's order: new_characters first, then
locations and scenes, and inside a scene its cast before its events. Without reasoning the
model must say who is new before it has planned anything, and in 240 saved plans it never
did. This replays saved, reasoning-free plan requests unchanged except for:

  current    the saved answer itself (no call)
  reordered  the same request, with scenes before new_characters and each scene's
             required_events before its character_ids in the output schema
  worded     reordered, and the instruction says that someone only heard (by phone, radio,
             a recording or in memory), or no longer alive, is a new person too
  reasoned            (--reasoning) the saved request with the model's reasoning on
  reasoned_reordered  (--reasoning) reordered, with reasoning on

Each replay keeps the saved seed. Nothing in the product changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.worker.generation import pipeline
from services.worker.generation.llm import write_json
from services.worker.generation.script_chapter_plan import ContinuedChapterPlan, FirstChapterPlan

JOBS = pipeline.ROOT / "services/worker/cache/m2/jobs"
OLD_RULE = "new_charactersは計画外で今回出演する新規人物だけ。不要なら空配列です。"
NEW_RULE = ("new_charactersは計画外で今回登場する新規人物です。電話・無線・録音や留守電・回想の声だけで"
            "姿を見せない人物や、亡くなった人物でも、台詞として声が聞こえるなら定義し、"
            "その場面のcharacter_idsに入れます。不要なら空配列です。")


def saved_plans() -> list[dict]:
    """Every saved reasoning-free chapter plan, with its work's title and chapter."""
    rows = []
    for path in JOBS.rglob("*script-plan*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            request = data["request"]
            answer = json.loads(data["response"]["choices"][0]["message"]["content"])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
        if request.get("reasoning_effort") != "none" or "new_characters" not in answer:
            continue
        system, user = request["messages"][0]["content"], request["messages"][-1]["content"]
        title = re.search(r"\n  title:\n    (.+)\n", user)
        chapter = re.search(r"の第(\d+)章を場面に分けます", system)
        if not title or not chapter or OLD_RULE not in system:
            continue
        rows.append({"path": path, "title": title.group(1), "chapter": int(chapter.group(1)),
                     "request": request, "answer": answer, "job": path.parents[2].name})
    return rows


def choose(rows: list[dict], focus: str, others: int) -> list[dict]:
    """Every chapter of the focused work, and one chapter (the second where there is one) of others."""
    # Each chapter is planned in its own job; a work may have been generated more than once.
    latest: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["title"], row["chapter"])
        if key not in latest or row["path"].stat().st_mtime > latest[key]["path"].stat().st_mtime:
            latest[key] = row
    by_title = defaultdict(list)
    for (title, _), row in sorted(latest.items()):
        by_title[title].append(row)
    chosen = []
    for title, plans in sorted(by_title.items()):
        if focus in title:
            chosen = plans + chosen
        elif sum(1 for row in chosen if focus not in row["title"]) < others:
            chosen.append(next((row for row in plans if row["chapter"] == 2), plans[0]))
    return chosen


def reorder(schema: dict) -> dict:
    """Scenes before new characters; in a scene, what happens before who is there."""
    schema = copy.deepcopy(schema)
    properties = schema["properties"]
    schema["properties"] = {key: properties[key] for key in [*[k for k in properties if k != "new_characters"],
                                                               "new_characters"]}
    for definition in schema.get("$defs", {}).values():
        fields = definition.get("properties", {})
        if "character_ids" in fields and "required_events" in fields:
            keys = [key for key in fields if key != "character_ids"]
            keys.insert(keys.index("required_events") + 1, "character_ids")
            definition["properties"] = {key: fields[key] for key in keys}
    return schema


def variants(row: dict, reasoning: bool = False) -> dict[str, dict]:
    request = row["request"]
    reordered = copy.deepcopy(request)
    reordered["response_format"]["json_schema"]["schema"] = reorder(request["response_format"]["json_schema"]["schema"])
    if reasoning:
        return {"reasoned": copy.deepcopy(request), "reasoned_reordered": reordered}
    worded = copy.deepcopy(reordered)
    worded["messages"][0]["content"] = worded["messages"][0]["content"].replace(OLD_RULE, NEW_RULE)
    return {"reordered": reordered, "worded": worded}


def summary(answer: dict) -> dict:
    """Who is new and where everyone appears, in a form that reads at a glance."""
    model = ContinuedChapterPlan if "continuation" in answer else FirstChapterPlan
    plan = model.model_validate(answer)
    added = {row.id: row.name for row in plan.new_characters}
    return {"new_characters": [{"id": row.id, "name": row.name, "role": row.role} for row in plan.new_characters],
            "scenes": [{"id": scene.id, "character_ids": scene.character_ids,
                        "events": [event.description for event in scene.required_events]} for scene in plan.scenes],
            "unused_new": [cid for cid in added if not any(cid in scene.character_ids for scene in plan.scenes)]}


def report(results: list[dict], output: Path) -> None:
    lines = ["# 場面計画で新しい人物を足せるか", "",
             "現行は保存済みの推論なしの回答。推論ありの版は、同じ依頼を推論ありで生成し直したもの。", "",
             "| 作品 | 章 | 現行 | 項目順の入れ替え | 入れ替え＋指示の明記 | 推論あり | 推論あり＋入れ替え |",
             "| --- | ---: | --- | --- | --- | --- | --- |"]
    for row in results:
        cells = []
        for arm in ("current", "reordered", "worded", "reasoned", "reasoned_reordered"):
            value = row["arms"].get(arm, {})
            if "error" in value:
                cells.append("失敗")
            elif "summary" in value:
                names = [c["name"] for c in value["summary"]["new_characters"]]
                cells.append("、".join(names) if names else "なし")
            else:
                cells.append("-")
        lines.append(f"| {row['title']} | {row['chapter']} | " + " | ".join(cells) + " |")
    lines += ["", "## 追加された人物と登場場面", ""]
    for row in results:
        for arm in ("reordered", "worded", "reasoned", "reasoned_reordered"):
            value = row["arms"].get(arm, {}).get("summary")
            if not value or not value["new_characters"]:
                continue
            lines.append(f"### {row['title']} 第{row['chapter']}章（{arm}）")
            for person in value["new_characters"]:
                scenes = [scene["id"] for scene in value["scenes"] if person["id"] in scene["character_ids"]]
                lines.append(f"- {person['name']}（{person['role']}）: 場面 {', '.join(scenes) or 'どこにも出ない'}")
            lines.append("")
    tokens = defaultdict(lambda: [0, 0, 0])
    for row in results:
        for arm, value in row["arms"].items():
            if "usage" in value:
                tokens[arm][0] += 1
                tokens[arm][1] += value["usage"].get("prompt_tokens") or 0
                tokens[arm][2] += value["usage"].get("completion_tokens") or 0
    lines += ["## トークン", "", "| 版 | 呼び出し | 入力合計 | 出力合計 |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {arm} | {n} | {given:,} | {made:,} |" for arm, (n, given, made) in tokens.items()]
    (output / "report.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus", default="凪の島", help="a work whose every chapter is replayed")
    parser.add_argument("--others", type=int, default=9, help="how many other works to sample")
    parser.add_argument("--dry", action="store_true", help="list the chosen plans without calling the model")
    parser.add_argument("--reasoning", default=None, help="run the reasoned arms at this reasoning level")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    chosen = choose(saved_plans(), args.focus, args.others)
    for row in chosen:
        print(row["title"], row["chapter"], row["path"].name, flush=True)
    if args.dry:
        return 0
    saved = output / "results.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else []
    done = {(row["path"], arm) for row in results for arm, value in row["arms"].items() if "error" not in value}
    from services.worker.generation.model_routing import RoutedLLM
    from services.worker.generation.processes import gpu_lock

    config = copy.deepcopy(pipeline.load_config())
    config["llm"]["total_timeout_seconds"] = 6 * 3600
    payload = {"seed": 20261016, "profile": {}, "profiles": {}, "story_workflow_version": 2,
               "workflow_policy": "script_continuation_v1"}
    with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
          RoutedLLM(pipeline.ROOT, config, payload, output / "llm") as llm):
        for row in chosen:
            record = next((value for value in results if value["path"] == str(row["path"])), None)
            if record is None:
                record = {"path": str(row["path"]), "title": row["title"], "chapter": row["chapter"],
                          "arms": {"current": {"summary": summary(row["answer"])}}}
                results.append(record)
            for arm, request in variants(row, bool(args.reasoning)).items():
                if (str(row["path"]), arm) in done:
                    continue
                start, trace_start = time.monotonic(), len(llm.trace)
                try:
                    llm.select_purpose("script-plan")
                    if args.reasoning:
                        # Thinking shares the saved output allowance, which still fits the context.
                        # As a work with common settings asks for it: only such a profile may think.
                        llm.profile = {**llm.profile, "reasoning_level": args.reasoning, "common_settings_version": 1}
                    llm._ensure_runtime()
                    reply = llm.chat("script-plan", request["messages"], allow_truncated=True,
                                     seed=request["seed"], response_format=request["response_format"],
                                     **({"max_tokens": 6144} if args.reasoning else {}))
                    answer = json.loads(reply.get("content") or "")
                    record["arms"][arm] = {"summary": summary(answer), "answer": answer}
                except (OSError, RuntimeError, ValueError, TypeError) as exc:
                    record["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                usage = [item for item in llm.trace[trace_start:] if item.get("type") == "llm_generation"]
                record["arms"][arm].update(seconds=round(time.monotonic() - start, 1),
                                           usage=usage[-1].get("usage", {}) if usage else {})
                write_json(saved, results)
                added = [c["name"] for c in record["arms"][arm].get("summary", {}).get("new_characters", [])]
                print(json.dumps({"title": row["title"], "chapter": row["chapter"], "arm": arm, "added": added,
                                  "error": record["arms"][arm].get("error"),
                                  "tokens": record["arms"][arm]["usage"]}, ensure_ascii=False), flush=True)
    report(results, output)
    print((output / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
