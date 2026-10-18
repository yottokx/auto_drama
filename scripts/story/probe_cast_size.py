"""How many supporting characters does the STEP4 cast plan make, and does a size anchor hold it back?

The cast instruction says "全3章では1〜2人を目安に世界・メイン人数へ合わせます". Saved casts sit at
about three people in total: one main gets one or two, three mains often get none. This
replays each work's latest saved, reasoning-free cast request unchanged except for:

  current       the saved request as it is
  unanchored    the size anchor is replaced by who is around the mains in this world and story
                (no number), mob-like people with lines are allowed, and a user's wish about the
                number of people counts in both directions
  surroundings  unanchored, and the answer first lists who is around the mains (by position
                and how they touch the story) in a new schema field before supporting_characters

Each arm runs with the saved seed and one fixed extra seed. Only the LLM runs. Nothing in the
product changes.
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
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.worker.generation import pipeline
from services.worker.generation.llm import write_json
from services.worker.generation.script_cast import CastPlan, check_cast

JOBS = pipeline.ROOT / "services/worker/cache/m2/jobs"
EXTRA_SEED = 20261017
ANCHOR = re.compile(r"全(\d+)章では1〜2人を目安に世界・メイン人数へ合わせます。最大8は容量であり目標ではありません。")
LIMIT_OLD = "登場人物を限定するユーザー設定は優先し、その場合は空配列も可能です。"
LIMIT_NEW = "登場人物の数や範囲についてのユーザー設定は、増やす場合も限定する場合も優先します。限定されていれば空配列も可能です。"


def unanchored_rule(chapters: str) -> str:
    return (f"人数の目安はありません。全{chapters}章の物語でメインの身近にいる人、出会う人、対立する人を"
            "この世界から考え、台詞と出番がある人を作ります。"
            "少人数の密な関係が魅力の作品なら少なく、学校・職場・軍・町など人が集まる場や、"
            "敵や競争相手がいる物語なら、それに見合う顔ぶれにします。"
            "一場面だけの店員・兵士・級友のような脇の人物も、台詞があれば作れます。最大8は容量です。")


SURROUNDINGS_RULE = "まずsurroundingsに挙げ、その中から台詞と出番がある人をsupporting_charactersにします。"
SURROUNDINGS_FIELD = {
    "type": "array", "maxItems": 12, "title": "Surroundings",
    "description": ("メインの身近にいる人・出会う人・対立する人を、この世界と物語に照らして"
                    "立場と物語での関わりで短く挙げます。メイン自身は含めません。"),
    "items": {"type": "string", "minLength": 1},
}


def saved_casts() -> list[dict]:
    """The latest saved cast request of every work whose instruction still has the size anchor."""
    latest: dict[str, dict] = {}
    for path in JOBS.glob("*/*/script/requests/cast-plan-*.json"):
        try:
            request = json.loads(path.read_text(encoding="utf-8"))["request"]
            job = path.parents[2]
            payload = json.loads((job / "job-request.json").read_text(encoding="utf-8"))
            payload = payload.get("payload", payload)
            snapshot = payload["approval_snapshot"]
            title = snapshot["world"]["result"]["title"]
            saved = json.loads((job / "script/cast-plan.json").read_text(encoding="utf-8"))["plan"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not isinstance(request, dict) or not ANCHOR.search(request["messages"][0]["content"]):
            continue
        # A few works were planned with reasoning; every replay here runs without it, as the product does.
        row = {"path": path, "title": title, "request": request, "saved": saved, "job": job.name,
               "saved_reasoning": request.get("reasoning_effort"),
               "main_ids": [c["result"]["id"] for c in snapshot["characters"]],
               "main_names": [c["result"]["name"] for c in snapshot["characters"]],
               "notes": snapshot["world"]["result"].get("notes", "")}
        if title not in latest or path.stat().st_mtime > latest[title]["path"].stat().st_mtime:
            latest[title] = row
    return sorted(latest.values(), key=lambda row: (len(row["main_ids"]), row["title"]))


def variants(row: dict) -> dict[str, dict]:
    request = row["request"]
    system = request["messages"][0]["content"]
    chapters = ANCHOR.search(system).group(1)
    unanchored = copy.deepcopy(request)
    text = ANCHOR.sub(unanchored_rule(chapters), system)
    if LIMIT_OLD not in text:
        raise ValueError(f"{row['title']}: the user-limit sentence is missing.")
    unanchored["messages"][0]["content"] = text.replace(LIMIT_OLD, LIMIT_NEW)
    surroundings = copy.deepcopy(unanchored)
    surroundings["messages"][0]["content"] = surroundings["messages"][0]["content"].replace(
        unanchored_rule(chapters), unanchored_rule(chapters) + SURROUNDINGS_RULE)
    schema = surroundings["response_format"]["json_schema"]["schema"]
    schema["properties"] = {"surroundings": SURROUNDINGS_FIELD, **schema["properties"]}
    schema["required"] = ["surroundings", *schema["required"]]
    return {"current": copy.deepcopy(request), "unanchored": unanchored, "surroundings": surroundings}


def summary(answer: dict, main_ids: list[str]) -> dict:
    answer = dict(answer)
    around = answer.pop("surroundings", None)
    plan = CastPlan.model_validate(answer)
    problem = None
    try:
        check_cast(plan, set(main_ids))
    except ValueError as exc:
        problem = str(exc)[:200]
    return {"supporting": [{"id": row.id, "name": row.name, "role": row.role, "age": row.age, "gender": row.gender}
                           for row in plan.supporting_characters],
            "surroundings": around, "check": problem}


def people(plan: dict) -> list[dict]:
    return [{"name": row["name"], "role": row.get("role", "")} for row in plan["supporting_characters"]]


ARMS = ("current", "unanchored", "surroundings")


def report(results: list[dict], output: Path) -> None:
    def count(row, arm, seed):
        value = row["arms"].get(f"{arm}@{seed}", {})
        return None if "summary" not in value else len(value["summary"]["supporting"])

    lines = ["# STEP4キャスト計画のサブキャラ人数", "",
             "保存済みの推論なしのキャスト計画依頼を、指示だけ変えて再生成した結果。各版を保存時のシードと追加シードの2回。", ""]
    by_mains = defaultdict(lambda: defaultdict(list))
    for row in results:
        for arm in ARMS:
            for seed in row["seeds"]:
                n = count(row, arm, seed)
                if n is not None:
                    by_mains[len(row["main_names"])][arm].append(n)
        by_mains[len(row["main_names"])]["saved"].append(len(row["saved"]))
    lines += ["## メイン人数ごとの平均サブ人数", "",
              "| メイン | 作品 | 保存済み | current | unanchored | surroundings |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mains in sorted(by_mains):
        cells = [f"{mean(by_mains[mains][arm]):.1f}" if by_mains[mains][arm] else "-" for arm in ("saved", *ARMS)]
        lines.append(f"| {mains} | {len(by_mains[mains]['saved'])} | " + " | ".join(cells) + " |")
    lines += ["", "## 作品ごとのサブ人数（シード2回）", "",
              "| 作品 | メイン | 保存済み | current | unanchored | surroundings |", "| --- | ---: | ---: | --- | --- | --- |"]
    for row in results:
        cells = []
        for arm in ARMS:
            values = [count(row, arm, seed) for seed in row["seeds"]]
            cells.append(" / ".join("失敗" if n is None else str(n) for n in values))
        lines.append(f"| {row['title']} | {len(row['main_names'])} | {len(row['saved'])} | " + " | ".join(cells) + " |")
    lines += ["", "## 作品ごとの顔ぶれ", ""]
    for row in results:
        lines.append(f"### {row['title']}（メイン{len(row['main_names'])}人: {'、'.join(row['main_names'])}）")
        if row["notes"]:
            lines.append(f"- 世界観のnotes: {row['notes'][:120]}")
        lines.append(f"- 保存済み（推論 {row.get('saved_reasoning')}）: " + ("、".join(f"{p['name']}（{p['role']}）" for p in row["saved"]) or "なし"))
        for arm in ARMS:
            for seed in row["seeds"]:
                value = row["arms"].get(f"{arm}@{seed}", {})
                if "error" in value:
                    lines.append(f"- {arm} seed{seed}: 失敗 {value['error'][:120]}")
                    continue
                if "summary" not in value:
                    continue
                result = value["summary"]
                names = "、".join(f"{p['name']}（{p['role']}）" for p in result["supporting"]) or "なし"
                lines.append(f"- {arm} seed{seed}: {names}" + (f" ※検査: {result['check']}" if result["check"] else ""))
                if result["surroundings"]:
                    lines.append(f"  - surroundings: {' / '.join(result['surroundings'])}")
        lines.append("")
    tokens = defaultdict(list)
    for row in results:
        for key, value in row["arms"].items():
            if value.get("usage", {}).get("completion_tokens"):
                tokens[key.split("@")[0]].append((value["usage"]["prompt_tokens"], value["usage"]["completion_tokens"]))
    lines += ["## トークン", "", "| 版 | 呼び出し | 入力の平均 | 出力の平均 | 出力の最大 |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        if tokens[arm]:
            lines.append(f"| {arm} | {len(tokens[arm])} | {mean(t[0] for t in tokens[arm]):,.0f} | "
                         f"{mean(t[1] for t in tokens[arm]):,.0f} | {max(t[1] for t in tokens[arm]):,} |")
    (output / "report.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--limit", type=int, default=0, help="only the first N works (0: all)")
    parser.add_argument("--titles", default="", help="only works whose title contains one of these (comma-separated)")
    parser.add_argument("--dry", action="store_true", help="list the chosen works without calling the model")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "results.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else []
    if args.report_only:
        report(results, output)
        return 0
    chosen = saved_casts()
    if args.titles:
        chosen = [row for row in chosen if any(part in row["title"] for part in args.titles.split(","))]
    if args.limit:
        chosen = chosen[:args.limit]
    for row in chosen:
        print(len(row["main_ids"]), len(row["saved"]["supporting_characters"]), row["title"], row["job"], flush=True)
    if args.dry:
        for row in chosen:
            variants(row)
        return 0
    arms = args.arms.split(",")
    from services.worker.generation.model_routing import RoutedLLM
    from services.worker.generation.processes import gpu_lock

    config = copy.deepcopy(pipeline.load_config())
    config["llm"]["total_timeout_seconds"] = 8 * 3600
    payload = {"seed": EXTRA_SEED, "profile": {}, "profiles": {}, "story_workflow_version": 2,
               "workflow_policy": "script_continuation_v1"}
    with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
          RoutedLLM(pipeline.ROOT, config, payload, output / "llm") as llm):
        # Seed-major order: one pass over every work and arm gives a first comparison early.
        for seed_index in range(2):
            for row in chosen:
                record = next((value for value in results if value["path"] == str(row["path"])), None)
                if record is None:
                    record = {"path": str(row["path"]), "title": row["title"], "job": row["job"],
                              "main_names": row["main_names"], "notes": row["notes"],
                              "saved_reasoning": row["saved_reasoning"],
                              "saved": people(row["saved"]), "seeds": [row["request"]["seed"], EXTRA_SEED],
                              "arms": {}}
                    results.append(record)
                seed = record["seeds"][seed_index]
                for arm, request in variants(row).items():
                    key = f"{arm}@{seed}"
                    if arm not in arms or "summary" in record["arms"].get(key, {}):
                        continue
                    start, trace_start = time.monotonic(), len(llm.trace)
                    try:
                        llm.select_purpose("script-cast")
                        reply = llm.chat("script-cast", request["messages"], seed=seed,
                                         response_format=request["response_format"],
                                         # A reasoned request saved a larger allowance than a plain one may use.
                                         max_tokens=min(request["max_tokens"], 8192))
                        answer = json.loads(reply.get("content") or "")
                        record["arms"][key] = {"summary": summary(answer, row["main_ids"]), "answer": answer}
                    except (OSError, RuntimeError, ValueError, TypeError) as exc:
                        record["arms"][key] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                    usage = [item for item in llm.trace[trace_start:] if item.get("type") == "llm_generation"]
                    record["arms"][key].update(seconds=round(time.monotonic() - start, 1),
                                               usage=usage[-1].get("usage", {}) if usage else {})
                    write_json(saved, results)
                    made = record["arms"][key].get("summary", {}).get("supporting", [])
                    print(json.dumps({"title": row["title"], "arm": arm, "seed": seed,
                                      "supporting": [f"{p['name']}（{p['role']}）" for p in made],
                                      "error": record["arms"][key].get("error"),
                                      "tokens": record["arms"][key]["usage"]}, ensure_ascii=False), flush=True)
            report(results, output)
    report(results, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
