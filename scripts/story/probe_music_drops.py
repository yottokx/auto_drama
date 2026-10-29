"""Where the version 3 chapter music plan stops the music, on the 35 chapters with a music plan.

The product's chapter music design (services/worker/generation/music_pipeline.py, planning_version 3)
now answers play/continue for every scene and the moments inside a scene where the music stops. Its
drop criteria come from the annotation trial (v3, docs/experiments/bgm-ambience-annotation-20261028.md),
which asked them together with environmental sound. This runs the product's own step on the newest
chapter-wide music request of every chapter, with the product's newest LLM settings, and compares the
drops with that trial (outputs/music-ambience-v3-20261028/a). Group captions are not asked; nothing is
saved to the database. Run it while the product's worker is stopped.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TRIAL = Path("outputs/music-ambience-v3-20261028/a/results.json")
KIND_LABELS = {"silence_in_text": "本文が静けさを書いている", "loss": "死や別れ", "premise_overturned": "前提が覆る真実"}
NEAR_LINES = 2


def chapters(connection) -> list[dict]:
    """The newest finished chapter-wide music request of every chapter (the trial's selection)."""
    rows = connection.execute(
        "SELECT j.id, j.project_id, j.payload, j.created_at, p.title FROM job j JOIN project p ON p.id=j.project_id "
        "WHERE j.kind='m3_music_plan' AND j.status='completed' ORDER BY j.created_at").fetchall()
    newest = {}
    for row in rows:
        payload = json.loads(row["payload"])
        if payload.get("planning_version") != 2 or payload.get("planning_scope") == "single_scene":
            continue
        newest[(row["project_id"], payload["chapter_number"])] = (row, payload)
    latest = {}
    for row, _ in newest.values():
        if row["title"] not in latest or row["created_at"] > latest[row["title"]][1]:
            latest[row["title"]] = (row["project_id"], row["created_at"])
    projects = {project for project, _ in latest.values()}
    return [{"key": f"{row['project_id'][:8]}-ch{number}", "title": row["title"], "chapter": number,
             "job_id": row["id"], "payload": {**payload, "planning_version": 3}}
            for (_, number), (row, payload) in sorted(
                ((key, value) for key, value in newest.items() if key[0] in projects),
                key=lambda item: (item[1][0]["title"], item[0][1]))]


def plan(llm, case: dict) -> dict:
    from services.worker.generation import music_pipeline

    before = len(llm.trace)
    try:
        decisions, drops = music_pipeline._chapter_decisions(case["payload"], llm, 3)
    except ValueError as exc:
        return {"key": case["key"], "title": case["title"], "chapter": case["chapter"], "error": str(exc)}
    usage = [row.get("usage") for row in llm.trace[before:] if row.get("type") == "llm_generation"]
    return {"key": case["key"], "title": case["title"], "chapter": case["chapter"],
            "scenes": [{"scene_id": row.scene_id, "action": row.action, "reason": row.reason} for row in decisions],
            "drops": drops, "usage": usage}


def trial_drops(path: Path) -> dict:
    if not path.is_file():
        return {}
    return {row["key"]: [{"start": drop["start_before"], "resume": drop["resume"], "kind": drop.get("kind")}
                         for drop in row["plan"]["drops"]]
            for row in json.loads(path.read_text(encoding="utf-8"))}


def summarize(rows: list[dict], trial: dict, cases: dict) -> dict:
    done = [row for row in rows if not row.get("error")]
    drops = [drop for row in done for drop in row["drops"]]
    lengths = []
    for row in done:
        order = {line["id"]: index for index, line in enumerate(
            line for scene in cases[row["key"]]["payload"]["context"]["narrative"]["scenes"] for line in scene["utterances"])}
        ends = {scene["utterances"][-1]["id"]: scene for scene in cases[row["key"]]["payload"]["context"]["narrative"]["scenes"]}
        for drop in row["drops"]:
            start = order[drop["start_utterance_id"]]
            end = order[drop["resume_utterance_id"]] if drop["resume_utterance_id"] else next(
                order[last] + 1 for last in ends if order[last] >= start)
            lengths.append(end - start)
    near = same = 0
    for row in done:
        ours = [drop["start_utterance_id"] for drop in row["drops"]]
        theirs = [drop["start"] for drop in trial.get(row["key"], [])]
        order = {line["id"]: index for index, line in enumerate(
            line for scene in cases[row["key"]]["payload"]["context"]["narrative"]["scenes"] for line in scene["utterances"])}
        near += sum(any(abs(order[a] - order[b]) <= NEAR_LINES for b in theirs if b in order) for a in ours)
        same += sum(a in theirs for a in ours)
    actions = Counter(scene["action"] for row in done for scene in row["scenes"])
    return {"chapters": len(rows), "errors": [row["key"] for row in rows if row.get("error")],
            "drops": len(drops), "chapters_with_drops": sum(bool(row["drops"]) for row in done),
            "kinds": dict(Counter(drop["kind"] for drop in drops)),
            "to_scene_end": sum(drop["resume_utterance_id"] is None for drop in drops),
            "lines_silent": sorted(lengths), "trial_drops": sum(len(value) for value in trial.values()),
            "same_line_as_trial": same, f"within_{NEAR_LINES}_lines_of_trial": near,
            "scene_actions": dict(actions),
            "max_prompt_tokens": max((usage.get("prompt_tokens", 0) for row in done for usage in row["usage"] if usage),
                                     default=0)}


def page(rows: list[dict], trial: dict, cases: dict, summary: dict) -> str:
    parts = ["<!doctype html><meta charset=utf-8><title>BGMを止める瞬間（v3計画）</title>",
             ("<style>body{font-family:sans-serif;margin:16px;line-height:1.6}h2{margin-top:28px}"
              ".line{margin:2px 0}.start{background:#fde7d6}.resume{background:#e3f0dc}.id{color:#888;font-size:12px}"
              "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 6px}</style>"),
             "<h1>章のBGM設計（v3）が止める瞬間</h1>",
             f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=1))}</pre>"]
    for row in rows:
        parts.append(f"<h2>{html.escape(row['title'])} 第{row['chapter']}章</h2>")
        if row.get("error"):
            parts.append(f"<p>失敗: {html.escape(row['error'])}</p>")
            continue
        parts.append("<p>場面: " + "、".join(f"{scene['scene_id']} {scene['action']}" for scene in row["scenes"]) +
                     "。試行（注釈実験 v3）の止める行: " +
                     (html.escape("、".join(drop["start"] for drop in trial.get(row["key"], []))) or "なし") + "</p>")
        if not row["drops"]:
            parts.append("<p>止める瞬間なし</p>")
            continue
        lines = [line for scene in cases[row["key"]]["payload"]["context"]["narrative"]["scenes"]
                 for line in scene["utterances"]]
        index = {line["id"]: number for number, line in enumerate(lines)}
        for drop in row["drops"]:
            start = index[drop["start_utterance_id"]]
            end = index[drop["resume_utterance_id"]] if drop["resume_utterance_id"] else start + 1
            parts.append(f"<h3>{KIND_LABELS[drop['kind']]}：{html.escape(drop['reason'])}</h3>")
            for number in range(max(0, start - NEAR_LINES), min(len(lines), end + NEAR_LINES + 1)):
                line = lines[number]
                mark = (" start" if number == start else " resume" if drop["resume_utterance_id"] and number == end else "")
                parts.append(f"<div class='line{mark}'><span class='id'>{line['id']}</span> "
                             f"{html.escape(str(line.get('display_text', '')))}</div>")
            if not drop["resume_utterance_id"]:
                parts.append("<div class='id'>（場面の終わりまで止めたまま）</div>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--context-size", type=int, default=16384)
    parser.add_argument("--only", default="", help="comma-separated chapter keys")
    parser.add_argument("--report-only", action="store_true", help="rebuild the page from saved results")
    args = parser.parse_args()

    from services.worker.generation import pipeline

    root = pipeline.ROOT
    output = args.output_dir.resolve()
    connection = sqlite3.connect(f"file:{root / 'data/auto_drama.db'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        cases = chapters(connection)
        if args.only:
            cases = [case for case in cases if case["key"] in set(args.only.split(","))]
        newest = connection.execute("SELECT payload FROM job WHERE kind='m3_music_plan' AND status='completed' "
                                    "ORDER BY created_at DESC LIMIT 1").fetchone()
        profile = {**json.loads(newest["payload"])["profile"], "context_size": args.context_size}
    finally:
        connection.close()
    by_key = {case["key"]: case for case in cases}
    trial = trial_drops(root / TRIAL)
    if args.report_only:
        rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    else:
        from services.worker.generation.llm import LocalLLM, write_json
        from services.worker.generation.processes import gpu_lock
        from services.worker.model_config import select_config

        payload = {"seed": args.seed, "profile": profile}
        config = copy.deepcopy(select_config(pipeline.load_config(), payload, root))
        config["llm"]["total_timeout_seconds"] = 6 * 3600
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        with (gpu_lock(root / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
              LocalLLM(root, config, payload, output / "llm") as llm):
            for case in cases:
                rows.append(plan(llm, case))
                write_json(output / "results.json", rows)
                print(json.dumps({"key": case["key"], "drops": len(rows[-1].get("drops", [])),
                                  "error": rows[-1].get("error")}, ensure_ascii=False), flush=True)
    summary = summarize(rows, trial, by_key)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (output / "index.html").write_text(page(rows, trial, by_key, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
