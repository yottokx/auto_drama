"""Let the scene writer give a voice the plan did not cast its own line, instead of borrowing a cast ID.

The writer's grammar allows only NARRATOR and the planned cast's IDs at the start of a line, so
a grandfather heard over a radio, but left out of the plan, speaks as the protagonist. This
replays saved scene-writing requests, unchanged except that the grammar also allows
"VOICE[name]: " and the instruction says to use it for a voice outside the planned cast:

  current  the saved answer itself (no call)
  voice    the replay with the VOICE line allowed

Scenes: the focused work's scenes where such a voice was heard, and the first scene of
another work's second chapter as a control. Nothing in the product changes.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_plan_new_characters as plans

from services.worker.generation import pipeline
from services.worker.generation.llm import write_json

RULE = "人物名ではなく指定IDを使い、半角コロン直後に半角空白1つを入れます。"
VOICE_RULE = ("計画の登場人物にいない人物の声（電話・無線・録音や留守電・映像・回想の声、故人など）を"
              "台詞にするときは、人物IDの代わりに『VOICE[呼び名]: 本文』とします。"
              "登場人物の台詞はVOICEにしません。")
LINE = re.compile(r"^(NARRATOR|VOICE\[[^\]\r\n]+\]|[^:\s]+): (.*)$")


def scene_requests(plan_path: Path) -> list[tuple[Path, dict]]:
    """The chapter's first-attempt scene-writing requests, in order, after its plan."""
    folder, number = plan_path.parent, int(plan_path.name.split("-", 1)[0])
    rows = []
    for path in sorted(folder.glob("*-script-scene-*.json"), key=lambda p: int(p.name.split("-", 1)[0])):
        if int(path.name.split("-", 1)[0]) <= number:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["request"].get("grammar", "").startswith("root ::= utterance"):
            rows.append((path, data))
    return rows


def with_voice(request: dict) -> dict:
    request = copy.deepcopy(request)
    grammar = request["grammar"]
    assert "utterance ::= (" in grammar
    request["grammar"] = grammar.replace("utterance ::= (", 'utterance ::= ("VOICE[" [^\\]\\r\\n]+ "]: " | ', 1)
    system = request["messages"][0]["content"]
    assert RULE in system
    request["messages"][0]["content"] = system.replace(RULE, RULE + VOICE_RULE, 1)
    return request


def lines(text: str) -> list[tuple[str, str]]:
    return [(match.group(1), match.group(2)) for line in text.splitlines()
            if (match := LINE.match(line.strip()))]


def report(results: list[dict], output: Path) -> None:
    out = ["# 本文で計画外の声を VOICE[呼び名] として書けるか（推論なし）", "",
           "| 作品 | 章 | 場面 | 現行の話者 | VOICE を許した話者 |", "| --- | ---: | --- | --- | --- |"]
    for row in results:
        cells = []
        for arm in ("current", "voice"):
            value = row["arms"].get(arm, {})
            cells.append("失敗" if "error" in value else ", ".join(f"{k}×{n}" for k, n in value.get("speakers", {}).items()))
        out.append(f"| {row['title']} | {row['chapter']} | {row['scene']} | {cells[0]} | {cells[1]} |")
    out += ["", "## VOICE の行", ""]
    for row in results:
        voiced = row["arms"].get("voice", {}).get("voice_lines", [])
        if voiced:
            out.append(f"### {row['title']} 第{row['chapter']}章 {row['scene']}")
            out += [f"- {speaker}: {text[:80]}" for speaker, text in voiced]
            out.append("")
    usage = [row["arms"]["voice"].get("usage", {}) for row in results if "voice" in row["arms"]]
    out += ["## トークン", "", (f"呼び出し {len(usage)} 回、入力合計 {sum(u.get('prompt_tokens') or 0 for u in usage):,}、"
                               f"出力合計 {sum(u.get('completion_tokens') or 0 for u in usage):,}。")]
    (output / "report.md").write_bytes(("\n".join(out) + "\n").encode("utf-8"))


def summarize(text: str) -> dict:
    parsed = lines(text)
    return {"speakers": dict(Counter(speaker for speaker, _ in parsed)),
            "voice_lines": [(speaker, body) for speaker, body in parsed if speaker.startswith("VOICE[")],
            "text": text}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus", default="凪の島")
    parser.add_argument("--focus-scenes", default="1:1,2:4,3:1", help="chapter:scene-number pairs of the focused work")
    parser.add_argument("--others", type=int, default=6)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    wanted = {tuple(map(int, pair.split(":"))) for pair in args.focus_scenes.split(",")}
    chosen = []
    for row in plans.choose(plans.saved_plans(), args.focus, args.others):
        scenes = scene_requests(row["path"])
        for number, (path, data) in enumerate(scenes, 1):
            focused = args.focus in row["title"]
            if (focused and (row["chapter"], number) in wanted) or (not focused and number == 1):
                chosen.append({"title": row["title"], "chapter": row["chapter"], "scene": f"s{number}",
                               "path": path, "data": data})
    for row in chosen:
        print(row["title"], row["chapter"], row["scene"], row["path"].name, flush=True)
    if args.dry:
        return 0
    saved = output / "results.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else []
    done = {row["path"] for row in results}
    from services.worker.generation.model_routing import RoutedLLM
    from services.worker.generation.processes import gpu_lock

    config = copy.deepcopy(pipeline.load_config())
    config["llm"]["total_timeout_seconds"] = 6 * 3600
    payload = {"seed": 20261016, "profile": {}, "profiles": {}, "story_workflow_version": 2,
               "workflow_policy": "script_continuation_v1"}
    with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
          RoutedLLM(pipeline.ROOT, config, payload, output / "llm") as llm):
        for row in chosen:
            if str(row["path"]) in done:
                continue
            original = row["data"]["response"]["choices"][0]["message"]["content"]
            record = {"path": str(row["path"]), "title": row["title"], "chapter": row["chapter"],
                      "scene": row["scene"], "arms": {"current": summarize(original)}}
            request = with_voice(row["data"]["request"])
            start, trace_start = time.monotonic(), len(llm.trace)
            try:
                llm.select_purpose("script-scene")
                llm._ensure_runtime()
                reply = llm.chat("script-scene", request["messages"], allow_truncated=True, seed=request["seed"],
                                 grammar=request["grammar"], max_tokens=request["max_tokens"])
                record["arms"]["voice"] = summarize(reply.get("content") or "")
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                record["arms"]["voice"] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
            usage = [item for item in llm.trace[trace_start:] if item.get("type") == "llm_generation"]
            record["arms"]["voice"].update(seconds=round(time.monotonic() - start, 1),
                                           usage=usage[-1].get("usage", {}) if usage else {})
            results.append(record)
            write_json(saved, results)
            print(json.dumps({"title": row["title"], "chapter": row["chapter"], "scene": row["scene"],
                              "speakers": record["arms"]["voice"].get("speakers"),
                              "error": record["arms"]["voice"].get("error")}, ensure_ascii=False), flush=True)
    report(results, output)
    print((output / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
