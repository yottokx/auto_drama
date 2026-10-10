"""Summarize how a text-only script experiment used its characters' alternate outfits.

Reads a finished or interrupted output directory of check_text.py; runs no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.contracts.outfits import DEFAULT_OUTFIT, snapshot_outfits

CLOTHING_WORDS = ("服", "着替", "変身", "姿", "衣装", "まとっ", "纏")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def summarize(output: Path) -> dict:
    manifest = _read(output / "experiment.json")
    state = _read(output / "draft-state.json") or {}
    outfits = snapshot_outfits(manifest["approval_snapshot"])
    names = {cid: {row.id: row.name for row in rows} for cid, rows in outfits.items()}
    alternates = [(cid, row.id) for cid, rows in outfits.items() for row in rows if row.id != DEFAULT_OUTFIT]
    plot = state.get("plot") or {"chapters": []}
    allocation = {row["number"]: row.get("outfit_appearances", []) for row in plot["chapters"]}
    allocated = {(row["character_id"], row["outfit_id"]) for rows in allocation.values() for row in rows}
    chapters = []
    for number in sorted(allocation):
        plan = _read(output / "chapters" / f"chapter-{number:03d}.plan.json")
        text = output / "chapters" / f"chapter-{number:03d}.md"
        scenes = [{"id": scene["id"], "location_id": scene["location_id"],
                   "character_ids": scene["character_ids"], "outfits": scene.get("outfits", [])}
                  for scene in (plan or {"scenes": []})["scenes"]]
        shown = {(row["character_id"], look) for scene in scenes for row in scene["outfits"]
                 for look in (row["outfit_id"], row.get("change_to"))}
        required = [(row["character_id"], row["outfit_id"]) for row in allocation[number]]
        words = (*CLOTHING_WORDS, *(name for looks in names.values() for name in looks.values()))
        chapters.append({
            "number": number, "allocated": allocation[number], "planned": plan is not None,
            "scenes": scenes, "allocated_but_not_planned": [pair for pair in required if pair not in shown],
            "clothing_lines": [line for line in text.read_text(encoding="utf-8").splitlines()
                               if any(word in line for word in words)] if text.exists() else []})
    steps = {key: {"attempts": len(step["attempts"]),
                   "rejected": sum(bool(row.get("validation_error")) for row in step["attempts"])}
             for key, step in state.get("steps", {}).items()
             if key.startswith(("story-chain", "chapter-allocation", "outline", "plot-batch", "plan-"))
             or key.endswith("-outfit")}
    moments = []
    for scope, record in sorted(state.get("outfit_moments", {}).items()):
        scene = state.get("scene_commits", {}).get(scope, {}).get("scene", {"utterances": []})
        text = {row["id"]: row["display_text"] for row in scene["utterances"]}
        for change in record["changes"]:
            chosen = record["moments"].get(change["character_id"])
            moments.append({"scope": scope, **change, "utterance_id": chosen, "text": text.get(chosen),
                            "error": record.get("error")})
    return {"outfits": {cid: [row.model_dump(mode="json") for row in rows] for cid, rows in outfits.items()},
            "moments": moments,
            "names": names, "never_allocated": [pair for pair in alternates if pair not in allocated],
            "chapters": chapters, "steps": steps,
            "dropped_choices": state.get("outfit_plan_normalizations", {})}


def render(summary: dict) -> str:
    names = summary["names"]

    def look(cid, oid):
        return names.get(cid, {}).get(oid, oid)

    lines = ["# 別バージョンの使用状況", "", "## 定義", ""]
    for cid, rows in summary["outfits"].items():
        for row in rows:
            lines.append(f"- {cid} / {row['id']}（{row['name']}）: {row['description']} {row['occasion']}".rstrip())
    lines += ["", "## 章への割り当て", ""]
    lines.append("どの章にも割り当てられていない姿: "
                 + (", ".join(f"{cid}/{oid}" for cid, oid in summary["never_allocated"]) or "なし"))
    for chapter in summary["chapters"]:
        lines += ["", f"### 第{chapter['number']}章", ""]
        for row in chapter["allocated"]:
            lines.append(f"- 割り当て: {row['character_id']} / {look(row['character_id'], row['outfit_id'])} — {row['occasion']}")
        if not chapter["allocated"]:
            lines.append("- 割り当て: なし")
        if not chapter["planned"]:
            lines.append("- 場面計画: 未作成")
            continue
        if chapter["allocated_but_not_planned"]:
            lines.append("- **計画に現れない割り当て**: "
                         + ", ".join(f"{cid}/{oid}" for cid, oid in chapter["allocated_but_not_planned"]))
        lines += ["", "| 場面 | 場所 | 登場人物 | 姿（開始 → 途中変化） |", "|---|---|---|---|"]
        for scene in chapter["scenes"]:
            worn = "、".join(f"{row['character_id']}: {look(row['character_id'], row['outfit_id'])}"
                            + (f" → {look(row['character_id'], row['change_to'])}" if row.get("change_to") else "")
                            for row in scene["outfits"]) or "全員が通常の姿"
            lines.append(f"| {scene['id']} | {scene['location_id']} | {', '.join(scene['character_ids'])} | {worn} |")
        if chapter["clothing_lines"]:
            lines += ["", "本文で服装・姿に触れている行:", ""]
            lines += [f"- {line}" for line in chapter["clothing_lines"]]
    lines += ["", "## 場面途中で姿が変わる位置", "",
              "| 場面 | 人物 | 変化 | 計画で宣言 | 選ばれた発話 | 本文 |", "|---|---|---|---|---|---|"]
    for row in summary["moments"]:
        chosen = row["utterance_id"] or ("失敗: " + row["error"] if row["error"] else "なし（次の登場で切替）")
        lines.append(f"| {row['scope']} | {row['character_id']} | {look(row['character_id'], row['from'])} → "
                     f"{look(row['character_id'], row['to'])} | {'はい' if row['expected'] else 'いいえ'} | "
                     f"{chosen} | {(row['text'] or '')[:60]} |")
    lines += ["", "## 検査", "", "| 工程 | 試行 | うち却下 |", "|---|---:|---:|"]
    lines += [f"| {key} | {row['attempts']} | {row['rejected']} |" for key, row in summary["steps"].items()]
    if summary["dropped_choices"]:
        lines += ["", "場面にいない人物への指定を落とした件数: "
                  + ", ".join(f"第{number}章 {len(rows)}件" for number, rows in summary["dropped_choices"].items())]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = summarize(args.output_dir)
    (args.output_dir / "outfits.md").write_text(render(summary), encoding="utf-8", newline="\n")
    print(json.dumps({key: summary[key] for key in ("never_allocated", "steps", "dropped_choices")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
