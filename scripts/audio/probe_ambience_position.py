"""Can an LLM tell where a scene's ambience sound is heard from, and so whether and how it plays?

The rain and wind the annotation runs chose include sounds heard indoors from outside, sounds made in
the room itself, sounds heard through a radio, and rain only seen through a window or talked about.
Instead of an indoor/outdoor rule, two judgments decide how a sound plays:

  position  same_space     the sound arises where the listener is            -> as it is
            beyond_wall    it reaches the listener through walls or windows  -> muffled, only if heard
            inside_device  it comes through a radio, phone or recording      -> not ambience
  shown     heard          a line says it is heard or how it sounds
            present        the thing happens around the listener, its sound unwritten
            not_shown      only seen from afar, only talked about, or absent

The cases are the scenes from that look (outputs/music-ambience-v5-20261028/a, the chapter material
the annotation used): each scene is asked alone, with its location settings, its lines and one sound.
Arms differ only in the seed. v2 (--criteria v2) first copies the words about the sound from the cited
line; heard stands only when those words are really there. Run it while the product's worker is stopped.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

POSITIONS = ("same_space", "beyond_wall", "inside_device")
SHOWN = ("heard", "present", "not_shown")
PLAYS = {"as_is": "そのまま", "muffled": "こもらせる", "none": "鳴らさない"}
LABELS = {"same_space": "同じ空間", "beyond_wall": "壁の向こう", "inside_device": "機械の中",
          "heard": "聞こえる", "present": "そこにある", "not_shown": "示されない"}

SYSTEM = (
    "You place one environmental sound in a scene of a visual novel. The user gives the scene's location "
    "settings, its lines and the sound. The listener is the person or people whose surroundings the narration "
    "describes; the location settings may name another place. First write where: one short Japanese sentence "
    "on where the sound arises and where the listener is. Then choose position: same_space when the sound "
    "arises in the space where the listener is, including sounds the walls, windows or roof around the listener "
    "make themselves; beyond_wall when it arises outside the walls, windows, roof or tent around the listener "
    "and reaches them through those; inside_device when it is heard through a radio, phone, television, "
    "recording or other device. Then choose shown: heard when a line says the sound is heard or describes how it "
    "sounds; present when the lines show the thing happening around the listener but say nothing about its "
    "sound; not_shown when the thing is only seen from where the listener is, only talked about, or not in the "
    "lines at all. Then give line: the ID of the line that best shows it, or none."
)
# v2: before choosing shown, the words about the sound are copied from the line; heard stands only when
# those words are really in it, otherwise it falls back to present.
SYSTEM_V2 = SYSTEM.split(" Then choose shown:")[0] + (
    " Then give line: the ID of the line that best shows the sound or the thing making it, or none. Then copy "
    "sound_words: the words in that line that say the sound is heard or describe how it sounds, copied exactly "
    "as written; leave it empty when the line shows only the thing itself, not its sound. Then choose shown: "
    "heard only when sound_words is not empty; present when the lines show the thing happening around the "
    "listener but say nothing about its sound; not_shown when the thing is only seen from where the listener "
    "is, only talked about, or not in the lines at all."
)
SYSTEMS = {"v1": SYSTEM, "v2": SYSTEM_V2}

# (chapter key, scene, sound, acceptable positions, acceptable shown, acceptable plays), expected first.
CASES = [
    ("e69c0508-ch1", "s1", "激しい雨と嵐の音", ("beyond_wall", "same_space"), ("heard",), ("muffled", "as_is")),
    ("e69c0508-ch2", "s1", "激しい雨音", ("beyond_wall",), ("heard",), ("muffled",)),
    ("e69c0508-ch2", "s2", "激しい風と波の音", ("inside_device",), ("heard",), ("none",)),
    ("e69c0508-ch2", "s3", "激しい波の音", ("inside_device",), ("heard",), ("none",)),
    ("e69c0508-ch2", "s3", "激しい雨音", ("beyond_wall",), ("heard",), ("muffled",)),
    ("e69c0508-ch3", "s1", "弱い雨音", ("beyond_wall",), ("heard",), ("muffled",)),
    ("bb3bace7-ch1", "s2", "小雨の音", ("beyond_wall",), ("not_shown",), ("none",)),
    ("691cc1fa-ch3", "s1", "雨音", ("same_space",), ("present", "heard"), ("as_is",)),
    ("691cc1fa-ch3", "s2", "雨音", ("beyond_wall",), ("not_shown",), ("none",)),
    ("691cc1fa-ch3", "s3", "降りしきる雨音", ("same_space",), ("heard",), ("as_is",)),
    ("99ccc78c-ch1", "s1", "隙間風の音", ("same_space",), ("present", "heard"), ("as_is",)),
    ("99ccc78c-ch5", "s4", "空虚な風の音", ("beyond_wall",), ("heard",), ("muffled",)),
    ("76388d87-ch1", "s1", "静かな風", ("beyond_wall",), ("not_shown",), ("none",)),
    ("76388d87-ch1", "s1", "不自然に激しい風", ("same_space",), ("present", "heard"), ("as_is",)),
]


def schema(line_ids: list[str], criteria: str = "v1") -> dict:
    fields = {"where": {"type": "string", "maxLength": 120},
              "position": {"type": "string", "enum": list(POSITIONS)},
              "shown": {"type": "string", "enum": list(SHOWN)},
              "line": {"type": "string", "enum": ["none", *line_ids]}}
    if criteria == "v2":
        order = ("where", "position", "line", "sound_words", "shown")
        fields["sound_words"] = {"type": "string", "maxLength": 60}
    else:
        order = ("where", "position", "shown", "line")
    return {"type": "object", "additionalProperties": False, "required": list(order),
            "properties": {name: fields[name] for name in order}}


def words_in(words: str, text: str) -> bool:
    words = words.strip().strip("「」『』\"'")
    return bool(words) and words in text


def play(position: str | None, shown: str | None) -> str | None:
    """Sounds in the listener's space play when they are there; sounds beyond a wall only when heard."""
    if position is None:
        return None
    if position == "same_space":
        return "as_is" if shown in ("heard", "present") else "none"
    if position == "beyond_wall":
        return "muffled" if shown == "heard" else "none"
    return "none"


def scene_of(chapters: dict, key: str, scene_id: str) -> dict:
    return next(scene for scene in chapters[key]["material"]["scenes"] if scene["scene_id"] == scene_id)


def prompt(scene: dict, sound: str) -> str:
    return json.dumps({"location": scene["location"], "lines": scene["lines"], "sound": sound},
                      ensure_ascii=False)


def judge(case: tuple, answer: dict | None, line_text: str = "") -> dict:
    """With sound_words (v2), heard stands only when the copied words are in the cited line."""
    key, scene_id, sound, positions, shown, plays = case
    position = answer["position"] if answer else None
    answered = answer["shown"] if answer else None
    seen = answered
    found = None
    if answer and "sound_words" in answer:
        found = words_in(answer["sound_words"], line_text)
        if seen == "heard" and not found:
            seen = "present"
    decided = play(position, seen)
    return {"key": key, "scene_id": scene_id, "sound": sound, "acceptable": {"position": list(positions),
            "shown": list(shown), "play": list(plays)}, "where": answer["where"] if answer else None,
            "position": position, "shown": seen, "answered_shown": answered,
            "sound_words": answer.get("sound_words") if answer else None, "words_found": found,
            "line": answer["line"] if answer else None, "play": decided,
            "position_ok": position in positions, "shown_ok": seen in shown, "play_ok": decided in plays}


def classify(llm, chapters: dict, case: tuple, criteria: str = "v1") -> dict:
    scene = scene_of(chapters, case[0], case[1])
    lines = {line["id"]: line["text"] for line in scene["lines"]}
    try:
        answer = llm.structured("ambience-position", [{"role": "system", "content": SYSTEMS[criteria]},
                                                      {"role": "user", "content": prompt(scene, case[2])}],
                                schema(list(lines), criteria))
    except ValueError as exc:
        return {**judge(case, None), "error": str(exc)}
    return judge(case, answer, lines.get(answer["line"], ""))


def summarize(rows: list[dict]) -> dict:
    return {"cases": len(rows), "errors": sum(1 for row in rows if row.get("error")),
            **{name: sum(row[f"{name}_ok"] for row in rows) for name in ("position", "shown", "play")},
            "misses": [f"{row['key']} {row['scene_id']} {row['sound']}: {row['position']}/{row['shown']} → "
                       f"{row['play']}（可: {'/'.join(row['acceptable']['play'])}）"
                       for row in rows if not row["play_ok"]]}


def agreement(arms: dict) -> dict:
    names = sorted(arms)
    if len(names) < 2:
        return {}
    pairs = list(zip(arms[names[0]], arms[names[1]], strict=True))
    same = [(a["position"], a["shown"]) == (b["position"], b["shown"]) for a, b in pairs]
    return {"same_answer": sum(same), "same_play": sum(a["play"] == b["play"] for a, b in pairs), "cases": len(pairs)}


def page(arms: dict, chapters: dict) -> str:
    names = sorted(arms)
    parts = ["<!doctype html><meta charset=utf-8><title>環境音の聞こえる位置</title>",
             ("<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}"
              "td,th{border:1px solid #ccc;padding:4px 6px;vertical-align:top}.ng{background:#fdd}"
              ".part{background:#ffd}small{color:#555}</style>"),
             "<h1>環境音の聞こえる位置と鳴らし方</h1><table><tr><th>場面</th><th>音</th><th>正解（可）</th>"]
    parts += [f"<th>{name}</th>" for name in names]
    parts.append("</tr>")
    for index, case in enumerate(CASES):
        key, scene_id, sound, positions, shown, plays = case
        scene = scene_of(chapters, key, scene_id)
        lines = {line["id"]: line["text"] for line in scene["lines"]}
        expected = (f"{' / '.join(LABELS[value] for value in positions)}<br>{' / '.join(LABELS[value] for value in shown)}"
                    f"<br>→ {' / '.join(PLAYS[value] for value in plays)}")
        cells = []
        for name in names:
            row = arms[name][index]
            style = "" if row["position_ok"] and row["shown_ok"] else (' class="part"' if row["play_ok"] else ' class="ng"')
            if row["play"] is None:
                cells.append(f"<td{style}>失敗</td>")
                continue
            quoted = html.escape(lines.get(row["line"], "")[:80])
            words = ""
            if row.get("sound_words") is not None:
                mark = "" if row["words_found"] else "（行に無い）"
                words = f"<br>音の言葉: 「{html.escape(row['sound_words'])}」{mark}"
            cells.append(f"<td{style}>{LABELS[row['position']]} / {LABELS[row['shown']]} → {PLAYS[row['play']]}"
                         f"<br><small>{html.escape(row['where'])}<br>{row['line']}: {quoted}{words}</small></td>")
        place = html.escape(f"{chapters[key]['title']} 第{chapters[key]['chapter']}章 {scene_id}・{scene['location']['name']}")
        parts.append(f"<tr><td>{place}</td><td>{html.escape(sound)}</td><td>{expected}</td>{''.join(cells)}</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--material", type=Path, default=Path("outputs/music-ambience-v5-20261028/a/results.json"))
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--arms", default="a,b", help="comma-separated arm names; each adds 1000 to the seed")
    parser.add_argument("--context-size", type=int, default=16384)
    parser.add_argument("--criteria", choices=sorted(SYSTEMS), default="v1",
                        help="v2: copy the words about the sound from the line; heard needs them to be there")
    args = parser.parse_args()

    from services.worker.generation import pipeline
    from services.worker.generation.llm import LocalLLM, write_json
    from services.worker.generation.processes import gpu_lock
    from services.worker.model_config import select_config

    root = pipeline.ROOT
    chapters = {row["key"]: row for row in json.loads((root / args.material).read_text(encoding="utf-8"))}
    output = args.output_dir.resolve()
    connection = sqlite3.connect(root / "data/auto_drama.db")
    try:
        # The product's newest LLM settings, as the music plan uses them.
        newest = connection.execute("SELECT payload FROM job WHERE kind='m3_music_plan' AND status='completed' "
                                    "ORDER BY created_at DESC LIMIT 1").fetchone()
    finally:
        connection.close()
    profile = {**json.loads(newest[0])["profile"], "context_size": args.context_size}
    output.mkdir(parents=True, exist_ok=True)
    arms = {}
    for number, arm in enumerate(value for value in args.arms.split(",") if value):
        payload = {"seed": args.seed + 1000 * number, "profile": profile}
        config = copy.deepcopy(select_config(pipeline.load_config(), payload, root))
        config["llm"]["total_timeout_seconds"] = 3600
        folder = output / arm
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        with (gpu_lock(root / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
              LocalLLM(root, config, payload, folder / "llm") as llm):
            for case in CASES:
                rows.append(classify(llm, chapters, case, args.criteria))
                write_json(folder / "results.json", rows)
        arms[arm] = rows
        print(arm, json.dumps(summarize(rows), ensure_ascii=False), flush=True)
    report = {"criteria": args.criteria, "profile": profile, "arms": {arm: summarize(rows) for arm, rows in arms.items()},
              "agreement": agreement(arms)}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (output / "index.html").write_text(page(arms, chapters), encoding="utf-8")
    print(json.dumps(report["agreement"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
