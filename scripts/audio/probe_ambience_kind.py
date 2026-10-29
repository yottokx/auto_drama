"""Can an LLM tell, from a Japanese description, how an ambience sound behaves and so how it is looped?

Experiment 2 of the ambience library (experiment 1, scripts/audio/probe_ambience_loops.py, showed that
each kind loops naturally when it is given). The four kinds are the four loop methods:

  continuous  an even mass or steady tone, drifting at random      texture loop
  swell       surging and drawing back in a slow repeated cycle    trough-to-trough loop
  ticks       short separate sounds at a steady interval           loop on whole ticks
  occasional  separate occurrences with quiet between them         cut once, scattered over a bed

The library-facing choice is only two-way (keeps sounding / heard now and then), so each answer is
scored both ways. The descriptions are the sounds the BGM-and-ambience annotation runs wrote for 35
chapters (outputs/music-ambience-*), one per distinct sound, plus the trial library's own labels;
music and things that are not sounds are left out. Each description is asked alone, as a library
entry would be. Arms differ only in the seed. Run it while the product's worker is stopped.
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

KINDS = ("continuous", "swell", "ticks", "occasional")
KIND_LABELS = {"continuous": "一様に続く", "swell": "寄せては返す", "ticks": "規則的に刻む", "occasional": "時々起きる"}
LASTING = {"continuous": "keeps", "swell": "keeps", "ticks": "keeps", "occasional": "now_and_then"}

KIND_SYSTEM = (
    "You prepare a sound for an ambience library. The user describes the sound in Japanese. Decide how the "
    "sound behaves over a minute of listening, because that decides how it is looped or played. First write "
    "over_a_minute: one short Japanese sentence on what is heard during that minute. Then choose kind: "
    "continuous when the sound goes on without gaps as an even mass or a steady tone, possibly drifting louder "
    "and softer at random; swell when it goes on while surging and drawing back in a slow cycle that repeats "
    "every few seconds; ticks when it is a series of short separate sounds at a steady interval; occasional "
    "when it is heard as separate occurrences with quiet between them, each lasting from a moment to tens of "
    "seconds."
)
# v2: a sound with a bed that keeps sounding is judged by the bed, not by details that come and go on it.
KIND_SYSTEM_V2 = KIND_SYSTEM + (
    " When something keeps sounding underneath and only details on top of it come now and then, judge the "
    "sound by what keeps sounding; occasional is only for a sound that stops entirely between occurrences."
)
CRITERIA = {"v1": KIND_SYSTEM, "v2": KIND_SYSTEM_V2}

# (description, acceptable kinds with the expected one first). Several kinds are acceptable where the
# description alone leaves the behaviour open, e.g. birdsong as a chorus or as single calls.
CASES = [
    ("しとしと降る雨", ("continuous",)),
    ("霧雨", ("continuous",)),
    ("小雨の音", ("continuous",)),
    ("本降りの雨", ("continuous",)),
    ("激しい雨音", ("continuous",)),
    ("嵐（激しい風雨、遠い雷）", ("continuous",)),
    ("窓を叩く雨音", ("continuous",)),
    ("規則的な雨音", ("continuous",)),
    ("不気味な残響を伴う雨音", ("continuous",)),
    ("弱まりつつある雨音", ("continuous",)),
    ("降りしきる雨", ("continuous",)),
    ("雨音と家屋の軋み", ("continuous", "occasional")),
    ("夜風の音", ("continuous",)),
    ("隙間風", ("continuous",)),
    ("冷たい強風", ("continuous",)),
    ("心地よい風", ("continuous",)),
    ("空虚な風の音", ("continuous",)),
    ("不自然に激しい風", ("continuous",)),
    ("雨上がりの湿った風", ("continuous",)),
    ("激しい波の音", ("swell",)),
    ("荒れた海", ("swell",)),
    ("穏やかな波が寄せる浜辺", ("swell",)),
    ("激しい風と波の音", ("swell", "continuous")),
    ("船のエンジン音と波を蹴る音", ("continuous", "swell")),
    ("焚き火のはぜる音", ("continuous",)),
    ("松明の燃える音", ("continuous",)),
    ("暖炉・松明の火", ("continuous",)),
    ("街の雑踏、人通り", ("continuous",)),
    ("休み時間の教室のざわめき", ("continuous",)),
    ("市場の喧騒", ("continuous",)),
    ("店内の話し声と食器の音", ("continuous",)),
    ("駅前の喧騒", ("continuous",)),
    ("戦場の喧騒", ("continuous",)),
    ("遠くの部活動の掛け声や生徒たちの話し声", ("continuous",)),
    ("兵士たちの歓声", ("continuous", "occasional")),
    ("機械の低い唸り（機関室、宇宙船）", ("continuous",)),
    ("巡視艇の穏やかな駆動音", ("continuous",)),
    ("エンジンのアイドリング音", ("continuous",)),
    ("電子レンジの作動音", ("continuous",)),
    ("コピー機の待機音", ("continuous",)),
    ("無線機の激しいノイズ", ("continuous",)),
    ("静かな夕方の空気", ("continuous",)),
    ("朝の森の鳥のさえずり", ("continuous", "occasional")),
    ("遠くで鳴く鳥の声", ("occasional",)),
    ("夜の虫の声", ("continuous", "ticks")),
    ("夏の昼の蝉の大合唱", ("continuous",)),
    ("モノノケの低い呻き", ("occasional", "continuous")),
    ("モノノケの咆哮", ("occasional",)),
    ("不快な耳鳴り", ("continuous",)),
    ("不気味なノイズ", ("continuous",)),
    ("呪力の嵐の轟音", ("continuous",)),
    ("不気味な泥の蠢き", ("continuous", "occasional")),
    ("モノノケの高周波振動", ("continuous",)),
    ("時計の秒針の音", ("ticks",)),
    ("時計のチクタク音", ("ticks",)),
    ("行軍の足音", ("ticks", "continuous")),
    ("重装歩兵の規則的な足音", ("ticks",)),
    ("足音、規則正しく", ("ticks",)),
    ("金属が擦れ合う規則的な音", ("ticks",)),
    ("馬の蹄の音", ("ticks", "occasional")),
    ("障壁への連続的な打撃音", ("ticks", "occasional")),
    ("門を叩く重い衝撃音", ("occasional", "ticks")),
    ("遠くの電車の走行音", ("occasional",)),
    ("電車の走行音", ("occasional", "continuous")),
    ("遠い車の走行音", ("occasional", "continuous")),
    ("車のクラクション", ("occasional",)),
    ("遠くで鳴るチャイム", ("occasional",)),
    ("扉が閉まる乾いた音", ("occasional",)),
    ("自動ドアの開閉音", ("occasional",)),
    ("ページをめくる音", ("occasional",)),
    ("電子レンジの完了音（チン）", ("occasional",)),
    ("兵士たちの悲鳴と殺戮の音", ("occasional", "continuous")),
    ("鈍い衝撃音", ("occasional",)),
    ("建物の崩壊音", ("occasional",)),
    ("錆びついた扉の軋み音", ("occasional",)),
    ("スマホの操作音", ("occasional",)),
    ("万年筆の筆記音", ("continuous", "occasional")),
    ("鋭い電子音", ("occasional",)),
    ("洞窟の水滴と反響", ("occasional", "ticks")),
]


def schema():
    return {"type": "object", "additionalProperties": False, "required": ["over_a_minute", "kind"],
            "properties": {"over_a_minute": {"type": "string", "maxLength": 120},
                           "kind": {"type": "string", "enum": list(KINDS)}}}


def judge(description: str, acceptable: tuple, answer: dict | None) -> dict:
    kind = answer["kind"] if answer else None
    lasting = {LASTING[value] for value in acceptable}
    return {"description": description, "acceptable": list(acceptable), "kind": kind,
            "over_a_minute": answer["over_a_minute"] if answer else None,
            "expected": kind == acceptable[0], "accepted": kind in acceptable,
            "lasting_ok": kind is not None and LASTING[kind] in lasting}


def classify(llm, description: str, acceptable: tuple, criteria: str = "v1") -> dict:
    try:
        answer = llm.structured("ambience-kind", [{"role": "system", "content": CRITERIA[criteria]},
                                                  {"role": "user", "content": description}], schema())
    except ValueError as exc:
        return {**judge(description, acceptable, None), "error": str(exc)}
    return judge(description, acceptable, answer)


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    return {"cases": count, "errors": sum(1 for row in rows if row.get("error")),
            "expected": sum(row["expected"] for row in rows), "accepted": sum(row["accepted"] for row in rows),
            "lasting_ok": sum(row["lasting_ok"] for row in rows),
            "kinds": dict(Counter(row["kind"] for row in rows)),
            "misses": [f"{row['description']}: {row['kind']}（可: {'/'.join(row['acceptable'])}）"
                       for row in rows if not row["accepted"]]}


def agreement(arms: dict) -> dict:
    names = sorted(arms)
    if len(names) < 2:
        return {}
    first, second = arms[names[0]], arms[names[1]]
    same = sum(a["kind"] == b["kind"] for a, b in zip(first, second, strict=True))
    return {"same_kind": same, "cases": len(first),
            "differ": [f"{a['description']}: {a['kind']} / {b['kind']}"
                       for a, b in zip(first, second, strict=True) if a["kind"] != b["kind"]]}


def page(arms: dict) -> str:
    names = sorted(arms)
    parts = ["<!doctype html><meta charset=utf-8><title>環境音の種類判定</title>",
             ("<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}"
              "td,th{border:1px solid #ccc;padding:4px 6px;vertical-align:top}.ng{background:#fdd}"
              ".alt{background:#ffd}</style>"),
             "<h1>説明文からの音の種類判定</h1><table><tr><th>説明</th><th>正解（可）</th>"]
    parts += [f"<th>{name}</th>" for name in names]
    parts.append("</tr>")
    for index, (description, acceptable) in enumerate(CASES):
        labels = " / ".join(KIND_LABELS[value] for value in acceptable)
        cells = []
        for name in names:
            row = arms[name][index]
            style = "" if row["expected"] else (' class="alt"' if row["accepted"] else ' class="ng"')
            kind = KIND_LABELS.get(row["kind"], "失敗")
            cells.append(f"<td{style}>{kind}<br><small>{html.escape(row['over_a_minute'] or '')}</small></td>")
        parts.append(f"<tr><td>{html.escape(description)}</td><td>{labels}</td>{''.join(cells)}</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--arms", default="a,b", help="comma-separated arm names; each adds 1000 to the seed")
    parser.add_argument("--context-size", type=int, default=16384)
    parser.add_argument("--criteria", choices=sorted(CRITERIA), default="v1",
                        help="v2: a sound is judged by what keeps sounding, not by details that come and go")
    args = parser.parse_args()

    from services.worker.generation import pipeline
    from services.worker.generation.llm import LocalLLM, write_json
    from services.worker.generation.processes import gpu_lock
    from services.worker.model_config import select_config

    root = pipeline.ROOT
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
            for description, acceptable in CASES:
                rows.append(classify(llm, description, acceptable, args.criteria))
                write_json(folder / "results.json", rows)
        arms[arm] = rows
        print(arm, json.dumps(summarize(rows), ensure_ascii=False), flush=True)
    report = {"criteria": args.criteria, "profile": profile, "arms": {arm: summarize(rows) for arm, rows in arms.items()},
              "agreement": agreement(arms)}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (output / "index.html").write_text(page(arms), encoding="utf-8")
    print(json.dumps(report["agreement"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
