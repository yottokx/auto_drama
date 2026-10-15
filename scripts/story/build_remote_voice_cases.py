"""Copy generated scenes that test remote-voice judgement into a labelled verification set.

Each case is a whole scene exactly as the staging step received it (setting, hints, plan and all
of its line batches), with the answers a remote-voice judgement should give:

  expected.lines    speaker lines whose answer is not plainly "present", plus chosen negatives.
                    voice:   accepted answers, preferred first: present / remote / recollection
                    channel: accepted voice treatments for a remote line, preferred first:
                             phone / voicemail / radio / speaker / telepathy / message
                    A speaker line that is not listed is expected to be "present".
  expected.absent   characters in the plan's cast who are never physically in the scene
  expected.arrives  characters who are heard first and come on stage later: accepted first
                    in-person lines; "start" means appearing from the scene's start is also fine

Nothing is written back to the source jobs. Synthetic scenes for patterns the generated works
lack live beside the output, in synthetic.json, with the same shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "services/worker/cache/m2/jobs"
OUTPUT = ROOT / "docs/experiments/assets/remote-voice-cases/real.json"

REMOTE = ["remote"]
UNSURE = ["remote", "present"]


def remote(*lines, channel, voice=REMOTE):
    return {line: {"voice": voice, "channel": channel} for line in lines}


def present(*lines, also=()):
    return {line: {"voice": ["present", *also]} for line in lines}


CASES = [
    {"id": "galaxy-bridge-monitor", "job": "d715ba4f", "scene": "c001-s1",
     "pattern": "ブリッジのメインコンソールに上官が映る通信",
     "why": "計画の登場人物に通信相手が入っている。21〜30行目のまとまりには通信を示す地の文が無い。",
     "lines": remote("s1-u10", "s1-u13", "s1-u16", "s1-u20", "s1-u22", "s1-u24", "s1-u29", "s1-u33",
                     channel=["speaker", "radio"]),
     "absent": ["ヴァレリウス"]},
    {"id": "hangar-intercom-then-face", "job": "725ca120", "scene": "c001-s1",
     "pattern": "同じ格納庫で通信機越しに話し、途中から対面する",
     "why": "相手は機体の下の点検口にいて姿が見えない。s1-u10で這い出して以降は対面。",
     "lines": {**remote("s1-u5", "s1-u8", channel=["radio"], voice=UNSURE),
               **present("s1-u4", "s1-u9")},
     "arrives": {"マルクス・ヴォーン": ["s1-u11", "start"]}},
    {"id": "mecha-duel-comm", "job": "725ca120", "scene": "c001-s2",
     "pattern": "機体同士が対峙し、パイロットが通信回線で話す",
     "why": "相手の機体は目の前にあるが本人の姿は見えない。どちらの扱いも成り立つ。",
     "lines": remote("s2-u8", "s2-u10", "s2-u12", "s2-u16", "s2-u19", "s2-u23", "s2-u25",
                     channel=["radio", "speaker"], voice=UNSURE),
     "absent": ["ゼクス・ヴァン・クロムウェル"], "absent_optional": True},
    {"id": "mecha-defeat-comm", "job": "80ab0552", "scene": "c002-s2",
     "pattern": "大破した機体へ、敵機から通信回線で語りかける",
     "why": "通信を示す地の文は s2-u3 と最後の s2-u20 だけ。",
     "lines": remote("s2-u2", "s2-u5", "s2-u7", "s2-u9", "s2-u12", "s2-u15", "s2-u17", "s2-u19",
                     channel=["radio", "speaker"], voice=UNSURE),
     "absent": ["ゼクス・ヴァン・クロムウェル"], "absent_optional": True},
    {"id": "earpiece-support", "job": "4ab9c10d", "scene": "c002-s3",
     "pattern": "耳の小型通信機で、離れた場所から戦闘を指示する",
     "why": "場面の最初から最後まで相手は来ない。本人が通信機に向かって話す行（s3-u36）はその場の台詞。",
     "lines": {**remote("s3-u4", "s3-u7", "s3-u9", "s3-u11", "s3-u17", "s3-u19", "s3-u21", "s3-u25",
                        "s3-u28", "s3-u30", "s3-u37", "s3-u40", channel=["radio"]),
               **present("s3-u36")},
     "absent": ["乾 奏太"]},
    {"id": "phone-alert-in-alley", "job": "a205a8cd", "scene": "c001-s1",
     "pattern": "スマートフォンへの緊急連絡",
     "why": "通話か、メッセージか、録音かは本文から決まらない。",
     "lines": remote("s1-u17", channel=["phone", "message", "voicemail", "speaker"]),
     "absent": ["佐伯 凛（さえき りん）"]},
    {"id": "chat-messages-at-night", "requests": "outputs/outfits/school-moments-a/requests", "scene": "c002-s2",
     "pattern": "メッセージアプリの文面とボイスメモ",
     "why": "声ではなく文面として届いている。計画の登場人物に送り主が入っている。",
     "lines": remote("s2-u9", "s2-u12", channel=["message", "voicemail"]),
     "absent": ["春日 ひなた"]},
    {"id": "voicemail-murmured", "job": "98841b93", "scene": "c001-s1",
     "pattern": "留守電の文面を本人がその場で呟く",
     "why": "直前の地の文が録音の声を描写している。s1-u9 は直後に「小さく呟き」とあり、その場の台詞。"
            "s1-u13 は手がかりが無く、録音を思い返しているとも読める。",
     "lines": {**present("s1-u9"), **present("s1-u13", also=("recollection",))}},
    {"id": "voicemail-retold", "job": "98841b93", "scene": "c001-s2",
     "pattern": "留守電の内容を相手に語って聞かせる",
     "why": "s2-u26 と s2-u28 は留守電の文面の引用だが、話しているのは目の前の本人。",
     "lines": present("s2-u24", "s2-u26", "s2-u28", "s2-u33")},
    {"id": "voice-of-the-void", "job": "6ed47483", "scene": "c001-scene-1-1",
     "pattern": "姿の無い管理者が空間そのものから話す",
     "why": "機器は無く、合成音声が方向なく響く。",
     "lines": remote("scene-1-1-u11", "scene-1-1-u14", "scene-1-1-u17", "scene-1-1-u19", "scene-1-1-u21",
                     "scene-1-1-u23", channel=["speaker", "telepathy"]),
     "absent": ["観測者（オブザーバー）"]},
    {"id": "system-voice-as-narration", "job": "cd3c953e", "scene": "c003-scene-2",
     "pattern": "頭の中に響くシステムの声が地の文の【】で書かれている",
     "why": "声の主は登場人物ではなく、話者の無い地の文。台詞として扱わない。",
     "lines": present("scene-2-u7", "scene-2-u14")},
    {"id": "voice-before-appearing", "job": "8c211db8", "scene": "c002-s2",
     "pattern": "霧の中から声が先に聞こえ、直後に姿を現す",
     "why": "声の主はその場にいて、まだ見えていないだけ。",
     "lines": present("s2-u4"),
     "arrives": {"朧（おぼろ）": ["s2-u7", "start"]}},
    {"id": "voice-from-behind", "job": "fc90e40c", "scene": "c001-s1",
     "pattern": "背後から声をかけられ、振り返ると相手がいる",
     "why": "その場にいる人物の、画面外からの声。",
     "lines": present("s1-u7"),
     "arrives": {"雨宮 栞": ["s1-u11", "start"]}},
    {"id": "letter-read-aloud", "job": "2e338422", "scene": "c003-scene-2",
     "pattern": "書いている手紙をその場で読み上げる",
     "why": "宛先の人物は不在だが、話しているのは書き手本人。直前の地の文に不在の友人の回想がある。",
     "lines": present("scene-2-u17", "scene-2-u24")},
]


def parse(path: Path) -> dict:
    body = json.loads(path.read_text(encoding="utf-8"))["request"]["messages"][1]["content"]
    rest = body[body.index("資料: ") + len("資料: "):]
    setting, rest = rest.split("\n読み上げない声の指示: ", 1)
    hints, rest = rest.split("\n計画: ", 1)
    plan, rest = rest.split("\n確定した本文とID: ", 1)
    lines = json.loads(rest[:rest.index("\n本文を再入力")])
    return {"setting": json.loads(setting), "hints": json.loads(hints), "plan": json.loads(plan), "lines": lines}


def requests_dir(case: dict) -> Path:
    if "requests" in case:
        return ROOT / case["requests"]
    return next(path / "script/requests" for path in sorted(JOBS.glob(f"*/{case['job']}*"))
                if (path / "script/requests").is_dir())


def scene(case: dict) -> dict:
    folder = requests_dir(case)
    batches = [parse(path) for path in folder.glob(f"{case['scene']}-staging-*-1.json")]
    if not batches:
        raise FileNotFoundError(f"No staging input for {case['scene']} in {folder}")
    batches.sort(key=lambda row: int(row["lines"][0]["id"].rsplit("-u", 1)[1]))
    first = batches[0]
    hints = {}
    for row in batches:
        hints.update(row["hints"])
    return {"source": str(folder.relative_to(ROOT)).replace("\\", "/"), "setting": first["setting"],
            "hints": hints, "plan": first["plan"],
            "batches": [[line["id"] for line in row["lines"]] for row in batches],
            "lines": [line for row in batches for line in row["lines"]]}


def build(case: dict) -> dict:
    data = scene(case)
    ids = {row["name"]: row["id"] for row in data["setting"]["characters"]}
    known = {line["id"]: line for line in data["lines"]}
    for line_id in case["lines"]:
        if line_id not in known or known[line_id]["speaker_id"] is None:
            raise ValueError(f"{case['id']}: {line_id} is not a speaker line of the scene")
    expected = {"lines": case["lines"], "absent": [ids[name] for name in case.get("absent", [])],
                "arrives": {ids[name]: lines for name, lines in case.get("arrives", {}).items()}}
    if case.get("absent_optional"):
        expected["absent_optional"] = True
    return {"id": case["id"], "kind": "generated", "title": data["setting"]["world"].get("title", ""),
            "pattern": case["pattern"], "why": case["why"], "source": {"requests": data["source"], "scene": case["scene"]},
            "expected": expected, "batches": data["batches"],
            **{key: data[key] for key in ("setting", "hints", "plan", "lines")}}


def synthetic(source: dict, real: dict[str, dict]) -> dict:
    """Expand a hand-written scene into the shape the staging step receives."""
    if "setting_from" in source:
        base = real[source["setting_from"]]
        setting, hints = base["setting"], base["hints"]
    else:
        setting = {"world": source["world"], "characters": source["characters"], "relationships": {"pairs": []},
                   "main_character_ids": [source["characters"][0]["id"]], "locations": [source["location"]]}
        hints = {}
    cast = source.get("cast") or [row["id"] for row in source["characters"]]
    plan = {"id": "s1", "location_id": source.get("location_id") or source["location"]["id"], "character_ids": cast,
            **{key: source["plan"][key] for key in ("objectives", "start_state")},
            "required_events": [{"id": f"event_{index}", "description": text}
                                for index, text in enumerate(source["plan"]["required_events"], 1)],
            **{key: source["plan"][key] for key in ("end_state", "atmosphere")}}
    speakers = {row["id"] for row in setting["characters"]}
    lines, labels, offset = [], {}, 0
    for index, row in enumerate(source["lines"], 1):
        speaker, text = row[0], row[1]
        if speaker is not None and speaker not in speakers:
            raise ValueError(f"{source['id']}: unknown speaker {speaker}")
        line_id = f"s1-u{index}"
        lines.append({"id": line_id, "speaker_id": speaker, "display_text": text, "spoken_text": text,
                      "source_start": offset, "source_end": offset + len(text), "inner_emotion": "未指定",
                      "voice_emotion": "neutral", "delivery": None})
        offset += len(text) + 1
        if len(row) > 2:
            if speaker is None:
                raise ValueError(f"{source['id']}: {line_id} is narration but has an expected answer")
            labels[line_id] = row[2]
    known = {line["id"] for line in lines}
    expected = {"lines": labels, **source["expected"]}
    for character in [*expected["absent"], *expected["arrives"]]:
        if character not in cast:
            raise ValueError(f"{source['id']}: {character} is not in the scene's cast")
    for accepted in expected["arrives"].values():
        if any(line != "start" and line not in known for line in accepted):
            raise ValueError(f"{source['id']}: arrival names an unknown line")
    return {"id": source["id"], "kind": "synthetic", "title": setting["world"].get("title", ""),
            "pattern": source["pattern"], "why": source["why"], "expected": expected,
            "batches": [[line["id"] for line in lines[start:start + 10]] for start in range(0, len(lines), 10)],
            "setting": setting, "hints": hints, "plan": plan, "lines": lines}


def write(path: Path, cases: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps({"schema_version": 1, "cases": cases}, ensure_ascii=False, indent=1)
                      + "\n").encode("utf-8"))
    for case in cases:
        print(f"{case['id']:<40} {case['title'][:18]:<18} lines={len(case['lines']):>3} "
              f"labelled={len(case['expected']['lines'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--synthetic-only", action="store_true",
                        help="expand synthetic-source.json against an existing real.json, without the job cache")
    args = parser.parse_args()
    folder = args.output.parent
    if args.synthetic_only:
        real = json.loads(args.output.read_text(encoding="utf-8"))["cases"]
    else:
        real = [build(case) for case in CASES]
        write(args.output, real)
    source = json.loads((folder / "synthetic-source.json").read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in real}
    write(folder / "synthetic.json", [synthetic(case, by_id) for case in source["cases"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
