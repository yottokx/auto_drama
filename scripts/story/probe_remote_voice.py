"""Ask the staging step, extended in this probe only, where each line's voice comes from.

The product's staging prompt and schema are taken from narrative._staging unchanged; this
probe adds two fields to every emotion annotation and one paragraph of instruction:

  voice_source   present / remote / recollection
  voice_channel  none / phone / voicemail / radio / speaker / telepathy / message

and checks the answers against docs/experiments/assets/remote-voice-cases (real.json and
synthetic.json, built by build_remote_voice_cases.py). Scenes are annotated in the product's
10-line batches. Arms:

  batch      the product's input for the batch, plus the added instruction
  preceding  the same, plus the scene's lines just before the batch as reference
  plain      the product's staging unchanged, run only when asked for; on the same seed as
             batch it shows what the added instruction changes in the existing annotations
             (side-effects.md)

The rest of each answer (emotions, expressions, directions) must still pass the product's own
checks, so a broken answer is reported as such rather than scored. Nothing in the product is
changed and nothing is written back to the source jobs.
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
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_staging_context as base

from packages.contracts.m3 import MappedUtterance, ScenePlan
from services.worker.generation import narrative, pipeline
from services.worker.generation.llm import write_json

CASES = pipeline.ROOT / "docs/experiments/assets/remote-voice-cases"
SOURCES = ["present", "remote", "recollection"]
CHANNELS = ["none", "phone", "voicemail", "radio", "speaker", "telepathy", "message"]
PRECEDING_LINES = 10
MARKER = "\n確定した本文とID: "
INSTRUCTION = (
    "\nさらに各emotionsに、その発話の声がどこから聞こえるかをvoice_sourceで付けます。"
    "present: 話者がその場にいて直接話す声。姿が見えていない画面外・扉越し・背後からの声、"
    "心の声、手紙やメッセージや留守電の文面を自分の口で読む・引用する・呟く場合もpresent。"
    "remote: 話者がその場におらず、電話・通信機・スピーカー・画面・録音や録画の再生・念話・"
    "文字のメッセージを通して届く声。"
    "recollection: いま話されている声ではなく、人物が思い出している過去の言葉。"
    "地の文（話者のない行）はpresent。"
    "voice_channelはremoteのときだけ次から選び、それ以外はnone。"
    "phone=電話の通話、voicemail=留守電や録音・録画の再生、radio=無線・通信機・インカム、"
    "speaker=館内放送・インターホン・モニターやホログラムなど映像や拡声器越しの通信、"
    "telepathy=機器を介さず頭に直接届く声、message=文字のメッセージやチャットの文面。"
    "計画の登場人物でも、その場にいるとは限りません。本文の状況から判断します。"
)


class VoiceEmotion(narrative.EmotionAnnotation):
    voice_source: Literal["present", "remote", "recollection"]
    voice_channel: Literal["none", "phone", "voicemail", "radio", "speaker", "telepathy", "message"]


class VoiceStaging(narrative.Staging):
    emotions: list[VoiceEmotion]


def load_cases(names: list[str] | None) -> list[dict]:
    cases = []
    for name in ("real.json", "synthetic.json"):
        cases += json.loads((CASES / name).read_text(encoding="utf-8"))["cases"]
    if names:
        missing = set(names) - {case["id"] for case in cases}
        if missing:
            raise SystemExit(f"Unknown case: {', '.join(sorted(missing))}")
        cases = [case for case in cases if case["id"] in names]
    return cases


def extend_schema(schema: dict) -> dict:
    schema = copy.deepcopy(schema)
    emotion = schema["$defs"]["EmotionAnnotation"]
    emotion["properties"]["voice_source"] = {"type": "string", "enum": SOURCES}
    emotion["properties"]["voice_channel"] = {"type": "string", "enum": CHANNELS}
    emotion["required"] = [*emotion["required"], "voice_source", "voice_channel"]
    return schema


def prompts(case: dict, index: int, preceding: int) -> tuple[dict, dict, list[MappedUtterance], ScenePlan]:
    """The product's prompt for one batch: as it is, extended, and extended with the lines before it.

    Returns the prompts and the schemas by arm.
    """
    plan = ScenePlan.model_validate(case["plan"])
    lines = [MappedUtterance.model_validate(row) for row in case["lines"]]
    by_id = {line.id: line for line in lines}
    batch = [by_id[line_id] for line_id in case["batches"][index]]
    context = (json.dumps(case["setting"], ensure_ascii=False) + "\n読み上げない声の指示: "
               + json.dumps(case["hints"], ensure_ascii=False))
    prompt, schema = base.current_prompt(context, plan, batch)
    head, tail = prompt.split(MARKER, 1)
    # The instruction goes after the product's own, where a product change would put it.
    arms = {"plain": prompt, "batch": prompt + INSTRUCTION}
    first = lines.index(batch[0])
    before = lines[max(0, first - preceding):first]
    if before and preceding:
        names = {row["id"]: row["name"] for row in case["setting"]["characters"]}
        reference = json.dumps([{"speaker": names.get(line.speaker_id, line.speaker_id) if line.speaker_id
                                 else "NARRATOR", "text": line.display_text} for line in before],
                               ensure_ascii=False)
        arms["preceding"] = (head + "\n直前の発話（参考。注釈の対象ではない）: " + reference + MARKER
                             + tail + INSTRUCTION)
    extended = extend_schema(schema)
    return arms, {arm: schema if arm == "plain" else extended for arm in arms}, batch, plan


def run_call(llm, arm: str, prompt: str, schema: dict, seed: int, batch, plan) -> dict:
    """One staging request as the product sends it, with its single structural retry."""
    model = narrative.Staging if arm == "plain" else VoiceStaging
    extra = {"seed": seed, "response_format": {"type": "json_schema", "json_schema": {
        "name": "staging", "strict": True, "schema": schema}}}
    result = {"arm": arm, "seed": seed, "status": "pending", "attempts": []}
    start, trace_start = time.monotonic(), len(llm.trace)
    feedback = ""
    for attempt in range(2):
        try:
            llm.select_purpose("script-staging")
            llm._ensure_runtime()
            messages = base.messages_for(llm, "current", prompt + feedback, extra)
            reply = llm.chat("script-staging", messages, allow_truncated=True, **extra)
            content = reply.get("content") or ""
            result["attempts"].append({"content": content, "finish_reason": reply.get("_finish_reason")})
            staging = model.model_validate_json(content)
            if [item.utterance_id for item in staging.emotions] != [line.id for line in batch]:
                raise ValueError("Emotion annotations must preserve every source ID in order.")
            # The answer must still be one the product would accept.
            narrative._directions(plan, batch, staging)
            result["status"] = "valid"
            if model is VoiceStaging:
                result["answers"] = {item.utterance_id: [item.voice_source, item.voice_channel]
                                     for item in staging.emotions}
            break
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}"[:600])
            feedback = "\n構造化だけ再試行します。本文は変更禁止。前回の不備: " + str(error)
    result["elapsed_seconds"] = time.monotonic() - start
    usage = [item for item in llm.trace[trace_start:] if item.get("type") == "llm_generation"]
    result["usage"] = [item.get("usage", {}) for item in usage]
    return result


# ── scoring ──

def predicted_scene(case: dict, answers: dict) -> dict:
    """Who the answers keep off the stage, and where a voice-first character comes on."""
    cast = case["plan"]["character_ids"]
    spoken = {character: [line for line in case["lines"] if line["speaker_id"] == character] for character in cast}
    absent, arrives = [], {}
    for character, lines in spoken.items():
        sources = [answers.get(line["id"], ["present"])[0] for line in lines]
        if lines and "present" not in sources:
            absent.append(character)
        elif lines:
            first = sources.index("present")
            arrives[character] = "start" if first == 0 else lines[first]["id"]
    return {"absent": absent, "arrives": arrives}


def score(case: dict, answers: dict) -> dict:
    expected = case["expected"]
    rows = []
    for line in case["lines"]:
        got = answers.get(line["id"])
        if got is None:
            continue
        want = expected["lines"].get(line["id"])
        if line["speaker_id"] is None:
            kind, voices, channels = "narration", ["present"], []
        elif want is None:
            kind, voices, channels = "plain", ["present"], []
        else:
            kind, voices, channels = "labelled", want["voice"], want.get("channel", [])
        voice_ok = got[0] in voices
        channel_ok = None
        if got[0] == "remote" and voices[0] == "remote" and channels:
            channel_ok = got[1] in channels
        rows.append({"id": line["id"], "kind": kind, "speaker": line["speaker_id"], "text": line["display_text"],
                     "expected": {"voice": voices, "channel": channels}, "got": got,
                     "voice_ok": voice_ok, "preferred": got[0] == voices[0],
                     "channel_ok": channel_ok, "channel_preferred": channel_ok and got[1] == channels[0]})
    scene = predicted_scene(case, answers)
    want_absent = set(expected["absent"])
    absent_ok = set(scene["absent"]) == want_absent or bool(
        expected.get("absent_optional") and set(scene["absent"]) <= want_absent)
    arrivals = {character: scene["arrives"].get(character, "absent") in accepted
                for character, accepted in expected["arrives"].items()}
    return {"lines": rows, "scene": scene, "absent_ok": absent_ok, "arrivals_ok": arrivals}


def summary(scored: list[dict]) -> dict:
    count = Counter()
    for item in scored:
        for row in item["lines"]:
            want = row["expected"]["voice"][0]
            count[f"{row['kind']}:{want}:total"] += 1
            count[f"{row['kind']}:{want}:ok"] += row["voice_ok"]
            if row["channel_ok"] is not None:
                count["channel:total"] += 1
                count["channel:ok"] += row["channel_ok"]
                count["channel:preferred"] += bool(row["channel_preferred"])
        count["scene:total"] += 1
        count["scene:absent_ok"] += item["absent_ok"]
        count["arrival:total"] += len(item["arrivals_ok"])
        count["arrival:ok"] += sum(item["arrivals_ok"].values())
    return dict(count)


def ratio(count: dict, ok: str, total: str) -> str:
    return f"{count.get(ok, 0)}/{count.get(total, 0)}" if count.get(total) else "-"


def report(cases: list[dict], results: list[dict], output: Path) -> None:
    by_case = {case["id"]: case for case in cases}
    # The product's unchanged answers carry no voice judgement; side_effects() compares them.
    arms = sorted({row["arm"] for row in results} - {"plain"}, key=["batch", "preceding"].index)
    scored, failures = {arm: [] for arm in arms}, Counter()
    for arm in arms:
        for case in cases:
            own = {row["batch"]: row for row in results if row["arm"] == arm and row["case"] == case["id"]}
            if not own:
                continue
            # A scene's first batch has no lines before it; both arms send the same prompt there.
            shared = {row["batch"]: row for row in results if row["arm"] == "batch" and row["case"] == case["id"]}
            calls = [own.get(index) or shared.get(index) for index in range(len(case["batches"]))]
            calls = [row for row in calls if row is not None]
            failures[arm] += sum(row["status"] != "valid" for row in calls)
            answers = {}
            for row in calls:
                answers.update(row.get("answers", {}))
            scored[arm].append({"case": case["id"], "kind": case["kind"], "complete": all(
                row["status"] == "valid" for row in calls), **score(case, answers)})
    lines = ["# その場にいない声の判定 結果", "",
             "台詞の声の出どころ（その場・通信・回想）と、通信の種類の判定結果。", "",
             "| 版 | 通信の台詞 | 回想の台詞 | 紛らわしいその場の台詞 | その他の台詞 | 地の文 | 通信の種類 | 不在の人物 | 途中の登場 | 失敗した呼び出し |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in arms:
        count = summary(scored[arm])
        lines.append(f"| {arm} | {ratio(count, 'labelled:remote:ok', 'labelled:remote:total')}"
                     f" | {ratio(count, 'labelled:recollection:ok', 'labelled:recollection:total')}"
                     f" | {ratio(count, 'labelled:present:ok', 'labelled:present:total')}"
                     f" | {ratio(count, 'plain:present:ok', 'plain:present:total')}"
                     f" | {ratio(count, 'narration:present:ok', 'narration:present:total')}"
                     f" | {ratio(count, 'channel:ok', 'channel:total')}"
                     f"（第一候補 {count.get('channel:preferred', 0)}）"
                     f" | {ratio(count, 'scene:absent_ok', 'scene:total')}"
                     f" | {ratio(count, 'arrival:ok', 'arrival:total')} | {failures[arm]} |")
    lines += ["", "「通信の台詞」などは、認める答えのどれかに当たった数／対象の数。",
              "「通信の種類」は、通信と正しく判定できた台詞のうち、種類も認める答えに入った数。", "",
              "## トークン", "",
              "再試行を含む、モデルへの全リクエストの合計。",
              "",
              "| 版 | 呼び出し | リクエスト | 入力合計 | 出力合計 | 入力平均 | 出力平均 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in arms:
        usage = [item for row in results if row["arm"] == arm for item in row.get("usage", [])]
        calls = sum(row["arm"] == arm for row in results)
        given = sum(item.get("prompt_tokens") or 0 for item in usage)
        made = sum(item.get("completion_tokens") or 0 for item in usage)
        lines.append(f"| {arm} | {calls} | {len(usage)} | {given:,} | {made:,} | "
                     f"{given / max(len(usage), 1):,.0f} | {made / max(len(usage), 1):,.0f} |")
    lines.append("")
    for arm in arms:
        lines += [f"## {arm} の誤り", ""]
        for item in scored[arm]:
            case = by_case[item["case"]]
            names = {row["id"]: row["name"] for row in case["setting"]["characters"]}
            wrong = [row for row in item["lines"] if not row["voice_ok"] or row["channel_ok"] is False]
            scene_notes = []
            if not item["absent_ok"]:
                scene_notes.append("不在の人物: 期待 " + "、".join(names.get(c, c) for c in case["expected"]["absent"])
                                   + " / 判定 " + "、".join(names.get(c, c) for c in item["scene"]["absent"]))
            for character, ok in item["arrivals_ok"].items():
                if not ok:
                    scene_notes.append(f"{names.get(character, character)}の登場: 期待 "
                                       + "・".join(case["expected"]["arrives"][character])
                                       + " / 判定 " + item["scene"]["arrives"].get(character, "不在"))
            if not wrong and not scene_notes and item["complete"]:
                continue
            lines.append(f"### {case['id']}（{case['title']}）" + ("" if item["complete"] else " ※一部の呼び出しが失敗"))
            lines += [f"- {note}" for note in scene_notes]
            for row in wrong:
                speaker = names.get(row["speaker"], "地の文") if row["speaker"] else "地の文"
                want = "/".join(row["expected"]["voice"]) + (
                    "・" + "/".join(row["expected"]["channel"]) if row["expected"]["channel"] else "")
                lines.append(f"- {row['id']} {speaker}「{row['text'][:60]}」 期待 {want} / 判定 {row['got'][0]}"
                             + (f"・{row['got'][1]}" if row["got"][0] == "remote" else ""))
            lines.append("")
    (output / "report.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    write_json(output / "scored.json", scored)


LATIN = re.compile(r"[A-Za-z]{3,}")
KANA_KANJI = re.compile(r"[぀-ヿ一-鿿]")


def english(text: str) -> bool:
    return bool(LATIN.search(text)) and not KANA_KANJI.search(text)


def side_effects(cases: list[dict], results: list[dict], output: Path) -> bool:
    """Compare the product's unchanged answers with the extended ones, batch by batch, on the same seed."""
    by_case = {case["id"]: case for case in cases}
    final = {}
    for row in results:
        if row["arm"] in ("plain", "batch") and row["case"] in by_case:
            final[(row["case"], row["batch"], row["arm"])] = row
    pairs = sorted({(case, index) for case, index, arm in final if arm == "plain"})
    if not pairs:
        return False
    count = Counter()
    kinds = {"plain": Counter(), "batch": Counter()}
    remote = []
    for case_id, index in pairs:
        rows = {arm: final.get((case_id, index, arm)) for arm in ("plain", "batch")}
        for arm, row in rows.items():
            if row is None:
                continue
            count[f"{arm}:calls"] += 1
            count[f"{arm}:retried"] += len(row["attempts"]) > 1
            count[f"{arm}:failed"] += row["status"] != "valid"
            count[f"{arm}:output"] += sum(item.get("completion_tokens") or 0 for item in row["usage"])
        if any(row is None or row["status"] != "valid" for row in rows.values()):
            continue
        answers = {arm: json.loads(row["attempts"][-1]["content"]) for arm, row in rows.items()}
        speakers = {line["id"]: line["speaker_id"] for line in by_case[case_id]["lines"]}
        emotions = {arm: {item["utterance_id"]: item for item in value["emotions"]} for arm, value in answers.items()}
        for line_id, plain in emotions["plain"].items():
            extended = emotions["batch"][line_id]
            dialogue = speakers[line_id] is not None
            count["lines"] += 1
            count["dialogue"] += dialogue
            if dialogue:
                count["voice_same"] += plain["voice_emotion"] == extended["voice_emotion"]
                count["expression_same"] += plain.get("expression", "neutral") == extended.get("expression", "neutral")
            for arm, item in (("plain", plain), ("batch", extended)):
                count[f"{arm}:english"] += english(item["inner_emotion"]) or english(item["delivery"])
            if extended.get("voice_source", "present") != "present":
                remote.append((case_id, line_id, extended["voice_source"], extended["voice_channel"]))
        for arm, value in answers.items():
            kinds[arm].update(item["kind"] for item in value["directions"])
    def share(ok: str, total: str) -> str:
        return f"{count[ok]}/{count[total]}（{count[ok] / count[total]:.0%}）" if count[total] else "-"

    lines = ["# 指示を足したことの副作用", "",
             ("同じまとまり・同じシードで、現行の製品の演出付け（現行）と、声の出どころの指示を足した版（追加）を比べた。"
              f"対象は {len(pairs)} まとまり・{count['lines']} 行（うち台詞 {count['dialogue']} 行）。"), "",
             "| 項目 | 結果 |", "| --- | --- |",
             f"| 声の感情の一致（台詞） | {share('voice_same', 'dialogue')} |",
             f"| 表情の一致（台詞） | {share('expression_same', 'dialogue')} |",
             f"| 内心・声の出し方が英語の行 | 現行 {count['plain:english']} / 追加 {count['batch:english']}（全 {count['lines']} 行） |",
             f"| 再試行した呼び出し | 現行 {count['plain:retried']} / 追加 {count['batch:retried']} |",
             f"| 失敗した呼び出し | 現行 {count['plain:failed']} / 追加 {count['batch:failed']} |",
             (f"| 出力トークン（1回平均） | 現行 {count['plain:output'] / max(count['plain:calls'], 1):,.0f}"
              f" / 追加 {count['batch:output'] / max(count['batch:calls'], 1):,.0f} |")]
    for kind in sorted(set(kinds["plain"]) | set(kinds["batch"])):
        lines.append(f"| 演出 {kind} の数 | 現行 {kinds['plain'][kind]} / 追加 {kinds['batch'][kind]} |")
    lines += ["", ("参考: 同じモデルでシードだけ変えた場合の一致率は、声の感情75%・表情84%"
                   "（[演出付けに渡す資料を絞る実験](../../docs/experiments/staging-context-probe-20261013.md)）。"), "",
              "## 追加した版で、その場の声以外と判定した行", ""]
    lines += [f"- {case_id} {line_id}: {source}" + (f"・{channel}" if source == "remote" else "")
              for case_id, line_id, source, channel in remote] or ["- なし"]
    (output / "side-effects.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", help="run only these case ids; repeatable")
    parser.add_argument("--arm", action="append", choices=["plain", "batch", "preceding"],
                        help="arms to run (default: batch and preceding); plain is the product's unchanged staging")
    parser.add_argument("--batch", action="append", metavar="CASE:INDEX",
                        help="run only these batches; repeatable")
    parser.add_argument("--preceding", type=int, default=PRECEDING_LINES, help="reference lines for the preceding arm")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--dry", action="store_true", help="build the prompts, save one sample and report sizes")
    parser.add_argument("--score-only", action="store_true", help="rescore saved results without the model")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.case)
    saved = output / "results.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else []
    wanted = set(args.arm or ["batch", "preceding"])
    chosen = {(name.rsplit(":", 1)[0], int(name.rsplit(":", 1)[1])) for name in args.batch or []}
    if chosen - {(case["id"], index) for case in cases for index in range(len(case["batches"]))}:
        raise SystemExit("Unknown batch in --batch.")
    if args.score_only:
        report(cases, results, output)
        print((output / "report.md").read_text(encoding="utf-8"))
        if side_effects(cases, results, output):
            print((output / "side-effects.md").read_text(encoding="utf-8"))
        return 0
    if args.dry:
        for case in cases:
            for index in range(len(case["batches"])):
                arms, schemas, batch, _ = prompts(case, index, args.preceding)
                print(case["id"], index, len(batch), {arm: len(text) for arm, text in arms.items()})
                if case is cases[0] and index == 0:
                    for arm, text in arms.items():
                        (output / f"sample-{arm}.txt").write_bytes(text.encode("utf-8"))
                    write_json(output / "sample-schema.json", schemas["batch"])
        return 0
    from services.worker.generation.model_routing import RoutedLLM
    from services.worker.generation.processes import gpu_lock

    config = copy.deepcopy(pipeline.load_config())
    # One model stays loaded for the whole comparison; the production limit is per job.
    config["llm"]["total_timeout_seconds"] = 6 * 3600
    payload = {"seed": args.seed, "profile": {}, "profiles": {}, "story_workflow_version": 2,
               "workflow_policy": "script_continuation_v1"}
    done = {(row["case"], row["batch"], row["arm"]) for row in results}
    with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
          RoutedLLM(pipeline.ROOT, config, payload, output / "llm") as llm):
        for case in cases:
            for index in range(len(case["batches"])):
                if chosen and (case["id"], index) not in chosen:
                    continue
                arms, schemas, batch, plan = prompts(case, index, args.preceding)
                for arm, prompt in arms.items():
                    if arm not in wanted or (case["id"], index, arm) in done:
                        continue
                    # Every arm of a batch shares a seed, so the prompt is the only difference.
                    seed = args.seed + 17 * index + sum(map(ord, case["id"]))
                    result = run_call(llm, arm, prompt, schemas[arm], seed, batch, plan)
                    result.update(case=case["id"], batch=index)
                    results.append(result)
                    write_json(saved, results)
                    tokens = {"input": sum(item.get("prompt_tokens") or 0 for item in result["usage"]),
                              "output": sum(item.get("completion_tokens") or 0 for item in result["usage"])}
                    print(json.dumps({**{key: result.get(key) for key in (
                        "case", "batch", "arm", "status", "elapsed_seconds", "error")}, "tokens": tokens},
                        ensure_ascii=False), flush=True)
        write_json(output / "runtime.json", {"profile": llm.profile, "provenance": llm.provenance()})
    report(cases, results, output)
    print((output / "report.md").read_text(encoding="utf-8"))
    if side_effects(cases, results, output):
        print((output / "side-effects.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
