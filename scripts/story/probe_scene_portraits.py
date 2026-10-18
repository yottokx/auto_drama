"""Which people should stand on stage in a scene of four or more?

Only three portraits fit on the stage. This replays the staging step on saved scenes of four
or more people (from check_text_with_cast.py --scene-cast runs) with one extra direction:

  portraits  from this line on, these one to three people stand on stage; everyone else
             shows only a face in the message window while speaking

--mode batch: the staging request is otherwise the product's (narrative._staging): the same
instruction, the same batches of ten lines and the same checks.
--mode entrances: as scene, but the people on stage change only where someone arrives or leaves
or the people at the centre move elsewhere; those who stay keep their place.
--mode central: one request per scene names the scene's central people (one to three) and the
lines where each comes and goes; only they stand on stage while there, and the stage follows
by code. Everyone else shows only a face, even while speaking.
--mode scene: one request per scene reads the plan and every line and names the people on stage
once, changing them only where the people at the centre clearly change. The line annotations
are left to the product's batches and are not replayed.

The material is the scene's setting taken from the saved scene-text request. Nothing in the
product changes.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_text_with_cast import open_scene_cast

from packages.contracts.m3 import NarrativeScene
from packages.contracts.script import Contract, Identifier
from services.worker.generation import narrative, pipeline
from services.worker.generation.llm import write_json
from services.worker.model_config import select_config

MIN_CAST = 4
STAGE = 3
SETTING_END = "\ncore:"
PORTRAITS_INSTRUCTION = (
    "この場面の登場人物は{count}人です。立ち絵で画面に出せるのは{stage}人までで、立ち絵にしない人物は、"
    "その人が話す行だけメッセージウィンドウに顔を表示します。directionsにkind=portraitsで、その行から"
    "立ち絵で見せる人物のcharacter_idsを1〜{stage}人指定します。場面の中心となる人物だけを選び、"
    "人数を{stage}人に揃える必要はありません（二人の口論が中心ならその二人、教師が授業をする場面なら"
    "教師一人）。遠くから聞こえる声や回想の声の人物は選びません。{first}中心となる人物が入れ替わる行で"
    "だけ追加します。reasonに選んだ理由を短く書きます。")
FIRST = "場面の最初の行（{uid}）には必ず指定し、"
SCENE_PORTRAITS_INSTRUCTION = (
    "\n本文は変更しません。この場面の登場人物は{count}人です。立ち絵で画面に出せるのは{stage}人までで、"
    "立ち絵にしない人物は、その人が話す行だけメッセージウィンドウに顔を表示します。場面全体を読み、"
    "立ち絵で見せる人物のcharacter_idsを1〜{stage}人決めてportraitsに書きます。場面の中心となる人物だけを"
    "選び、人数を{stage}人に揃える必要はありません（二人の口論が中心ならその二人、教師が授業をする場面"
    "なら教師一人）。遠くから聞こえる声や回想の声の人物は選びません。portraitsの最初は場面の最初の行"
    "（{uid}）から始めます。その後は、中心となる人物がはっきり入れ替わるとき（人物が来る・去る、会話の"
    "相手が変わる、別の人物たちの出来事に移る）だけ、その行から追加します。数行ごとの話者の交代や"
    "短い相づちでは変えません。reasonに選んだ理由を短く書きます。")


ENTRANCE_PORTRAITS_INSTRUCTION = (
    "\n本文は変更しません。この場面の登場人物は{count}人です。立ち絵で画面に出せるのは{stage}人までで、"
    "立ち絵にしない人物は、その人が話す行だけメッセージウィンドウに顔を表示します。場面全体を読み、"
    "場面を通して中心となる人物を1〜{stage}人選んでportraitsに書きます。人数を{stage}人に揃える必要は"
    "ありません（二人の口論が中心ならその二人、教師が授業をする場面なら教師一人）。遠くから聞こえる声や"
    "回想の声の人物は選びません。portraitsの最初は場面の最初の行（{uid}）から始めます。立ち絵を変えるのは、"
    "本文で人物がその場に来る行、その場を去る行、中心の人物が別の場所へ移る行だけです。会話の相手や"
    "注目が移るだけでは変えません（話している人は顔の表示と明るさで分かります）。変えるときも、"
    "その場に残る人物は立ち絵に残します。reasonに選んだ理由を短く書きます。")


class PortraitAnnotation(Contract):
    utterance_id: Identifier
    kind: Literal["portraits"]
    character_ids: list[str] = Field(min_length=1, max_length=STAGE)
    reason: str = Field(min_length=1, max_length=200)


CENTRAL_INSTRUCTION = (
    "\n本文は変更しません。この場面の登場人物は{count}人です。立ち絵で画面に出せるのは{stage}人までです。"
    "場面全体を読み、この場面の中心となる人物を1〜{stage}人選んでcentralに書きます。人数を{stage}人に"
    "揃える必要はありません（二人の口論が中心ならその二人、教師が授業をする場面なら教師一人）。"
    "中心となる人物だけが、その場にいる間、立ち絵で表示されます。それ以外の人物は、その場にいても、"
    "話す行でもメッセージウィンドウに顔だけを表示します。遠くから聞こえる声や回想の声の人物は選びません。"
    "各人物について、本文でその場に来る行をenters_at、その場を去る行をleaves_atに書きます。場面の最初から"
    "その場にいる人物のenters_atと、場面の最後までその場にいる人物のleaves_atは空文字にします。"
    "reasonにその人物を中心とした理由を短く書きます。")


class CentralCharacter(Contract):
    character_id: str
    enters_at: str
    leaves_at: str
    reason: str = Field(min_length=1, max_length=200)


class SceneCentral(Contract):
    central: list[CentralCharacter] = Field(min_length=1, max_length=STAGE)


class ScenePortraits(Contract):
    portraits: list[PortraitAnnotation] = Field(min_length=1, max_length=20)


class PortraitStaging(Contract):
    emotions: list[narrative.EmotionAnnotation]
    directions: list[narrative.FocusAnnotation | narrative.PauseAnnotation | narrative.ExpressionAnnotation
                     | PortraitAnnotation] = Field(max_length=50)


def setting_text(run: Path, scope: str) -> str:
    request = json.loads((run / "requests" / f"{scope}-text-1.json").read_text(encoding="utf-8"))["request"]
    user = request["messages"][-1]["content"]
    head, found, _ = user.partition(SETTING_END)
    if not found:
        raise ValueError(f"{scope}: the scene-text request has no setting section.")
    return head.strip()


def portrait_staging(llm, context: str, plan, utterances) -> dict:
    """narrative._staging with the portraits direction; returns the annotations of every batch."""
    emotions, directions = [], []
    instruction = narrative.VOICE_SOURCE_INSTRUCTION
    for start in range(0, len(utterances), 10):
        batch = utterances[start:start + 10]
        schema = PortraitStaging.model_json_schema()
        emotion = schema["$defs"]["EmotionAnnotation"]
        emotion["properties"]["utterance_id"]["enum"] = [u.id for u in batch]
        emotion["required"] = [*emotion["required"], "expression", "voice_source", "voice_channel"]
        schema["properties"]["emotions"].update(minItems=len(batch), maxItems=len(batch))
        expression = schema["$defs"]["ExpressionAnnotation"]["properties"]
        expression["utterance_id"]["enum"] = [u.id for u in batch]
        expression["character_id"]["enum"] = plan.character_ids
        focus = schema["$defs"]["FocusAnnotation"]["properties"]
        focus["character_id"]["enum"] = plan.character_ids
        dialogue_ids = [u.id for u in batch if u.speaker_id is not None]
        portraits = schema["$defs"]["PortraitAnnotation"]["properties"]
        portraits["utterance_id"]["enum"] = [u.id for u in batch]
        portraits["character_ids"]["items"]["enum"] = plan.character_ids
        portraits["character_ids"]["uniqueItems"] = True
        if dialogue_ids:
            focus["utterance_id"]["enum"] = dialogue_ids
        else:
            schema["properties"]["directions"]["items"] = {"anyOf": [
                {"$ref": "#/$defs/PauseAnnotation"}, {"$ref": "#/$defs/ExpressionAnnotation"},
                {"$ref": "#/$defs/PortraitAnnotation"}]}
        schema["$defs"]["PauseAnnotation"]["properties"]["utterance_id"]["enum"] = [u.id for u in batch]
        first = FIRST.format(uid=batch[0].id) if start == 0 else ""
        extra = PORTRAITS_INSTRUCTION.format(count=len(plan.character_ids), stage=STAGE, first=first)
        feedback = ""
        for attempt in range(2):
            try:
                staging = narrative._structured(llm, "staging", "資料: " + context
                    + "\n計画: " + narrative._data(plan) + "\n確定した本文とID: "
                    + narrative._data([narrative._staged_line(item) for item in batch])
                    + "\n本文を再入力・修正せず、各utterance_idについて内心inner_emotion、"
                    "声に出すvoice_emotion、声の出し方deliveryを別々に設定します。"
                    "地の文の声はneutral、deliveryが不要なら空文字。emotionsは提示した発話を"
                    "順番通り1回ずつ含めます。演出は任意のfocus/pause/blackout/expressionのみ。"
                    "focusのcharacter_idは必ず登場人物IDで空文字禁止。NARRATORの行には"
                    "focusを付けません。pause/blackoutのcharacter_idだけは空文字。"
                    "pause/blackoutは100〜1500msの正の長さ。演出が不要ならdirections=[]。"
                    "登退場はプログラムが処理します。"
                    "expressionはその行を話す人物の顔に見える表情で、声の感情とは別に決めます。"
                    "地の文のexpressionはneutral。"
                    "話者以外の登場人物の表情がその行で変わるとき（聞き手の反応、地の文に書かれた"
                    "表情の変化）だけ、directionsにkind=expressionでその人物のcharacter_idと表情を"
                    "追加します。表情は次に変わるまで続くので、変わらない行には付けません。"
                    + instruction + extra + feedback, PortraitStaging, schema=schema)
                if [e.utterance_id for e in staging.emotions] != [u.id for u in batch]:
                    raise ValueError("Emotion annotations must preserve every source ID in order.")
                narrative._voice_sources_agree(staging.emotions)
                chosen = [row for row in staging.directions if isinstance(row, PortraitAnnotation)]
                if start == 0 and not any(row.utterance_id == batch[0].id for row in chosen):
                    raise ValueError(f"Name who stands on stage (kind=portraits) at the scene's first line {batch[0].id}.")
                if any(len(set(row.character_ids)) != len(row.character_ids) for row in chosen):
                    raise ValueError("Name each person on stage once.")
                rest = [row for row in staging.directions if not isinstance(row, PortraitAnnotation)]
                narrative._directions(plan, batch, narrative.Staging(emotions=staging.emotions, directions=rest))
                emotions.extend(staging.emotions)
                directions.extend(staging.directions)
                break
            except ValueError as error:
                if attempt:
                    raise
                feedback = "\n構造化だけ再試行します。本文は変更禁止。前回の不備: " + str(error)
    return {"emotions": [row.model_dump(mode="json") for row in emotions],
            "directions": [row.model_dump(mode="json") for row in directions]}


def scene_portraits(llm, context: str, plan, utterances, instruction: str = SCENE_PORTRAITS_INSTRUCTION) -> dict:
    """Name the people on stage once for the whole scene; repeats are folded by code."""
    schema = ScenePortraits.model_json_schema()
    item = schema["$defs"]["PortraitAnnotation"]["properties"]
    item["utterance_id"]["enum"] = [u.id for u in utterances]
    item["character_ids"]["items"]["enum"] = plan.character_ids
    order = {u.id: n for n, u in enumerate(utterances)}
    feedback = ""
    for attempt in range(2):
        try:
            answer = narrative._structured(llm, "staging", "資料: " + context
                + "\n計画: " + narrative._data(plan) + "\n確定した本文とID: "
                + narrative._data([narrative._staged_line(u) for u in utterances])
                + instruction.format(count=len(plan.character_ids), stage=STAGE,
                                                     uid=utterances[0].id) + feedback,
                ScenePortraits, schema=schema)
            rows = sorted(answer.portraits, key=lambda row: order[row.utterance_id])
            if rows[0].utterance_id != utterances[0].id:
                raise ValueError(f"Start portraits at the scene's first line {utterances[0].id}.")
            break
        except ValueError as error:
            if attempt:
                raise
            feedback = "\n構造化だけ再試行します。前回の不備: " + str(error)
    kept, previous, folded = [], None, 0
    for row in rows:
        people = list(dict.fromkeys(row.character_ids))
        if frozenset(people) == previous:
            folded += 1
            continue
        kept.append({**row.model_dump(mode="json"), "character_ids": people})
        previous = frozenset(people)
    return {"directions": kept, "folded_repeats": folded, "answered": len(rows)}


def central_portraits(llm, context: str, plan, utterances) -> dict:
    """The scene's central people and where each comes and goes; the stage follows by code."""
    ids = [u.id for u in utterances]
    order = {uid: n for n, uid in enumerate(ids)}
    schema = SceneCentral.model_json_schema()
    item = schema["$defs"]["CentralCharacter"]["properties"]
    item["character_id"]["enum"] = plan.character_ids
    item["enters_at"]["enum"] = ["", *ids]
    item["leaves_at"]["enum"] = ["", *ids]
    feedback = ""
    for attempt in range(2):
        try:
            answer = narrative._structured(llm, "staging", "資料: " + context
                + "\n計画: " + narrative._data(plan) + "\n確定した本文とID: "
                + narrative._data([narrative._staged_line(u) for u in utterances])
                + CENTRAL_INSTRUCTION.format(count=len(plan.character_ids), stage=STAGE) + feedback,
                SceneCentral, schema=schema)
            people = list({row.character_id: row for row in answer.central}.values())
            for row in people:
                start = order[row.enters_at] if row.enters_at else 0
                end = order[row.leaves_at] if row.leaves_at else len(ids)
                if end <= start:
                    raise ValueError(f"{row.character_id} leaves at or before the line they come in.")
            break
        except ValueError as error:
            if attempt:
                raise
            feedback = "\n構造化だけ再試行します。前回の不備: " + str(error)
    spans = [(row.character_id, order[row.enters_at] if row.enters_at else 0,
              order[row.leaves_at] if row.leaves_at else len(ids)) for row in people]
    marks = sorted({0, *(start for _, start, _ in spans), *(end for _, _, end in spans if end < len(ids))})
    directions, previous = [], None
    for line in marks:
        here = [cid for cid, start, end in spans if start <= line < end]
        if here == previous:
            continue
        came = [cid for cid in here if previous is not None and cid not in previous]
        went = [cid for cid in (previous or []) if cid not in here]
        why = "場面の最初" if previous is None else "、".join(
            [f"{cid}が来る" for cid in came] + [f"{cid}が去る" for cid in went])
        directions.append({"utterance_id": ids[line], "kind": "portraits", "character_ids": here, "reason": why})
        previous = here
    return {"directions": directions, "central": [row.model_dump(mode="json") for row in people],
            "answered": len(answer.central)}


def scenes_of(run: Path) -> list[tuple[str, NarrativeScene]]:
    state = json.loads((run / "draft-state.json").read_text(encoding="utf-8"))
    found = []
    for scope, row in state["scene_commits"].items():
        if len(row["scene"]["plan"]["character_ids"]) >= MIN_CAST:
            found.append((scope, NarrativeScene.model_validate(row["scene"])))
    return found


def names_of(run: Path) -> dict[str, str]:
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    names = {row["result"]["id"]: row["result"]["name"] for row in experiment["approval_snapshot"]["characters"]}
    cast = json.loads((run / "cast-plan.json").read_text(encoding="utf-8"))["plan"]
    names.update({row["id"]: row["name"] for row in cast["supporting_characters"]})
    state = json.loads((run / "draft-state.json").read_text(encoding="utf-8"))
    for plan in state["chapter_plans"].values():
        names.update({row["id"]: row["name"] for row in plan["plan"].get("new_characters", [])})
    return names


def report(results: list[dict], output: Path) -> None:
    lines = ["# 4人以上の場面で立ち絵に出す人物", "",
             "保存済みの場面に、演出注釈の portraits（その行から立ち絵で見せる1〜3人）を足して演出を付け直した結果。", ""]
    for row in results:
        lines.append(f"## {row['run']} {row['scope']}（{row['model']}）")
        lines.append(f"配役: {'・'.join(row['cast'])}　台詞: {row['lines']}")
        if "error" in row:
            lines += [f"- 失敗: {row['error']}", ""]
            continue
        for item in row["portraits"]:
            lines.append(f"- {item['from']}〜（{item['line']}）: {'・'.join(item['people'])} — {item['reason']}")
        lines.append("")
    (output / "report.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def short(name: str) -> str:
    return name.split("（")[0].split("(")[0].split("・")[0].split()[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path, help="script runs made with --scene-cast open/acting")
    parser.add_argument("--mode", choices=("batch", "scene", "entrances", "central"), default="batch")
    parser.add_argument("--scenes", nargs="*", default=[], help="only these, as RUN:SCOPE")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    open_scene_cast()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for run in args.runs:
        experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
        names = names_of(run)
        for scope, scene in scenes_of(run):
            if args.scenes and f"{run.name}:{scope}" not in args.scenes:
                continue
            setting_text(run, scope)
            jobs.append({"run": run.name, "path": run, "scope": scope, "scene": scene, "names": names,
                         "profile": experiment["profile"]})
            print(run.name, scope, len(scene.plan.character_ids), "people", len(scene.utterances), "lines")
    if args.dry:
        return 0
    from services.worker.generation.model_routing import RoutedLLM
    from services.worker.generation.processes import gpu_lock

    saved = output / "results.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else []
    by_model: dict[tuple, list[dict]] = {}
    for job in jobs:
        by_model.setdefault((job["profile"]["model_id"], job["profile"]["context_size"]), []).append(job)
    base = pipeline.load_config()
    for (model, _), group in by_model.items():
        profile = group[0]["profile"]
        config = copy.deepcopy(select_config(base, {"profile": profile}, pipeline.ROOT))
        config["llm"]["total_timeout_seconds"] = 4 * 3600
        payload = {"seed": 20261018, "profile": profile, "profiles": {}, "story_workflow_version": 2,
                   "workflow_policy": "script_continuation_v1"}
        with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
              RoutedLLM(pipeline.ROOT, config, payload, output / "llm" / model) as llm):
            for job in group:
                if any(row["run"] == job["run"] and row["scope"] == job["scope"] and "portraits" in row
                       for row in results):
                    continue
                scene, names = job["scene"], job["names"]
                spoken = collections.Counter(short(names.get(u.speaker_id, u.speaker_id))
                                             for u in scene.utterances if u.speaker_id)
                row = {"run": job["run"], "scope": job["scope"], "model": model,
                       "cast": [short(names.get(cid, cid)) for cid in scene.plan.character_ids],
                       "lines": dict(spoken)}
                start = time.monotonic()
                try:
                    llm.select_purpose("script-staging")
                    staging = {"scene": scene_portraits, "batch": portrait_staging,
                               "entrances": lambda *a: scene_portraits(*a, ENTRANCE_PORTRAITS_INSTRUCTION),
                               "central": central_portraits}[args.mode]
                    answer = staging(llm, setting_text(job["path"], job["scope"]), scene.plan, scene.utterances)
                    order = {u.id: n for n, u in enumerate(scene.utterances, 1)}
                    row["answer"] = answer
                    row["lines_total"] = len(scene.utterances)
                    row["portraits"] = [
                        {"from": item["utterance_id"], "line": order[item["utterance_id"]],
                         "people": [short(names.get(cid, cid)) for cid in item["character_ids"]],
                         "reason": item["reason"]}
                        for item in answer["directions"] if item["kind"] == "portraits"]
                except (OSError, RuntimeError, ValueError, TypeError) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"[:500]
                row["seconds"] = round(time.monotonic() - start, 1)
                results = [value for value in results
                           if not (value["run"] == row["run"] and value["scope"] == row["scope"])] + [row]
                write_json(saved, results)
                print(json.dumps({key: row.get(key) for key in ("run", "scope", "portraits", "error")},
                                 ensure_ascii=False), flush=True)
    report(results, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
