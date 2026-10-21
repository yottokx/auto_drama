"""Each chapter's own steps around its skeleton, for long-form plans (ScriptOptions.long_form).

The chain of events is the skeleton: it carries the story's turns but not enough to fill many
chapters. After the chain is allocated, each chapter in turn, seeing the whole skeleton and the
chapters already expanded, gets steps added around its skeleton steps, which stay as they are.
The wording and the answer format are the ones tried in docs/experiments/long-form-expand-
20261020.md (scripts/story/check_text_long.py --expand --expansion-steps 6 --distinct-steps
--fresh-roles --closing-step-last --closure-end).
"""

from __future__ import annotations

from pydantic import Field

from packages.contracts.script import Contract

from .script_plot import ChainEvent, ChainStep

# The most steps a chapter reaches with the added ones; more made chapters of three far longer
# than ten minutes.
MOST_CHAPTER_STEPS = 6


class AddedStep(ChainStep):
    after_step: int = Field(ge=0, le=60, strict=True,
        description="当章の骨組みの手順番号のどれの後に置くか。0は最初の手順の前。")


class ChapterExpansion(Contract):
    added_steps: list[AddedStep] = Field(max_length=8,
        description="当章に加える手順。骨組みの手順は書き直さず、ここに含めない。")


EXPANSION_INSTRUCTION = (
    "会話中心のビジュアルノベルの全体プロットを、章ごとに肉付けしています。今回は当章だけです。"
    "skeletonは全章の骨組み（各章の題名・役割・予定の手順）、expanded_chaptersは肉付け済みの前の章、"
    "current_chapter_stepsは当章の骨組みの手順（番号付き）です。すべて未実施の予定です。"
    "当章が1話（約10分）の会話と出来事を支えられるよう、added_stepsに当章で新しく起きる手順を加えます。"
    "骨組みの手順は自動で残るので、書き直さず、added_stepsにも含めません。"
    "加える手順は、骨組みの手順の前後や間で起きる、骨組みに無い行動・発見・応答です。"
    "サブや脇の人物自身の用事・関心・対立による行動や、メインとサブの具体的なやり取りを中心にします。"
    "骨組みの手順と同じ行動・同じ提案・同じ決定を、言い換えや前振りとして足しません。"
    "骨組みの手順の準備を足すなら、その手順とは別の人物か別の用事にします。"
    "少人数の密な話なら、主要な人物同士の具体的な試行と応答で構いません。"
    "after_stepは、その手順を置く位置（当章の骨組みの手順番号の後。0は最初の前）です。"
    "骨組みの手順の結果と矛盾させず、当章の最後の到達状態を変えません。"
    "他の章の骨組みにある出来事を先取りせず、前の章で済んだ出来事や同じ主張の確認を繰り返しません。"
    "目安は、骨組みと合わせて当章が4〜6手順になる量です。骨組みだけで十分なら空配列も可能です。"
    "stepsは登録人物のactionと具体的なresultで、各1〜2文。全台詞や細かな演出は書きません。"
)
# A person's added step may not repeat the role they had in an earlier chapter.
FRESH_ROLES_RULE = (
    "expanded_chaptersで、その人物が前の章で担った役回り（同じ働き、同じ種類の転機）を繰り返しません。"
    "その人物の新しい面や、前の章の結果を受けた次の段階を描くか、"
    "当章では手順を足さずに、別の人物の手順にします。全員を毎章登場させる必要はありません。"
    "場面の外で起きることや、本人が知覚できないことは手順にしません。"
)
# The last chapter's closing step stays last; the answer format holds it there too.
FINAL_EXPANSION_RULE = "最終章です。骨組みの最後の手順は作品の締めなので、その後には手順を足しません。"
# Put before the last chapter's scene plan, so the work ends on its closing step.
FINAL_END_RULE = "最終章です。最後の場面の終了状態を、当章の予定の最後の手順の結果にします。\n"


def expansion_prompt(count: int, number: int) -> str:
    return (f"全{count}章の第{number}章を肉付けします。" + EXPANSION_INSTRUCTION + FRESH_ROLES_RULE
            + (FINAL_EXPANSION_RULE if number == count else ""))


def expansion_schema(available: set[str], steps: int, last: bool) -> tuple[dict, int]:
    """The answer format for a chapter of ``steps`` skeleton steps, and how many may be added."""
    schema = ChapterExpansion.model_json_schema()
    step_schema = schema["$defs"]["AddedStep"]
    step_schema["properties"]["character_id"]["enum"] = sorted(available)
    # The last chapter's closing step stays last: nothing is placed after it.
    step_schema["properties"]["after_step"]["maximum"] = steps - 1 if last else steps
    room = max(0, MOST_CHAPTER_STEPS - steps)
    schema["properties"]["added_steps"]["maxItems"] = room
    # The position first, as the instruction reads it.
    step_schema["properties"] = {"after_step": step_schema["properties"].pop("after_step"),
                                 **step_schema["properties"]}
    return schema, room


def expanded_events(events: list[ChainEvent], added: list[AddedStep]) -> list[ChainEvent]:
    """The chapter's skeleton steps with the added ones in place, as events of up to six steps."""
    skeleton = [step for event in events for step in event.steps]
    if any(row.after_step > len(skeleton) for row in added):
        raise ValueError(f"after_step must be between 0 and {len(skeleton)}.")
    steps = [ChainStep(**row.model_dump(exclude={"after_step"})) for row in added if row.after_step == 0]
    for number, step in enumerate(skeleton, 1):
        steps.append(step)
        steps.extend(ChainStep(**row.model_dump(exclude={"after_step"}))
                     for row in added if row.after_step == number)
    result, condition = [], events[0].start_condition
    for start in range(0, len(steps), 6):
        chunk = steps[start:start + 6]
        result.append(ChainEvent(start_condition=condition, steps=chunk))
        condition = chunk[-1].result
    return result
