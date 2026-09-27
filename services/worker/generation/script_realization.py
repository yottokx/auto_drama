"""Place the single planned action sequence into scenes without paraphrasing it."""

from typing import Literal

from pydantic import Field

from packages.contracts.m2 import CharacterResult
from packages.contracts.m3 import Location, ScenePlan
from packages.contracts.script import Contract, Identifier

from .script_cast import ScriptOptions
from .script_plot import ChainEvent, ChainStep, CurrentFacts, RouteAdjustment


class StepPlacement(Contract):
    step_number: int = Field(ge=1, strict=True)
    handling: Literal["stage", "already_done", "adjust"]
    scene_number: int = Field(ge=0, le=8, strict=True, description="描く場面の1始まりの位置。実施済みだけ0。")
    reason_from_source: str = Field(description="実施済み・調整の場合は実台本の行動や結果を記す。初章のみ承認設定。通常は空文字。")
    adjustment: ChainStep | None = Field(description="adjustの場合のみ、実績に合わせる行動と結果。通常はnull。")


class StagedStep(Contract):
    step_number: int = Field(ge=1, strict=True)
    handling: Literal["stage"]
    scene_number: int = Field(ge=1, le=8, strict=True)


class CompletedStep(Contract):
    step_number: int = Field(ge=1, strict=True)
    handling: Literal["already_done"]
    reason_from_source: str = Field(min_length=1)


class AdjustedStep(Contract):
    step_number: int = Field(ge=1, strict=True)
    handling: Literal["adjust"]
    scene_number: int = Field(ge=1, le=8, strict=True)
    reason_from_source: str = Field(min_length=1)
    adjustment: ChainStep


class SceneSetup(Contract):
    id: Identifier
    location_id: Identifier
    character_ids: list[Identifier] = Field(min_length=1, max_length=3)
    interaction: str = Field(min_length=1, max_length=300, description="誰と誰が何を持ちかけ、どんな応答・掛け合い・距離感を描くか。1〜3文。")
    interaction_end: str = Field(description="主要行動のない交流場面を閉じる状況。行動のある場面では空文字。")
    length_weight: int = Field(ge=1, le=10, strict=True, description="章内の分量の相対比。重要なやり取りには厚みを与える。")
    staging: str = Field(min_length=1, max_length=200, description="割当行動を具体化する場所・所作・会話の工夫。行動や結果を別の筋へ置き換えない。")
    atmosphere: str = Field(min_length=1)
    follow_through: list[ChainStep] = Field(max_length=3,
        description="割当行動の実績から必要な追加の実行・後始末。不要なら空配列。同じ議論や結論を繰り返さない。")


class ChainAdjustment(Contract):
    chapter_number: int = Field(ge=1, le=100, strict=True)
    reason_from_source: str = Field(min_length=1)
    events: list[ChainEvent] = Field(min_length=1, max_length=9)


class ChapterRealization(Contract):
    current_facts: CurrentFacts
    placements: list[StepPlacement] = Field(min_length=1, max_length=54)
    future_route_changes: list[ChainAdjustment] = Field(max_length=99)
    unresolved_core_gaps: list[str] = Field(max_length=10)
    new_characters: list[CharacterResult] = Field(max_length=3)
    locations: list[Location] = Field(min_length=1, max_length=8)
    scenes: list[SceneSetup] = Field(min_length=1, max_length=8)


class ChapterRealizationResponse(ChapterRealization):
    # anyOf of closed objects is supported by the JSON-schema decoder. The
    # internal/export representation remains unchanged, but invalid combinations
    # are impossible in the generated representation.
    placements: list[StagedStep | CompletedStep | AdjustedStep] = Field(min_length=1, max_length=54)

    def as_realization(self) -> ChapterRealization:
        data = self.model_dump(mode="json", exclude={"placements"})
        placements = []
        for row in self.placements:
            item = row.model_dump(mode="json")
            item.setdefault("scene_number", 0)
            item.setdefault("reason_from_source", "")
            item.setdefault("adjustment", None)
            placements.append(StepPlacement(**item))
        return ChapterRealization(**data, placements=placements)


def numbered_steps(events: list[ChainEvent]) -> list[dict]:
    result = []
    for event in events:
        condition = event.start_condition
        for step in event.steps:
            result.append({"step_number": len(result) + 1, "before": condition,
                           **step.model_dump(mode="json")})
            condition = step.result
    return result


def realize_chapter(draft: ChapterRealization, events: list[ChainEvent], options: ScriptOptions | None = None) -> dict:
    """Check only mapping/shape. No semantic judge or retry pipeline is added."""
    source = numbered_steps(events)
    if [row.step_number for row in draft.placements] != list(range(1, len(source) + 1)):
        raise ValueError("Placements must account for every planned step exactly once in order.")
    groups = {n: [] for n in range(1, len(draft.scenes) + 1)}
    previous_scene = 0
    for placement, original in zip(draft.placements, source, strict=True):
        if placement.handling == "already_done":
            if placement.scene_number != 0 or placement.adjustment is not None or not placement.reason_from_source.strip():
                raise ValueError("An already-done step needs source facts, no scene and no adjustment.")
            continue
        if placement.scene_number not in groups or placement.scene_number < previous_scene:
            raise ValueError("Staged steps must retain order within existing scenes.")
        previous_scene = placement.scene_number
        if placement.handling == "adjust":
            if placement.adjustment is None or not placement.reason_from_source.strip():
                raise ValueError("An adjusted step needs a replacement and source facts.")
            step = placement.adjustment
        else:
            if placement.adjustment is not None:
                raise ValueError("An unchanged step cannot have a replacement.")
            step = ChainStep(**{key: original[key] for key in ("character_id", "action", "result")})
        groups[placement.scene_number].append((f"step{placement.step_number}", step))
    scenes, realized = [], []
    condition = draft.current_facts.situation
    destination = condition
    options = options or ScriptOptions()
    total_weight = sum(scene.length_weight for scene in draft.scenes)
    for n, setup in enumerate(draft.scenes, 1):
        steps = groups[n] + [(f"follow{n}_{i}", step) for i, step in enumerate(setup.follow_through, 1)]
        if not steps and not setup.interaction_end.strip():
            raise ValueError("An interaction-only scene needs an ending situation.")
        if any(step.character_id not in setup.character_ids for _, step in steps):
            raise ValueError("A staged action's actor must be present in its scene.")
        required = [{"id": key,
                "description": step.character_id + ": " + step.action + " → " + step.result}
                for key, step in steps] or [{"id": f"interaction{n}", "description": setup.interaction}]
        end = steps[-1][1].result if steps else setup.interaction_end
        body = round(options.target_body_characters * setup.length_weight / total_weight)
        dialogue = round(options.target_dialogue_characters * setup.length_weight / total_weight)
        objectives = (f"【会話】{setup.interaction}\n【場面・動作】{setup.staging}\n"
                      f"【分量の補助目安】この場面の本文約{body}文字、うち台詞約{dialogue}文字。"
                      "会話と応答を十分に描く。字数検査や機械的な水増しはしない。")
        scenes.append(ScenePlan(id=setup.id, location_id=setup.location_id,
            character_ids=setup.character_ids, objectives=objectives,
            start_state=condition, required_events=required, end_state=end, atmosphere=setup.atmosphere))
        # A scene can contain several source events. Keep event chunks bounded
        # without changing action order or inventing another endpoint.
        for start in range(0, len(steps), 6):
            chunk = [step for _, step in steps[start:start + 6]]
            realized.append(ChainEvent(start_condition=condition, steps=chunk))
            condition = chunk[-1].result
        if steps:
            destination = end
        condition = end
    changes = []
    for change in draft.future_route_changes:
        # route is only a legacy view; events are authoritative for this path.
        changes.append(RouteAdjustment(chapter_number=change.chapter_number,
            reason_from_source=change.reason_from_source, route=change.events[-1].as_route(), events=change.events))
    return {"current_facts": draft.current_facts, "destination": destination,
            "bridge": "割当行動を場面のrequired_eventsへ直接引き継ぐ。",
            "character_actions": [], "future_route_changes": changes,
            "unresolved_core_gaps": draft.unresolved_core_gaps,
            "new_characters": draft.new_characters, "locations": draft.locations, "scenes": scenes,
            "step_placements": draft.placements, "realized_events": realized}


REALIZATION_INSTRUCTION = (
    "会話中心のビジュアルノベルとして、当章の番号付きの主要行動列を場面化します。主筋の行動と結果は保持します。"
    "各話の分量方針とconversation_topicsを参照し、適した話題を当場面のinteractionへ具体化します。全項目の消化は不要です。"
    "サブは主筋に不可欠でなくても、本人の用事や掛け合い、日常や相手の別の面を描くため登場できます。"
    "interactionに誰と誰が何を持ちかけ、どんな応答や寄り道を描くかを1〜3文で記します。"
    "全台詞を固定したり主筋の別案を作ったりせず、執筆で膨らませる具体的な接点を示します。"
    "交流だけの場面も可能です。その場合は割当行動やfollow_throughを捏造せずinteraction_endに閉じる状況を記します。"
    "主要行動のある場面ではinteraction_endは空文字で、終了状態は最後の行動結果です。"
    "length_weightは章内での相対的な分量比1〜10です。全章合計ではなく各話の量を場面へ配分します。"
    "current_factsは実台本の現在地・選択・未決事項と、人物に帰属する評価や推測を分けて記録します。初章のみ設定を使います。"
    "placementsに全step_numberを順番どおり一度ずつ記します。通常はstageとし、描くscene_numberを指定します。"
    "本文で既に実施した行動だけalready_doneとし、具体的な実績をreason_from_sourceへ記します。場面番号は不要です。"
    "実台本と予定が合わないときはadjustで、核心へ向かう行動・結果の調整と原文に基づく理由を記します。"
    "過去の資料がないことを実施済みの根拠にせず、未実施の当章計画も実績にしません。"
    "stageではstep_number・handling・scene_numberだけを出します。元の行動と結果は自動的に引き継がれ、adjustmentは出しません。"
    "scenesは会話・行動のまとまりから決めます。掛け合いや休憩を描く場面も置けます。"
    "一続きの試行と応答はまとめ、新たな情報・結果・選択、場所や時間の変化で区切ります。上限8は目標ではありません。"
    "一場面に割り当てる行動とfollow_throughは合計8件以内です。"
    "stagingは場所や必要な動作を短く補います。沈黙や手元の逐次説明で分量を作りません。"
    "follow_throughは必要な実行や後始末のみ。全て実施済みなら残る帰結を描けます。不要なら空配列です。"
    "核心が実行完了を求めているのに準備・約束で止まる場合は、その履行まで場面化します。"
    "理解・発見を最初から済んだ状態にせず、得る過程を行動と結果で示します。"
    "他者の内心を既知にせず、全員の成長・相談・和解を強制しません。"
    "future_route_changesは実台本により必要になった後続章の新しいeventsだけ。不要なら空配列。"
    "核心と実績を両立できない点はunresolved_core_gapsに記します。合否判定はしません。"
    "計画済みサブはIDをcharacter_idsに指定するだけで登場でき、new_charactersで再作成しません。"
    "new_charactersは計画外で当章に出演する新規人物だけ。不要なら空配列です。"
    "計画された将来の登場をcurrent_factsや過去の発言にしません。初登場でも既知の間柄は維持します。"
    "各sceneのcharacter_idsは行動する人物を含め最大3人。locationsは使う場所だけ。"
    "既存の場所IDの名前を変えず、背景の状態は今回に合わせます。別の場所へ移るなら場面を分けます。"
    "image_promptは人物のいない背景の英語描写です。"
)
