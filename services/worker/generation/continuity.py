"""Extract chapter state without rewriting adopted prose or previous artifacts."""

import copy
import json

from packages.contracts.m3 import NarrativeResult, NarrativeScene, StoryOutline
from packages.contracts.m4 import ContinuityReview, StoryState
from packages.narrative.continuity import (
    story_state_hash,
    validate_continuity,
    validate_story_state,
)


class InferredStateReviewError(RuntimeError):
    """Repair inferred predecessor state instead of regenerating only new prose."""


def _expected_foreshadowing(outline: StoryOutline, chapter_number: int) -> list[dict]:
    """Represent the immutable schedule, never evidence that an event occurred."""
    return [{"outline_index": index,
             "status": "resolved" if chapter_number >= item.payoff_chapter else
                       "planted" if chapter_number >= item.setup_chapter else "pending",
             "changed_chapter": item.payoff_chapter if chapter_number >= item.payoff_chapter else
                                item.setup_chapter if chapter_number >= item.setup_chapter else 0}
            for index, item in enumerate(outline.foreshadowing)]


def _state_schema(character_ids: set[str], *, chapter_number: int | None = None,
                  outline: StoryOutline | None = None) -> dict:
    schema = StoryState.model_json_schema()
    schema["$defs"]["CharacterStoryState"]["properties"]["character_id"]["enum"] = sorted(character_ids)
    if chapter_number is not None:
        schema["properties"]["chapter_number"]["const"] = chapter_number
        if outline is not None:
            expected = _expected_foreshadowing(outline, chapter_number)
            clues = schema["properties"]["foreshadowing"]
            clues.update(minItems=len(expected), maxItems=len(expected))
            variants = []
            for item in expected:
                variant = copy.deepcopy(schema["$defs"]["ForeshadowingState"])
                for key, value in item.items():
                    variant["properties"][key]["const"] = value
                evidence = variant["properties"]["evidence_utterance_ids"]
                if item["status"] == "pending":
                    evidence["maxItems"] = 0
                else:
                    evidence["minItems"] = 1
                variants.append(variant)
            if variants:
                clues["items"] = {"anyOf": variants}
    return schema


def _source(scenes: list[NarrativeScene]) -> list[dict]:
    """Keep every adopted word and evidence ID, without rendering metadata.

    Display and spoken text are usually identical. The omitted spoken_text is
    defined as display_text; a distinct spoken form is always retained. Source
    offsets, voice annotations and delivery hints remain in the artifact, but
    are not additional story evidence for state extraction or review.
    """
    return [{"plan": scene.plan.model_dump(), "utterances": [
        {"id": utterance.id, "speaker_id": utterance.speaker_id,
         "display_text": utterance.display_text,
         **({"spoken_text": utterance.spoken_text}
            if utterance.spoken_text != utterance.display_text else {})}
        for utterance in scene.utterances]} for scene in scenes]


def _source_json(scenes: list[NarrativeScene]) -> str:
    return json.dumps(_source(scenes), ensure_ascii=False, separators=(",", ":"))


SOURCE_FORMAT = ("\n本文の各項目は原文のid・speaker_id・display_textです。"
                 "spoken_text省略時はdisplay_textと同一であり、本文の省略ではありません。"
                 "speaker_id=nullは地の文です。提示順序とIDを根拠に使います。")


def start_state(llm, context: str, outline: StoryOutline, character_ids: set[str],
                previous: NarrativeResult | None) -> StoryState:
    from .narrative import _data, _structured

    if previous is not None and previous.end_state is not None:
        return previous.end_state.model_copy(deep=True)
    chapter = previous.chapter_number if previous else 0
    scenes = previous.scenes if previous else []
    utterances = {u.id for s in previous.scenes for u in s.utterances} if previous else set()
    instruction = (f"\nchapter_number={chapter}, schema_version=1の開始状態を抽出します。"
        "本文を変更せず、全キャストの現在地・関係性・所持品・負傷・約束・内心を記録します。"
        "factsは今までに成立した事実・秘密の固定IDと内容です。未来のプロットを既知の事実にしません。"
        "各人物のknowledgeにはその人物が知るfact_idだけを含め、他人だけが知る秘密は含めません。"
        "承認設定から本編開始前に知ると確認できるものはacquired_chapter=0、evidence_utterance_ids=[]。"
        "本文で知ったものはacquired_chapterに章番号、evidence_utterance_idsに認識の根拠となる発話ID。"
        "不明な知識を推測で追加しません。facts/knowledgeは不要なら空配列。"
        "foreshadowingはoutline_index（0始まり）で全伏線を追跡し、設置前pending、設置章以降planted、"
        "回収章以降resolved。pendingはchanged_chapter=0、evidence_utterance_ids=[]。"
        "設置/回収した項目はchanged_chapterにその章番号と実本文の根拠IDを保存します。"
        "無い所持品・負傷・約束等は空配列。設定や本文で確認できない出来事を補完しません。")
    # An exact old request may already have a validated response. Keep that
    # request available for cache-only replay, never for a new oversized call.
    legacy_source = [{"plan": scene.plan.model_dump(),
                      "utterances": [u.model_dump() for u in scene.utterances]}
                     for scene in scenes]
    legacy_prompt = (context + "\n固定された全体設計: " + _data(outline)
        + "\n採用済みの前章本文（空なら本編開始前）: " + _data(legacy_source) + instruction)
    return _structured(llm, "story_state", context
        + "\n採用済みの前章本文（空なら本編開始前）: " + _source_json(scenes)
        + SOURCE_FORMAT + instruction,
        StoryState, schema=_state_schema(character_ids),
        legacy_prompt=legacy_prompt,
        validate=lambda value: validate_story_state(value, chapter_number=chapter,
            character_ids=character_ids, outline=outline, utterance_ids=utterances))


def _review_legacy_state(llm, payload: dict, context: str, initial: StoryState,
                         previous: NarrativeResult, character_ids: set[str]) -> None:
    """Verify an inferred predecessor independently of the newly written chapter."""
    from .narrative import SceneContentError, StructuredGenerationError, _data, _structured

    validate_story_state(initial, chapter_number=previous.chapter_number,
        character_ids=character_ids, outline=previous.outline,
        utterance_ids={u.id for scene in previous.scenes for u in scene.utterances})

    def check(review):
        if not review.passed or review.issues:
            raise SceneContentError("Predecessor state review rejected source/state: "
                                    + "; ".join(review.issues))
        if (len(review.checked_character_ids) != len(character_ids)
                or set(review.checked_character_ids) != character_ids
                or sorted(review.checked_foreshadowing_indices)
                    != list(range(len(previous.outline.foreshadowing)))):
            raise ValueError("Predecessor state review must check every character and foreshadowing item.")

    try:
        review = _structured(llm, "continuity_review", context
            + f"\n以下は第{previous.chapter_number}章の変更禁止の採用本文です: "
            + _source_json(previous.scenes) + SOURCE_FORMAT
            + "\nこの本文から抽出した終了状態（次章の開始状態）の候補: " + _data(initial)
            + "\n既存章本文と抽出状態だけを照合します。checked_character_idsに全人物、"
            "checked_foreshadowing_indicesに全伏線の0始まり番号を1回ずつ記録します。"
            "各人物のknowledgeの獲得時期・根拠IDと発話順序を照合し、未取得の秘密を知る、"
            "その場にいない人物へ知識が漏れる、根拠と内容が一致しない場合はpassed=false、"
            "具体的な問題と発話IDをissuesへ記録します。設定からの初期知識と本文で獲得した"
            "知識を区別し、未来の全体プロットを既知の事実として追加することを認めません。"
            "場所、関係、所持品、負傷、約束、感情、factsが設定と実本文に一致するか、"
            "伏線の設置/回収時期・根拠が一致するかも検査します。要約や創作による状態の"
            "補完を認めません。全条件を満たすときだけpassed=true、issues=[]。"
            "本文・状態を再出力/改変しません。",
            ContinuityReview, validate=check)
    except StructuredGenerationError as error:
        raise InferredStateReviewError(str(error)) from error
    llm.trace.append({"type": "legacy_state_review", "chapter_number": previous.chapter_number,
        "previous_narrative_artifact_id": payload.get("previous_narrative_artifact_id"),
        "state_hash": story_state_hash(initial), "review": review.model_dump(mode="json")})


def finish_narrative(llm, payload: dict, context: str, result: NarrativeResult,
                      initial: StoryState, previous: NarrativeResult | None) -> NarrativeResult:
    from .narrative import SceneContentError, _data, _structured

    character_ids = {c.character_id for c in initial.characters}
    if previous is not None and previous.end_state is None:
        _review_legacy_state(llm, payload, context, initial, previous, character_ids)
    common = (context + "\n章開始時の状態: " + _data(initial)
              + "\n変更禁止の採用本文（順序通り）: " + _source_json(result.scenes) + SOURCE_FORMAT)
    end = _structured(llm, "story_state", common
        + f"\n第{result.chapter_number}章終了時のStoryStateをschema_version=1、"
        f"chapter_number={result.chapter_number}で抽出します。開始状態の章番号は返しません。"
        "全人物の現在地・関係性・所持品・負傷・約束・内心を本文に従い更新します。"
        "factsの既存ID/内容、既知のknowledgeを削除・変更せず引き継ぎます。"
        "新しい知識だけ現在の章番号acquired_chapterと本文のevidence_utterance_idsを記録します。"
        "knowledgeにはその人物が実際に知った事実のみ追加し、傍にいない人物や秘密を知らない人物に"
        "知識を共有しません。新しい事実は一意のfact_idで追加します。"
        "foreshadowingは全outline_indexを保持し、設置章でplanted、回収章でresolvedとし、"
        "changed_chapterにその章番号と本文根拠IDを記録します。"
        "進展がない伏線は開始状態のレコードをそのまま保存します。"
        "本文で描かれていない出来事や未来の事実を追加しません。"
        + "\n承認プロットから計算した全伏線の必須状態（outline_index/status/changed_chapter）: "
        + _data(_expected_foreshadowing(result.outline, result.chapter_number))
        + "\nこの一覧は設置・回収の予定であり、実際に描かれた証拠ではありません。"
        "予定だけを根拠に出来事・知識を補完せず、根拠IDは実本文の該当箇所を引用します。"
        "根拠がない場合に無関係な発話IDを当てはめてはいけません。",
        StoryState, schema=_state_schema(character_ids, chapter_number=result.chapter_number,
                                         outline=result.outline),
        validate=lambda value: validate_story_state(value, chapter_number=result.chapter_number,
            character_ids=character_ids, outline=result.outline,
            utterance_ids={u.id for s in result.scenes for u in s.utterances}, previous=initial))
    result = result.model_copy(update={
        "storyline_id": payload.get("storyline_id"), "start_state": initial, "end_state": end,
        "previous_narrative_artifact_id": payload.get("previous_narrative_artifact_id"),
        "previous_state_hash": story_state_hash(initial) if previous else None,
        "predecessor_state_inferred": previous is not None and previous.end_state is None,
    })

    def check(review):
        if not review.passed or review.issues:
            raise SceneContentError("Chapter continuity review rejected source/state: " + "; ".join(review.issues))
        validate_continuity(result.model_copy(update={"continuity_review": review}), character_ids,
                            previous_narrative=previous, require_state=True)

    review = _structured(llm, "continuity_review", common
        + "\n終了状態の候補: " + _data(end)
        + "\n本文と状態更新を検査します。checked_character_idsに全人物、"
        "checked_foreshadowing_indicesに全伏線の0始まり番号を1回ずつ記録します。"
        "各人物のknowledgeと発話順序を照合し、まだ知らない秘密を話す・知る前に利用する・"
        "その場にいない人物へ知識が漏れる場合はpassed=false、具体的な問題と発話IDをissuesへ。"
        "開始/終了状態の場所、関係、所持品、負傷、約束、感情が実本文に一致するかも検査します。"
        "本文による知識獲得の根拠、伏線設置/回収の実在と時期を検査し、要約や創作による"
        "状態の補完を認めません。未来のプロットの情報を人物の既知情報と混同しません。"
        "全条件を満たすときだけpassed=true、issues=[]。本文・状態を再出力/改変しません。",
        ContinuityReview, validate=check)
    return result.model_copy(update={"continuity_review": review})
