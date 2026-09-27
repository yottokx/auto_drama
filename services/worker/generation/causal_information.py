"""Extract information exchanges; assign ledger identities in application code.

The model judges current prose and recipients, never record IDs or preconditions.
Current utterance aliases are expanded by the caller before semantic review.
"""

from __future__ import annotations

import copy
from typing import Literal

from pydantic import Field

from packages.contracts.m2 import CharacterId
from packages.contracts.script import Contract, Identifier
from packages.contracts.story_workflow import ChapterMemory, StoryText

from . import narrative as legacy
from .causal_context import build_context
from .causal_runtime import digest, encoded


class KnowledgeChange(Contract):
    character_id: CharacterId
    kind: Literal["knowledge", "belief", "uncertain", "retracted"]
    change: Literal["learned", "revised"]
    acquired_at: StoryText


class ReaderIntroduction(Contract):
    kind: Literal["character", "location", "fact", "relationship"]
    entity_ids: list[CharacterId] = Field(min_length=1, max_length=100)
    description: StoryText
    audience_character_ids: list[CharacterId] = Field(max_length=100)


class InformationObservation(Contract):
    fact_id: Identifier | None
    content: StoryText
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=20)
    event_ids: list[Identifier] = Field(max_length=24)
    knowledge_changes: list[KnowledgeChange] = Field(max_length=100)
    reader_introduction: ReaderIntroduction | None
    existing_record_ids: list[Identifier] = Field(max_length=50)
    reason: str = Field(min_length=1, max_length=2000)


class KnowledgeObservation(InformationObservation):
    fact_id: Identifier
    knowledge_changes: list[KnowledgeChange] = Field(min_length=1, max_length=100)


class NonKnowledgeObservation(InformationObservation):
    knowledge_changes: list[KnowledgeChange] = Field(max_length=0)


class InformationAnnotation(Contract):
    inspected_utterance_ids: list[Identifier] = Field(min_length=1, max_length=1000)
    observations: list[KnowledgeObservation | NonKnowledgeObservation] = Field(min_length=1, max_length=32)


INSTRUCTION = """
あなたは今回本文の情報交換を独立に抽出する担当です。全発話を順に読み、情報ごとのobservations行を返します。
current_candidate_sourceだけが今回誰に何が伝わったかの原文です。current_event_indexは参照専用の出来事索引です。
新event・知識レコード・紹介レコードのIDやsupersedesは生成しません。アプリが保持・確定します。
event_idsはその情報交換を表す今回の既存eventだけを選び、該当がなければ[]で原文evidence_idsを示します。
全台詞をイベント化せず、同じ情報の説明と応答は一つの行へまとめます。

previous_observed_informationは採用済みの知識・紹介の索引です。author_secrets_not_shared_knowledgeは
初期作者事実とknown_by_character_idsの参照専用資料です。その人物は開始前から知っています。
作者事実の全文を、他人物が今回獲得したcontentへコピーしません。設定に複数の秘密や細部が並んでも、
今回の発話で実際に伝わった部分だけを記録します。原文で言っていない人物名・日付・理由を足しません。
同じ命題には既存fact_idを使い、言い換えで新IDを作りません。別の命題だけ新しい短いsnake_caseのfact_idにします。
場所・人物の紹介や情報変化のない確認ではfact_idはnullにできます。knowledge_changesがある行では必須です。
人物紹介と知識更新が同じ行にあっても、知識の命題を識別するfact_idを設定します。
fact_id=nullの行はknowledge_changes=[]専用です。実際の情報伝達を空配列にして落とさず、適切なfact_idの行へ記録します。

- knowledge_changesには実際に情報を受け取った人物だけを記録します。人物別にkindを判断します。
  観測した事実=knowledge、噂・信念=belief、未確定=uncertain、過去の誤りの撤回=retractedです。
  「Aがそう説明した」と聞いた事実と、説明の中身が真実であることを区別し、不確かさを消しません。
  acquired_atは知った時点です。回想の過去時点と今回読者が知る時点を混ぜません。
- 初取得はchange=learned。採用済みknowledgeまたは初期known_byで既知の同じfactならlearnedにしません。
  新しい細部・訂正・不確かさの変化が本文にあるときだけchange=revisedとし、reasonで何が変わったか説明します。
  完全に同じ内容や言い換えだけの再確認はknowledge_changes=[]、existing_record_idsとreasonへ記録します。
  同じ情報でも既知の人と新たな受け手を分け、(人物,fact_id)は一回だけ更新します。
- 内心・地の文・相手不在の発話を、その場の全員の知識へ移しません。作者資料にあるだけでは読者も知りません。
  読者だけへの開示はknowledge_changes=[]でreader_introductionを返し、そのaudience_character_idsは[]です。
  人物には既知でも読者へ初めて示す場合は紹介を作れます。読者に紹介済みでも新たな人物は知識を獲得できます。
- reader_introductionは今回初めて読者に示された人物・場所・事実・関係のみ。
  紹介済みの場所や人物の再訪・再登場にはnull。同じ説明を毎回初紹介しません。
  新しい細部は既存人物の再紹介でなく、その事実の紹介として記録します。
  audience_character_idsはその紹介を実際に受けた人物だけです。出演者全員を自動で列挙しません。
- evidence_idsは今回の発話IDだけ。過去のq IDやhashは生成しません。既知の参照はexisting_record_idsへ入れます。
  過去との関係が不明ならreasonに具体的に記し、後段で過去原文を取得して確認できるようにします。
- inspected_utterance_idsは今回の全発話IDを順番通り返します。全フィールド・配列は必須、該当なしは[]またはnull。
  情報の更新・紹介がなくてもobservationsを空にせず、確認した内容・原文根拠・追加不要の具体的理由を一行以上残します。
"""


def _previous_information(memory: ChapterMemory, scene):
    context = build_context(memory, scope="extraction", character_ids=scene.plan.character_ids,
                            location_ids=[scene.plan.location_id], include_author_facts=True)
    if context["status"] != "ready":
        raise ValueError("Information extraction is missing adopted context: "
                         + "; ".join(context["missing_information"]))
    return {"memory_hash": context["memory_hash"],
            "retrieval_instructions": context["retrieval_instructions"],
            "knowledge": context["knowledge"], "introductions": context["already_introduced"],
            "event_index": context["event_index"],
            "author_secrets_not_shared_knowledge": context["author_secrets_not_shared_knowledge"]}


def _introduction_key(item):
    return item.kind, tuple(sorted(item.entity_ids))


def _validate(annotation: InformationAnnotation, *, memory: ChapterMemory, scene, primary: dict):
    utterance_ids = [item.id for item in scene.utterances]
    if annotation.inspected_utterance_ids != utterance_ids:
        raise ValueError("Information audit must inspect every current utterance exactly once in order.")
    available_evidence = set(utterance_ids)
    old_ids = {item.id for group in (memory.events, memory.knowledge, memory.introductions, memory.author_facts)
               for item in group}
    current_events = {event["id"] for event in primary.get("events", [])}
    scene_cast = set(scene.plan.character_ids)
    latest = {(item.character_id, item.fact_id): item for item in memory.knowledge}
    canon = {item.id: item for item in memory.author_facts}
    old_introductions = {_introduction_key(item): item for item in memory.introductions}
    seen_knowledge, seen_introductions = set(), set()
    for observation in annotation.observations:
        if (len(observation.evidence_ids) != len(set(observation.evidence_ids))
                or not set(observation.evidence_ids).issubset(available_evidence)):
            raise ValueError("Information evidence must resolve to current source utterances.")
        if (len(observation.event_ids) != len(set(observation.event_ids))
                or not set(observation.event_ids).issubset(current_events)):
            raise ValueError("Information can reference only existing current events.")
        if not set(observation.existing_record_ids).issubset(old_ids):
            raise ValueError("Information observation refers to unavailable history.")
        for change in observation.knowledge_changes:
            if change.character_id not in scene_cast or observation.fact_id is None:
                raise ValueError("Knowledge must name a current participant and a stable fact ID.")
            key = (change.character_id, observation.fact_id)
            if key in seen_knowledge:
                raise ValueError("A character/fact pair cannot be updated twice in one information batch.")
            seen_knowledge.add(key)
            old = latest.get(key)
            initial = canon.get(observation.fact_id)
            initially_known = initial is not None and change.character_id in initial.known_by_character_ids
            if change.change == "learned" and (old or initially_known):
                raise ValueError("Previously known information cannot be learned for the first time again.")
            if change.change == "revised" and not (old or initially_known):
                raise ValueError("A revision requires previous knowledge or an initially known fact.")
            if old and (observation.content, change.kind) == (old.content, old.kind):
                raise ValueError("Unchanged knowledge is a recall, not a new knowledge update.")
            if initially_known and not old and observation.content == initial.content and change.kind == "knowledge":
                raise ValueError("Unchanged initial knowledge is a recall, not a new knowledge update.")
            if change.kind == "retracted" and old is None:
                raise ValueError("A retraction requires a previously recorded belief or knowledge.")
        introduction = observation.reader_introduction
        if introduction is None:
            continue
        if (len(introduction.entity_ids) != len(set(introduction.entity_ids))
                or len(introduction.audience_character_ids) != len(set(introduction.audience_character_ids))
                or not set(introduction.audience_character_ids).issubset(scene_cast)):
            raise ValueError("Introduction identities and audience must be distinct actual participants.")
        if introduction.kind == "character" and not set(introduction.entity_ids).issubset(scene_cast):
            raise ValueError("Character introductions must identify current participants.")
        if introduction.kind == "location" and introduction.entity_ids != [scene.plan.location_id]:
            raise ValueError("Location introductions must identify the current location.")
        key = _introduction_key(introduction)
        if key in seen_introductions:
            raise ValueError("An information batch cannot introduce the same subject twice.")
        seen_introductions.add(key)
        old = old_introductions.get(key)
        if old and (introduction.kind in {"character", "location"}
                    or old.description == introduction.description):
            raise ValueError("Previously introduced information is a recall, not another first introduction.")


def _merge(annotation: InformationAnnotation, *, memory: ChapterMemory, scene, primary: dict, chapter: int):
    combined = copy.deepcopy(primary)
    combined["knowledge_updates"], combined["introductions"] = [], []
    latest = {(item.character_id, item.fact_id): item for item in memory.knowledge}
    prefix = f"c{chapter}-{scene.id}-info"
    if len(prefix) > 45:
        prefix = f"c{chapter}-s{digest(scene.id)[:12]}-info"
    for observation in annotation.observations:
        for change in observation.knowledge_changes:
            key = (change.character_id, observation.fact_id)
            old = latest.get(key)
            combined["knowledge_updates"].append({
                "id": prefix + "-k-" + digest(key)[:16], "character_id": change.character_id,
                "fact_id": observation.fact_id, "content": observation.content, "kind": change.kind,
                "acquired_at": change.acquired_at, "event_ids": observation.event_ids,
                "supersedes_id": old.id if old else None, "evidence": observation.evidence_ids})
        introduction = observation.reader_introduction
        if introduction is not None:
            combined["introductions"].append({
                "id": prefix + "-i-" + digest(_introduction_key(introduction))[:16],
                **introduction.model_dump(mode="json"), "reader_visible": True,
                "evidence": observation.evidence_ids})
    generated = [item["id"] for group in (combined["knowledge_updates"], combined["introductions"])
                 for item in group]
    reserved = {item.id for group in (memory.events, memory.knowledge, memory.introductions, memory.author_facts)
                for item in group} | {item["id"] for item in primary.get("events", [])}
    if len(generated) != len(set(generated)) or reserved.intersection(generated):
        raise ValueError("Deterministic information identities collided with an existing record.")
    return combined


def information_schema(memory: ChapterMemory, scene, primary: dict, *, previous=None) -> dict:
    """Restrict source/participant/event aliases; record identities are not model output."""
    schema = InformationAnnotation.model_json_schema()
    definitions = schema["$defs"]
    utterances = [u.id for u in scene.utterances]

    def enum_array(properties, key, values):
        values = sorted(set(values))
        if values:
            properties[key]["items"] = {"type": "string", "enum": values}
        else:
            properties[key]["maxItems"] = 0

    previous = previous if previous is not None else _previous_information(memory, scene)
    old_ids = [item["id"] for name in ("knowledge", "introductions", "event_index",
               "author_secrets_not_shared_knowledge") for item in previous[name]]
    for name in ("KnowledgeObservation", "NonKnowledgeObservation"):
        properties = definitions[name]["properties"]
        enum_array(properties, "evidence_ids", utterances)
        enum_array(properties, "event_ids", [event["id"] for event in primary.get("events", [])])
        enum_array(properties, "existing_record_ids", old_ids)
    definitions["KnowledgeChange"]["properties"]["character_id"] = {
        "type": "string", "enum": list(scene.plan.character_ids)}
    enum_array(definitions["ReaderIntroduction"]["properties"],
               "audience_character_ids", scene.plan.character_ids)
    enum_array(schema["properties"], "inspected_utterance_ids", utterances)
    schema["properties"]["inspected_utterance_ids"].update(minItems=len(utterances), maxItems=len(utterances))
    return schema


def annotate_information(run, memory: ChapterMemory, scene, raw_extraction, *,
                         validate_combined=None) -> dict:
    """Keep primary events/state/threads; independently replace knowledge/introductions."""
    memory = ChapterMemory.model_validate(memory)
    primary = (raw_extraction.model_dump(mode="json") if hasattr(raw_extraction, "model_dump")
               else copy.deepcopy(raw_extraction))
    number = run.payload["chapter_number"]
    previous = _previous_information(memory, scene)
    inputs = {"chapter_number": number, "scene_id": scene.id,
        "scene_character_ids": scene.plan.character_ids, "scene_location_id": scene.plan.location_id,
        "previous_observed_information": previous,
        "current_candidate_source": [{"id": u.id, "speaker": u.speaker_id, "text": u.display_text}
                                     for u in scene.utterances],
        "current_event_index": primary.get("events", [])}
    schema = information_schema(memory, scene, primary, previous=previous)

    def merge(value):
        return _merge(value, memory=memory, scene=scene, primary=primary, chapter=number)

    def validate(value):
        _validate(value, memory=memory, scene=scene, primary=primary)
        combined = merge(value)
        # Replay the whole batch before caching a successful information node.
        if validate_combined is not None:
            validate_combined(combined)

    annotation = run.node("information-" + scene.id,
        {"inputs": inputs, "instruction": INSTRUCTION, "schema": schema, "information_version": 7},
        InformationAnnotation,
        lambda: legacy._structured(run.llm, "information-" + scene.id,
            INSTRUCTION + "\n資料: " + encoded(inputs), InformationAnnotation,
            schema=schema, validate=validate))
    validate(annotation)
    result = merge(annotation)
    run.llm.trace.append({"type": "information_annotation", "chapter_number": number,
        "scene_id": scene.id, "inspected_utterance_ids": annotation.inspected_utterance_ids,
        "observations": [item.model_dump(mode="json") for item in annotation.observations],
        "knowledge_count": len(result["knowledge_updates"]),
        "introduction_count": len(result["introductions"])})
    return result
