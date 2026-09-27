"""A plot may propose future transitions but cannot invent independent past starts."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from services.worker.generation.script_plot import (
    ChapterAllocation,
    ConversationTopic,
    StoryChain,
    StoryChainDraft,
    allocate_chain,
    check_chain,
    future_material,
)


def draft_data():
    return {
        "core": {
            "central_question": "離れた仲間へ約束した観測結果を届けられるか。",
            "external_resolution": "古い観測所から通信を復旧し、記録を届ける。",
            "relationship_resolution": "観測を続ける人と街へ戻る人が別々の役目を選ぶ。",
            "characters": [{
                "character_id": "observer",
                "initial_behavior": "一人で装置を直そうとする。",
                "enduring_value": "記録を待つ人との約束を守る。",
                "final_behavior": "街へ戻る仲間に地上の連絡を任せる。",
            }],
        },
        "resolution_basis": [{
            "earlier_experience": "装置を調べ、観測所なら通信を復旧できると知る。",
            "later_application": "観測所から送れる結果を使い、二人が山と街に分かれて記録を送り受信を確かめる。",
        }],
        "opening_condition": "街に戻った二人が観測装置の故障を確かめる。",
        "events": [
            {"steps": [
                {"character_id": "observer", "action": "街の装置を調べる。",
                 "result": "交換部品が山の観測所にあると分かる。"},
                {"character_id": "guide", "action": "古い登山道を案内する。",
                 "result": "二人が観測所に到着し、鍵のかかった扉を見つける。"},
            ]},
            {"steps": [
                {"character_id": "observer", "action": "管理人へ事情を説明し鍵を借りる。",
                 "result": "装置を調べ、観測所からなら通信を復旧できると分かる。"},
            ]},
            {"steps": [
                {"character_id": "guide", "action": "地上へ戻り連絡を受ける役を引き受ける。",
                 "result": "二人が別の場所から通信を試せる状態になる。"},
                {"character_id": "observer", "action": "観測所の装置を修理して記録を送る。",
                 "result": "仲間から受信の返答が届き、約束が果たされる。"},
            ]},
        ],
    }


def allocation():
    return ChapterAllocation(chapters=[{
        "number": number,
        "last_event": number,
        "title": f"第{number}章",
        "role": "予定した出来事を会話と試行で描く。",
        "conversation_topics": [],
    } for number in (1, 2, 3)], foreshadowing=[])


def test_generated_chain_has_one_opening_and_keeps_all_causal_steps_through_export():
    data = draft_data()
    original = deepcopy(data)
    chain = StoryChainDraft.model_validate(data).as_chain()
    check_chain(chain, {"observer"}, {"observer", "guide"})
    assert [event.start_condition for event in chain.events] == [
        data["opening_condition"],
        data["events"][0]["steps"][-1]["result"],
        data["events"][1]["steps"][-1]["result"],
    ]
    assert [event.model_dump()["steps"] for event in chain.events] == [
        event["steps"] for event in data["events"]]
    assert data == original

    # Allocation and the existing export view must not create a new start or
    # drop the journey that made the later destination available.
    plot = allocate_chain(chain, allocation())
    assert [chapter.route.start_condition for chapter in plot.chapters] == [
        event.start_condition for event in chain.events]
    assert "二人が観測所に到着" in plot.as_outline().chapters[0].summary
    assert "鍵を借りる" in plot.as_outline().chapters[1].summary
    assert StoryChain.model_validate_json(chain.model_dump_json()) == chain


def test_generated_schema_cannot_accept_independent_later_start_conditions():
    schema = StoryChainDraft.model_json_schema()
    assert "opening_condition" in schema["required"]
    assert "start_condition" not in schema["$defs"]["ChainEventDraft"]["properties"]
    assert schema["$defs"]["ChainEventDraft"]["additionalProperties"] is False
    data = draft_data()
    data["events"][1]["start_condition"] = "必要な移動を描かず別の街へ到着済み。"
    with pytest.raises(ValidationError, match="start_condition"):
        StoryChainDraft.model_validate(data)


def test_resolution_basis_is_saved_in_draft_but_never_becomes_a_second_plot_source():
    data = draft_data()
    data["resolution_basis"] = [{"earlier_experience": "DRAFT_ONLY_EXPERIENCE",
                                 "later_application": "DRAFT_ONLY_APPLICATION"}]
    original = deepcopy(data)
    draft = StoryChainDraft.model_validate(data)
    restored = StoryChainDraft.model_validate_json(draft.model_dump_json())
    assert restored == draft
    chain = restored.as_chain()
    plot = allocate_chain(chain, allocation())
    # A design note is neither a new event nor a historical accomplishment.
    for downstream in (chain.model_dump_json(), plot.model_dump_json(),
                       plot.as_outline().model_dump_json(), str(future_material(plot, "plot", 2, []))):
        assert "resolution_basis" not in downstream and "DRAFT_ONLY_" not in downstream
    assert [event.model_dump()["steps"] for event in chain.events] == [row["steps"] for row in data["events"]]
    assert data == original
    # Persisted StoryChain remains readable without the new generation-only field.
    assert StoryChain.model_validate_json(chain.model_dump_json()) == chain


def test_resolution_basis_is_required_in_new_generation_but_may_be_empty():
    schema = StoryChainDraft.model_json_schema()
    assert list(schema["properties"]) == ["core", "resolution_basis", "opening_condition", "events"]
    assert "resolution_basis" in schema["required"]
    assert schema["properties"]["resolution_basis"]["maxItems"] == 2
    assert set(schema["$defs"]["ResolutionBasis"]["properties"]) == {
        "earlier_experience", "later_application"}
    data = draft_data()
    data["resolution_basis"] = []
    assert StoryChainDraft.model_validate(data).as_chain().events
    del data["resolution_basis"]
    with pytest.raises(ValidationError, match="resolution_basis"):
        StoryChainDraft.model_validate(data)


@pytest.mark.parametrize("field", ["earlier_experience", "later_application"])
def test_resolution_basis_has_bounded_fields_without_prefix_cropping(field):
    data = draft_data()
    data["resolution_basis"][0][field] = "長" * 181
    original = deepcopy(data)
    with pytest.raises(ValidationError, match=field):
        StoryChainDraft.model_validate(data)
    assert data == original


def test_resolution_basis_requires_both_experience_and_application_without_reinterpreting_old_drafts():
    data = draft_data()
    data["resolution_basis"] = [{"earlier_experience": "道具を試して限界を知る。"}]
    with pytest.raises(ValidationError, match="later_application"):
        StoryChainDraft.model_validate(data)
    data["resolution_basis"] = [{"means": "相手を助ける。", "required_condition": "相手が危機に陥る。",
                                 "planned_setup": "危機を目撃する。"}]
    original = deepcopy(data)
    with pytest.raises(ValidationError, match="earlier_experience"):
        StoryChainDraft.model_validate(data)
    assert data == original


def test_resolution_basis_does_not_require_matching_every_event():
    data = draft_data()
    data["resolution_basis"] *= 2
    # Duplicate or unfulfilled intentions are a reading concern, not a new gate.
    chain = StoryChainDraft.model_validate(data).as_chain()
    check_chain(chain, {"observer"}, {"observer", "guide"})
    data["resolution_basis"].append(data["resolution_basis"][0])
    with pytest.raises(ValidationError, match="resolution_basis"):
        StoryChainDraft.model_validate(data)


def test_conversation_schema_generates_exchange_and_reads_legacy_text_without_rewriting_it():
    schema = ConversationTopic.model_json_schema()
    assert set(schema["properties"]) == {"character_ids", "topic", "exchange"}
    assert "exchange" in schema["required"] and "relationship_aspect" not in schema["required"]
    legacy = {"character_ids": ["observer", "guide"], "topic": "道具を返したい。",
              "relationship_aspect": "旧記録では気安い友人同士と説明されていた。"}
    original = deepcopy(legacy)
    topic = ConversationTopic.model_validate(legacy)
    assert topic.exchange == legacy["relationship_aspect"]
    assert topic.model_dump() == {"character_ids": legacy["character_ids"], "topic": legacy["topic"],
                                 "exchange": legacy["relationship_aspect"]}
    assert ConversationTopic.model_validate_json(topic.model_dump_json()) == topic
    assert legacy == original
    with pytest.raises(ValidationError, match="relationship_aspect"):
        ConversationTopic.model_validate({**legacy, "exchange": "別のやり取り"})


def test_legacy_conversation_is_a_future_material_view_and_does_not_invent_executed_actions():
    planned_allocation = allocation().model_dump()
    legacy = {"character_ids": ["observer", "guide"], "topic": "昼食の相談。",
              "relationship_aspect": "LEGACY_RELATIONSHIP_ONLY"}
    planned_allocation["chapters"][1]["conversation_topics"] = [legacy]
    original = deepcopy(planned_allocation)
    chain = StoryChainDraft.model_validate(draft_data()).as_chain()
    plot = allocate_chain(chain, ChapterAllocation.model_validate(planned_allocation))
    future = future_material(plot, "legacy-plot", 2, [])
    assert future["routes"][0]["conversation_topics"][0]["exchange"] == legacy["relationship_aspect"]
    assert "LEGACY_RELATIONSHIP_ONLY" not in str(future["routes"][0]["events"])
    assert plot.chapters[1].events == chain.events[1:2]
    assert planned_allocation == original


@pytest.mark.parametrize("events", [True, False])
def test_future_input_labels_start_as_planned_without_changing_saved_contract(events):
    plot = allocate_chain(StoryChainDraft.model_validate(draft_data()).as_chain(), allocation())
    if not events:
        plot = plot.model_copy(update={"chapters": [
            chapter.model_copy(update={"events": []}) for chapter in plot.chapters]})
    original = plot.model_dump()
    material = future_material(plot, "saved-plot", 2, [])
    assert [row["number"] for row in material["routes"]] == [2, 3]
    for expected, row in zip(plot.chapters[1:], material["routes"], strict=True):
        projected = row["events"][0] if events else row["route"]
        assert projected["planned_start_condition"] == expected.route.start_condition
        assert "start_condition" not in projected
    assert plot.model_dump() == original


def test_draft_projection_does_not_add_semantic_acceptance_gates():
    # Structural inheritance is deliberately not a new all-event semantic test.
    data = draft_data()
    data["events"][1]["steps"][0]["result"] = data["events"][0]["steps"][-1]["result"]
    chain = StoryChainDraft.model_validate(data).as_chain()
    check_chain(chain, {"observer"}, {"observer", "guide"})
    assert len(allocate_chain(chain, allocation()).chapters) == 3
