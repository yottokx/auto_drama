"""Check classification retries and the speech-to-staging boundary without a model."""

import json

import pytest

from packages.contracts.m3 import ScenePlan
from packages.narrative import parse_scene_text, validate_narrative
from services.worker.generation.narrative import (
    StructuredGenerationError,
    generate_narrative,
    separate_speech,
)


class FakeLLM:
    def __init__(self, responses=()):
        self.config = {"llm": {"model_id": "fake", "temperature": 0.5, "max_tokens": 3072}}
        self.payload, self.trace, self.calls = {}, [], []
        self.responses = iter(responses)
        self.requests = 0

    def chat(self, *args, **kwargs):
        self.requests += 1
        self.calls.append((args, kwargs))
        return next(self.responses)


def response(value):
    return {"content": json.dumps(value, ensure_ascii=False)}


def annotation(identifier, kind="spoken", *, narration="", delivery=""):
    return {"candidate_id": identifier, "kind": kind, "narration": narration, "delivery": delivery}


def classification(*annotations):
    return response({"annotations": list(annotations)})


@pytest.fixture
def plan():
    return ScenePlan.model_validate({
        "id": "s1", "location_id": "home", "character_ids": ["human", "dog"],
        "objectives": "帰宅した犬を迎える", "start_state": "犬が玄関にいる",
        "required_events": [{"id": "welcome", "description": "犬が招かれて落ち着く"}],
        "end_state": "犬が家族のそばに落ち着く", "atmosphere": "穏やか",
    })


def classify(llm, plan, raw):
    return separate_speech(llm, "陽介と犬のポチ。ポチは人語を話せない。", plan, raw,
                           {"human": "陽介", "dog": "ポチ"})


def test_plain_speech_and_narrator_parentheses_do_not_invoke_the_model(plan):
    raw = "human: おかえり。\nNARRATOR: ポチ（犬）は駆け寄った。\ndog: ワン！"
    llm = FakeLLM()
    result, hints = classify(llm, plan, raw)
    assert result.raw_text == raw
    assert hints == {}
    assert llm.calls == []


@pytest.mark.parametrize("body", [
    '「(仮)」を外してほしい。', '「（小声で）」と書いてあるね。', '「（満面の笑みで）」という注記だよ。',
])
def test_literal_spoken_parentheses_remain_verbatim_and_audible(plan, body):
    raw = "human: " + body
    llm = FakeLLM([classification(annotation("s1-u1-p1"))])
    result, hints = classify(llm, plan, raw)
    assert result.raw_text == raw
    assert hints == {}
    assert parse_scene_text(result.raw_text, "s1", {"human"})[0].spoken_text == raw[7:]
    schema = llm.calls[0][1]["response_format"]["json_schema"]["schema"]
    assert schema["$defs"]["SpeechMeaning"]["properties"]["candidate_id"]["enum"] == ["s1-u1-p1"]


@pytest.mark.parametrize("cue", [
    "小声で", "明るい声で", "少し小声で、ゆっくりと", "ごく小さな声で,とても丁寧に", "穏やかに",
])
def test_pure_vocal_cues_remain_supported_as_hidden_delivery(plan, cue):
    raw = "human: （" + cue + "）おかえり。"
    llm = FakeLLM([classification(annotation("s1-u1-p1", "delivery", delivery=cue))])
    result, hints = classify(llm, plan, raw)
    assert result.raw_text == "human: おかえり。"
    assert hints == {"s1-u1": cue}
    assert len(llm.calls) == 1
    assert not any(item["type"] == "speech_separation_rejected" for item in llm.trace)


@pytest.mark.parametrize("cue,narration", [
    ("眉をひそめ、小声で", "陽介は眉をひそめ、小声で話した。"),
    ("満面の笑みで", "陽介は満面の笑みを浮かべた。"),
    ("小声で、ポチを抱きしめる", "陽介は小声で話し、ポチを抱きしめた。"),
    ("少し眉をひそめ、明るい声で", "陽介は少し眉をひそめながら、明るい声で話した。"),
])
def test_delivery_misclassification_cannot_hide_visible_actions(plan, cue, narration):
    raw = "human: （" + cue + "）おかえり。"
    llm = FakeLLM([
        classification(annotation("s1-u1-p1", "delivery", delivery="小声で")),
        classification(annotation("s1-u1-p1", "stage_direction", narration=narration)),
    ])
    result, hints = classify(llm, plan, raw)
    assert result.raw_text == "NARRATOR: " + narration + "\nhuman: おかえり。"
    assert hints == {}
    assert [call[0][0] for call in llm.calls] == ["speech_separation", "speech_separation"]
    feedback = llm.calls[1][0][1][1]["content"]
    assert cue in feedback
    assert "動作・表情を隠さないためstage_directionに分類" in feedback
    parsed = parse_scene_text(result.raw_text, "s1", {"human"})
    assert [(item.speaker_id, item.spoken_text) for item in parsed] == [
        (None, narration), ("human", "おかえり。"),
    ]
    trace = next(item for item in llm.trace if item["type"] == "speech_separation")
    assert trace["original_raw_text"] == raw
    assert trace["segments"][0]["text"] == "（" + cue + "）"
    assert trace["segments"][0]["display_text"] == narration


def test_repeated_delivery_misclassification_fails_instead_of_erasing_action(plan):
    raw = "human: （満面の笑みで）おかえり。"
    invalid = classification(annotation("s1-u1-p1", "delivery", delivery="明るい声で"))
    llm = FakeLLM([invalid, invalid])
    with pytest.raises(StructuredGenerationError, match="動作・表情を隠さない"):
        classify(llm, plan, raw)
    assert [call[0][0] for call in llm.calls] == ["speech_separation", "speech_separation"]
    assert not any(item["type"] == "speech_separation" for item in llm.trace)
    assert llm.failure_request == 1


def test_visible_actions_become_narration_and_delivery_becomes_hidden_hints(plan):
    raw = "human: （小声で）おかえり。（椅子を引く）どうぞ。\ndog: （しっぽを振る）"
    llm = FakeLLM([classification(
        annotation("s1-u1-p1", "delivery", delivery="小声で"),
        annotation("s1-u1-p2", "stage_direction", narration="陽介は椅子を引いた。"),
        annotation("s1-u2-p1", "stage_direction", narration="ポチはしっぽを振った。"),
    )])
    result, hints = classify(llm, plan, raw)
    parsed = parse_scene_text(result.raw_text, "s1", {"human", "dog"})
    assert [(item.speaker_id, item.spoken_text) for item in parsed] == [
        ("human", "おかえり。"), (None, "陽介は椅子を引いた。"),
        ("human", "どうぞ。"), (None, "ポチはしっぽを振った。"),
    ]
    assert hints == {"s1-u1": "小声で", "s1-u3": "小声で"}
    trace = next(item for item in llm.trace if item["type"] == "speech_separation")
    assert trace["original_raw_text"] == raw
    assert trace["normalized_raw_text"] == result.raw_text
    assert trace["delivery_hints"] == hints
    assert any(part["kind"] == "delivery" and part["utterance_id"] is None
               for part in trace["segments"])


def test_whole_dog_action_has_no_speech_while_actual_bark_is_preserved(plan):
    raw = "dog: （しっぽを振る）\ndog: ワン！"
    llm = FakeLLM([classification(annotation("s1-u1-p1", "stage_direction",
                                           narration="ポチはしっぽを振った。"))])
    result, hints = classify(llm, plan, raw)
    parsed = parse_scene_text(result.raw_text, "s1", {"dog"})
    assert [(item.speaker_id, item.spoken_text) for item in parsed if item.speaker_id] == [
        ("dog", "ワン！"),
    ]
    assert hints == {}


@pytest.mark.parametrize("invalid", [
    [annotation("unknown", "stage_direction", narration="陽介はうなずいた。")],
    [annotation("s1-u1-p1"), annotation("s1-u1-p1")],
    [annotation("s1-u1-p1", "stage_direction")],
    [annotation("s1-u1-p1", "stage_direction", narration="陽介はうなずいた。", delivery="静かに")],
    [annotation("s1-u1-p1", "spoken", narration="書き換える")],
    [annotation("s1-u1-p1", "delivery")],
])
def test_invalid_classification_retries_only_the_annotations(plan, invalid):
    raw = "human: （うなずく）はい。"
    valid = annotation("s1-u1-p1", "stage_direction", narration="陽介はうなずいた。")
    llm = FakeLLM([classification(*invalid), classification(valid)])
    result, _ = classify(llm, plan, raw)
    assert result.raw_text == "NARRATOR: 陽介はうなずいた。\nhuman: はい。"
    assert [call[0][0] for call in llm.calls] == ["speech_separation", "speech_separation"]
    assert raw in llm.calls[1][0][1][1]["content"]
    assert "元本文を変更せず分類だけを再試行" in llm.calls[1][0][1][1]["content"]


def test_whole_line_delivery_retries_as_natural_narration(plan):
    raw = "dog: （甘えるように）"
    llm = FakeLLM([
        classification(annotation("s1-u1-p1", "delivery", delivery="甘えるように")),
        classification(annotation("s1-u1-p1", "stage_direction", narration="ポチは甘えるしぐさをした。")),
    ])
    result, hints = classify(llm, plan, raw)
    assert result.raw_text == "NARRATOR: ポチは甘えるしぐさをした。"
    assert hints == {}
    assert len(llm.calls) == 2
    assert "entire speech line" in llm.calls[1][0][1][1]["content"]


def test_repeated_invalid_ids_fail_at_classification_checkpoint_without_rewriting(plan):
    invalid = classification(annotation("not-a-source-candidate"))
    llm = FakeLLM([invalid, invalid])
    llm.requests = 7
    with pytest.raises(StructuredGenerationError, match="cover every candidate"):
        classify(llm, plan, "human: （うなずく）はい。")
    assert llm.failure_request == 8
    assert [call[0][0] for call in llm.calls] == ["speech_separation", "speech_separation"]
    assert sum(item["type"] == "speech_separation_rejected" for item in llm.trace) == 2


def test_invalid_structured_output_fails_without_a_writer_retry(plan):
    llm = FakeLLM([{"content": "broken"}, {"content": "broken"}])
    with pytest.raises(StructuredGenerationError):
        classify(llm, plan, "human: （うなずく）はい。")
    assert llm.failure_request == 1
    assert [call[0][0] for call in llm.calls] == ["speech_separation", "speech_separation"]


def test_more_than_ten_candidates_are_classified_in_bounded_batches(plan):
    raw = "\n".join(f"human: 用語（補足{index}）です。" for index in range(11))
    annotations = [annotation(f"s1-u{index}-p1") for index in range(1, 12)]
    llm = FakeLLM([classification(*annotations[:10]), classification(*annotations[10:])])
    result, hints = classify(llm, plan, raw)
    assert result.raw_text == raw
    assert hints == {}
    assert len(llm.calls) == 2
    schemas = [call[1]["response_format"]["json_schema"]["schema"] for call in llm.calls]
    assert [len(schema["$defs"]["SpeechMeaning"]["properties"]["candidate_id"]["enum"])
            for schema in schemas] == [10, 1]


@pytest.mark.parametrize("invalid_classification", [False, True])
def test_full_generation_normalizes_before_staging_and_retains_delivery_metadata(
    plan, invalid_classification,
):
    snapshot = {"world": {"result": {"chapterCount": 1}}, "characters": [
        {"result": {"id": "human", "name": "陽介"}},
        {"result": {"id": "dog", "name": "ポチ", "voice": "人語を話せない犬"}},
    ]}
    raw = "human: （穏やかに）おかえり。（椅子を引く）\ndog: （しっぽを振る）\nhuman: ここにおいで。\ndog: ワン！"
    outline = {"ending": "家族で落ち着く", "character_arcs": [
        {"character_id": cid, "change": "安心する"} for cid in ("human", "dog")],
        "chapters": [{"number": 1, "title": "おかえり", "role": "導入", "summary": "犬を迎える"}],
        "foreshadowing": []}
    location = {"id": "home", "name": "家", "description": "食卓のある居間", "time_of_day": "夕方",
                "atmosphere": "穏やか", "image_prompt": "cozy dining room, no people"}
    responses = [
        response(outline), response({"characters": []}),
        response({"locations": [location], "scenes": [plan.model_dump()]}),
        {"content": raw, "_finish_reason": "stop"},
        classification(
            annotation("s1-u1-p1", "delivery", delivery="穏やかに"),
            annotation("s1-u1-p2", "stage_direction", narration="陽介は椅子を引いた。"),
            annotation("s1-u2-p1", "stage_direction", narration="ポチはしっぽを振った。"),
        ),
        response({"emotions": [
            {"utterance_id": f"s1-u{index}", "inner_emotion": "安心", "voice_emotion": "neutral",
             "delivery": ""} for index in range(1, 6)], "directions": []}),
        response({"passed": True, "issues": [], "events": {"welcome": {
            "dramatized": True, "dialogue_utterance_ids": ["s1-u1", "s1-u4", "s1-u5"],
            "action_utterance_ids": ["s1-u2", "s1-u3"], "reason": "迎える言葉と応える犬の動作がある。",
        }}}),
    ]
    if invalid_classification:
        invalid = classification(annotation("unknown"))
        llm = FakeLLM([*responses[:4], invalid, invalid])
        with pytest.raises(StructuredGenerationError, match="cover every candidate"):
            generate_narrative({"approval_snapshot": snapshot}, llm)
        assert [call[0][0] for call in llm.calls] == [
            "story_outline", "supporting_character", "scene_plan", "scene-text-s1",
            "speech_separation", "speech_separation",
        ]
        assert llm.failure_request == 5
        return
    llm = FakeLLM(responses)
    result = generate_narrative({"approval_snapshot": snapshot}, llm)
    validated = validate_narrative(result, snapshot)
    utterances = validated.scenes[0].utterances
    assert [item.spoken_text for item in utterances if item.speaker_id] == [
        "おかえり。", "ここにおいで。", "ワン！",
    ]
    assert utterances[0].delivery == "穏やかに"
    assert [call[0][0] for call in llm.calls] == [
        "story_outline", "supporting_character", "scene_plan", "scene-text-s1",
        "speech_separation", "staging", "quality_review",
    ]
    staging_prompt = llm.calls[5][0][1][1]["content"]
    staged = json.loads(staging_prompt.split("\n確定した本文とID: ", 1)[1].split("\n本文を", 1)[0])
    assert [item["spoken_text"] for item in staged if item["speaker_id"]] == [
        "おかえり。", "ここにおいで。", "ワン！",
    ]
    assert '"s1-u1": "穏やかに"' in staging_prompt
    assert next(item for item in llm.trace if item["type"] == "scene_text")["raw_text"] == raw
