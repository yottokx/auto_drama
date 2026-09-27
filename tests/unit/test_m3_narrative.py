from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from contextlib import nullcontext

import httpx
import pytest

from packages.contracts.m3 import EMOTION_TAGS, ScenePlan
from packages.narrative import parse_scene_text, script_character_id, validate_narrative
from services.worker.client import WorkerClient
from services.worker.generation import m3_pipeline
from services.worker.generation.llm import LocalLLM
from services.worker.generation.narrative import (
    SceneContentError,
    StructuredGenerationError,
    _legacy_review_prompt,
    _messages,
    _outline,
    _review_scene,
    _review_schema,
    _scene_text,
    _staging,
    generate_narrative,
)


def narrative_fixture():
    snapshot = {"world": {"result": {"chapterCount": 3}},
                "characters": [{"result": {"id": "Hero"}}, {"result": {"id": "keeper"}}]}
    raw = ("Hero: 扉を開けてくれ。\r\nNARRATOR: 門番が鍵を握りしめ、首を横に振った。\r\n"
           "keeper: 約束を破った人は通せない。\r\nHero: 確かに遅れた。この薬だけでも届けてほしい。\r\n"
           "NARRATOR: 差し出された薬袋を見て、門番の指が緩んだ。\r\n"
           "keeper: わかった。一緒に届けよう。\r\n")
    utterances = [item.model_dump() for item in parse_scene_text(raw, "s1", {"Hero", "keeper"})]
    plan = {"id": "s1", "location_id": "gate", "character_ids": ["Hero", "keeper"],
            "objectives": "門番を説得する", "start_state": "扉は閉まっている",
            "required_events": [{"id": "persuasion", "description": "反発を受け、行動で信頼を取り戻す"}],
            "end_state": "門番が協力を決める", "atmosphere": "緊張から安堵へ"}
    result = {"schema_version": 1, "chapter_number": 1, "title": "閉じた扉",
        "outline": {"ending": "薬が届く", "character_arcs": [
            {"character_id": cid, "change": "信頼を得る"} for cid in ("Hero", "keeper")],
            "chapters": [{"number": n, "title": f"第{n}章", "role": "変化", "summary": "薬を届ける"}
                         for n in range(1, 4)], "foreshadowing": [
                             {"setup_chapter": 1, "payoff_chapter": 3, "detail": "薬袋の刻印"}]},
        "supporting_characters": [], "locations": [{"id": "gate", "name": "城門",
            "description": "石の門", "time_of_day": "夕方", "atmosphere": "緊張",
            "image_prompt": "stone gate at dusk, no people"}],
        "scenes": [{"id": "s1", "plan": plan, "raw_text": raw, "utterances": utterances,
            "directions": [{"id": f"s1-d{i}", "utterance_id": "s1-u1", "kind": "enter",
                            "timing": "before", "character_id": cid, "position": position,
                            "duration_ms": 0}
                           for i, cid, position in ((1, "Hero", "left"), (2, "keeper", "right"))],
            "review": {"passed": True, "issues": [], "events": [{"event_id": "persuasion",
                "dramatized": True, "evidence_utterance_ids": [f"s1-u{i}" for i in range(1, 7)],
                "reason": "拒絶、具体的な証拠、行動と判断の変化が描かれている"}]}}]}
    return result, snapshot


def test_exact_source_mapping_preserves_crlf_unicode_and_spaces():
    raw = "Hero:  本当に？　\r\n\r\nNARRATOR: 彼は頷いた。\r\n"
    parsed = parse_scene_text(raw, "s1", {"Hero"})
    assert parsed[0].display_text == " 本当に？　"
    assert [raw[u.source_start:u.source_end] for u in parsed] == [u.display_text for u in parsed]
    assert parsed[1].speaker_id is None


@pytest.mark.parametrize("raw", ["誰か: 台詞", "Hero: ", "説明だけ。", "other: 台詞", ""])
def test_ambiguous_or_unmapped_source_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_scene_text(raw, "s1", {"Hero"})


def test_narrative_preserves_approved_outline_count_and_event_evidence():
    result, snapshot = narrative_fixture()
    assert validate_narrative(result, snapshot).scenes[0].raw_text == result["scenes"][0]["raw_text"]
    assert script_character_id("Hero") == script_character_id("Hero")
    assert script_character_id("Hero") != script_character_id("hero")


@pytest.mark.parametrize("mutation", [
    lambda r: r["outline"]["chapters"].pop(),
    lambda r: r["scenes"][0]["utterances"].pop(),
    lambda r: r["scenes"][0]["utterances"][0].update(display_text="違う台詞"),
    lambda r: r["scenes"][0]["utterances"][0].update(spoken_text="読み替えた台詞"),
    lambda r: r["scenes"][0]["utterances"][0].update(speaker_id="keeper"),
    lambda r: r["scenes"][0]["utterances"][0].update(source_start=0),
    lambda r: r["scenes"][0]["review"]["events"][0].update(dramatized=False),
    lambda r: r["scenes"][0]["review"]["events"][0].update(evidence_utterance_ids=["s1-u1", "s1-u3"]),
    lambda r: r["scenes"][0]["review"]["events"][0].update(evidence_utterance_ids=["s1-u2", "s1-u5"]),
    lambda r: r["scenes"][0]["review"]["events"][0].update(evidence_utterance_ids=["s1-u1", "s2-u1"]),
    lambda r: r["scenes"][0]["directions"][0].update(kind="javascript"),
    lambda r: r["scenes"][0]["directions"][0].update(character_id="intruder"),
    lambda r: r["scenes"][0]["directions"].pop(),
    lambda r: r.update(title="長" * 201),
    lambda r: r["scenes"][0]["utterances"][0].update(delivery="長" * 201),
])
def test_inconsistent_or_digest_only_results_cannot_be_adopted(mutation):
    result, snapshot = narrative_fixture()
    mutation(result)
    with pytest.raises(ValueError):
        validate_narrative(result, snapshot)


class FakeLLM:
    def __init__(self, responses):
        self.config = {"llm": {"model_id": "fake", "temperature": 0.5, "max_tokens": 3072}}
        self.payload, self.trace, self.calls, self.responses = {}, [], [], iter(responses)
        self.requests = 0

    def chat(self, *args, **kwargs):
        self.requests += 1
        self.calls.append((args, kwargs))
        self.trace.append({"type": "llm_generation", "request": self.requests, "cache_hit": False,
                           "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
        return next(self.responses)


def test_truncated_body_is_continued_only_at_exact_source_anchor():
    result, _ = narrative_fixture()
    plan = ScenePlan.model_validate(result["scenes"][0]["plan"])
    prefix = "Hero: その薬を"
    llm = FakeLLM([{"content": prefix, "_finish_reason": "length"},
                   {"content": prefix + "届けたい。\nNARRATOR: 袋を差し出した。", "_finish_reason": "stop"}])
    raw, mapped = _scene_text(llm, "write", plan)
    assert raw == prefix + "届けたい。\nNARRATOR: 袋を差し出した。"
    assert len(mapped) == 2
    assert len([t for t in llm.trace if t["type"] == "scene_text"]) == 2


def test_continuation_with_wrong_connection_is_rejected():
    result, _ = narrative_fixture()
    plan = ScenePlan.model_validate(result["scenes"][0]["plan"])
    llm = FakeLLM([{"content": "Hero: その薬", "_finish_reason": "length"},
                   {"content": "Hero: 違う話。", "_finish_reason": "stop"}])
    with pytest.raises(ValueError, match="anchor"):
        _scene_text(llm, "write", plan)


def test_staging_retries_mapping_without_rewriting_source():
    result, _ = narrative_fixture()
    scene = result["scenes"][0]
    utterances = parse_scene_text(scene["raw_text"], "s1", {"Hero", "keeper"})
    emotions = [{"utterance_id": u.id, "inner_emotion": "緊張", "voice_emotion": "neutral",
                 "delivery": ""} for u in utterances]
    llm = FakeLLM([{"content": json.dumps({"emotions": emotions[:-1], "directions": []})},
                   {"content": json.dumps({"emotions": emotions, "directions": []})}])
    staging = _staging(llm, "context", ScenePlan.model_validate(scene["plan"]), utterances)
    assert len(staging.emotions) == 6
    assert len(llm.calls) == 2
    assert all(call[0][0] == "staging" for call in llm.calls)
    assert len([entry for entry in llm.trace if entry["type"] == "staging_rejected"]) == 1
    schema = llm.calls[0][1]["response_format"]["json_schema"]["schema"]
    assert schema["$defs"]["FocusAnnotation"]["properties"]["character_id"]["enum"] == ["Hero", "keeper"]
    assert "s1-u2" not in schema["$defs"]["FocusAnnotation"]["properties"]["utterance_id"]["enum"]


def evidence_review(scene):
    utterances = {u["id"]: u for u in scene["utterances"]}
    review = scene["review"]
    return {"passed": review["passed"], "issues": review["issues"], "events": {
        event["event_id"]: {"dramatized": event["dramatized"], "reason": event["reason"],
            "dialogue_utterance_ids": [uid for uid in event["evidence_utterance_ids"]
                                       if utterances[uid]["speaker_id"] is not None],
            "action_utterance_ids": [uid for uid in event["evidence_utterance_ids"]
                                     if utterances[uid]["speaker_id"] is None]}
        for event in review["events"]}}


def narrative_responses(result, *, rejected_review=False):
    scene = result["scenes"][0]
    def response(value):
        return {"content": json.dumps(value, ensure_ascii=False)}
    emotions = [{"utterance_id": u["id"], "inner_emotion": "緊張", "voice_emotion": "neutral",
                 "delivery": ""} for u in scene["utterances"]]
    staging = response({"emotions": emotions, "directions": []})
    responses = [response(result["outline"]), response({"characters": []}),
                 response({"locations": result["locations"], "scenes": [scene["plan"]]}),
                 {"content": scene["raw_text"], "_finish_reason": "stop"}, staging]
    if rejected_review:
        review = evidence_review(scene)
        review.update(passed=False, issues=["具体的な反発が不足しています。"])
        responses.extend([response(review), {"content": scene["raw_text"], "_finish_reason": "stop"}, staging])
    responses.append(response(evidence_review(scene)))
    return responses


def test_complete_narrative_pipeline_uses_plaintext_and_lightweight_support_path():
    result, snapshot = narrative_fixture()
    llm = FakeLLM(narrative_responses(result))
    generated = generate_narrative({"approval_snapshot": snapshot}, llm)
    assert generated["scenes"][0]["raw_text"] == result["scenes"][0]["raw_text"]
    assert [c[0][0] for c in llm.calls] == ["story_outline", "supporting_character", "scene_plan",
                                         "scene-text-s1", "staging", "quality_review"]
    assert "response_format" not in llm.calls[3][1]
    assert len(generated["outline"]["chapters"]) == 3


@pytest.mark.parametrize("rejected_review", [False, True])
def test_audition_lines_do_not_become_story_inputs_or_review_requirements(rejected_review):
    from tests.integration.test_m2 import CHARACTER

    result, snapshot = narrative_fixture()
    hero = {**copy.deepcopy(CHARACTER), "id": "Hero", "name": "旅人",
            "settings": "門の外にいる薬売り。秘密はまだ知らない。",
            "voice": "一人称は僕。相手の話を聞きながら慎重に言葉を選ぶ。",
            "selfIntroduction": "試聴専用の自己紹介であり物語の本文ではない。",
            "sampleLines": ["例文だけの遠い未来の宣言。", "例文だけの勝利の言葉。", "例文だけの誓い。"]}
    keeper = {**copy.deepcopy(CHARACTER), "id": "keeper", "name": "門番",
              "voice": "一人称は私。簡潔で落ち着いた口調。",
              "selfIntroduction": "門番の試聴専用の自己紹介。",
              "sampleLines": ["門番の例文一。", "門番の例文二。", "門番の例文三。"]}
    snapshot["characters"] = [{"result": hero}]
    snapshot["relationships"] = {"result": {"summary": "門番は薬売りを疑っている。"}}
    original = copy.deepcopy(snapshot)
    responses = narrative_responses(result, rejected_review=rejected_review)
    responses[1] = {"content": json.dumps({"characters": [keeper]}, ensure_ascii=False)}
    llm = FakeLLM(responses)
    generated = generate_narrative({"approval_snapshot": snapshot}, llm)

    examples = [value for character in (hero, keeper)
                for value in [character["selfIntroduction"], *character["sampleLines"]]]
    for args, _ in llm.calls:
        messages = args[1]
        sent = json.dumps(messages, ensure_ascii=False)
        assert all(example not in sent for example in examples), args[0]
        assert hero["settings"] in sent and hero["voice"] in sent
        assert "門番は薬売りを疑っている。" in sent
        if args[0] not in {"story_outline", "supporting_character"}:
            assert keeper["voice"] in sent
    assert snapshot == original
    assert generated["supporting_characters"][0]["sampleLines"] == keeper["sampleLines"]
    assert generated["supporting_characters"][0]["selfIntroduction"] == keeper["selfIntroduction"]
    assert generated["scenes"][0]["raw_text"] == result["scenes"][0]["raw_text"]


@pytest.mark.parametrize("count", [9, 17, 100])
def test_large_outlines_are_batched_and_preserve_all_requested_chapters(count):
    result, _ = narrative_fixture()
    design = {key: value for key, value in result["outline"].items() if key != "chapters"}
    chapters = [{"number": n, "title": f"第{n}章", "role": "進展", "summary": "具体的な進展。"}
                for n in range(1, count + 1)]
    responses = [{"content": json.dumps(design)}] + [
        {"content": json.dumps({"chapters": chapters[start:start + 8]})}
        for start in range(0, count, 8)
    ]
    llm = FakeLLM(responses)
    outline = _outline(llm, "世界・人物の承認済み設定", count)
    assert [chapter.number for chapter in outline.chapters] == list(range(1, count + 1))
    assert len(llm.calls) == 1 + (count + 7) // 8
    assert llm.calls[0][0][0] == "story_design"


def test_content_repair_receives_specific_review_issues_and_preserves_revisions():
    result, snapshot = narrative_fixture()
    llm = FakeLLM(narrative_responses(result, rejected_review=True))
    generate_narrative({"approval_snapshot": snapshot}, llm)
    text_calls = [c for c in llm.calls if c[0][0] == "scene-text-s1"]
    assert len(text_calls) == 2
    assert "具体的な反発が不足" in text_calls[1][0][1][1]["content"]
    assert len([event for event in llm.trace if event["type"] == "scene_text"]) == 2


def test_repeated_review_format_failure_does_not_rewrite_scene_body():
    result, snapshot = narrative_fixture()
    responses = narrative_responses(result)[:-1] + [{"content": "broken"}, {"content": "broken"}]
    llm = FakeLLM(responses)
    with pytest.raises(StructuredGenerationError):
        generate_narrative({"approval_snapshot": snapshot}, llm)
    assert len([c for c in llm.calls if c[0][0] == "scene-text-s1"]) == 1
    assert llm.failure_request == 6


@pytest.mark.parametrize("mutation", [
    lambda v: v["events"]["persuasion"].update(action_utterance_ids=[]),
    lambda v: v["events"]["persuasion"].update(action_utterance_ids=["s1-u1"]),
    lambda v: v["events"]["persuasion"].update(dialogue_utterance_ids=["s1-u2"]),
    lambda v: v["events"]["persuasion"].update(action_utterance_ids=["s2-u1"]),
    lambda v: v["events"]["persuasion"].update(action_utterance_ids=["s1-u2", "s1-u2"]),
    lambda v: v.update(events={}),
    lambda v: v["events"].update(extra=copy.deepcopy(v["events"]["persuasion"])),
])
def test_review_mapping_repair_keeps_original_source_and_schema_categories(mutation):
    result, snapshot = narrative_fixture()
    valid = evidence_review(result["scenes"][0])
    invalid = copy.deepcopy(valid)
    mutation(invalid)
    responses = narrative_responses(result)[:-1] + [
        {"content": json.dumps(invalid)}, {"content": json.dumps(valid)}]
    llm = FakeLLM(responses)
    generated = generate_narrative({"approval_snapshot": snapshot}, llm)
    assert generated["scenes"][0]["raw_text"] == result["scenes"][0]["raw_text"]
    assert len([c for c in llm.calls if c[0][0] == "scene-text-s1"]) == 1
    reviews = [c for c in llm.calls if c[0][0] == "quality_review"]
    assert len(reviews) == 2
    events = reviews[0][1]["response_format"]["json_schema"]["schema"]["properties"]["events"]
    assert events["required"] == ["persuasion"]
    accepted = events["properties"]["persuasion"]["anyOf"][0]
    assert accepted["properties"]["action_utterance_ids"]["items"]["enum"] == ["s1-u2", "s1-u5"]
    assert accepted["properties"]["dialogue_utterance_ids"]["items"]["enum"] == [
        "s1-u1", "s1-u3", "s1-u4", "s1-u6"]
    assert accepted["properties"]["action_utterance_ids"]["minItems"] == 1


def test_repeated_review_mapping_failure_preserves_text_and_review_checkpoint():
    result, snapshot = narrative_fixture()
    invalid = evidence_review(result["scenes"][0])
    invalid["events"]["persuasion"]["action_utterance_ids"] = []
    llm = FakeLLM(narrative_responses(result)[:-1] + [{"content": json.dumps(invalid)}] * 2)
    with pytest.raises(StructuredGenerationError, match="action_utterance_ids"):
        generate_narrative({"approval_snapshot": snapshot}, llm)
    assert len([c for c in llm.calls if c[0][0] == "scene-text-s1"]) == 1
    assert llm.failure_request == 6


def test_absent_visible_action_is_reported_as_content_failure_without_inventing_evidence():
    result, _ = narrative_fixture()
    scene = result["scenes"][0]
    plan = ScenePlan.model_validate(scene["plan"])
    utterances = parse_scene_text("Hero: 説得して和解した。", "s1", {"Hero"})
    review = {"passed": False, "issues": ["結果だけで行動と反応が描かれていない。"],
        "events": {"persuasion": {"dramatized": False, "dialogue_utterance_ids": ["s1-u1"],
                    "action_utterance_ids": [], "reason": "目に見える行動が存在しない。"}}}
    llm = FakeLLM([{"content": json.dumps(review)}])
    with pytest.raises(SceneContentError, match="行動"):
        _review_scene(llm, "承認資料", plan, utterances)
    variants = _review_schema(plan, utterances)["properties"]["events"]["properties"]["persuasion"]["anyOf"]
    assert len(variants) == 1
    assert variants[0]["properties"]["dramatized"]["const"] is False
    assert variants[0]["properties"]["action_utterance_ids"]["maxItems"] == 0


def test_existing_valid_review_cache_is_replayed_without_a_new_request(tmp_path):
    from packages.contracts.m3 import SceneReview
    result, _ = narrative_fixture()
    scene = result["scenes"][0]
    plan = ScenePlan.model_validate(scene["plan"])
    utterances = parse_scene_text(scene["raw_text"], "s1", {"Hero", "keeper"})
    config, payload = m3_pipeline.pipeline.load_config(), {"seed": 19}
    first = LocalLLM(m3_pipeline.pipeline.ROOT, config, payload, tmp_path)
    first.request = lambda *_: {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(scene["review"], ensure_ascii=False)}}]}
    first.chat("quality_review", _messages(_legacy_review_prompt("context", plan, utterances)),
        response_format={"type": "json_schema", "json_schema": {"name": "quality_review",
            "strict": True, "schema": SceneReview.model_json_schema()}})
    saved = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    resumed = LocalLLM(m3_pipeline.pipeline.ROOT, config, payload, tmp_path)
    resumed.request = lambda *_: pytest.fail("Accepted saved review must not call the model")
    actual = _review_scene(resumed, "context", plan, utterances)
    assert actual.model_dump() == scene["review"]
    assert resumed.requests == 1
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.json")} == saved


def test_cached_action_omission_repairs_review_only_and_keeps_legacy_record(tmp_path):
    from packages.contracts.m3 import SceneReview
    result, _ = narrative_fixture()
    scene = result["scenes"][0]
    plan = ScenePlan.model_validate(scene["plan"])
    utterances = parse_scene_text(scene["raw_text"], "s1", {"Hero", "keeper"})
    invalid = copy.deepcopy(scene["review"])
    invalid["events"][0].update(evidence_utterance_ids=["s1-u1", "s1-u3", "s1-u4"],
                               reason="行動はs1-u2とs1-u5に具体的に描かれている。")
    config, payload = m3_pipeline.pipeline.load_config(), {"seed": 20}
    first = LocalLLM(m3_pipeline.pipeline.ROOT, config, payload, tmp_path)
    first.request = lambda *_: {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(invalid, ensure_ascii=False)}}]}
    first.chat("quality_review", _messages(_legacy_review_prompt("context", plan, utterances)),
        response_format={"type": "json_schema", "json_schema": {"name": "quality_review",
            "strict": True, "schema": SceneReview.model_json_schema()}})
    path = next(tmp_path.glob("*.json"))
    saved = path.read_bytes()
    calls = []
    resumed = LocalLLM(m3_pipeline.pipeline.ROOT, config, payload, tmp_path)
    def repair(_, request):
        calls.append(request)
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(evidence_review(scene), ensure_ascii=False)}}]}
    resumed.request = repair
    actual = _review_scene(resumed, "context", plan, utterances)
    assert "s1-u2" in actual.events[0].evidence_utterance_ids
    assert len(calls) == 1
    assert resumed.requests == 2
    assert path.read_bytes() == saved
    assert "本文は絶対に書き換えません" in calls[0]["messages"][1]["content"]


def test_cached_lookup_miss_does_not_consume_request_or_call_model(tmp_path):
    llm = LocalLLM(m3_pipeline.pipeline.ROOT, m3_pipeline.pipeline.load_config(), {"seed": 2}, tmp_path)
    llm.request = lambda *_: pytest.fail("Cache-only lookup must not perform inference")
    assert llm.cached_chat("quality_review", _messages("unknown request")) is None
    assert llm.requests == 0


def test_emotion_is_applied_only_to_tts_request(monkeypatch, tmp_path):
    seen = []
    def media(job, work):
        seen.append(job)
        return m3_pipeline.pipeline._bundle({"schema_version": 1, "kind": "m2_voice_clone",
            "result": {}, "provenance": {"voice": {"text":
                EMOTION_TAGS[job["payload"]["tts_emotion"]] + job["payload"]["dialogue_text"]}},
            "trace": []}, {"voice.wav": b"wav"})
    monkeypatch.setattr(m3_pipeline.pipeline, "generate_job", media)
    original = {"schema_version": 1, "seed": 2, "voice_emotion": "happy", "dialogue_text": "ありがとう。"}
    bundle = m3_pipeline.generate_job({"kind": "m3_voice_clone", "payload": original}, tmp_path)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        envelope = json.loads(archive.read("result.json"))
    assert original["dialogue_text"] == "ありがとう。"
    assert seen[0]["payload"]["dialogue_text"] == "ありがとう。"
    assert seen[0]["payload"]["tts_emotion"] == "happy"
    assert envelope["provenance"]["voice"]["spoken_text"] == "ありがとう。"
    assert envelope["kind"] == "m3_voice_clone"


def test_m3_job_caches_result_and_pins_generation_config(monkeypatch, tmp_path):
    config = {"gpu_lock_timeout_seconds": 1, "max_zip_bytes": 1024 * 1024}
    seen = []
    monkeypatch.setattr(m3_pipeline.pipeline, "load_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr(m3_pipeline, "gpu_lock", lambda *args: nullcontext())
    monkeypatch.setattr(m3_pipeline, "generate_background",
                        lambda payload, work, settings: (seen.append(settings) or b"png", {}))
    job = {"kind": "m3_background", "payload": {"schema_version": 1, "seed": 9}}
    result = m3_pipeline.generate_job(job, tmp_path)
    config["gpu_lock_timeout_seconds"] = 90
    assert m3_pipeline.generate_job(job, tmp_path) == result
    assert len(seen) == 1
    with pytest.raises(ValueError, match="another input"):
        m3_pipeline.generate_job({**job, "payload": {"schema_version": 1, "seed": 10}}, tmp_path)


def test_failed_annotation_retry_resamples_only_failed_tail_and_keeps_accepted_raw(tmp_path):
    config = m3_pipeline.pipeline.load_config()
    payload = {"seed": 57}
    messages = [{"role": "user", "content": "Write the scene."}]
    def response(text):
        return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
    first = LocalLLM(m3_pipeline.pipeline.ROOT, config, payload, tmp_path)
    first.request = lambda *_: response("Hero: 保存した本文。")
    original = first.chat("scene", messages)
    first.request = lambda *_: response("invalid annotation")
    first.chat("staging", messages)
    first.retry_failed_from(2)
    seeds = []
    second = LocalLLM(m3_pipeline.pipeline.ROOT, config, payload, tmp_path)
    def new_request(path, request):
        seeds.append(request["seed"])
        return response("valid annotation")
    second.request = new_request
    assert second.chat("scene", messages) == original
    assert seeds == []
    assert second.chat("staging", messages)["content"] == "valid annotation"
    assert seeds == [57 + 2 + 104729]
    second.chat("review", messages)
    second.retry_failed_from(3)
    third = LocalLLM(m3_pipeline.pipeline.ROOT, config, payload, tmp_path)
    third.request = lambda *_: pytest.fail("both completed stages must use their accepted cache")
    assert third.chat("scene", messages) == original
    assert third.chat("staging", messages)["content"] == "valid annotation"


def test_emotion_annotation_does_not_reduce_the_1000_character_source_limit(monkeypatch, tmp_path):
    reference = b"reference bytes verified before synthesis"
    (tmp_path / "reference-voice.wav").write_bytes(reference)
    requests = []
    def run(command, *_args, **_kwargs):
        from pathlib import Path
        output = Path(command[command.index("--output-dir") + 1])
        requests.append(json.loads((output / "request.json").read_text(encoding="utf-8")))
        (output / "voice.wav").write_bytes(b"wave")
        (output / "result.json").write_text('{"decoded_and_non_silent":true}', encoding="utf-8")
    monkeypatch.setattr(m3_pipeline.pipeline, "run_process", run)
    text = "あ" * 999 + "。"
    m3_pipeline.pipeline.generate_voice_clone({"seed": 1, "dialogue_text": text,
        "tts_emotion": "happy", "reference_voice": {"text": "元の声", "artifact_id": "reference",
        "sha256": hashlib.sha256(reference).hexdigest()}}, tmp_path, m3_pipeline.pipeline.load_config())
    assert requests[0]["text"] == "😊" + text
    assert len(requests[0]["text"]) == 1001


def test_worker_posts_m3_completion_to_its_own_route(tmp_path):
    requests = []
    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200, json={})
    with httpx.Client(base_url="http://coordinator", transport=httpx.MockTransport(handler)) as client:
        worker = WorkerClient(client, generation_runner=lambda j, p: b"zip",
                              generation_kinds=["m3_narrative"], work_dir=tmp_path)
        worker.worker_id = "worker"
        worker._complete({"kind": "m3_narrative", "id": "job", "lease_id": "lease"}, b"zip")
    assert requests == ["/api/m3/jobs/job/complete"]
