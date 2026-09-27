"""Exercise source retention, staged dialogue, continuation and real Tyrano export."""

import copy
import json
import zipfile
from contextlib import nullcontext
from pathlib import Path

import pytest

from packages.narrative import parse_scene_text, script_character_id
from services.worker.generation import script_continuation as runner
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.llm import write_json
from services.worker.generation.model_routing import RoutedLLM

FIRST = ("aoi: 年は空欄のまま、この写真を出そう。\n"
         "NARRATOR: 葵はゆず色の付箋を裏返し、写真の横に置いた。\n"
         "ren: それなら隣の写真との間を空けよう。\n")
SECOND = ("ren: さっき空けた間に、小さい紙を一枚置いてみた。\n"
          "NARRATOR: 葵は紙を持ち上げ、写真と離して貼り直した。\n"
          "aoi: これで写真と説明が混ざらない。片付けよう。\n")
HANDOFF = "葵は年を空欄にして写真を出すと決めた。蓮は隣の写真との間を空けた。配置全体は未完了。"


def snapshot():
    source = json.loads((Path(__file__).parents[1] / "fixtures" / "story-debug-relationship.json")
                        .read_text(encoding="utf-8"))
    source["world"]["result"]["chapterCount"] = 2
    return source


def route():
    return {"start_condition": "展示を準備する。", "attempt": "札を仮置きする。",
            "consequence": "対応を誤読される。", "choice": "距離を変えて確かめる。",
            "next_state": "配置の判断を共有する。", "core_progress": "相手の判断を待つ。"}


def outline():
    return {"core": {"central_question": "不明な日付をどう示すか。",
            "external_resolution": "展示の準備を終える。", "relationship_resolution": "判断を互いに待つ。",
            "characters": [{"character_id": cid, "initial_behavior": "先に決める。",
                            "enduring_value": "正確に伝える。", "turning_experience": "誤読に気付く。",
                            "final_behavior": "相手の判断を待つ。"} for cid in ("aoi", "ren")],
            "foreshadowing": []},
            "chapters": [{"number": 1, "title": "空欄の選択", "role": "選択", "route": route(), "events": [], "conversation_topics": []},
                         {"number": 2, "title": "写真との間", "role": "決着", "route": route(), "events": [], "conversation_topics": []}]}


def legacy_chapter_plan(*scene_ids):
    return {"current_facts": {"situation": "写真の年は不明。", "decisions": "日付を捏造しない。",
                              "unresolved": "写真と札の対応が見分けにくい。", "attributed_views": []},
        "destination": "札と写真を区別して展示する。",
        "bridge": "仮置きで誤読に気付き、距離を離す。",
        "future_route_changes": [], "unresolved_core_gaps": [],
        "character_actions": [{"character_id": "ren", "action_and_trigger":
            "配置の誤読に気付き、固定する前に葵へ見え方を尋ねて判断を待つ。"}],
        "new_characters": [], "locations": [{"id": "room", "name": "展示室",
        "description": "作業台と展示パネルがある部屋。", "time_of_day": "夕方",
        "atmosphere": "落ち着いた緊張", "image_prompt": "community hall exhibition room, no people"}],
        "scenes": [{"id": sid, "location_id": "room", "character_ids": ["aoi", "ren"],
            "objectives": "未確定の写真をどう見せるか判断する。", "start_state": "札が白紙のまま。",
            "required_events": [{"id": "choose", "description": "二人が扱いを決める。"}],
            "end_state": "判断に沿って手を動かす。", "atmosphere": "静かな対話"} for sid in scene_ids]}


def chapter_plan(*scene_ids, continued=False):
    old = legacy_chapter_plan(*scene_ids)
    result = {key: old[key] for key in ("new_characters", "locations", "scenes")}
    if continued:
        result["continuation"] = "蓮が写真同士の間へ小さい紙を仮置きし、葵へ読みやすいか尋ねる。"
    return result


def staging(raw, scene_id="s1"):
    utterances = parse_scene_text(raw, scene_id, {"aoi", "ren"})
    return {"emotions": [{"utterance_id": row.id, "inner_emotion": "落ち着き",
                           "voice_emotion": "calm" if row.speaker_id else "neutral", "delivery": ""}
                          for row in utterances],
            "directions": [
                {"utterance_id": utterances[0].id, "kind": "focus", "character_id": utterances[0].speaker_id,
                 "timing": "start", "duration_ms": 0},
                {"utterance_id": utterances[1].id, "kind": "pause", "character_id": "",
                 "timing": "after", "duration_ms": 300},
                {"utterance_id": utterances[-1].id, "kind": "blackout", "character_id": "",
                 "timing": "after", "duration_ms": 200},
            ]}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_text(call):
    return "\n".join(row["content"] for row in call["messages"])


@pytest.fixture
def runtime(monkeypatch):
    control = {"responses": [outline(), chapter_plan("s1"), FIRST, staging(FIRST),
                              HANDOFF, chapter_plan("s1", continued=True), SECOND, staging(SECOND)]}
    calls = []

    class FakeLLM(RoutedLLM):
        def select_purpose(self, purpose):
            super().select_purpose(purpose)
            self.purpose = purpose

        def _ensure_runtime(self):
            pass

        def check_context(self, stage, request):
            budget = self.context_budget(prompt_tokens=500, output_tokens=request["max_tokens"])
            self.trace.append({"type": "context_budget", "stage": stage, **budget})
            return budget

        def chat(self, stage, messages, *, allow_truncated=False, **extra):
            assert allow_truncated, "Source chunks must survive reaching the output limit."
            calls.append({"stage": stage, "purpose": self.purpose, "messages": copy.deepcopy(messages),
                          "profile": copy.deepcopy(self.profile), "extra": copy.deepcopy(extra)})
            assert control["responses"], f"Unexpected additional model call: {stage}"
            value = control["responses"].pop(0)
            self.requests += 1
            self.trace.append({"type": "llm_generation", "stage": stage, "request": self.requests,
                              "cache_hit": False, "elapsed_seconds": 0.01,
                              "usage": {"prompt_tokens": 500, "completion_tokens": 30}})
            if isinstance(value, BaseException):
                raise value
            if isinstance(value, dict) and "_finish_reason" in value:
                return value
            return {"content": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                    "_finish_reason": "stop"}

        def __exit__(self, *args):
            write_json(self.output / "llm-metrics.json", self.trace)
            self._entered = False

    monkeypatch.setattr(runner, "RoutedLLM", FakeLLM)
    monkeypatch.setattr(runner, "gpu_lock", lambda *_args, **_kwargs: nullcontext())
    return calls, control


def test_two_chapters_keep_speakers_staging_and_tyrano_source_without_semantic_reviews(runtime, tmp_path):
    calls, _ = runtime
    report = runner.run_script_debug(snapshot(), tmp_path)
    assert report["status"] == "script_complete"
    assert report["saved_chapter_count"] == report["exported_chapter_count"] == 2
    assert report["semantic_review"] == report["quality_acceptance"] == "not_evaluated"
    assert [call["purpose"] for call in calls] == [
        "script-outline", "script-plan", "script-scene", "script-staging",
        "script-handoff", "script-plan", "script-scene", "script-staging"]
    assert all(call["profile"]["model_id"].startswith("gemma-") for call in calls)
    assert all(call["profile"]["reasoning_level"] == "none" for call in calls)
    assert all(call["profile"]["context_size"] == 16384 for call in calls)
    assert all(FIRST in prompt_text(calls[index]) for index in (4, 5, 6))
    location = chapter_plan("s1")["locations"][0]
    for call in calls:
        if call["purpose"] in {"script-scene", "script-staging"}:
            assert all(location[field] in prompt_text(call)
                       for field in ("name", "description", "time_of_day", "image_prompt"))
    for number, raw in enumerate((FIRST, SECOND), 1):
        exported = tmp_path / "exports" / f"chapter-{number:03d}"
        script = read_json(exported / "script.json")
        parsed = parse_scene_text(raw, "s1", {"aoi", "ren"})
        assert [(row["speaker_id"], row["display_text"], row["spoken_text"]) for row in script["utterances"]] == [
            (script_character_id(row.speaker_id) if row.speaker_id else None,
             row.display_text, row.spoken_text) for row in parsed]
        assert all(row["audio_asset_id"] is None for row in script["utterances"])
        assert {row["kind"] for row in script["directions"]} >= {
            "background", "enter", "exit", "focus", "pause", "blackout"}
        assert (exported / "sources" / "s1.txt").read_text(encoding="utf-8") == raw
        with zipfile.ZipFile(exported / "tyrano-source.zip") as archive:
            scenario = archive.read("data/scenario/first.ks").decode("utf-8")
            assert "[chara_show " in scenario and "[bg " in scenario and "[wait time=\"300\"]" in scenario
            assert "[mask color=" in scenario and "[playse " not in scenario
            assert all(row.display_text in scenario for row in parsed)
        requirements = read_json(exported / "asset-requirements.json")
        assert requirements["placeholder_assets_are_production_assets"] is False


def test_script_runner_works_without_a_qwen_route(runtime, monkeypatch, tmp_path):
    calls, _ = runtime
    config = copy.deepcopy(runner.pipeline.load_config())
    config.pop("model_routing", None)
    monkeypatch.setattr(runner.pipeline, "load_config", lambda: config)
    report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    assert report["status"] == "chapter_limit_reached" and len(calls) == 4
    assert all(call["profile"]["model_id"].startswith("gemma-") for call in calls)


def test_resume_preserves_the_completed_chapter_and_its_export(runtime, tmp_path):
    calls, _ = runtime
    first_report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    assert first_report["exported_chapter_count"] == 1 and len(calls) == 4
    chapter = tmp_path / "chapters" / "chapter-001.md"
    export = tmp_path / "exports" / "chapter-001" / "tyrano-source.zip"
    saved, bundle = chapter.read_bytes(), export.read_bytes()
    modified = chapter.stat().st_mtime_ns, export.stat().st_mtime_ns
    report = runner.run_script_debug(snapshot(), tmp_path, resume=True)
    assert report["status"] == "script_complete" and len(calls) == 8
    assert chapter.read_bytes() == saved and export.read_bytes() == bundle
    assert (chapter.stat().st_mtime_ns, export.stat().st_mtime_ns) == modified
    report = runner.run_script_debug(snapshot(), tmp_path, resume=True)
    assert report["status"] == "script_complete" and len(calls) == 8


def test_resume_reuses_a_committed_scene_after_the_next_scene_is_cancelled(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [outline(), chapter_plan("s1", "s2"), FIRST, staging(FIRST),
                              GenerationCancelled("cancel scene 2")]
    with pytest.raises(GenerationCancelled):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    first_source = tmp_path / "sources" / "c001-s1.raw.txt"
    saved, modified = first_source.read_bytes(), first_source.stat().st_mtime_ns
    assert read_json(tmp_path / "report.json")["status"] == "interrupted"
    before = len(calls)
    control["responses"] = [SECOND, staging(SECOND, "s2")]
    report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert report["status"] == "chapter_limit_reached"
    assert [call["purpose"] for call in calls[before:]] == ["script-scene", "script-staging"]
    assert FIRST in prompt_text(calls[before])
    assert first_source.read_bytes() == saved and first_source.stat().st_mtime_ns == modified
    narrative = read_json(tmp_path / "exports" / "chapter-001" / "narrative.json")
    assert [row["id"] for row in narrative["scenes"]] == ["s1", "s2"]


@pytest.mark.parametrize("purpose", ["speech", "staging"])
def test_failed_source_annotations_preserve_raw_script_and_do_not_regenerate_it(runtime, tmp_path, purpose):
    calls, control = runtime
    raw = FIRST.replace("年は", "（小声で）年は") if purpose == "speech" else FIRST
    invalid = {"annotations": []} if purpose == "speech" else {"emotions": [], "directions": []}
    control["responses"] = [outline(), chapter_plan("s1"), raw, invalid, invalid]
    with pytest.raises((RuntimeError, ValueError)):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    report = read_json(tmp_path / "report.json")
    assert report["status"] == "failed" and report["saved_chapter_count"] == 0
    source = tmp_path / "sources" / "c001-s1.raw.txt"
    assert source.read_text(encoding="utf-8") == raw
    assert raw in (tmp_path / "story.md").read_text(encoding="utf-8")
    assert [call["purpose"] for call in calls].count("script-scene") == 1
    assert [call["purpose"] for call in calls].count("script-" + purpose) == 2
    if purpose == "speech":
        location = chapter_plan("s1")["locations"][0]
        speech = next(call for call in calls if call["purpose"] == "script-speech")
        assert location["description"] in prompt_text(speech)
        assert location["image_prompt"] in prompt_text(speech)
    before = len(calls)
    with pytest.raises((RuntimeError, ValueError)):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert len(calls) == before


def test_unknown_speaker_is_rejected_without_losing_the_returned_source(runtime, tmp_path):
    calls, control = runtime
    raw = FIRST.replace("aoi:", "unregistered:", 1)
    control["responses"] = [outline(), chapter_plan("s1"), raw]
    with pytest.raises(ValueError, match="Unknown scene speaker"):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    report = read_json(tmp_path / "report.json")
    assert report["status"] == "failed" and report["exported_chapter_count"] == 0
    assert (tmp_path / "sources" / "c001-s1.raw.txt").read_text(encoding="utf-8") == raw
    assert len(calls) == 3


def test_empty_handoff_retries_once_then_uses_the_saved_script(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [outline(), chapter_plan("s1"), FIRST, staging(FIRST), "", "",
                              chapter_plan("s1", continued=True), SECOND, staging(SECOND)]
    report = runner.run_script_debug(snapshot(), tmp_path)
    assert report["status"] == "script_complete" and report["exported_chapter_count"] == 2
    assert len(calls) == 9 and len(report["fallbacks"]) == 1
    assert [call["purpose"] for call in calls].count("script-handoff") == 2
    assert FIRST in prompt_text(calls[6]) and FIRST in prompt_text(calls[7])


def test_changed_export_prevents_resume_without_new_model_requests(runtime, tmp_path):
    calls, _ = runtime
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    script = tmp_path / "exports" / "chapter-001" / "script.json"
    script.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Tyrano export"):
        runner.run_script_debug(snapshot(), tmp_path, resume=True)
    assert len(calls) == 4


def test_scene_scope_retains_future_actions_and_matches_each_exported_location(runtime, tmp_path):
    calls, control = runtime
    plan = chapter_plan("s1", "s2")
    plan["locations"].append({**plan["locations"][0], "id": "corridor", "name": "廊下",
                              "description": "展示室の外の廊下。", "image_prompt": "empty hallway at night"})
    plan["scenes"][0]["end_state"] = "最後の説明札を固定した。まだ消灯していない。"
    plan["scenes"][1].update(location_id="corridor", objectives="施錠前に鍵を返す。",
                             start_state="展示作業を終えて廊下に出た。", end_state="鍵を返した。")
    control["responses"] = [outline(), plan, FIRST, staging(FIRST), SECOND, staging(SECOND, "s2")]
    report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    assert report["status"] == "chapter_limit_reached"
    writers = [call for call in calls if call["purpose"] == "script-scene"]
    scopes = [json.JSONDecoder().raw_decode(prompt_text(call).split(
        "執筆範囲と後続場面の担当（実績ではない）:\n", 1)[1])[0] for call in writers]
    assert scopes[0]["following_scenes_not_yet_executed"] == [{k: v for k, v in scene.items() if k not in {"objectives", "atmosphere"}} for scene in plan["scenes"][1:]]
    assert scopes[1]["following_scenes_not_yet_executed"] == []
    assert all(set(scope) == {"following_scenes_not_yet_executed"} for scope in scopes)
    for call, scene, location in zip(writers, plan["scenes"], plan["locations"], strict=True):
        text = prompt_text(call)
        assert text.count(location["description"]) == 1
        assert text.count(scene["end_state"]) == 1
        other = next(row for row in plan["locations"] if row["id"] != location["id"])
        assert other["description"] not in text
    assert FIRST in prompt_text(writers[1])
    script = read_json(tmp_path / "exports/chapter-001/script.json")
    backgrounds = [row for row in script["directions"] if row["kind"] == "background"]
    assert [row["utterance_id"] for row in backgrounds] == ["s1-u1", "s2-u1"]
    assert len({row["asset_id"] for row in backgrounds}) == 2
    assert len(calls) == 6  # Scope material does not introduce planning/review calls.


def test_handoff_is_source_only_and_does_not_become_instruction_when_full_script_fits(runtime, tmp_path):
    calls, control = runtime
    memo = ('【人物の選択】aoi: 年は空欄のまま、この写真を出そう。\n'
            '【変わっていないこと・未決事項】不明な年を断定しない。\n'
            '【未決事項】全体の配置は未完了。')
    control["responses"][4] = memo
    runner.run_script_debug(snapshot(), tmp_path)
    handoff = next(call for call in calls if call["purpose"] == "script-handoff")
    assert FIRST in prompt_text(handoff)
    assert "前から認めていたことと新しい判断を区別" in prompt_text(handoff)
    assert outline()["core"]["external_resolution"] not in prompt_text(handoff)
    assert "全体構成（予定" not in prompt_text(handoff)
    assert "今回の作業依頼" not in prompt_text(handoff)
    assert "【次章の予定】" not in prompt_text(handoff)
    assert (tmp_path / "notes/chapter-001.md").read_text(encoding="utf-8") == memo
    for call in calls[5:7]:
        assert memo not in prompt_text(call) and FIRST in prompt_text(call)
    for key in ("plan-002", "c002-s1-text"):
        selection = read_json(tmp_path / "requests" / f"{key}-1.json")["selection"]
        assert selection["notes"] == [{"number": 1, "included": False}]
        assert selection["chapters"][0]["excluded_ranges"] == []
    assert len(calls) == 8


def test_direct_scene_plan_survives_resume_without_duplicate_volume_guidance(runtime, tmp_path):
    calls, control = runtime
    plan = chapter_plan("s1", "s2")
    control["responses"] = [outline(), plan, FIRST, staging(FIRST),
                            GenerationCancelled("cancel scene 2")]
    with pytest.raises(GenerationCancelled):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    saved_plan = (tmp_path / "chapters/chapter-001.plan.json").read_bytes()
    control["responses"] = [SECOND, staging(SECOND, "s2")]
    report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert report["technical_validation"] == "passed"
    assert (tmp_path / "chapters/chapter-001.plan.json").read_bytes() == saved_plan
    for call in calls:
        if call["purpose"] == "script-scene":
            assert plan["scenes"][0]["objectives"] in prompt_text(call)
    assert [call["purpose"] for call in calls].count("script-plan") == 1
    planning = next(call for call in calls if call["purpose"] == "script-plan")
    properties = planning["extra"]["response_format"]["json_schema"]["schema"]["properties"]
    assert set(properties) == {"new_characters", "locations", "scenes"}


def test_invalid_scene_character_is_repaired_without_changing_output_contract(runtime, tmp_path):
    calls, control = runtime
    invalid = chapter_plan("s1")
    invalid["scenes"][0]["character_ids"] = ["unknown"]
    control["responses"] = [outline(), invalid, chapter_plan("s1"), FIRST, staging(FIRST)]
    report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    assert report["technical_validation"] == "passed"
    assert [call["purpose"] for call in calls].count("script-plan") == 2
    assert (tmp_path / "exports/chapter-001/sources/s1.txt").read_text(encoding="utf-8") == FIRST


def test_old_prompt_protocol_cannot_resume_into_changed_generation(runtime, monkeypatch, tmp_path):
    from services.worker.generation import workflow_version

    calls, _ = runtime
    current = copy.deepcopy(workflow_version.SCRIPT_CONTINUATION_PROTOCOL)
    with monkeypatch.context() as patch:
        patch.setattr(workflow_version, "SCRIPT_CONTINUATION_PROTOCOL",
                      {**current, "prompt": 2, "implementation": 2, "contract_revision": 1})
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    saved = (tmp_path / "draft-state.json").read_bytes()
    with pytest.raises(ValueError, match="protocol"):
        runner.run_script_debug(snapshot(), tmp_path, resume=True)
    assert len(calls) == 4 and (tmp_path / "draft-state.json").read_bytes() == saved


def test_actual_history_and_unchanged_future_plot_reach_next_chapter_after_resume(runtime, tmp_path):
    calls, _ = runtime
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    saved = read_json(tmp_path / "draft-state.json")
    runner.run_script_debug(snapshot(), tmp_path, resume=True)
    assert FIRST in prompt_text(calls[5])
    assert route()["attempt"] in prompt_text(calls[5])
    assert read_json(tmp_path / "plot.json") == saved["plot"] == outline()
    assert len(calls) == 8
    later = calls[5]["extra"]["response_format"]["json_schema"]["schema"]
    assert "continuation" in later["required"]
    assert "current_facts" not in later["properties"]


@pytest.mark.parametrize("changed", ["plot", "plan"])
def test_resume_rejects_changed_plot_or_plan(runtime, tmp_path, changed):
    calls, _ = runtime
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    state = read_json(tmp_path / "draft-state.json")
    if changed == "plot":
        state["plot"]["core"]["central_question"] = "CHANGED"
    else:
        state["chapter_plans"]["1"]["plan"]["destination"] = "CHANGED"
    write_json(tmp_path / "draft-state.json", state)
    with pytest.raises(ValueError, match="changed"):
        runner.run_script_debug(snapshot(), tmp_path, resume=True)
    assert len(calls) == 4


def test_nine_chapter_plot_is_batched_without_changing_core(runtime, tmp_path):
    calls, control = runtime
    source = snapshot()
    source["world"]["result"]["chapterCount"] = 9
    planned = [{**outline()["chapters"][0], "number": n} for n in range(1, 10)]
    control["responses"] = [outline()["core"], {"chapters": planned[:8]}, {"chapters": planned[8:]},
                            chapter_plan("s1"), FIRST, staging(FIRST)]
    runner.run_script_debug(source, tmp_path, chapter_limit=1)
    value = read_json(tmp_path / "plot.json")
    assert value["core"] == outline()["core"]
    assert [row["number"] for row in value["chapters"]] == list(range(1, 10))
    assert len(calls) == 6
    from services.worker.generation.script_plot import DetailedPlot
    assert read_json(tmp_path / "outline.json") == DetailedPlot.model_validate(value).as_outline().model_dump(mode="json")


def empty_cast():
    return {"supporting_characters": [], "everyday_context": [], "connections": []}


def three_chapter_snapshot():
    source = snapshot()
    source["world"]["result"]["chapterCount"] = 3
    return source


def story_chain():
    core = {key: value for key, value in outline()["core"].items() if key != "foreshadowing"}
    for character in core["characters"]:
        character.pop("turning_experience")
    return {"core": core, "events": [
        {"start_condition": "作業中。", "steps": [
            {"character_id": "ren", "action": f"EVENT_{n}_ATTEMPT", "result": f"EVENT_{n}_EXPERIENCE"},
            {"character_id": "ren", "action": f"EVENT_{n}_CHOICE", "result": f"EVENT_{n}_RESULT"}]}
        for n in range(1, 5)]}


def story_chain_draft(chain=None):
    saved = copy.deepcopy(story_chain() if chain is None else chain)
    return {"core": saved["core"], "resolution_basis": [],
            "opening_condition": saved["events"][0]["start_condition"],
            "events": [{"steps": event["steps"]} for event in saved["events"]]}


def realization_plan(count):
    legacy = legacy_chapter_plan("s1")
    return {"current_facts": legacy["current_facts"],
            "placements": [{"step_number": n, "handling": "stage", "scene_number": 1} for n in range(1, count + 1)],
            "future_route_changes": [], "unresolved_core_gaps": [], "new_characters": [],
            "locations": legacy["locations"],
            "scenes": [{"id": "s1", "location_id": "room", "character_ids": ["aoi", "ren"],
                        "staging": "作業台で行動と反応を見せる。", "atmosphere": "静か", "follow_through": [],
                        "interaction": "互いの仕事の癖を話してから、提案に応じる。",
                        "interaction_end": "", "length_weight": 1}]}


def allocation():
    return {"chapters": [{"number": n, "title": f"章{n}", "role": "選択の結果を実行する。",
                          "last_event": end} for n, end in enumerate((1, 3, 4), 1)],
            "foreshadowing": []}


def test_three_events_have_code_assigned_boundaries_and_resume(runtime, tmp_path):
    calls, control = runtime
    chain = story_chain()
    chain["events"] = chain["events"][:3]
    presentation = {f"chapter_{n}": {"title": f"章{n}", "role": "結果を受けて進む。",
                                    "conversation_topics": [{"character_ids": ["aoi", "ren"], "topic": f"会話{n}", "exchange": "軽口を返す"}]} for n in (1, 2, 3)}
    presentation["foreshadowing"] = []
    control["responses"] = [empty_cast(), story_chain_draft(chain), presentation, chapter_plan("s1"), FIRST, staging(FIRST)]
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1)
    assert report["exported_chapter_count"] == 1
    saved = read_json(tmp_path / "chapter-allocation.json")
    assert [row["last_event"] for row in saved["chapters"]] == [1, 2, 3]
    plot = read_json(tmp_path / "plot.json")
    for index, chapter in enumerate(plot["chapters"]):
        assert chapter["events"][0]["steps"] == chain["events"][index]["steps"]
        expected_start = (chain["events"][index - 1]["steps"][-1]["result"]
                          if index else chain["events"][0]["start_condition"])
        assert chapter["events"][0]["start_condition"] == expected_start
        assert chapter["conversation_topics"][0]["topic"] == f"会話{index + 1}"
    schema = calls[2]["extra"]["response_format"]["json_schema"]["schema"]
    assert "last_event" not in json.dumps(schema)
    assert set(schema["properties"]) == {"chapter_1", "chapter_2", "chapter_3", "foreshadowing"}
    assert '"event_number": 1' in prompt_text(calls[2])
    count = len(calls)
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert len(calls) == count


def test_fixed_allocation_rejects_model_supplied_boundaries():
    from pydantic import ValidationError

    from services.worker.generation.script_plot import FixedChapterAllocation

    data = {f"chapter_{n}": {"title": "章", "role": "役割", "conversation_topics": [],
                             "last_event": 3} for n in (1, 2, 3)}
    data["foreshadowing"] = []
    with pytest.raises(ValidationError, match="Extra inputs"):
        FixedChapterAllocation.model_validate(data)


def test_three_chapters_use_same_chain_without_rewriting_or_extra_reviews(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [empty_cast(), story_chain_draft(), allocation(), chapter_plan("s1"), FIRST, staging(FIRST),
                            HANDOFF, chapter_plan("s1", continued=True), SECOND, staging(SECOND),
                            HANDOFF, chapter_plan("s1", continued=True), SECOND, staging(SECOND)]
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path)
    assert report["status"] == "script_complete" and report["exported_chapter_count"] == 3
    assert len(calls) == 14
    assert [call["purpose"] for call in calls].count("script-outline") == 1
    assert [call["purpose"] for call in calls].count("script-allocation") == 1
    chain = read_json(tmp_path / "story-chain.json")
    plot = read_json(tmp_path / "plot.json")
    assert [event for chapter in plot["chapters"] for event in chapter["events"]] == chain["events"]
    assert [len(row["events"]) for row in plot["chapters"]] == [1, 2, 1]
    summary = read_json(tmp_path / "outline.json")["chapters"][1]["summary"]
    assert summary.index("EVENT_2_CHOICE") < summary.index("EVENT_3_ATTEMPT")
    assert all(row["number"] == n for n, row in enumerate(plot["chapters"], 1))
    for ordinal, number in ((3, 1), (7, 2), (11, 3)):
        planning = prompt_text(calls[ordinal])
        for event in plot["chapters"][number - 1]["events"]:
            assert event["steps"][1]["action"] in planning
            assert event["steps"][0]["result"] in planning
    assert "EVENT_1_CHOICE" not in prompt_text(calls[11])  # Completed plans are not source facts.
    assert FIRST in prompt_text(calls[7]) and FIRST in prompt_text(calls[11])
    assert SECOND in prompt_text(calls[11])
    for ordinal in (3, 7, 11):
        schema = calls[ordinal]["extra"]["response_format"]["json_schema"]["schema"]
        assert "placements" not in json.dumps(schema)
    saved_plan = read_json(tmp_path / "chapters/chapter-001.plan.json")
    assert saved_plan["scenes"][0]["required_events"] == chapter_plan("s1")["scenes"][0]["required_events"]
    assert (tmp_path / "exports/chapter-003/sources/s1.txt").read_text(encoding="utf-8") == SECOND


def test_interrupted_allocation_resumes_saved_chain_without_regeneration(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [empty_cast(), story_chain_draft(), GenerationCancelled("allocation interrupted")]
    with pytest.raises(GenerationCancelled):
        runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1)
    saved = (tmp_path / "story-chain.json").read_bytes()
    control["responses"] = [allocation(), chapter_plan("s1"), FIRST, staging(FIRST)]
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert report["status"] == "chapter_limit_reached"
    assert (tmp_path / "story-chain.json").read_bytes() == saved
    assert len(calls) == 7  # Cast, chain, interrupted/resumed allocation, plan, text, staging.
    assert read_json(tmp_path / "draft-state.json")["request_ordinal"] == 7


@pytest.mark.parametrize("ends", [(1, 1, 4), (2, 1, 4), (1, 2, 3), (1, 3, 5)])
def test_invalid_boundaries_do_not_drop_duplicate_or_reorder_events(ends):
    from services.worker.generation.script_plot import ChapterAllocation, StoryChain, allocate_chain
    data = allocation()
    for row, end in zip(data["chapters"], ends, strict=True):
        row["last_event"] = end
    with pytest.raises(ValueError, match="partition"):
        allocate_chain(StoryChain.model_validate(story_chain()), ChapterAllocation.model_validate(data))


def test_changed_three_chapter_allocation_is_rejected_on_resume(runtime, tmp_path):
    _, control = runtime
    control["responses"] = [empty_cast(), story_chain_draft(), allocation(), chapter_plan("s1"), FIRST, staging(FIRST)]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1)
    state = read_json(tmp_path / "draft-state.json")
    state["chapter_allocation"]["chapters"][0]["last_event"] = 2
    write_json(tmp_path / "draft-state.json", state)
    with pytest.raises(ValueError, match="differs"):
        runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1, resume=True)


def test_adjusted_route_replaces_original_chain_events_in_future_input():
    from services.worker.generation.script_plot import (
        ChapterAllocation,
        StoryChain,
        allocate_chain,
        future_material,
    )
    plot = allocate_chain(StoryChain.model_validate(story_chain()), ChapterAllocation.model_validate(allocation()))
    plan = chapter_plan("s1")
    plan["future_route_changes"] = [{"chapter_number": 3, "reason_from_source": "実台本で進んだ。",
                                     "route": {**route(), "attempt": "ADJUSTED_ONLY"}}]
    material = future_material(plot, "original", 3, [{"number": 2, "plan": plan, "sha256": "changed"}])
    assert material["routes"][0]["route"]["attempt"] == "ADJUSTED_ONLY"
    assert "events" not in material["routes"][0]
    assert "EVENT_4_ATTEMPT" not in json.dumps(material)


@pytest.mark.parametrize("actor,other,experience,action", [
    ("pilot", "guide", "案内人が戻れない状況で自力で経路を確かめる。", "独力で出発する。"),
    ("guard", "envoy", "提示された条件が守る相手を危険にさらすと知る。", "交渉を断って去る。"),
    ("artisan", "owner", "試作品を動かしても懸念した危険が残ると確かめる。", "従来の安全基準を守る。"),
])
def test_turning_synopsis_is_derived_from_events_for_different_landings(actor, other, experience, action):
    from services.worker.generation.script_plot import (
        ChapterAllocation,
        StoryChain,
        allocate_chain,
        check_chain,
        future_material,
    )
    data = story_chain()
    for row, cid in zip(data["core"]["characters"], (actor, other), strict=True):
        row["character_id"] = cid
        row["final_behavior"] = action if cid == actor else "立ち去る相手を見送る。"
    for event in data["events"]:
        event["steps"] = [{"character_id": actor, "action": action, "result": experience}]
    chain = StoryChain.model_validate(data)
    check_chain(chain, {actor, other})
    plot = allocate_chain(chain, ChapterAllocation.model_validate(allocation()))
    assert plot.core.characters[0].turning_experience == "\n".join(
        f"出来事{index}: {actor}: {action} → {experience}" for index in range(1, 5))
    assert plot.core.characters[1].turning_experience == "本人の転機は予定していない。"
    assert plot.core.characters[0].final_behavior == action
    material = future_material(plot, "new-plot", 3, [])
    assert all("turning_experience" not in row for row in material["core"]["characters"])
    assert material["routes"][0]["events"][0]["steps"] == data["events"][-1]["steps"]
    assert experience in plot.as_outline().character_arcs[0].change
    assert "turning_experience" not in StoryChain.model_json_schema()["$defs"]["ChainCharacter"]["properties"]


def test_short_continuation_is_saved_and_reaches_writer_with_original_source(runtime, tmp_path):
    calls, _ = runtime
    runner.run_script_debug(snapshot(), tmp_path)
    plan = read_json(tmp_path / "chapters/chapter-002.plan.json")
    assert plan["continuation"] == chapter_plan("s1", continued=True)["continuation"]
    assert plan["continuation"] in prompt_text(calls[6])
    assert FIRST in prompt_text(calls[6])
    assert "current_facts" not in plan and "step_placements" not in plan


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "unknown_scene", "backwards_scene", "absent_actor"])
def test_realization_rejects_only_broken_mapping(mutation):
    from services.worker.generation.script_plot import StoryChain
    from services.worker.generation.script_realization import (
        ChapterRealizationResponse,
        realize_chapter,
    )
    draft = realization_plan(2)
    if mutation == "missing":
        draft["placements"].pop()
    elif mutation == "duplicate":
        draft["placements"][1]["step_number"] = 1
    elif mutation == "reordered":
        draft["placements"].reverse()
    elif mutation == "unknown_scene":
        draft["placements"][0]["scene_number"] = 2
    elif mutation == "backwards_scene":
        draft["scenes"].append({**draft["scenes"][0], "id": "s2"})
        draft["placements"][0]["scene_number"] = 2
    else:
        draft["scenes"][0]["character_ids"] = ["aoi"]
    with pytest.raises(ValueError):
        realize_chapter(ChapterRealizationResponse.model_validate(draft).as_realization(), StoryChain.model_validate(story_chain()).events[:1])


def test_direct_scene_is_only_immediate_instruction_even_when_plot_differs(runtime, tmp_path):
    calls, control = runtime
    draft = chapter_plan("s1")
    draft["scenes"][0]["required_events"] = [{"id": "adapt", "description": "DIRECT_SCENE_ACTION"}]
    control["responses"] = [empty_cast(), story_chain_draft(), allocation(), draft, FIRST, staging(FIRST)]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1)
    writing = prompt_text(calls[4])
    assert "DIRECT_SCENE_ACTION" in writing
    assert "EVENT_1_ATTEMPT" not in writing and "EVENT_1_CHOICE" not in writing
    plan = read_json(tmp_path / "chapters/chapter-001.plan.json")
    assert plan["scenes"][0]["required_events"] == draft["scenes"][0]["required_events"]
    assert "step_placements" not in plan


def test_completed_chapter_actions_can_lead_to_concrete_follow_through():
    from services.worker.generation.script_plot import StoryChain
    from services.worker.generation.script_realization import (
        ChapterRealizationResponse,
        realize_chapter,
    )
    draft = realization_plan(2)
    for placement in draft["placements"]:
        placement.pop("scene_number")
        placement.update(handling="already_done", reason_from_source="前章本文で実施済み。")
    draft["scenes"][0]["follow_through"] = [{"character_id": "aoi", "action": "結果を届ける。", "result": "受取人へ届く。"}]
    plan = realize_chapter(ChapterRealizationResponse.model_validate(draft).as_realization(), StoryChain.model_validate(story_chain()).events[:1])
    assert len(plan["scenes"][0].required_events) == 1
    assert plan["destination"] == "受取人へ届く。"


def test_interrupted_direct_planning_resumes_without_regenerating_chain(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [empty_cast(), story_chain_draft(), allocation(), chapter_plan("s1"), GenerationCancelled("paused")]
    with pytest.raises(GenerationCancelled):
        runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1)
    plan = read_json(tmp_path / "chapters/chapter-001.plan.json")
    control["responses"] = [FIRST, staging(FIRST)]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert plan == read_json(tmp_path / "chapters/chapter-001.plan.json")
    assert len(calls) == 7
