"""CPU-only contracts for LLM-chosen cast and camera: limits, numbering and provenance."""

import io
import json
from contextlib import contextmanager
from copy import deepcopy

import pytest

from scripts.audio import llm_runtime
from scripts.image_edit import cg_proposals, cg_tab, gui, llm_prompts, llm_runner
from scripts.image_edit.common import write_json
from tests.unit.test_image_edit_gui import app, reference, tkinter_root  # noqa: F401

PROMPT = (
    "Low-angle close-up from just behind the shoulder of <B>, whose back and raised arm fill the "
    "left foreground, dark and out of focus. Beyond the shoulder, <A> stands under a single "
    "station lamp, face lit from above, eyes wide and wet, mouth open in the middle of a shout, "
    "one hand pulled back against a tight grip on the wrist. Empty ticket gates recede into dusk "
    "behind, orange light on wet concrete, long shadows, cold violet sky. The frame is tilted "
    "slightly, the gripped wrist near its center, tension running along the stretched arm."
)
LAYOUT = ("A row of ticket gates stands behind them. <A> faces the camera two steps from the gates "
          "and <B> stands between <A> and the camera, back to it, holding the wrist of <A>.")


def proposal(**updates):
    value = {"moment": "透が美緒の手首を掴んで引き止める瞬間。二人の決裂が形になる。",
             "visual_hook": "肩越しの低い視点で、掴まれた手首と泣き顔だけに光が当たる。",
             "cast": [{"tag": "A", "visibility": "face_front", "face_items": ""},
                      {"tag": "B", "visibility": "back", "face_items": "thin black glasses."}],
             "shot": "close_up", "angle": "over_shoulder", "display_from": 1, "display_to": 3,
             "layout": LAYOUT, "light": "Dusk outdoors, lit by one station lamp overhead.",
             "extras": "",
             "english_prompt": PROMPT}
    value.update(updates)
    return value


def answer(*rows):
    return {"proposals": list(rows or [proposal(), proposal(shot="wide"), proposal(angle="low")])}


def portraits():
    return [
        {"character_id": "mio", "name": "篠原 美緒", "artifact_id": "p-mio", "chapter_numbers": [1, 2],
         "source": {"sha256": "secret-hash"}},
        {"character_id": "toru", "name": "水瀬 透", "artifact_id": "p-toru-1", "chapter_numbers": [1]},
        {"character_id": "toru", "name": "水瀬 透", "artifact_id": "p-toru-2", "chapter_numbers": [2]},
        {"character_id": "absent", "name": "別の章の人物", "artifact_id": "p-x", "chapter_numbers": [3]},
    ]


def scene(lines=3):
    rows = [{"id": f"u{n}", "speaker_id": ("mio", "toru", None, "clerk")[n % 4],
             "display_text": f"発話{n}の本文。", "inner_emotion": "焦り"} for n in range(1, lines + 1)]
    return {"id": "s3", "label": "3. 駅の入口", "preview": "本文", "context": {
        "chapter_number": 2, "project_title": "留守電", "chapter_title": "第2章",
        "plan": {"character_ids": ["mio", "toru", "clerk"], "objectives": "引き止める"},
        "location": {"name": "駅の入口", "time_of_day": "夕暮れ"}, "utterances": rows,
        "cast_appearance": {"toru": "黒髪に細い黒縁の四角いメガネ。紺のパーカー。"},
        "story_context": {"brief": {"genre": "青春"}, "chapter": {"role": "決裂"},
                          "cast": [{"id": "mio", "name": "美緒", "role": "ヒロイン"},
                                   {"id": "clerk", "name": "駅員"}]}}}


def test_candidates_follow_scene_order_and_chapter_portrait_and_list_unreferenced():
    found = cg_proposals.scene_candidates(scene(), portraits())
    assert [(row["tag"], row["character_id"]) for row in found["candidates"]] == [
        ("A", "mio"), ("B", "toru")]
    assert found["candidates"][1]["portrait"]["artifact_id"] == "p-toru-2"
    assert found["candidates"][0]["role"] == "ヒロイン"
    assert [row["lines"] for row in found["candidates"]] == [0, 1]
    assert found["unreferenced"] == ["駅員"]


def test_valid_proposals_are_normalized_and_cast_is_limited_to_three():
    rows = cg_proposals.normalize_proposals(answer(), ["A", "B"])
    assert len(rows) == 3 and rows[0]["english_prompt"] == PROMPT
    four = [{"tag": tag, "visibility": "back", "face_items": ""} for tag in "ABCD"]
    assert rows[0]["cast"][1]["face_items"] == "thin black glasses"
    with pytest.raises(ValueError, match="1〜3人"):
        cg_proposals.normalize_proposals(
            answer(proposal(cast=four), proposal(), proposal()), list("ABCD"))
    assert cg_proposals.schema(list("ABCDE"))["properties"]["proposals"]["items"][
        "properties"]["cast"]["maxItems"] == 3


@pytest.mark.parametrize("updates, message", [
    ({"english_prompt": PROMPT.replace("<B>", "the other one")}, "一致しません"),
    ({"english_prompt": PROMPT + " <C> watches."}, "一致しません"),
    ({"english_prompt": PROMPT.replace("<A>", "Reference image 1")}, "英語の画像指示"),
    ({"english_prompt": "手首を掴む。" + PROMPT}, "英語の画像指示"),
    ({"english_prompt": "<A> and <B> stand."}, "80〜200語"),
    ({"english_prompt": PROMPT + " A red sign reads 'OVERDUE' above."}, "引用符"),
    ({"layout": 'A hologram labelled "OVERDUE" floats between <A> and <B>.'}, "引用符"),
    ({"layout": LAYOUT + " <C> waits by the gates."}, "一致しません"),
    ({"layout": "改札の前に立つ。"}, "英語の画像指示"),
    ({"light": ""}, "英語の画像指示"),
    ({"light": "A neon sign saying 'OPEN' glows above <A>."}, "引用符"),
    ({"display_from": 3, "display_to": 2}, "発話番号"),
    ({"display_to": 4}, "発話番号"),
    ({"cast": [{"tag": "A", "visibility": "face_front", "face_items": ""},
               {"tag": "A", "visibility": "back", "face_items": ""}]}, "重複なく"),
    ({"cast": [{"tag": "Z", "visibility": "back", "face_items": ""}]}, "重複なく"),
    ({"cast": [{"tag": "A", "visibility": "face_front"},
               {"tag": "B", "visibility": "back", "face_items": ""}]}, "重複なく"),
    ({"cast": [{"tag": "A", "visibility": "face_front", "face_items": "細いメガネ"},
               {"tag": "B", "visibility": "back", "face_items": ""}]}, "英語の画像指示"),
    ({"cast": [{"tag": "A", "visibility": "face_front", "face_items": "the glasses of <B>"},
               {"tag": "B", "visibility": "back", "face_items": ""}]}, "face_items"),
    ({"shot": "bird"}, "分類"),
    ({"moment": "English only moment"}, "日本語"),
])
def test_unusable_proposals_are_rejected(updates, message):
    with pytest.raises(ValueError, match=message):
        cg_proposals.normalize_proposals(
            answer(proposal(**updates), proposal(), proposal()), ["A", "B"], utterances=3)


def test_possessive_tags_are_not_mistaken_for_quoted_lettering():
    text = PROMPT + " The hand of <B> grips <A>'s wrist while <A>'s other hand trembles."
    rows = cg_proposals.normalize_proposals(
        answer(proposal(english_prompt=text), proposal(), proposal()), ["A", "B"])
    assert rows[0]["english_prompt"] == text


def test_extras_follow_the_user_choice():
    rows = answer(proposal(extras="改札の奥に通行人の人影"), proposal(), proposal())
    assert cg_proposals.normalize_proposals(rows, ["A", "B"])[0]["extras"]
    with pytest.raises(ValueError, match="立ち絵のない人物"):
        cg_proposals.normalize_proposals(rows, ["A", "B"], allow_extras=False)


def test_compiled_prompt_numbers_cast_order_and_matches_identity_to_visibility():
    row = proposal(cast=[{"tag": "B", "visibility": "back", "face_items": "thin black glasses"},
                         {"tag": "A", "visibility": "face_front", "face_items": "a red hair clip"},
                         {"tag": "C", "visibility": "partial", "face_items": "an eyepatch"}],
                   english_prompt=PROMPT + " The hand of <C> enters from the right edge.")
    prompt = cg_proposals.compile_prompt(row)
    assert "<" not in prompt
    assert prompt.splitlines()[1].startswith("Spatial layout: A row of ticket gates")
    assert prompt.splitlines()[2] == "Lighting: Dusk outdoors, lit by one station lamp overhead."
    assert "The Reference image 2 character faces the camera" in prompt
    assert "shoulder of the Reference image 1 character" in prompt
    assert "Reference image 2 character stands under" in prompt
    assert "The Reference image 1 character is seen from behind" in prompt
    assert "Do not turn the face toward the viewer" in prompt
    assert "Preserve the Reference image 2 character's face" in prompt
    assert "Only part of the Reference image 3 character is visible" in prompt
    assert "Reference image 1 character's face" not in prompt
    assert "No additional people." in prompt and "lower quarter" not in prompt
    assert "always has" not in prompt
    shown = cg_proposals.compile_prompt(row, face_items=True)
    assert "The Reference image 2 character always has a red hair clip: draw this clearly" in shown
    assert "glasses" not in shown and "eyepatch" not in shown
    with_extras = cg_proposals.compile_prompt({**row, "extras": "人影"}, message_area=True)
    assert "anonymous figures" in with_extras and "No additional people" not in with_extras
    assert "lower quarter" in with_extras
    assert "No additional people." in cg_proposals.compile_prompt(
        {**row, "extras": "人影"}, allow_extras=False)


def mock_responses(monkeypatch, answers):
    calls = []

    def open_response(request, timeout):
        calls.append(json.loads(request.data))
        value = answers.pop(0)
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return io.BytesIO(json.dumps({"choices": [
            {"message": {"content": content}, "finish_reason": "stop"}]}).encode())

    monkeypatch.setattr(llm_prompts, "urlopen", open_response)
    return calls


def request(value=None, **options):
    value = value or scene()
    found = cg_proposals.scene_candidates(value, portraits())
    return cg_proposals.request_proposals(
        value, found["candidates"], found["unreferenced"], "雨を強調", "http://llm/v1", "local", **options)


def test_request_keeps_every_utterance_of_a_long_scene_and_hides_portrait_sources(monkeypatch):
    value = scene(lines=80)
    for row in value["context"]["utterances"]:
        row["display_text"] = row["display_text"] + "長い本文。" * 80
    original = deepcopy(value)
    calls = mock_responses(monkeypatch, [answer()])
    rows = request(value)
    assert len(rows) == 3
    mock_responses(monkeypatch, [answer(proposal(display_to=81), proposal(), proposal())] * 2)
    with pytest.raises(ValueError, match="1〜80"):
        request(value)
    payload = calls[0]
    source = json.loads(payload["messages"][1]["content"])
    assert [row["n"] for row in source["scene"]["utterances"]] == list(range(1, 81))
    assert source["scene"]["utterances"][40]["text"].startswith("発話41")
    assert {row["speaker"] for row in source["scene"]["utterances"]} == {
        "篠原 美緒", "水瀬 透", "narration", "other"}
    assert source["candidates"] == [
        {"tag": "A", "name": "篠原 美緒", "role": "ヒロイン", "lines": 20},
        {"tag": "B", "name": "水瀬 透", "lines": 20,
         "appearance": "黒髪に細い黒縁の四角いメガネ。紺のパーカー。"}]
    assert source["unreferenced_characters"] == ["駅員"] and source["director_note"] == "雨を強調"
    text = json.dumps(payload, ensure_ascii=False)
    assert "secret-hash" not in text and "p-mio" not in text
    system = payload["messages"][0]["content"]
    for phrase in ("3 alternative illustrations", "stays on screen", "central exchange",
                   "can be held on screen", "side by side", "choose 1 to 2 characters",
                   "anonymous figures", "ONLY by its tag", "behind the register",
                   "at least three specific things", "no quoted words",
                   "still lit by its own lamps", "Smaller details are yours to shape",
                   "must not influence the cast, the camera"):
        assert phrase in system
    assert payload["temperature"] == cg_proposals.TEMPERATURE
    assert value == original


def test_forbidden_extras_change_the_instruction(monkeypatch):
    calls = mock_responses(monkeypatch, [answer()])
    request(allow_extras=False)
    system = calls[0]["messages"][0]["content"]
    assert "Do not show any person other than the chosen candidates" in system
    assert "anonymous figures" not in system


def test_invalid_answer_gets_one_corrective_retry_then_fails_closed(monkeypatch):
    bad = answer(proposal(english_prompt=PROMPT.replace("<B>", "someone")), proposal(), proposal())
    calls = mock_responses(monkeypatch, [bad, "```json\n" + json.dumps(answer()) + "\n```"])
    assert len(request()) == 3
    assert "Correct this problem" in calls[1]["messages"][2]["content"]
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]
    mock_responses(monkeypatch, [bad, "not json"])
    with pytest.raises(ValueError, match="1回再試行済み"):
        request()


def test_runner_requests_structured_proposals_and_publishes_after_release(tmp_path, monkeypatch):
    value = scene()
    found = cg_proposals.scene_candidates(value, portraits())
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"task": "cg_proposals", "scene": value, **found, "instruction": "",
                                "allow_extras": False, "count": 3, "backend": "llama_cpp",
                                "local_model": "local-gguf"}, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "run"

    @contextmanager
    def native(root, model, directory, *, status, cancelled):
        try:
            yield "http://127.0.0.1:9000/v1", "local"
        finally:
            assert not (output / "result.json").exists()

    def generate(scene_value, candidates, unreferenced, instruction, url, model, **kwargs):
        assert [row["tag"] for row in candidates] == ["A", "B"] and unreferenced == ["駅員"]
        assert kwargs["allow_extras"] is False and kwargs["count"] == 3
        assert kwargs["utterances"] == 3
        options = kwargs["request_options"]
        assert options["chat_template_kwargs"] == {"enable_thinking": False}
        schema = options["response_format"]["json_schema"]["schema"]
        assert schema == cg_proposals.schema(["A", "B"], 3, 3)
        assert schema["properties"]["proposals"]["items"]["properties"]["display_to"]["maximum"] == 3
        return cg_proposals.normalize_proposals(answer(), ["A", "B"], allow_extras=False)

    monkeypatch.setattr(llm_runtime, "managed_llama", native)
    monkeypatch.setattr(llm_runner, "request_proposals", generate)
    assert llm_runner.run_request(path, output) == 0
    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert saved["ok"] and saved["llm_released"] and len(saved["proposals"]) == 3


def cg_run(app, monkeypatch):  # noqa: F811
    app.portraits = portraits()
    app.chapters = [{"number": 2, "title": "第2章", "scenes": [scene(), {**scene(), "id": "s4", "label": "4. 別"}]}]
    app.chapter_box["values"] = ["2章: 第2章"]
    app._cg_set_chapters()
    calls = []

    def start(command, directory, kind, data):
        calls.append((command, directory, kind, data))
        app.run_dir = directory

    monkeypatch.setattr(app, "_start_process", start)
    app.create_cg_proposals()
    return calls


def cg_result(**overrides):
    rows = cg_proposals.normalize_proposals(answer(), ["A", "B"])
    return {"ok": True, "proposals": rows, "backend": "llama_cpp", "llm_released": True, **overrides}


def test_tab_requests_proposals_without_any_user_selected_reference(app, monkeypatch):  # noqa: F811
    assert "A=篠原 美緒" not in app.cg_cast.get()
    command, _, kind, data = cg_run(app, monkeypatch)[0]
    assert "A=篠原 美緒（0発話） / B=水瀬 透（1発話）" in app.cg_cast.get()
    assert "駅員" in app.cg_cast.get() and data["prompt_version"] == cg_proposals.PROMPT_VERSION
    assert command[4].endswith("llm_runner.py") and kind == "cg_proposals"
    assert data["task"] == "cg_proposals" and data["count"] == 3 and data["allow_extras"] is True
    assert [row["tag"] for row in data["candidates"]] == ["A", "B"]
    assert app.references["scene"] == []


def test_selected_proposal_loads_only_its_cast_in_order_and_records_provenance(
        app, monkeypatch, tmp_path):  # noqa: F811
    calls = cg_run(app, monkeypatch)
    rows = cg_result()["proposals"]
    rows[1]["cast"] = [{"tag": "B", "visibility": "face_profile", "face_items": "thin black glasses"}]
    rows[1]["english_prompt"] = PROMPT.replace("<A>", "a distant lamp")
    rows[1]["layout"] = LAYOUT.replace("<A>", "a lamp post")
    app._finish_cg_proposals(0, {**cg_result(), "proposals": rows})
    assert len(app.cg_list.get_children()) == 3
    assert app.cg_list.item("0", "values")[3] == "1〜3（3発話）"
    assert "Reference image 1 character" in app.cg_prompt.get("1.0", "end")
    assert "描く内容" in app.cg_detail.get() and "顔の小物" not in app.cg_detail.get()
    app.cg_list.selection_set("1")
    app.cg_proposal_selected()
    assert "顔の小物: 水瀬 透=thin black glasses" in app.cg_detail.get()
    assert "always has thin black glasses" in app.cg_prompt.get("1.0", "end")
    app.cg_face_items.set(False)
    app._cg_recompile()
    assert "always has" not in app.cg_prompt.get("1.0", "end")
    app.cg_face_items.set(True)
    app._cg_recompile()
    app.cg_prompt.insert("end", "\nRain falls.")
    downloads = []

    class Catalog:
        def download_reference(self, portrait, directory):
            downloads.append(portrait["artifact_id"])
            return {**reference(tmp_path, portrait["name"]), "character_id": portrait["character_id"]}

    app.catalog = Catalog()
    work = []
    monkeypatch.setattr(app, "_async", lambda kind, revision, job: work.append((kind, revision, job)))
    monkeypatch.setattr(app, "_runtime_python", lambda: "runtime-python")
    app.tabs.select(app.cg_frame)
    app.generate()
    kind, revision, job = work[0]
    assert kind == "cg_references" and revision == (app.catalog_revision, app.cg_revision)
    app._cg_references_ready(job(), None)
    assert downloads == ["p-toru-2"]
    command, directory, kind, data = calls[1]
    assert command[0] == "runtime-python" and command[4].endswith("runner.py") and kind == "generate"
    assert directory.name.startswith("cg-")
    assert [row["character_id"] for row in data["references"]] == ["toru"]
    assert data["prompt"].endswith("Rain falls.") and (data["width"], data["height"]) == (1536, 1024)
    record = data["context"]["cg_proposal"]
    assert record["index"] == 1 and record["manually_edited"] is True
    assert data["text_encoder_offload"] == "layers" and data["reference_resolution"] == 1024
    assert record["prompt_version"] == cg_proposals.PROMPT_VERSION
    assert (record["proposal"]["display_from"], record["proposal"]["display_to"]) == (1, 3)
    assert record["proposal"]["layout"] == LAYOUT.replace("<A>", "a lamp post")
    assert record["proposal"]["cast"] == [
        {"tag": "B", "visibility": "face_profile", "face_items": "thin black glasses"}]
    assert record["face_items"] is True and "always has thin black glasses" in data["prompt"]
    assert record["candidates"][0] == {"tag": "A", "character_id": "mio", "name": "篠原 美緒"}


def test_stale_or_unreleased_proposals_are_not_applied(app, monkeypatch):  # noqa: F811
    cg_run(app, monkeypatch)
    app._finish_cg_proposals(0, cg_result(llm_released=False))
    assert app.cg_proposals == [] and "GPU解放" in app.status.get()
    app.create_cg_proposals()
    app.cg_scene_box.current(1)
    app.cg_scene_changed()
    app._finish_cg_proposals(0, cg_result())
    assert app.cg_proposals == [] and "反映しませんでした" in app.status.get()


def test_all_proposals_run_in_sequence_and_stop_on_failure(app, monkeypatch, tmp_path):  # noqa: F811
    calls = cg_run(app, monkeypatch)
    app._finish_cg_proposals(0, cg_result())
    started = []
    monkeypatch.setattr(app, "_cg_next", lambda: started.append(app.cg_queue.pop(0)))
    app.generate_cg_all()
    assert started == [0] and app.cg_queue == [1, 2]
    app.run_dir, app.run_kind = tmp_path, "generate"
    (tmp_path / "result.json").write_text('{"ok": true, "outputs": []}', encoding="utf-8")
    app._finish_process(0)
    assert started == [0, 1] and app.cg_queue == [2]
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: errors.append(args))
    (tmp_path / "result.json").write_text('{"ok": false, "error": "失敗"}', encoding="utf-8")
    app._finish_process(1)
    assert started == [0, 1] and app.cg_queue == [] and errors
    assert len(calls) == 1


def test_preview_button_prepares_off_thread_and_opens_the_local_page(app, monkeypatch, tmp_path):  # noqa: F811
    from PIL import Image

    def run(name, context):
        directory = tmp_path / name
        directory.mkdir()
        Image.new("RGBA", (96, 64), "red").save(directory / "image.png")
        write_json(directory / "request.json", {"mode": "scene", "references": [], "context": context})
        app._add_result(directory, {"ok": True, "outputs": [
            {"path": str(directory / "image.png"), "seed": 1}]})
        return directory

    work, opened = [], []
    monkeypatch.setattr(app, "_async", lambda kind, revision, job: work.append((kind, revision, job)))
    run("scene-plain", {})
    app.preview_cg()
    assert work == [] and "印象的な一枚" in app.status.get()
    directory = run("cg-one", {"published_build_id": "build-1", "cg_proposal": {
        "index": 0, "proposal": {"shot": "wide", "angle": "low", "moment": "場面"}}})
    app.server.set("http://coordinator:8000")
    app.preview_cg()
    kind, revision, job = work[0]
    assert kind == "cg_preview" and revision == app.cg_preview_revision
    monkeypatch.setattr(cg_tab, "prepare", lambda *values: values)
    record, path, server, transition = job()
    assert transition == ("dissolve", 500)
    assert record["context"]["published_build_id"] == "build-1"
    assert path == directory / "image.png" and server == "http://coordinator:8000"
    monkeypatch.setattr(app.cg_preview_server, "add", lambda upstream, files: "http://127.0.0.1:9/t/")
    monkeypatch.setattr(cg_tab.webbrowser, "open", opened.append)
    app.cg_transition.set("なし（現在の本編と同じ）")
    app.cg_transition_ms.set("0")
    app.preview_cg()
    assert work[1][2]()[3] == ("cut", 0)
    app.cg_transition_ms.set("half a second")
    app.preview_cg()
    assert len(work) == 2 and "ミリ秒" in app.status.get()
    app._cg_preview_ready(None, "作品サーバーに接続できません")
    assert opened == [] and "接続できません" in app.status.get()
    app._cg_preview_ready(("http://coordinator:8000/player/build-1/", {}), None)
    assert opened == ["http://127.0.0.1:9/t/"] and "つづきから" in app.status.get()


def test_vram_presets_set_the_three_memory_options_and_reach_the_request(app, tmp_path):  # noqa: F811
    assert app.text_encoder_layers.get() and app.reference_resolution.get() == "1024"
    app.cpu_offload.set(False)
    app.vram_preset.set("24GB: 参照768・再利用あり")
    app.apply_vram_preset()
    assert (app.reference_resolution.get(), app.use_kv_cache.get(), app.text_encoder_layers.get(),
            app.cpu_offload.get()) == ("768", True, True, True)
    app.add_reference("portrait", reference(tmp_path))
    app.prompts["portrait"].insert("1.0", "Smile.")
    request = app.build_request("portrait")
    assert request["reference_resolution"] == 768 and request["text_encoder_offload"] == "layers"
    assert request["transformer_storage"] == "native" and request["vae_tiling"] is False
    app.vram_preset.set("12GB: 8ビット・参照768・再利用なし")
    app.apply_vram_preset()
    request = app.build_request("portrait")
    assert (request["transformer_storage"], request["vae_tiling"], request["use_kv_cache"],
            request["reference_resolution"]) == ("fp8", True, False, 768)
    app.vram_preset.set("24GB: 参照1024・再利用なし（約2倍の時間）")
    app.apply_vram_preset()
    assert (app.reference_resolution.get(), app.use_kv_cache.get()) == ("1024", False)
    app.text_encoder_layers.set(False)
    assert app.build_request("portrait")["text_encoder_offload"] == "model"


def test_text_encoder_mode_is_validated_and_defaults_to_whole_model(app, tmp_path):  # noqa: F811
    from scripts.image_edit import engine

    app.add_reference("portrait", reference(tmp_path))
    app.prompts["portrait"].insert("1.0", "Smile.")
    request = app.build_request("portrait")
    request.pop("text_encoder_offload")
    assert engine.validate_request(request)["text_encoder_offload"] == "model"
    with pytest.raises(ValueError, match="text_encoder_offload"):
        engine.validate_request({**request, "text_encoder_offload": "blocks"})
    for key in ("transformer_storage", "vae_tiling"):
        request.pop(key)
    checked = engine.validate_request(request)
    assert checked["transformer_storage"] == "native" and checked["vae_tiling"] is False
    with pytest.raises(ValueError, match="transformer_storage"):
        engine.validate_request({**request, "transformer_storage": "int4"})
    with pytest.raises(ValueError, match="vae_tiling"):
        engine.validate_request({**request, "vae_tiling": "yes"})


def test_layered_text_encoder_keeps_whole_model_swaps_for_transformer_and_vae(monkeypatch):
    import sys
    from types import SimpleNamespace

    from scripts.image_edit import backend

    calls = []

    def hook(model, device, prev_module_hook=None):
        calls.append(("swap", model, str(device), prev_module_hook))
        return model, "hook-" + model

    def group(model, **options):
        calls.append(("layers", model, options["offload_type"], str(options["onload_device"]),
                      str(options["offload_device"])))

    class Part(str):
        def enable_layerwise_casting(self, **options):
            calls.append(("fp8", str(self), options["storage_dtype"], options["compute_dtype"]))

        def enable_tiling(self, **options):
            calls.append(("tiles", str(self), options["tile_sample_min_width"],
                          options["tile_sample_stride_width"]))

    class Pipeline:
        text_encoder, transformer, vae = "encoder", Part("transformer"), Part("vae")

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def enable_model_cpu_offload(self):
            calls.append(("stock",))

    monkeypatch.setitem(sys.modules, "accelerate", SimpleNamespace(cpu_offload_with_hook=hook))
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(QwenImage21Pipeline=Pipeline))
    monkeypatch.setitem(sys.modules, "diffusers.hooks", SimpleNamespace(apply_group_offloading=group))

    class Torch:
        bfloat16, float8_e4m3fn = "bf16", "fp8-storage"

        @staticmethod
        def device(name):
            return name

        class inference_mode:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

    request = {"model_path": "model", "dtype": "bfloat16", "cpu_offload": True}
    backend.load_pipeline({**request, "text_encoder_offload": "layers"}, Torch)
    assert calls == [("layers", "encoder", "leaf_level", "cuda", "cpu"),
                     ("swap", "transformer", "cuda", None),
                     ("swap", "vae", "cuda", "hook-transformer")]
    calls.clear()
    backend.load_pipeline(request, Torch)
    backend.load_pipeline({**request, "text_encoder_offload": "model"}, Torch)
    assert calls == [("stock",), ("stock",)]
    calls.clear()
    backend.load_pipeline({**request, "transformer_storage": "fp8", "vae_tiling": True}, Torch)
    assert calls == [("fp8", "transformer", "fp8-storage", "bf16"), ("tiles", "vae", 768, 640),
                     ("stock",)]
