"""Exercise asynchronous GUI results without downloading models or calling APIs."""

import gc
import io
import json
import queue
import tkinter
from pathlib import Path

import pytest

from scripts.audio import gui
from scripts.audio.catalog import Scene
from scripts.audio.gui import AudioTestApp


class FakeWindowsChildJob:
    def __init__(self):
        self.assigned = []
        self.closed = False

    def assign(self, process):
        self.assigned.append(process)

    def close(self):
        self.closed = True


class FakePlaybackProcess:
    pid = 12345

    def __init__(self):
        self.returncode = None
        self.stderr = io.BytesIO()
        self.stopped = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.stopped = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture(scope="module")
def gui_app():
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    instance = AudioTestApp(root, auto_refresh=False)
    yield instance
    root.destroy()
    # Release Tk variables on the UI thread before later CPU tests start workers.
    instance.__dict__.clear()
    gc.collect()


@pytest.fixture
def app(gui_app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui, "WindowsChildJob", FakeWindowsChildJob, raising=False)
    monkeypatch.setattr(gui, "PLAYBACK_ROOT", tmp_path / "playback")
    gui_app._cleanup_playback_session()
    gui_app._playback_key = None
    gui_app._playback_mode = None
    gui_app._playback_sequence = 0
    gui_app._timeline_key = None
    gui_app.timeline.reset()
    gui_app.events = queue.SimpleQueue()
    gui_app.catalog_revision = 0
    gui_app.results.clear()
    gui_app.history.delete(*gui_app.history.get_children())
    gui_app.playback_process = None
    for name in ("_process_job", "_playback_job"):
        job = getattr(gui_app, name, None)
        if job is not None:
            job.close()
        setattr(gui_app, name, None)
    gui_app.loop_source_metadata = {}
    gui_app._loop_source_path = None
    gui_app.loop_source.set("")
    gui_app.loop_method.set(gui.LOOP_METHODS["smart_region"])
    gui_app.loop_duration.set("60")
    gui_app.loop_fade.set("0.25")
    gui_app.loop_candidates.set("3")
    gui_app.loop_bridge.set("4")
    gui_app.loop_context.set("15")
    gui_app.loop.set(False)
    gui_app.loop_points.set("")
    gui_app.output_format.set("mp3")
    gui_app.mp3_bitrate.set("192")
    gui_app.keep_wav.set(False)
    gui_app.model.set("medium")
    gui_app.ace_bpm.set("0")
    gui_app.ace_keyscale.set("")
    gui_app.ace_timesignature.set("")
    gui_app.ace_planner_enabled.set(True)
    gui_app.ace_planner_path.set(str(gui.RUNTIME / "models" / "ace15_planner_1_7b"))
    gui_app.ace_continuous.set(True)
    gui_app._model_changed()
    for key, variable in gui_app.paths.items():
        variable.set(str(gui.RUNTIME / "models" / key))
    gui_app.duration.set("30")
    gui_app.steps.set("8")
    gui_app.seed.set("-1")
    gui_app.device.set("auto")
    gui_app.dtype.set("auto")
    gui_app.offload.set(False)
    gui_app.quality_preset.set(next(iter(gui.QUALITY_PRESETS)))
    gui_app.rating.set("未評価")
    gui_app.note.set("")
    gui_app.pending_llm = None
    gui_app.llm_run_revision = None
    gui_app.prompt_revision = 0
    gui_app._active_scene_key = None
    gui_app.auto_scene_prompt.set(False)
    gui_app.process = None
    gui_app.run_dir = None
    gui_app.run_kind = None
    gui_app.stop_at = None
    gui_app.close_pending = False
    gui_app.output.set(str(tmp_path))
    gui_app.llm_backend.set("llama_cpp")
    gui_app.llm_local_model.set("registered.gguf")
    gui_app.llm_url.set("http://127.0.0.1:8080/v1")
    gui_app.llm_model.set("local-model")
    gui_app.tempo.set("0")
    gui_app.mood_box.current(0)
    gui_app.style_box.current(0)
    gui_app._replace_prompt(gui.DEFAULT_PROMPT)
    for button in (gui_app.generate_button, gui_app.ai_button, gui_app.setup_button,
                   gui_app.prepare_button, gui_app.auth_button, gui_app.loop_button):
        button.configure(state="normal")
    gui_app.stop_button.configure(state="disabled")
    gui_app.scenes = [Scene("s1", "夜の駅", {"plan": {"atmosphere": "緊張"}}, "緊張するシーン")]
    gui_app.scene_box["values"] = ["夜の駅"]
    gui_app.scene_box.current(0)
    monkeypatch.setattr(gui_app, "_async", lambda *args: None)
    monkeypatch.setattr(gui_app, "_start_process", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_app, "_save_settings", lambda: None)
    yield gui_app
    gui_app.stop_playback()
    gui_app.progress.stop()
    if gui_app._log_stream is not None:
        gui_app._log_stream.close()
        gui_app._log_stream = None
    gui_app.process = None
    for name in ("_process_job", "_playback_job"):
        job = getattr(gui_app, name, None)
        if job is not None:
            job.close()
        setattr(gui_app, name, None)


def test_later_manual_edit_is_preserved_over_llm_reply(app):
    app.draft_ai()
    revision = app.prompt_revision
    app._replace_prompt("Manually revised music prompt")
    app.events.put(("prompt", revision, "Old LLM prompt", None))
    app.poll()
    assert app.prompt.get("1.0", "end").strip() == "Manually revised music prompt"
    assert app.pending_llm is None
    assert str(app.ai_button["state"]) == "normal"


def test_llm_scene_interpretation_is_displayed_with_its_prompt(app):
    app.draft_ai()
    interpretation = "尊大な人物と淡々とした対応の対比で笑わせる場面。短い木管と弦の掛け合いで表す。"
    app.events.put(("prompt", app.prompt_revision,
                    {"prompt": "Genre: Comedy Soundtrack. Instruments: Bassoon, Pizzicato Strings.",
                     "scene_interpretation": interpretation}, None))
    app.poll()
    assert app.prompt_source.get() == "LLM作成"
    assert app.scene_interpretation.get() == interpretation
    assert "Bassoon" in app.prompt.get("1.0", "end")


def test_reselecting_scene_preserves_llm_but_new_scene_requires_new_prompt(app):
    app.scene_changed()
    app._replace_prompt("Comic bassoon and plucked strings.", source="llm", interpretation="ギャグ場面")
    app.scene_changed()
    assert app.prompt_source.get() == "LLM作成"
    assert "Comic bassoon" in app.prompt.get("1.0", "end")
    app.scenes.append(Scene("s2", "別の場面", {"plan": {"atmosphere": "悲しみ"}}, "別の場面"))
    app.scene_box["values"] = [scene.label for scene in app.scenes]
    app.scene_box.current(1)
    app.scene_changed()
    assert app.prompt.get("1.0", "end").strip() == ""
    assert app.prompt_source.get() == "未作成"
    assert app.scene_interpretation.get() == ""


def test_simple_scene_prompt_is_only_inserted_when_enabled(app):
    app.auto_scene_prompt.set(True)
    app.scene_changed()
    assert app.prompt_source.get() == "簡易作成"
    assert "TrackType: Music" in app.prompt.get("1.0", "end")


def test_work_context_change_invalidates_llm_for_the_same_scene(app):
    scene = app.selected_scene()
    scene.context["story_context"] = {"brief": {"genre": "現代コメディ", "mood": "明るい"},
                                      "outline": {"ending": "気楽な日常に戻る"}}
    app.scene_changed()
    app._replace_prompt("Genre: Pop. Instruments: Guitar, Bass.", source="llm", interpretation="明るい場面")
    app.scene_changed()
    assert app.prompt_source.get() == "LLM作成"
    scene.context["story_context"]["brief"]["mood"] = "陰鬱なホラー"
    app.scene_changed()
    assert app.prompt_source.get() == "未作成"
    assert app.prompt.get("1.0", "end").strip() == ""
    assert app.scene_interpretation.get() == ""


def test_work_context_change_discards_a_pending_llm_reply(app):
    app.scene_changed()
    app.draft_ai()
    revision = app.prompt_revision
    app.selected_scene().context["story_context"] = {"brief": {"genre": "現代コメディ"}}
    app.scene_changed()
    app.events.put(("prompt", revision, {"prompt": "Old orchestral prompt",
                                       "scene_interpretation": "古い解釈"}, None))
    app.poll()
    assert app.prompt.get("1.0", "end").strip() == ""
    assert app.pending_llm is None


def test_gui_shows_which_work_and_chapter_materials_reach_the_llm(app):
    app.selected_scene().context["story_context"] = {
        "status": "complete", "brief": {"genre": "現代コメディ／ゆるファンタジー"},
        "outline": {"ending": "食べ物を巡り仲良く会計する"},
        "chapter": {"role": "魔王が店員を認める"},
    }
    app.scene_changed()
    assert "作品設定・全体プロット・章の役割・シーン" in app.story_context_hint.get()
    assert "現代コメディ／ゆるファンタジー" in app.story_context_hint.get()
    app.selected_scene().context["story_context"] = {"outline": {"ending": "会計する"}}
    app.scene_changed()
    assert "作品設定未取得" in app.story_context_hint.get()
    assert "全体プロット・シーン" in app.story_context_hint.get()
    app.selected_scene().context["story_context"] = {
        "outline": {"ending": "", "chapters": [], "character_arcs": [], "foreshadowing": []},
        "missing": ["brief", "outline"],
    }
    app.scene_changed()
    assert app.story_context_hint.get() == "LLM参照: シーンのみ（作品設定・全体プロット未取得）"


def test_manual_prompt_edit_invalidates_llm_interpretation(app):
    app._replace_prompt("Original comic score.", source="llm", interpretation="ギャグ場面")
    app.prompt.insert("end", " A melancholy cello melody.")
    app._prompt_edited()
    assert app.prompt_source.get() == "手動入力・編集"
    assert app.scene_interpretation.get() == ""


def test_music_request_records_llm_scene_interpretation(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: calls.append(kwargs["request"]))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app._replace_prompt("Comic bassoon and pizzicato strings.", source="llm", interpretation="シュールなギャグ")
    app.generate()
    assert calls[0]["context"]["prompt_generation"] == {
        "method": "llm", "scene_interpretation": "シュールなギャグ"}
    assert "prompt_generation" not in app.selected_scene().context


def test_setup_uses_console_python_without_powershell(app, monkeypatch, tmp_path):
    python = tmp_path / "Python with spaces/python.exe"
    python.parent.mkdir()
    python.touch()
    monkeypatch.setattr(gui.sys, "executable", str(python.with_name("pythonw.exe")))
    app.output.set(str(tmp_path / "outputs"))
    commands = []
    monkeypatch.setattr(app, "_start_process", lambda command, *args: commands.append(command))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app.setup_runtime()
    assert len(commands) == 1
    assert commands[0][0] == str(python)
    assert Path(commands[0][4]).name == "setup_runtime.py"
    assert "--status-dir" in commands[0]


def test_later_preset_is_preserved_over_llm_reply(app):
    app.draft_ai()
    revision = app.prompt_revision
    initial = app.prompt.get("1.0", "end").strip()
    app.mood_box.current(3)
    app.events.put(("prompt", revision, "Old LLM prompt", None))
    app.poll()
    assert app.prompt.get("1.0", "end").strip() == initial


def test_quality_control_displaces_pending_scene_prompt(app):
    app.draft_ai()
    revision = app.prompt_revision
    app.model.set("small")
    app.steps.set("16")
    app.seed.set("0")
    app.quality_preset.set(list(gui.QUALITY_PRESETS)[1])
    app.draft_quality()
    control = app.prompt.get("1.0", "end").strip()
    app.events.put(("prompt", revision, "Old dark scene prompt", None))
    app.poll()
    assert app.prompt.get("1.0", "end").strip() == control
    assert "TrackType: Music, VocalType: Instrumental" in control
    assert app.model.get() == "medium"
    assert app.steps.get() == "8"
    assert app.seed.get() == "-1"
    assert app.pending_llm is None


def test_history_includes_quality_comparisons_and_skips_prompt_runs(app, monkeypatch, tmp_path):
    for name in ("bgm-existing", "quality-medium-seed0", "prompt-existing"):
        run = tmp_path / name
        run.mkdir()
        (run / "result.json").write_text('{"ok": true}', encoding="utf-8")
        if not name.startswith("prompt-"):
            (run / "output.wav").touch()
    loaded = []
    monkeypatch.setattr(app, "_add_result", lambda directory, result: loaded.append(directory.name))
    app.load_history()
    assert loaded == ["bgm-existing", "quality-medium-seed0"]


def test_old_reply_cannot_enable_button_for_new_request(app):
    app.draft_ai()
    old_revision = app.prompt_revision
    app.draft_ai()
    current_revision = app.prompt_revision
    app.events.put(("prompt", old_revision, "Old LLM prompt", None))
    app.poll()
    assert app.pending_llm[0] == current_revision
    assert str(app.ai_button["state"]) == "disabled"
    app.events.put(("prompt", current_revision, "Current LLM prompt", None))
    app.poll()
    assert app.prompt.get("1.0", "end").strip() == "Current LLM prompt"
    assert str(app.ai_button["state"]) == "normal"


def test_project_refresh_discards_pending_scene_prompt(app):
    app.draft_ai()
    revision = app.prompt_revision
    initial = app.prompt.get("1.0", "end").strip()
    app.refresh()
    app.events.put(("prompt", revision, "Old scene LLM prompt", None))
    app.poll()
    assert app.prompt.get("1.0", "end").strip() == initial
    assert app.pending_llm is None


@pytest.mark.parametrize("action", ["refresh", "project_changed"])
def test_catalog_reload_immediately_clears_scene_dependent_prompt(app, action):
    app.scene_changed()
    app._replace_prompt("Genre: Pop. Instruments: Guitar.", source="llm", interpretation="元作品の喜劇")
    getattr(app, action)()
    assert app.prompt.get("1.0", "end").strip() == ""
    assert app.scene_interpretation.get() == ""
    assert app.prompt_source.get() == "未作成"
    assert app._active_scene_key is None


def test_catalog_reload_preserves_independent_quality_control(app):
    app.draft_quality()
    control = app.prompt.get("1.0", "end").strip()
    app.refresh()
    assert app.prompt.get("1.0", "end").strip() == control
    assert app.prompt_source.get() == "品質確認用"


def test_prompt_uses_root_console_python_and_scene_snapshot(app, monkeypatch, tmp_path):
    python = tmp_path / "Python with spaces/python.exe"
    python.parent.mkdir()
    python.touch()
    monkeypatch.setattr(gui.sys, "executable", str(python.with_name("pythonw.exe")))
    calls = []
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: calls.append((args, kwargs)))
    app.llm_backend.set("ollama")
    app.llm_url.set("http://127.0.0.1:11434/v1")
    app.llm_model.set("qwen3:8b")
    app.draft_ai()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0][0] == str(python)
    assert Path(args[0][4]).name == "llm_runner.py"
    assert args[2] == "prompt"
    assert kwargs["request"]["backend"] == "ollama"
    assert kwargs["request"]["scene"] == {
        "id": "s1", "label": "夜の駅", "context": {"plan": {"atmosphere": "緊張"}},
        "preview": "緊張するシーン",
    }
    assert kwargs["request"]["options"]["tempo"] == 0
    assert app.llm_run_revision == app.pending_llm[0]


def test_prompt_cannot_start_during_music_generation(app, monkeypatch):
    app.process = object()
    app.run_kind = "generate"
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: errors.append(args))
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: pytest.fail("overlapping process"))
    app.draft_ai()
    assert errors and "現在の処理" in errors[0][1]
    assert app.pending_llm is None


def test_prompt_result_is_applied_only_after_process_exit(app, tmp_path):
    app.draft_ai()
    app.run_dir = tmp_path
    app.run_kind = "prompt"
    app.process = object()
    app._log_stream = io.BytesIO()
    (tmp_path / "result.json").write_text(
        json.dumps({"ok": True, "prompt": "Quiet instrumental strings and piano."}), encoding="utf-8")
    app._finish_process(0)
    assert app.process is None
    assert app.pending_llm is not None
    assert app.prompt.get("1.0", "end").strip() == gui.DEFAULT_PROMPT
    with pytest.raises(ValueError, match="LLMの結果"):
        AudioTestApp._start_process(app, ["python"], tmp_path / "early-audio", "generate")
    assert not (tmp_path / "early-audio").exists()
    app.poll()
    assert app.prompt.get("1.0", "end").strip() == "Quiet instrumental strings and piano."
    assert app.pending_llm is None


def test_prompt_cancel_discards_reply_and_allows_next_process(app, tmp_path):
    app.draft_ai()
    app.run_dir = tmp_path
    app.run_kind = "prompt"
    app.process = object()
    app._log_stream = io.BytesIO()
    app.stop_at = 1.0
    (tmp_path / "result.json").write_text(
        json.dumps({"ok": True, "prompt": "Cancelled reply"}), encoding="utf-8")
    app._finish_process(0)
    assert app.process is None
    assert app.pending_llm is None
    assert app.prompt.get("1.0", "end").strip() == gui.DEFAULT_PROMPT
    assert str(app.ai_button["state"]) == "normal"


@pytest.mark.parametrize(
    ("backend", "url", "expected"),
    [("ollama", "http://192.168.1.10:11434/v1", True),
     ("llama_cpp", "http://127.0.0.1:11434/v1", True),
     ("llama_cpp", "http://localhost:11434/v1", True),
     ("llama_cpp", "http://[::1]:11434/v1", True),
     ("llama_cpp", "http://127.0.0.1:8080/v1", False),
     ("openai", "http://127.0.0.1:11434/v1", False)],
)
def test_music_request_releases_selected_or_legacy_ollama(app, backend, url, expected):
    app.llm_backend.set(backend)
    app.llm_url.set(url)
    release = app._llm_release_request()
    if expected:
        assert release == {"base_url": url, "model": "local-model"}
    else:
        assert release is None


def test_start_process_serializes_prompt_and_audio_controls(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.subprocess, "Popen", lambda *args, **kwargs: object())
    run = tmp_path / "prompt-run"
    AudioTestApp._start_process(app, ["python", "llm_runner.py"], run, "prompt", request={"backend": "llama_cpp"})
    try:
        assert app.run_kind == "prompt"
        assert isinstance(app._process_job, FakeWindowsChildJob)
        assert app._process_job.assigned == [app.process]
        assert str(app.stop_button["state"]) == "normal"
        for button in [app.generate_button, app.setup_button, app.prepare_button, app.auth_button, app.ai_button]:
            assert str(button["state"]) == "disabled"
        with pytest.raises(ValueError, match="現在の処理"):
            AudioTestApp._start_process(app, ["python"], tmp_path / "overlap", "generate")
        assert not (tmp_path / "overlap").exists()
    finally:
        app._log_stream.close()
        app._log_stream = None
        app.process = None
        app.progress.stop()


def test_prompt_stop_creates_cancel_signal(app, tmp_path):
    app.process = object()
    app.run_dir = tmp_path
    app.run_kind = "prompt"
    app.stop()
    assert (tmp_path / "stop.request").is_file()
    assert app.stop_at is not None


def test_close_waits_for_prompt_cleanup(app, tmp_path):
    app.process = object()
    app.run_dir = tmp_path
    app.run_kind = "prompt"
    app.close()
    assert app.close_pending is True
    assert (tmp_path / "stop.request").is_file()


def test_native_registry_failure_keeps_gui_available(app, monkeypatch):
    import scripts.audio.llm_runtime as runtime

    monkeypatch.setattr(runtime, "list_native_models", lambda root: (_ for _ in ()).throw(ValueError("unavailable")))
    app.reload_llm_models()
    assert "unavailable" in app.llm_hint.get()
    assert app.model.get() in {"small", "medium"}


def saved_audio_result(directory, *, suffix="mp3", legacy=False, raw=False):
    directory.mkdir(parents=True, exist_ok=True)
    audio = directory / ("output." + suffix)
    audio.write_bytes(b"unit test media placeholder")
    metadata = {
        "audio_path": str(audio), "settings": {"model": "medium", "seed": 4,
                                              "prompt": "Genre: Pop. Instruments: Piano. Upbeat instrumental."},
        "context": {"scene_label": "元作品", "story_context": {"sources": {"narrative_artifact_id": "fixed-version"}}},
        "duration_seconds": 60, "sample_rate": 44100, "channels": 2,
    }
    if raw:
        original = directory / "output-float.wav"
        original.write_bytes(b"original PCM placeholder")
        metadata["float_audio_path"] = str(original)
    result = {"ok": True} if legacy else {"ok": True, "audio_path": str(audio), "metadata": metadata}
    (directory / "generation.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result, metadata


def test_generation_request_defaults_to_mp3_without_retaining_wav(app, monkeypatch):
    captured = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: captured.append(kwargs["request"]))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app.generate()
    assert captured[0]["output_format"] == "mp3"
    assert captured[0]["mp3_bitrate"] == 192
    assert captured[0]["keep_wav"] is False


def test_generation_request_honors_explicit_wav_retention_and_mp3_bitrate(app, monkeypatch):
    captured = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: captured.append(kwargs["request"]))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app.mp3_bitrate.set("320")
    app.keep_wav.set(True)
    app.generate()
    assert captured[0]["output_format"] == "mp3"
    assert captured[0]["mp3_bitrate"] == 320
    assert captured[0]["keep_wav"] is True


def test_nested_loop_history_excludes_aggregate_failed_and_prompt_results_and_retains_old_wav(app, tmp_path):
    old = tmp_path / "old-wav"
    saved_audio_result(old, suffix="wav", legacy=True)
    regular = tmp_path / "new-mp3"
    saved_audio_result(regular)
    aggregate = tmp_path / "loop-experiment"
    candidates = []
    for number in (1, 2, 3):
        directory = aggregate / f"candidate-{number:02d}"
        candidate, _ = saved_audio_result(directory)
        candidates.append({**candidate, "directory": str(directory)})
    # Aggregate references the first candidate's audio but must not add a duplicate.
    (aggregate / "result.json").write_text(json.dumps({**candidates[0], "candidates": candidates}))
    failed, _ = saved_audio_result(aggregate / "candidate-failed")
    failed["ok"] = False
    (aggregate / "candidate-failed/result.json").write_text(json.dumps(failed))
    prompt = tmp_path / "prompt-only"
    prompt.mkdir()
    (prompt / "result.json").write_text('{"ok":true,"prompt":"English prompt only"}')
    app.load_history()
    expected = {str(old.resolve()), str(regular.resolve()), *(str(Path(row["directory"]).resolve()) for row in candidates)}
    assert set(app.results) == expected
    assert len(app.history.get_children()) == 5
    assert app.results[str(old.resolve())][0] == old.resolve()
    assert app._result_audio_path(*app.results[str(old.resolve())]).suffix == ".wav"
    app.load_history()
    assert set(app.results) == expected
    assert len(app.history.get_children()) == 5


@pytest.mark.parametrize("parent_state", ["missing", "failed", "cancelled", "not-an-aggregate"])
def test_history_hides_partial_loop_candidates_without_successful_aggregate(app, tmp_path, parent_state):
    aggregate = tmp_path / "incomplete-loop"
    candidate, _ = saved_audio_result(aggregate / "candidate-01")
    if parent_state != "missing":
        parent_result = {
            "ok": parent_state == "not-an-aggregate",
            "cancelled": parent_state == "cancelled",
        }
        if parent_state != "not-an-aggregate":
            parent_result["candidates"] = [candidate]
        (aggregate / "result.json").write_text(json.dumps(parent_result))
    app.load_history()
    assert app.results == {}
    assert app.history.get_children() == ()


@pytest.mark.parametrize("raw_available", [False, True])
def test_loop_input_prefers_original_float_pcm_when_available(app, tmp_path, raw_available):
    directory = tmp_path / "source"
    result, metadata = saved_audio_result(directory, raw=raw_available)
    app._add_result(directory, result)
    app.choose_loop_source()
    expected = directory / ("output-float.wav" if raw_available else "output.mp3")
    assert Path(app.loop_source.get()) == expected
    assert app._loop_source_path == expected.resolve()
    assert app.loop_source_metadata == metadata


def test_manual_loop_input_path_does_not_reuse_old_result_provenance(app, monkeypatch, tmp_path):
    directory = tmp_path / "source"
    result, _ = saved_audio_result(directory)
    app._add_result(directory, result)
    app.choose_loop_source()
    replacement = directory / "unrelated-edit.wav"
    replacement.touch()
    app.loop_source.set(str(replacement))
    captured = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: captured.append(kwargs["request"]))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app.create_loop()
    assert captured[0]["source_audio"] == str(replacement.resolve())
    assert captured[0]["source_generation"] == {}
    assert app.loop_source_metadata["context"]["scene_label"] == "元作品"


def test_manually_selected_other_audio_uses_its_own_saved_generation(app, monkeypatch, tmp_path):
    original = tmp_path / "source"
    old_result, _ = saved_audio_result(original)
    app._add_result(original, old_result)
    app.choose_loop_source()
    replacement = tmp_path / "replacement"
    _, fresh_metadata = saved_audio_result(replacement)
    fresh_metadata["context"]["scene_label"] = "別作品"
    fresh_metadata["context"]["story_context"]["sources"]["narrative_artifact_id"] = "other-version"
    (replacement / "generation.json").write_text(json.dumps(fresh_metadata, ensure_ascii=False), encoding="utf-8")
    app.loop_source.set(str(replacement / "output.mp3"))
    captured = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: captured.append(kwargs["request"]))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app.create_loop()
    assert captured[0]["source_generation"]["context"]["scene_label"] == "別作品"
    assert captured[0]["source_generation"]["context"]["story_context"]["sources"]["narrative_artifact_id"] == "other-version"


@pytest.mark.parametrize("method", ["whole_crossfade", "smart_region", "ai_bridge", "smart_ai_bridge"])
def test_loop_request_forwards_ai_controls_and_releases_ollama_only_for_ai(app, monkeypatch, tmp_path, method):
    result, metadata = saved_audio_result(tmp_path / "source")
    app._add_result(tmp_path / "source", result)
    app.choose_loop_source()
    app.loop_method.set(gui.LOOP_METHODS[method])
    app.llm_backend.set("ollama")
    app.llm_url.set("http://127.0.0.1:11434/v1")
    app.llm_model.set("selected-model")
    app.model.set("medium")
    app.paths["medium"].set("prepared/medium")
    app.device.set("cuda")
    app.dtype.set("bfloat16")
    app.offload.set(True)
    app.steps.set("12")
    app.seed.set("123")
    app.loop_bridge.set("6")
    app.loop_context.set("18")
    calls = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app.create_loop()
    args, kwargs = calls[0]
    request = kwargs["request"]
    assert args[2] == "loop" and Path(args[0][4]).name == "loop_runner.py"
    assert request["method"] == method and request["source_generation"] == metadata
    assert request["output_format"] == "mp3" and request["keep_wav"] is False
    if method in {"ai_bridge", "smart_ai_bridge"}:
        assert request["llm_release"] == {"base_url": "http://127.0.0.1:11434/v1", "model": "selected-model"}
        assert (request["model"], request["model_path"], request["device"], request["dtype"]) == (
            "medium", "prepared/medium", "cuda", "bfloat16")
        assert request["cpu_offload"] is True and request["steps"] == 12 and request["seed"] == 123
        assert request["bridge_seconds"] == 6 and request["bridge_context_seconds"] == 18
    else:
        assert "llm_release" not in request


def test_finished_loop_candidates_use_audio_parent_when_directory_is_missing(app, tmp_path):
    aggregate = tmp_path / "loop-run"
    directory = aggregate / "candidate-01"
    candidate, metadata = saved_audio_result(directory)
    (aggregate / "result.json").write_text(json.dumps({"ok": True, "candidates": [candidate]}))
    app.run_dir, app.run_kind, app.process = aggregate, "loop", object()
    app._log_stream = io.BytesIO()
    job = FakeWindowsChildJob()
    job.assign(app.process)
    app._process_job = job
    app._finish_process(0)
    assert app.process is None
    assert app._process_job is None and job.closed
    assert set(app.results) == {str(directory)}
    assert app.results[str(directory)][2] == metadata
    assert "候補" in app.loop_hint.get()


@pytest.mark.parametrize("loop_enabled,seam_preview", [(False, False), (True, False), (True, True)])
def test_playback_uses_pcm_player_and_exclusive_loop_or_seam_preview_flags(app, monkeypatch, tmp_path, loop_enabled, seam_preview):
    directory = tmp_path / "source"
    result, _ = saved_audio_result(directory)
    app._add_result(directory, result)
    app.loop.set(loop_enabled)
    calls = []

    def popen(command, **kwargs):
        process = FakePlaybackProcess()
        calls.append((command, kwargs, process))
        return process

    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(gui.subprocess, "Popen", popen)
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app.play(seam_preview=seam_preview)
    command, options, process = calls[0]
    assert Path(command[4]).name == "player.py"
    assert command[command.index("--input") + 1] == str(directory / "output.mp3")
    assert ("--seam-preview" in command) is seam_preview
    assert ("--loop" in command) is (loop_enabled and not seam_preview)
    assert options["stdin"] == gui.subprocess.DEVNULL
    assert app.playback_process is process
    job = app._playback_job
    assert isinstance(job, FakeWindowsChildJob) and job.assigned == [process]
    app.stop_playback()
    assert process.stopped and process.stderr.closed and app.playback_process is None
    assert job.closed and app._playback_job is None


@pytest.mark.parametrize("seam_preview", [False, True])
def test_loop_candidate_displays_and_passes_intro_then_loop_points(app, monkeypatch, tmp_path, seam_preview):
    directory = tmp_path / "intro-loop"
    result, metadata = saved_audio_result(directory)
    metadata["loop"] = {"method": "smart_region", "format_version": 2,
                        "start_sample": 10 * 44100, "end_sample": 60 * 44100}
    app._add_result(directory, result)
    assert app.loop.get() is True
    assert "0.00 → 60.00" in app.loop_points.get()
    assert "10.00 → 60.00" in app.loop_points.get()
    calls = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(gui.subprocess, "Popen", lambda command, **kwargs: (
        calls.append(command) or FakePlaybackProcess()))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app.play(seam_preview=seam_preview)
    command = calls[0]
    assert command[command.index("--loop-start-sample") + 1] == str(10 * 44100)
    assert command[command.index("--loop-end-sample") + 1] == str(60 * 44100)
    assert ("--loop" in command) is not seam_preview
    assert ("--seam-preview" in command) is seam_preview
    assert app.timeline._seeking_enabled is not seam_preview
    assert ("継ぎ目だけ試聴中" in app.loop_points.get()) is seam_preview
    app.stop_playback()
    assert "継ぎ目だけ試聴中" not in app.loop_points.get()


def test_legacy_sliced_loop_restores_original_intro_for_normal_playback(app, monkeypatch, tmp_path):
    _, original = saved_audio_result(tmp_path / "original", raw=True)
    directory = tmp_path / "legacy-loop"
    result, metadata = saved_audio_result(directory)
    metadata.update({"source_audio": original["float_audio_path"], "source_generation": original,
                     "duration_seconds": 50, "preview_gain": 0.6,
                     "loop": {"method": "smart_region", "source_start_seconds": 10,
                              "source_end_seconds": 60}})
    app._add_result(directory, result)
    assert "0.00 → 60.00" in app.loop_points.get()
    assert "10.00 → 60.00" in app.loop_points.get()
    calls = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(gui.subprocess, "Popen", lambda command, **kwargs: (
        calls.append(command) or FakePlaybackProcess()))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app.play()
    command = calls[0]
    assert command[command.index("--intro-input") + 1] == original["float_audio_path"]
    assert command[command.index("--intro-samples") + 1] == str(10 * 44100)
    assert float(command[command.index("--intro-gain") + 1]) == pytest.approx(0.6)
    assert command[command.index("--loop-start-sample") + 1] == str(10 * 44100)
    assert command[command.index("--loop-end-sample") + 1] == str(60 * 44100)


def test_nonloop_result_clears_previous_candidate_points_and_repeat_mode(app, tmp_path):
    result, metadata = saved_audio_result(tmp_path / "loop")
    metadata["loop"] = {"method": "smart_region", "start_sample": 10 * 44100, "end_sample": 60 * 44100}
    app._add_result(tmp_path / "loop", result)
    assert app.loop.get()
    ordinary, _ = saved_audio_result(tmp_path / "ordinary")
    app._add_result(tmp_path / "ordinary", ordinary)
    assert app.loop.get() is False
    assert "終端で停止" in app.loop_points.get()
    assert "10.00" not in app.loop_points.get()


def test_invalid_loop_metadata_is_visible_and_cannot_start_playback(app, monkeypatch, tmp_path):
    result, metadata = saved_audio_result(tmp_path / "invalid")
    metadata["loop"] = {"method": "smart_region", "start_sample": -1, "end_sample": 60 * 44100}
    app._add_result(tmp_path / "invalid", result)
    assert "読み込めません" in app.loop_points.get()
    errors = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(gui.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("invalid loop was started"))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: errors.append(args))
    app.play()
    assert errors


def test_legacy_wav_metadata_is_recognized_when_primary_path_only_in_result(app, tmp_path):
    directory = tmp_path / "legacy-wav"
    result, metadata = saved_audio_result(directory, suffix="wav", raw=True)
    del metadata["audio_path"]
    (directory / "generation.json").write_text(json.dumps(metadata), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    assert app._metadata_for_audio(directory / "output.wav") == metadata
    unrelated = directory / "unrelated.wav"
    unrelated.touch()
    assert app._metadata_for_audio(unrelated) == {}


def test_timeline_shows_loop_and_ai_edit_ranges_and_retains_position_on_reselection(app, tmp_path):
    result, metadata = saved_audio_result(tmp_path / "candidate-01")
    metadata["loop"] = {"method": "smart_ai_bridge", "start_sample": 10 * 44100,
                        "end_sample": 60 * 44100, "edited_file_start_sample": 55 * 44100,
                        "edited_file_end_sample": 60 * 44100}
    app._add_result(tmp_path / "candidate-01", result)
    assert app.timeline._duration == 60
    assert (app.timeline._loop_start, app.timeline._loop_end) == (10, 60)
    assert (app.timeline._edit_start, app.timeline._edit_end) == (55, 60)
    app.timeline.set_position(40)
    app.result_selected()
    assert app.timeline._position == 40


def test_seeking_during_playback_sends_control_without_restarting_decoder(app, monkeypatch, tmp_path):
    result, _ = saved_audio_result(tmp_path / "source")
    app._add_result(tmp_path / "source", result)
    calls = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(gui.subprocess, "Popen", lambda command, **kwargs: (
        calls.append(command) or FakePlaybackProcess()))
    app.play()
    assert "--status-file" in calls[0] and "--control-file" in calls[0]
    app._seek_playback(12.5)
    state = gui.read_json(app._playback_control_path)
    assert state == {"sequence": 1, "seek_sample": round(12.5 * 44100)}
    app._seek_playback(6)
    assert gui.read_json(app._playback_control_path) == {"sequence": 2, "seek_sample": 6 * 44100}
    assert len(calls) == 1


def test_timeline_click_when_stopped_starts_from_requested_position(app, monkeypatch, tmp_path):
    result, _ = saved_audio_result(tmp_path / "source")
    app._add_result(tmp_path / "source", result)
    calls = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(gui.subprocess, "Popen", lambda command, **kwargs: (
        calls.append(command) or FakePlaybackProcess()))
    app._seek_playback(15)
    assert calls[0][calls[0].index("--start-sample") + 1] == str(15 * 44100)
    assert app.timeline._position == 15


def test_cursor_ignores_player_status_before_latest_seek_is_acknowledged(app, monkeypatch, tmp_path):
    result, _ = saved_audio_result(tmp_path / "source")
    app._add_result(tmp_path / "source", result)
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(gui.subprocess, "Popen", lambda *args, **kwargs: FakePlaybackProcess())
    app.play()
    app._seek_playback(50)
    gui.write_json(app._playback_status_path, {"state": "playing", "position_sample": 3 * 44100,
                                             "sample_rate": 44100, "command_sequence": 0})
    app._poll_playback_status()
    assert app.timeline._position == 50
    gui.write_json(app._playback_status_path, {"state": "playing", "position_sample": round(50.1 * 44100),
                                             "sample_rate": 44100, "command_sequence": 1})
    app._poll_playback_status()
    assert app.timeline._position == pytest.approx(50.1)


def test_actual_player_status_moves_cursor_back_to_loop_start(app, monkeypatch, tmp_path):
    result, metadata = saved_audio_result(tmp_path / "loop")
    metadata["loop"] = {"method": "smart_region", "start_sample": 10 * 44100, "end_sample": 60 * 44100}
    app._add_result(tmp_path / "loop", result)
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(gui.subprocess, "Popen", lambda *args, **kwargs: FakePlaybackProcess())
    app.play()
    for seconds in (59.9, 10.1):
        gui.write_json(app._playback_status_path, {"state": "playing", "position_sample": round(seconds * 44100),
                                                 "sample_rate": 44100, "total_samples": 60 * 44100})
        app._poll_playback_status()
        assert app.timeline._position == pytest.approx(seconds)


def test_switching_audio_stops_old_player_and_resets_cursor(app, monkeypatch, tmp_path):
    first, _ = saved_audio_result(tmp_path / "first")
    app._add_result(tmp_path / "first", first)
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    old_player = FakePlaybackProcess()
    monkeypatch.setattr(gui.subprocess, "Popen", lambda *args, **kwargs: old_player)
    app.play()
    app.timeline.set_position(30)
    second, _ = saved_audio_result(tmp_path / "second")
    app._add_result(tmp_path / "second", second)
    assert old_player.stopped and app.playback_process is None
    assert app.timeline._position == 0


def test_clearing_audio_selection_stops_player_and_clears_timeline(app, monkeypatch, tmp_path):
    result, _ = saved_audio_result(tmp_path / "source")
    app._add_result(tmp_path / "source", result)
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    process = FakePlaybackProcess()
    monkeypatch.setattr(gui.subprocess, "Popen", lambda *args, **kwargs: process)
    app.play()
    app.history.selection_remove(*app.history.selection())
    app.result_selected()
    assert process.stopped and app.playback_process is None
    assert app.timeline._duration == 0


def test_repeat_checkbox_changes_active_playback_mode_at_current_position(app, monkeypatch, tmp_path):
    result, metadata = saved_audio_result(tmp_path / "loop")
    metadata["loop"] = {"method": "smart_region", "start_sample": 10 * 44100, "end_sample": 60 * 44100}
    app._add_result(tmp_path / "loop", result)
    calls = []
    processes = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")

    def popen(command, **kwargs):
        calls.append(command)
        processes.append(FakePlaybackProcess())
        return processes[-1]

    monkeypatch.setattr(gui.subprocess, "Popen", popen)
    app.play()
    app.timeline.set_position(35)
    app.loop.set(False)
    app._loop_mode_changed()
    assert processes[0].stopped
    assert "--loop" in calls[0] and "--loop" not in calls[1]
    assert calls[1][calls[1].index("--start-sample") + 1] == str(35 * 44100)


def test_generation_failure_does_not_leave_old_audio_selected(app, monkeypatch, tmp_path):
    previous, _ = saved_audio_result(tmp_path / "previous")
    app._add_result(tmp_path / "previous", previous)
    monkeypatch.setattr(gui.subprocess, "Popen", lambda *args, **kwargs: FakePlaybackProcess())
    run = tmp_path / "new-loop"
    AudioTestApp._start_process(app, ["python"], run, "loop", request={})
    assert app.selected_result() is None and app.timeline._duration == 0
    (run / "result.json").write_text('{"ok":false,"error":"AI failed"}', encoding="utf-8")
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: errors.append(args))
    app._finish_process(1)
    assert errors and app.selected_result() is None
    assert len(app.results) == 1


def test_selecting_ace_model_shows_human_label_and_requests_gemma_metadata(app):
    app.model_choice.set(gui.MODEL_SPECS["ace15_turbo"]["label"])
    app._model_selected()
    assert app.model.get() == "ace15_turbo"
    assert app.prompt_options()["music_backend"] == "ace_step15"
    assert "インスト固定" in app.generation_hint.get()
    assert app.ace_controls.grid_info()


def test_ace_gemma_result_applies_caption_bpm_key_and_meter(app):
    app.model.set("ace15_turbo")
    app._model_changed()
    app.draft_ai()
    app.events.put(("prompt", app.prompt_revision, {
        "prompt": "Instrumental background music. Genre: Quirky Pop. Instruments: Guitar, Bass, Drums.",
        "scene_interpretation": "現代の日常喜劇として、軽快なポップで場面を支えます。",
        "ace_metadata": {"bpm": 115, "keyscale": "Bb major", "timesignature": "4"},
    }, None))
    app.poll()
    assert app.ace_bpm.get() == "115"
    assert app.tempo.get() == "0" and app.prompt_options()["tempo"] == 0
    assert app.ace_keyscale.get() == "Bb major"
    assert app.ace_timesignature.get() == "4"
    assert app.prompt_source.get() == "LLM作成"


def test_changing_music_backend_discards_old_gemma_reply(app):
    app.draft_ai()
    revision = app.prompt_revision
    app.model_choice.set(gui.MODEL_SPECS["ace15_turbo"]["label"])
    app._model_selected()
    app.events.put(("prompt", revision, {"prompt": "Old music prompt", "scene_interpretation": "古い指定"}, None))
    app.poll()
    assert app.prompt.get("1.0", "end").strip() == gui.DEFAULT_PROMPT
    assert app.tempo.get() == "0"


def test_ace_generation_forwards_visible_metadata_and_mp3_settings(app, monkeypatch):
    app.model.set("ace15_turbo")
    app.ace_bpm.set("120")
    app.ace_keyscale.set("G major")
    app.ace_timesignature.set("4")
    captured = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: captured.append(kwargs["request"]))
    app.generate()
    request = captured[0]
    assert request["model"] == "ace15_turbo"
    assert (request["bpm"], request["keyscale"], request["timesignature"]) == (120, "G major", "4")
    assert request["output_format"] == "mp3" and request["keep_wav"] is False
    assert "lyrics" not in request and "audio_codes" not in request
    assert request["ace_planner"] is True
    assert request["planner_model_path"] == app.ace_planner_path.get()
    assert request["ace_continuous"] is True


def test_continuity_toggle_preserves_music_settings_for_comparison(app, monkeypatch):
    app.model.set("ace15_turbo")
    app.seed.set("42")
    captured = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: captured.append(kwargs["request"]))
    app.ace_continuous.set(True)
    app.generate()
    app.ace_continuous.set(False)
    app.generate()
    assert captured[0]["ace_continuous"] is True and captured[1]["ace_continuous"] is False
    assert {key: value for key, value in captured[0].items() if key != "ace_continuous"} == {
        key: value for key, value in captured[1].items() if key != "ace_continuous"
    }


def test_result_displays_measured_long_silence_without_changing_audio_duration(app, tmp_path):
    result, metadata = saved_audio_result(tmp_path / "silence")
    metadata["silence_analysis"] = {"count": 6, "total_seconds": 42.5, "longest_seconds": 8.0}
    app._add_result(tmp_path / "silence", result)
    assert "ほぼ無音 6区間・計42.5秒" in app.metrics.get()
    assert "最長8.0秒" in app.metrics.get()
    assert app.timeline._duration == metadata["duration_seconds"]


def test_planner_toggle_keeps_gemma_caption_metadata_and_seed_for_comparison(app, monkeypatch):
    app.model.set("ace15_turbo")
    app._replace_prompt("Instrumental. Genre: Funk. Instruments: Slap Bass, Drums.", source="llm")
    app.ace_bpm.set("132")
    app.ace_keyscale.set("C major")
    app.ace_timesignature.set("4")
    app.seed.set("42")
    captured = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: captured.append(kwargs["request"]))
    app.ace_planner_enabled.set(True)
    app._model_changed()
    app.generate()
    assert "専用planner" in app.generation_hint.get()
    app.ace_planner_enabled.set(False)
    app._model_changed()
    app.generate()
    assert "plannerなし" in app.generation_hint.get()
    assert captured[0]["ace_planner"] is True and captured[1]["ace_planner"] is False
    assert {key: value for key, value in captured[0].items() if key != "ace_planner"} == {
        key: value for key, value in captured[1].items() if key != "ace_planner"
    }


def test_stable_audio_does_not_receive_hidden_ace_planner_options(app, monkeypatch):
    app.ace_planner_enabled.set(True)
    captured = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: captured.append(kwargs["request"]))
    app.generate()
    assert "ace_planner" not in captured[0] and "planner_model_path" not in captured[0]
    assert "ace_continuous" not in captured[0]
    assert not app.ace_planner_controls.grid_info()


def test_planner_preparation_targets_separate_lm_directory(app, monkeypatch):
    captured = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: captured.append((args, kwargs)))
    app.prepare_planner()
    command, _, kind = captured[0][0]
    assert kind == "prepare" and "ace_planner_prepare.py" in command[4]
    assert command[command.index("--output-dir") + 1] == app.ace_planner_path.get()
    assert "--model" not in command


def test_ace_quality_control_keeps_selected_model(app):
    app.model.set("ace15_turbo")
    app.draft_quality()
    assert app.model.get() == "ace15_turbo" and app.steps.get() == "8"


def test_added_model_path_does_not_overlap_output_settings_row(app):
    settings_page = app.tabs.nametowidget(app.tabs.tabs()[3])
    rows = {}
    for child in settings_page.content.winfo_children():
        if isinstance(child, tkinter.ttk.Entry):
            rows[str(child.cget("textvariable"))] = int(child.grid_info()["row"])
    assert rows[str(app.paths["ace15_turbo"])] != rows[str(app.output)]
    assert rows[str(app.ace_planner_path)] != rows[str(app.output)]


def test_48000_hz_ace_audio_uses_seconds_for_seek_display(app, tmp_path):
    result, metadata = saved_audio_result(tmp_path / "ace")
    metadata.update({"sample_rate": 48000, "sample_count": 30 * 48000, "duration_seconds": 30,
                     "settings": {"model": "ace15_turbo"}})
    app._add_result(tmp_path / "ace", result)
    assert app.timeline._duration == 30


@pytest.mark.parametrize("meter,display", [("2", "2/4"), ("3", "3/4"), ("4", "4/4"), ("6", "6/8")])
def test_switching_gemma_caption_to_medium_keeps_visible_music_metadata(app, meter, display):
    app.model.set("ace15_turbo")
    app._replace_prompt("Instrumental background music. Genre: Pop. Instruments: Guitar, Bass.",
                        source="llm", interpretation="軽快な日常場面です。")
    app.ace_bpm.set("115")
    app.ace_keyscale.set("Eb major")
    app.ace_timesignature.set(meter)
    app.model_choice.set(gui.MODEL_SPECS["medium"]["label"])
    app._model_selected()
    prompt = app.prompt.get("1.0", "end").strip()
    assert "BPM: 115." in prompt and "Key: Eb major." in prompt
    assert f"Time signature: {display}." in prompt and "VocalType: Instrumental" in prompt
    app.model_choice.set(gui.MODEL_SPECS["ace15_turbo"]["label"])
    app._model_selected()
    assert "VocalType:" not in app.prompt.get("1.0", "end")


def test_ace_gui_preparation_uses_bfloat16_without_changing_stable_defaults(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda command, *args, **kwargs: calls.append(command))
    app.model.set("ace15_turbo")
    app.prepare_model()
    assert calls[-1][calls[-1].index("--dtype") + 1] == "bfloat16"
    app.model.set("medium")
    app.prepare_model()
    assert "--dtype" not in calls[-1]


def test_repeated_model_comparison_replaces_music_metadata_without_growing_caption(app):
    original = "Instrumental background music. Genre: Pop. Instruments: Guitar, Bass."
    app.model.set("ace15_turbo")
    app._replace_prompt(original, source="llm", interpretation="明るい場面です。")
    for bpm, key, meter in [("115", "Eb major", "4"), ("130", "C major", "6"), ("90", "A minor", "3")]:
        app.ace_bpm.set(bpm)
        app.ace_keyscale.set(key)
        app.ace_timesignature.set(meter)
        app.model_choice.set(gui.MODEL_SPECS["medium"]["label"])
        app._model_selected()
        caption = app.prompt.get("1.0", "end").strip()
        assert caption.count("BPM:") == caption.count("Key:") == caption.count("Time signature:") == 1
        assert f"BPM: {bpm}." in caption and f"Key: {key}." in caption
        app.model_choice.set(gui.MODEL_SPECS["ace15_turbo"]["label"])
        app._model_selected()
        assert app.prompt.get("1.0", "end").strip() == original
        assert app.scene_interpretation.get() == "明るい場面です。"


@pytest.mark.parametrize("requested,returned", [("0", 112), ("140", 140)])
def test_ace_scene_change_keeps_requested_tempo_and_discards_previous_generated_metadata(app, monkeypatch, requested, returned):
    app.model.set("ace15_turbo")
    app.tempo.set(requested)
    calls = []
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: calls.append(kwargs["request"]))
    app.draft_ai()
    app.events.put(("prompt", app.prompt_revision, {
        "prompt": "Instrumental. Genre: Pop. Instruments: Guitar, Bass, Drums.",
        "scene_interpretation": "活発なやりとりを軽快なポップで支えます。",
        "ace_metadata": {"bpm": returned, "keyscale": "C major", "timesignature": "4"},
    }, None))
    app.poll()
    assert app.tempo.get() == requested and app.ace_bpm.get() == str(returned)
    app.scenes.append(Scene("s2", "危険な追跡", {"plan": {"atmosphere": "緊迫した追跡"}}, "迫る追跡者"))
    app.scene_box["values"] = [scene.label for scene in app.scenes]
    app.scene_box.current(1)
    app.scene_changed()
    assert app.ace_bpm.get() == "0" and app.ace_keyscale.get() == app.ace_timesignature.get() == ""
    app.draft_ai()
    assert calls[-1]["options"]["tempo"] == int(requested)
    assert calls[-1]["scene"]["id"] == "s2"


def test_manual_ace_caption_can_use_explicit_requested_tempo(app, monkeypatch):
    app.model.set("ace15_turbo")
    app.tempo.set("150")
    captured = []
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    monkeypatch.setattr(app, "_start_process", lambda *args, **kwargs: captured.append(kwargs["request"]))
    app.generate()
    assert captured[-1]["bpm"] == 150


@pytest.mark.parametrize("title", list(gui.QUALITY_METADATA))
def test_contrast_quality_presets_apply_matching_native_metadata_without_locking_next_scene(app, title):
    app.model.set("ace15_turbo")
    app.quality_preset.set(title)
    app.draft_quality()
    expected = gui.QUALITY_METADATA[title]
    assert int(app.ace_bpm.get()) == expected["bpm"]
    assert app.ace_keyscale.get() == expected["keyscale"]
    assert app.ace_timesignature.get() == expected["timesignature"]
    assert str(expected["bpm"]) in app.prompt.get("1.0", "end")
    assert expected["keyscale"] in app.prompt.get("1.0", "end")
    assert app.prompt_options()["tempo"] == 0


def test_quality_control_does_not_inherit_requested_scene_tempo(app):
    app.model.set("ace15_turbo")
    app.tempo.set("160")
    app.quality_preset.set(next(iter(gui.QUALITY_PRESETS)))
    app.draft_quality()
    assert app._ace_generation_bpm() == "0"
    assert app.tempo.get() == "160"
