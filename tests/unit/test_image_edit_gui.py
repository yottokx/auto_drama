"""Verify image experiment selection, provenance, stale results and process ownership."""

import io
import subprocess
import sys
import threading
import time
import tkinter

import pytest
from PIL import Image

from scripts.image_edit import gui
from scripts.image_edit.common import read_json, write_json


@pytest.fixture(scope="module")
def tkinter_root():
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    yield root
    tkinter.Tk.destroy(root)


@pytest.fixture
def app(tkinter_root, monkeypatch, tmp_path):
    monkeypatch.setattr(gui, "SETTINGS", tmp_path / "settings.json")
    root = tkinter_root
    instance = gui.ImageEditTestApp(root, auto_refresh=False)
    instance.output.set(str(tmp_path / "experiments"))
    monkeypatch.setattr(instance, "_async", lambda *args: None)
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    yield instance
    for callback in root.tk.call("after", "info"):
        root.tk.call("after", "cancel", callback)
    for child in root.winfo_children():
        child.destroy()


def reference(tmp_path, name="Alice", color="red"):
    path = tmp_path / f"{name}.png"
    Image.new("RGBA", (32, 64), color).save(path)
    return {
        "path": str(path.resolve()),
        "name": name,
        "character_id": name.lower(),
        "source": {"kind": "local", "original_path": str(path)},
    }


def result_run(tmp_path, references, *, seeds=(0, 1), partial=False):
    directory = tmp_path / "scene-test"
    directory.mkdir()
    outputs = []
    for seed in seeds:
        path = directory / f"seed-{seed}.png"
        Image.new("RGBA", (32, 64), (10, 20, 30, 128)).save(path)
        outputs.append({"path": str(path.resolve()), "seed": seed, "alpha_range": [128, 128]})
    write_json(
        directory / "request.json",
        {"mode": "scene", "references": references, "width": 1024, "height": 576, "steps": 40},
    )
    result = {"ok": not partial, "cancelled": partial, "outputs": outputs}
    write_json(directory / "result.json", result)
    return directory, result


def test_import_does_not_load_inference_libraries():
    code = (
        "import sys; from scripts.image_edit import gui; "
        "assert not {'torch', 'diffusers', 'transformers'} & set(sys.modules)"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        cwd=gui.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_reference_order_names_and_duplicates_are_explicit(app, tmp_path):
    alice = reference(tmp_path)
    bob = reference(tmp_path, "Bob", "blue")
    assert app.add_reference("scene", alice)
    assert not app.add_reference("scene", alice)
    assert "すでに" in app.status.get()
    assert app.add_reference("scene", bob)
    app.move_reference("scene", -1)
    assert [row["name"] for row in app.references["scene"]] == ["Bob", "Alice"]
    app.reference_names["scene"].set("Robert")
    app.rename_reference("scene")
    assert app.references["scene"][0]["name"] == "Robert"
    app.remove_reference("scene")
    assert app.references["scene"] == [alice]


def test_catalog_duplicate_identity_survives_new_download_path(app, tmp_path):
    first = reference(tmp_path)
    first["source"]["artifact_id"] = "fixed-artifact"
    second = {**first, "path": str(tmp_path / "other-download.png")}
    assert app.add_reference("portrait", first)
    assert not app.add_reference("portrait", second)
    assert len(app.references["portrait"]) == 1


def test_request_captures_independent_reference_snapshot_and_comparison_seeds(app, tmp_path):
    app.add_reference("scene", reference(tmp_path))
    app.add_reference("scene", reference(tmp_path, "Bob", "blue"))
    app.replace_prompt("scene", "Two friends talking beside a lake.")
    app.three_seeds.set(True)
    request = app.build_request("scene")
    assert request["seeds"] == [0, 1, 2]
    assert (request["width"], request["height"]) == (1024, 576)
    assert request["references"][0]["name"] == "Alice"
    app.references["scene"][0]["source"]["original_path"] = "later-change"
    assert request["references"][0]["source"]["original_path"] != "later-change"
    assert app.references["portrait"] == []


def test_portrait_requires_exactly_one_reference(app, tmp_path):
    app.replace_prompt("portrait", "Smile.")
    with pytest.raises(ValueError, match="1枚"):
        app.build_request("portrait")
    app.add_reference("portrait", reference(tmp_path))
    assert app.build_request("portrait")["transparent"] is True
    app.add_reference("portrait", reference(tmp_path, "Bob", "blue"))
    with pytest.raises(ValueError, match="1枚"):
        app.build_request("portrait")


def test_scene_change_and_reference_edits_preserve_final_prompt(app, monkeypatch, tmp_path):
    app.replace_prompt("scene", "User's final composition.")
    app.scenes = [{"id": "scene-1", "label": "昼の公園", "preview": "二人が話す", "context": {}}]
    app.scene_box["values"] = ["昼の公園"]
    app.scene_box.current(0)
    app.scene_changed()
    app.add_reference("scene", reference(tmp_path))
    assert app.prompts["scene"].get("1.0", "end").strip() == "User's final composition."
    calls = []
    monkeypatch.setattr(app, "_start_process", lambda *args: calls.append(args))
    app.create_scene_prompt()
    assert calls[0][2] == "prompt"
    assert calls[0][3]["scene"]["label"] == "昼の公園"
    assert app.prompts["scene"].get("1.0", "end").strip() == "User's final composition."


def test_old_catalog_and_download_replies_are_discarded(app, tmp_path):
    app.catalog_revision = 3
    app.events.put(("portraits", 2, [{"name": "old"}], None))
    app.events.put(("reference:scene", (2, 0), reference(tmp_path), None))
    app.events.put(("reference:scene", (3, -1), reference(tmp_path), None))
    app.poll()
    assert app.portraits == []
    assert app.references["scene"] == []
    app.events.put(("reference:scene", (3, 0), reference(tmp_path), None))
    app.poll()
    assert len(app.references["scene"]) == 1


def test_api_404_suggests_coordinator_restart(app):
    app.events.put(("portraits", 0, None, "HTTP 404"))
    app.poll()
    assert "再起動" in app.catalog_status.get()


def test_project_refresh_uses_one_new_snapshot_for_both_catalog_calls(app, monkeypatch):
    instances, calls = [], []

    class Catalog:
        def __init__(self, url):
            instances.append(self)

        def portraits(self, project_id):
            calls.append((self, "portraits", project_id))
            return []

        def chapters(self, project_id):
            calls.append((self, "chapters", project_id))
            return []

    monkeypatch.setattr(gui, "CoordinatorImageCatalog", Catalog)
    monkeypatch.setattr(app, "_async", lambda kind, revision, work: work())
    app.projects = [{"id": "p", "title": "Project"}]
    app.project_box["values"] = ["Project"]
    app.project_box.current(0)
    app.project_changed()
    assert [row[1] for row in calls] == ["portraits", "chapters"]
    assert calls[0][0] is calls[1][0] is app.catalog
    app.project_changed()
    assert len(instances) == 2


class OwnedProcess:
    def __init__(self, completed=False):
        self.completed = completed
        self.terminated = self.killed = self.waited = 0

    def poll(self):
        return 130 if self.completed else None

    def terminate(self):
        self.terminated += 1
        self.completed = True

    def kill(self):
        self.killed += 1
        self.completed = True

    def wait(self, timeout=None):
        self.waited += 1
        return 130


def test_stop_signals_then_reaps_only_owned_process(app, tmp_path):
    process = OwnedProcess()
    app.process = process
    app.run_dir = tmp_path
    app.run_kind = "generate"
    app._log_stream = io.BytesIO()
    app.stop()
    assert (tmp_path / "stop.request").exists()
    assert process.terminated == 0
    app.stop_at = time.monotonic() - 6
    app.poll()
    assert process.terminated == process.waited == 1
    assert app.process is None
    assert "停止" in app.status.get()


def test_close_during_setup_keeps_installer_running(app, monkeypatch, tmp_path):
    process = OwnedProcess()
    app.process, app.run_dir, app.run_kind = process, tmp_path, "setup"
    destroyed = []
    monkeypatch.setattr(app.root, "destroy", lambda: destroyed.append(True))
    app.close()
    assert app.close_pending
    assert not destroyed
    assert process.terminated == process.waited == 0
    assert not (tmp_path / "stop.request").exists()
    assert "環境準備" in app.status.get()


def test_start_serializes_operations_and_logs_without_shell(app, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        gui.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)) or OwnedProcess(),
    )
    monkeypatch.setattr(gui, "hub_environment", lambda path, inherited: dict(inherited))
    run_dir = tmp_path / "run with spaces"
    app._start_process(
        ["Python with spaces/python.exe", "runner.py"], run_dir, "generate", {"schema_version": 1}
    )
    try:
        assert calls[0][0][0] == "Python with spaces/python.exe"
        assert calls[0][1].get("shell", False) is False
        assert calls[0][1]["env"]["PYTHONUTF8"] == "1"
        assert read_json(run_dir / "request.json") == {"schema_version": 1}
        with pytest.raises(ValueError, match="現在の処理"):
            app._start_process(["another.exe"], tmp_path / "second", "prepare")
    finally:
        app._log_stream.close()
        app._log_stream = None
        app.process = None


def test_partial_history_uses_frozen_reference_and_preserves_generated_lineage(app, tmp_path):
    alice = reference(tmp_path)
    directory, result = result_run(tmp_path, [alice], seeds=(0,), partial=True)
    frozen = directory / "frozen.png"
    Image.new("RGBA", (32, 64), "blue").save(frozen)
    write_json(
        directory / "generation.json", {"references": [{**alice, "normalized_path": str(frozen)}]}
    )
    app._add_result(directory, result)
    selected = app.selected_result()
    assert selected["request"]["references"][0]["path"] == str(frozen)
    assert "中断" in app.history.item(app.history.selection()[0], "values")[0]
    app.use_result_reference("portrait")
    source = app.references["portrait"][0]["source"]
    assert source["kind"] == "generated"
    assert source["seed"] == 0
    assert source["parents"][0]["path"] == str(frozen)
    assert app.references["portrait"][0]["character_id"] == "alice"


def test_review_is_separate_for_each_output_and_character(app, tmp_path):
    directory, result = result_run(tmp_path, [reference(tmp_path)])
    app._add_result(directory, result)
    rows = app.history.get_children()
    app.history.selection_set(rows[0])
    app.result_selected()
    app.ratings["face"].set("5")
    app.save_review()
    app.review_target_box.current(1)
    app.load_review()
    app.ratings["hair"].set("3")
    app.save_review()
    app.history.selection_set(rows[1])
    app.result_selected()
    app.ratings["face"].set("2")
    app.save_review()
    evaluations = read_json(directory / "review.json")["evaluations"]
    assert evaluations["output_0:overall"]["ratings"]["face"] == 5
    assert evaluations["output_0:reference_1"]["ratings"]["hair"] == 3
    assert evaluations["output_1:overall"]["ratings"]["face"] == 2


def test_transparent_instruction_is_visible_and_saved_only_on_explicit_generate(
    app, monkeypatch, tmp_path
):
    app.add_reference("portrait", reference(tmp_path))
    app.replace_prompt("portrait", "Change only the expression to surprise.")
    app.transparent["portrait"].set(True)
    assert "RGBA" not in app.prompts["portrait"].get("1.0", "end")
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    requests = []
    monkeypatch.setattr(
        app, "_start_process", lambda command, directory, kind, request: requests.append(request)
    )
    app.generate()
    actual = app.prompts["portrait"].get("1.0", "end").strip()
    assert actual == requests[0]["prompt"]
    assert gui.TRANSPARENCY_INSTRUCTION in actual
    app.generate()
    assert app.prompts["portrait"].get("1.0", "end").count(gui.TRANSPARENCY_INSTRUCTION) == 1


def test_local_scene_instruction_can_include_reference_mapping_without_scene(app, tmp_path):
    app.add_reference("scene", reference(tmp_path))
    app.add_reference("scene", reference(tmp_path, "Bob", "blue"))
    app.replace_prompt("scene", "Two friends greet each other.")
    app.create_reference_prompt()
    prompt = app.prompts["scene"].get("1.0", "end")
    assert "Reference image 1: Alice" in prompt
    assert "Reference image 2: Bob" in prompt
    assert "Two friends greet" in prompt


def test_generated_portrait_replaces_existing_baseline_and_keeps_parents(app, tmp_path):
    alice = reference(tmp_path)
    app.add_reference("portrait", alice)
    directory, result = result_run(tmp_path, [alice], seeds=(0,))
    request = read_json(directory / "request.json")
    request["mode"] = "portrait"
    write_json(directory / "request.json", request)
    app._add_result(directory, result)
    app.replace_prompt("portrait", "Next edit.")
    app.use_result_reference("portrait")
    assert len(app.references["portrait"]) == 1
    replacement = app.references["portrait"][0]
    assert replacement["path"] == result["outputs"][0]["path"]
    assert replacement["source"]["parents"][0] == alice
    assert app.prompts["portrait"].get("1.0", "end").strip() == "Next edit."


def test_generated_mapping_tracks_reorder_rename_and_removal_without_changing_body(app, tmp_path):
    for name, color in (("Alice", "red"), ("Bob", "blue"), ("Cathy", "green")):
        app.add_reference("scene", reference(tmp_path, name, color))
    app.replace_prompt("scene", "Alice greets the others; retain the fountain in the background.")
    app.create_reference_prompt()
    widget = app.prompts["scene"]
    custom_body = "\nMy own note: Reference image 2 should stand near the fountain."
    widget.insert("end-1c", custom_body)
    original_body = widget.get("4.0", "end-1c")
    app.move_reference("scene", -1)
    app.move_reference("scene", -1)
    app.reference_names["scene"].set("Carol")
    app.rename_reference("scene")
    app.reference_lists["scene"].selection_set("2")
    app.remove_reference("scene")
    assert widget.get("1.0", "2.0") == "Reference image 1: Alice.\n"
    app.prepare_final_prompt("scene")
    request = app.build_request("scene")
    assert request["prompt"].startswith("Reference image 1: Carol.\nReference image 2: Alice.\n")
    assert "Reference image 3:" not in request["prompt"]
    assert widget.get("3.0", "end-1c") == original_body
    assert request["prompt"] == widget.get("1.0", "end").strip()
    assert [row["name"] for row in request["references"]] == ["Carol", "Alice"]
    assert custom_body.strip() in request["prompt"]


def test_mapping_refresh_preserves_user_prefix_and_user_authored_numbering(app, tmp_path):
    app.add_reference("scene", reference(tmp_path))
    app.replace_prompt("scene", "User body: Reference image 1 belongs in the foreground.")
    app.create_reference_prompt()
    widget = app.prompts["scene"]
    prefix = "User prefix: Reference image 1 uses dramatic lighting.\n"
    widget.insert("1.0", prefix)
    app.reference_names["scene"].set("Alicia")
    app.rename_reference("scene")
    app.prepare_final_prompt("scene")
    actual = widget.get("1.0", "end")
    assert actual.startswith(prefix + "Reference image 1: Alicia.\n")
    assert "User body: Reference image 1 belongs in the foreground." in actual


def test_manual_edit_to_generated_mapping_becomes_user_owned(app, tmp_path):
    app.add_reference("scene", reference(tmp_path))
    app.replace_prompt("scene", "A garden scene.")
    app.create_reference_prompt()
    widget = app.prompts["scene"]
    widget.delete("1.0", "1.end")
    widget.insert("1.0", "My own reference image 1 is the narrator.")
    app.reference_names["scene"].set("Alicia")
    app.rename_reference("scene")
    before = widget.get("1.0", "end")
    app.prepare_final_prompt("scene")
    assert widget.get("1.0", "end") == before


def test_transparency_toggle_off_removes_only_gui_sentence_and_preserves_manual_body(
    app, monkeypatch, tmp_path
):
    app.add_reference("portrait", reference(tmp_path))
    original = "Change the outfit to a coat.\nPreserve the earrings."
    app.replace_prompt("portrait", original)
    monkeypatch.setattr(app, "_runtime_python", lambda: "python.exe")
    requests = []
    monkeypatch.setattr(
        app, "_start_process", lambda command, directory, kind, request: requests.append(request)
    )
    app.generate()
    widget = app.prompts["portrait"]
    widget.insert("end-1c", "\nKeep the embroidered cuffs.")
    app.transparent["portrait"].set(False)
    app.generate()
    assert gui.TRANSPARENCY_INSTRUCTION in requests[0]["prompt"]
    assert gui.TRANSPARENCY_INSTRUCTION not in requests[1]["prompt"]
    assert requests[1]["transparent"] is False
    assert requests[1]["prompt"] == original + "\nKeep the embroidered cuffs."
    assert requests[1]["prompt"] == widget.get("1.0", "end").strip()


def test_user_transparency_instruction_is_retained_when_checkbox_is_off(app, tmp_path):
    app.add_reference("portrait", reference(tmp_path))
    manual = "Smile.\n" + gui.TRANSPARENCY_INSTRUCTION
    app.replace_prompt("portrait", manual)
    app.transparent["portrait"].set(True)
    app.prepare_final_prompt("portrait")
    app.transparent["portrait"].set(False)
    app.prepare_final_prompt("portrait")
    assert app.prompts["portrait"].get("1.0", "end").strip() == manual


def test_outfit_not_applicable_and_art_style_ratings_round_trip(app, tmp_path):
    directory, result = result_run(tmp_path, [reference(tmp_path)], seeds=(0,))
    app._add_result(directory, result)
    app.ratings["clothing"].set("N/A")
    app.ratings["art_style"].set("4")
    app.ratings["anatomy"].set("2")
    app.save_review()
    app.load_review()
    assert app.ratings["clothing"].get() == "N/A"
    assert app.ratings["art_style"].get() == "4"
    assert app.ratings["anatomy"].get() == "2"


def test_network_work_runs_off_ui_thread_and_returns_through_queue(app):
    main_thread = threading.get_ident()
    finished = threading.Event()
    called = []

    def work():
        called.append(threading.get_ident())
        finished.set()
        return [{"id": "p", "title": "Project"}]

    gui.ImageEditTestApp._async(app, "projects", app.catalog_revision, work)
    assert finished.wait(timeout=3)
    # The worker may set the event immediately before publishing its queue item.
    deadline = time.monotonic() + 3
    while app.events.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert called[0] != main_thread
    assert app.projects == []
    app.poll()
    assert app.projects == [{"id": "p", "title": "Project"}]


def test_checker_preview_retains_file_alpha_and_shows_transparency(tmp_path):
    path = tmp_path / "transparent.png"
    Image.new("RGBA", (40, 40), (255, 0, 0, 0)).save(path)
    preview = gui.checker_preview(path, (40, 40))
    assert preview.getpixel((0, 0)) == (238, 238, 238)
    assert preview.getpixel((16, 0)) == (204, 204, 204)
    with Image.open(path) as original:
        assert original.mode == "RGBA"
        assert original.getchannel("A").getextrema() == (0, 0)


def test_invalid_request_does_not_spawn_generation(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_start_process", lambda *args: calls.append(args))
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: errors.append(args))
    app.generate()
    assert calls == []
    assert "1枚" in errors[0][1]


def scene_llm_run(app, monkeypatch, tmp_path):
    """Capture the actual runner boundary without starting a model process."""
    app.add_reference("scene", reference(tmp_path))
    app.scenes = [
        {
            "id": "bridge-1",
            "label": "操縦室",
            "preview": "Alice: 左へ旋回して！\n艦橋に警報が響く。",
            "context": {"script": "台本全文は画像モデルへの指示ではない。"},
        },
        {"id": "bridge-2", "label": "格納庫", "preview": "格納庫へ戻る。", "context": {}},
    ]
    app.scene_box["values"] = [scene["label"] for scene in app.scenes]
    app.scene_box.current(0)
    app.scene_changed()
    app.scene_instruction.set("操縦桿を握る一瞬を描いて")
    app.replace_prompt("scene", "Keep my current composition until the LLM finishes.")
    calls = []

    def start(command, directory, kind, request):
        calls.append((command, directory, kind, request))
        app.run_dir = directory

    monkeypatch.setattr(app, "_start_process", start)
    app.create_scene_prompt()
    assert len(calls) == 1
    return calls[0]


def scene_llm_result(**overrides):
    return {
        "ok": True,
        "prompt": "Alice grips the control stick in a spacecraft cockpit, tense and focused.",
        "scene_interpretation": "警報を受けて操縦桿を握る一瞬。",
        "backend": "llama_cpp",
        "local_model": "gemma-test.gguf",
        "llm_released": True,
        **overrides,
    }


def test_scene_llm_uses_root_runner_and_frozen_script_source(app, monkeypatch, tmp_path):
    command, directory, kind, request = scene_llm_run(app, monkeypatch, tmp_path)
    assert command[0] == app._console_python()
    assert command[0] != app.python.get()
    assert command[1:4] == ["-u", "-X", "utf8"]
    assert command[4].endswith("llm_runner.py")
    assert command[5:] == [
        "--request",
        str(directory / "request.json"),
        "--output-dir",
        str(directory),
    ]
    assert kind == "prompt"
    assert request["scene"] == app.selected_scene()
    assert request["instruction"] == "操縦桿を握る一瞬を描いて"
    assert request["references"][0]["name"] == "Alice"
    assert "台本全文" not in app.prompts["scene"].get("1.0", "end")
    app.references["scene"][0]["source"]["original_path"] = "later-change"
    app.scenes[0]["context"]["script"] = "changed script"
    assert request["references"][0]["source"]["original_path"] != "later-change"
    assert request["scene"]["context"]["script"] != "changed script"
    assert len(app.tabs.tabs()) == 6


def test_scene_llm_completion_uses_visual_prompt_and_records_provenance(app, monkeypatch, tmp_path):
    _, directory, _, _ = scene_llm_run(app, monkeypatch, tmp_path)
    source = app.pending_llm
    result = scene_llm_result()
    app._finish_scene_prompt(0, result)
    final = app.prompts["scene"].get("1.0", "end").strip()
    assert final.startswith("Reference image 1: Alice.")
    assert result["prompt"] in final
    assert "Do not render the prompt" in final
    assert "台本全文" not in final
    assert "左へ旋回して" not in final
    assert app.pending_llm is None
    details = app.scene_prompt_details
    assert details["source"] == source
    assert details["final_prompt"] == final
    assert details["run_dir"] == str(directory.resolve())
    assert details["local_model"] == "gemma-test.gguf"
    assert details["llm_released"] is True
    assert result["scene_interpretation"] in app.scene_interpretation.get()
    request = app.build_request("scene")
    assert request["context"]["prompt_generation"] == {**details, "manually_edited": False}
    request["context"]["prompt_generation"]["source"]["scene"]["id"] = "mutated"
    assert app.scene_prompt_details["source"]["scene"]["id"] == "bridge-1"


@pytest.mark.parametrize("change", ["scene", "reference", "instruction", "manual_prompt"])
def test_scene_llm_stale_completion_preserves_current_input(app, monkeypatch, tmp_path, change):
    scene_llm_run(app, monkeypatch, tmp_path)
    if change == "scene":
        app.scene_box.current(1)
        app.scene_changed()
    elif change == "reference":
        app.reference_names["scene"].set("Alicia")
        app.rename_reference("scene")
    elif change == "instruction":
        app.scene_instruction.set("窓の外を向く")
    else:
        app.prompts["scene"].insert("end-1c", "\nMy revised framing.")
    current = app.prompts["scene"].get("1.0", "end").strip()
    app._finish_scene_prompt(0, scene_llm_result())
    assert app.prompts["scene"].get("1.0", "end").strip() == current
    assert app.scene_prompt_details is None
    assert app.pending_llm is None
    assert "反映しません" in app.status.get()


@pytest.mark.parametrize("reason", ["failed", "cancelled", "stopped"])
def test_scene_llm_failure_or_cancellation_preserves_current_input(
    app, monkeypatch, tmp_path, reason
):
    scene_llm_run(app, monkeypatch, tmp_path)
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: errors.append(args))
    current = app.prompts["scene"].get("1.0", "end").strip()
    code, result = 0, scene_llm_result()
    if reason == "failed":
        code, result = 1, {"ok": False, "error": "LLM unavailable"}
    elif reason == "cancelled":
        result["cancelled"] = True
    else:
        app.stop_at = time.monotonic()
    app._finish_scene_prompt(code, result)
    assert app.prompts["scene"].get("1.0", "end").strip() == current
    assert app.scene_prompt_details is None
    assert app.pending_llm is None
    assert bool(errors) is (reason == "failed")


@pytest.mark.parametrize("backend", ["llama_cpp", "ollama"])
def test_scene_llm_local_gpu_must_be_released_before_applying_prompt(
    app, monkeypatch, tmp_path, backend
):
    scene_llm_run(app, monkeypatch, tmp_path)
    current = app.prompts["scene"].get("1.0", "end").strip()
    app._finish_scene_prompt(0, scene_llm_result(backend=backend, llm_released=False))
    assert app.prompts["scene"].get("1.0", "end").strip() == current
    assert app.scene_prompt_details is None
    assert "GPU解放" in app.status.get()


def test_scene_llm_external_api_does_not_require_local_gpu_release(app, monkeypatch, tmp_path):
    scene_llm_run(app, monkeypatch, tmp_path)
    app._finish_scene_prompt(0, scene_llm_result(backend="openai", llm_released=False))
    assert app.scene_prompt_details["backend"] == "openai"
    assert "control stick" in app.prompts["scene"].get("1.0", "end")


@pytest.mark.parametrize("change", ["scene", "reference", "instruction"])
def test_scene_llm_unchanged_prompt_rejects_changed_source_until_manually_edited(
    app, monkeypatch, tmp_path, change
):
    scene_llm_run(app, monkeypatch, tmp_path)
    app._finish_scene_prompt(0, scene_llm_result())
    if change == "scene":
        app.scene_box.current(1)
        app.scene_changed()
    elif change == "reference":
        app.reference_names["scene"].set("Alicia")
        app.rename_reference("scene")
    else:
        app.scene_instruction.set("Add strong rim lighting.")
    with pytest.raises(ValueError, match="LLMで指示を作り直し"):
        app.build_request("scene")
    with pytest.raises(ValueError, match="LLMで指示を作り直し"):
        app.prepare_final_prompt("scene")
    app.prompts["scene"].insert("end-1c", "\nI have checked this revised composition.")
    app.prepare_final_prompt("scene")
    request = app.build_request("scene")
    assert "I have checked this revised composition." in request["prompt"]
    assert request["context"]["prompt_generation"]["manually_edited"] is True


def test_scene_llm_transparency_decoration_keeps_stale_reference_guard(app, monkeypatch, tmp_path):
    scene_llm_run(app, monkeypatch, tmp_path)
    app.add_reference("scene", reference(tmp_path, "Bob", "blue"))
    # Capture the two-reference source before returning the fake LLM response.
    app.pending_llm = app._scene_prompt_snapshot()
    app._finish_scene_prompt(0, scene_llm_result())
    body = app.scene_prompt_details["prompt_body"]
    app.transparent["scene"].set(True)
    app.prepare_final_prompt("scene")
    assert gui.TRANSPARENCY_INSTRUCTION in app.prompts["scene"].get("1.0", "end")
    assert app._scene_prompt_body() == body
    assert app.build_request("scene")["context"]["prompt_generation"]["manually_edited"] is False
    app.move_reference("scene", -1)
    assert [row["name"] for row in app.references["scene"]] == ["Bob", "Alice"]
    with pytest.raises(ValueError, match="LLMで指示を作り直し"):
        app.prepare_final_prompt("scene")
    with pytest.raises(ValueError, match="LLMで指示を作り直し"):
        app.build_request("scene")
    app.prompts["scene"].insert("end-1c", "\nBob is now the pilot; Alice stands beside him.")
    app.prepare_final_prompt("scene")
    request = app.build_request("scene")
    assert request["prompt"].startswith("Reference image 1: Bob.")
    assert request["context"]["prompt_generation"]["manually_edited"] is True
