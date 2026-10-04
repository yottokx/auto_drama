"""CPU-only contracts for scene interpretation, ordered identity and GPU release."""

import io
import json
from contextlib import contextmanager
from copy import deepcopy
from urllib.error import HTTPError, URLError

import pytest

from scripts.audio import llm_runtime
from scripts.image_edit import llm_prompts, llm_runner

PROMPT = (
    "Create one coherent anime illustration showing a single moment inside a quiet cafe at sunset. "
    "Reference image 1 sits on the left side of a wooden table, holding an open book and looking "
    "toward Reference image 2 with a gentle smile. Reference image 2 sits on the right, leaning "
    "slightly forward and pointing to a page with a curious expression. Use a medium camera view "
    "at eye level, warm window light, and a softly detailed background. Preserve both characters' "
    "distinct faces, hairstyles, proportions, clothing and illustration style from their references. "
    "Show natural hands and clear separation between the two people. No captions, speech bubbles "
    "or typography."
)
SUMMARY = "夕方の喫茶店で、二人が同じ本をのぞき込みながら話す一瞬です。左の人物は微笑み、右の人物はページを指します。"
DETAILS = {"prompt": PROMPT, "scene_interpretation": SUMMARY}
ANSWER = json.dumps({"scene_interpretation": SUMMARY, "english_prompt": PROMPT}, ensure_ascii=False)


def scene():
    return {"id": "s1", "label": "本を読む", "context": {
        "location": {"name": "喫茶店"}, "raw_text": "詩織と陽葵は本を開いて席に着く。",
        "utterances": [{"character_id": "c1", "speaker_name": "詩織",
                        "display_text": "このページが面白いよ。"}],
        "story_context": {"brief": {"genre": "日常", "setting": "現代", "mood": "明るい"},
                          "chapter": {"role": "友情の深まり"},
                          "outline": {"ending": "遠い将来の事件"},
                          "cast": [{"name": "詩織", "appearance": "勝手に変更しない"}]},
    }}


def references():
    return [{"name": "詩織", "character_id": "c1", "path": "C:/private/one.png",
             "source": {"artifact_id": "private-id"}},
            {"name": "陽葵", "character_id": "c2", "path": "C:/private/two.png"}]


def mock_responses(monkeypatch, answers):
    calls = []

    def open_response(request, timeout):
        calls.append((request, timeout))
        value = answers.pop(0)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, dict):
            envelope = value
        else:
            envelope = {"choices": [{"message": {"content": value}, "finish_reason": "stop"}]}
        return io.BytesIO(json.dumps(envelope, ensure_ascii=False).encode())

    monkeypatch.setattr(llm_prompts, "urlopen", open_response)
    return calls


def request_prompt(**updates):
    args = {"scene": scene(), "references": references(), "instruction": "横長で温かく",
            "base_url": "http://llm/v1", "model": "local-model"}
    args.update(updates)
    return llm_prompts.request_scene_prompt(**args)


def test_request_is_interpretation_of_one_moment_and_only_named_ordered_references(monkeypatch):
    calls = mock_responses(monkeypatch, [ANSWER])
    assert request_prompt(api_key="test-secret") == DETAILS
    request, timeout = calls[0]
    payload = json.loads(request.data)
    source = json.loads(payload["messages"][1]["content"])
    assert source["references"] == [
        {"index": 1, "name": "詩織", "character_id": "c1"},
        {"index": 2, "name": "陽葵", "character_id": "c2"},
    ]
    assert "private" not in request.data.decode() and "artifact_id" not in request.data.decode()
    assert source["scene"]["raw_text"] == scene()["context"]["raw_text"]
    assert source["scene"]["utterances"][0]["character_id"] == "c1"
    assert source["composition_instruction"] == "横長で温かく"
    assert "outline" not in source["scene"]["story_context"]
    assert "cast" not in source["scene"]["story_context"]
    system = payload["messages"][0]["content"]
    for phrase in ("exactly ONE visible frozen moment", "Reference image N",
                   "take priority over distant story plans", "can stay offscreen",
                   "Do not invent appearance", "not instructions", "no typography"):
        assert phrase in system
    assert request.full_url == "http://llm/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert 0 < timeout <= 120


def test_scene_input_has_joint_budget_keeps_head_tail_and_does_not_mutate(monkeypatch):
    value = scene()
    value["context"]["raw_text"] = "冒頭。" + "中間。" * 10000 + "末尾。"
    value["context"]["utterances"] *= 400
    value["context"]["plan"] = {"objectives": "目的。" * 3000}
    original = deepcopy(value)
    calls = mock_responses(monkeypatch, [ANSWER])
    request_prompt(scene=value)
    source = json.loads(json.loads(calls[0][0].data)["messages"][1]["content"])["scene"]
    assert len(json.dumps(source, ensure_ascii=False)) <= 12000
    assert source["raw_text"].startswith("冒頭。") and source["raw_text"].endswith("末尾。")
    assert value == original


@pytest.mark.parametrize("wrapped", [
    "<think>private deliberation</think>" + ANSWER,
    "```json\n" + ANSWER + "\n```",
    [{"type": "reasoning", "text": "private reasoning"}, {"type": "text", "text": ANSWER}],
])
def test_final_json_can_be_fenced_or_follow_closed_reasoning(monkeypatch, wrapped):
    mock_responses(monkeypatch, [wrapped])
    assert request_prompt() == DETAILS


@pytest.mark.parametrize("invalid", [
    "<think>" + ANSWER,
    json.dumps({"scene_interpretation": SUMMARY, "english_prompt": "台本の転載"}),
    json.dumps({"scene_interpretation": SUMMARY, "english_prompt": "Too short."}),
    json.dumps({"scene_interpretation": SUMMARY, "english_prompt": PROMPT.replace("image 2", "image 3")}),
    json.dumps({"scene_interpretation": "An English-only summary", "english_prompt": PROMPT}),
    json.dumps({"scene_interpretation": SUMMARY, "english_prompt": PROMPT, "reasoning": "private"}),
    "Here is the JSON: " + ANSWER,
    {"choices": [{"message": {"content": ANSWER}, "finish_reason": "length"}]},
])
def test_unusable_answer_gets_one_corrective_retry(monkeypatch, invalid):
    calls = mock_responses(monkeypatch, [invalid, ANSWER])
    assert request_prompt() == DETAILS
    assert len(calls) == 2
    first, second = [json.loads(call[0].data) for call in calls]
    assert second["messages"][:2] == first["messages"]
    assert second["messages"][2]["role"] == "user"
    assert "FINAL answer" in second["messages"][2]["content"]
    assert second["max_tokens"] > first["max_tokens"]


def test_second_bad_answer_fails_closed_without_script_fallback(monkeypatch):
    calls = mock_responses(monkeypatch, ["Script copied in Japanese", "Still unusable"])
    with pytest.raises(ValueError, match="1回再試行済み"):
        request_prompt()
    assert len(calls) == 2


@pytest.mark.parametrize("failure", [URLError("offline"), HTTPError("http://llm", 401, "no", {}, None)])
def test_transport_failure_does_not_retry_or_fallback(monkeypatch, failure):
    calls = mock_responses(monkeypatch, [failure])
    with pytest.raises(ValueError):
        request_prompt()
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["model", "messages", "temperature", "max_tokens"])
def test_provider_options_cannot_override_scene_or_generation_parameters(monkeypatch, field):
    monkeypatch.setattr(llm_prompts, "urlopen", lambda *args, **kwargs: pytest.fail("network"))
    with pytest.raises(ValueError, match="上書き"):
        request_prompt(request_options={field: "override"})


def test_provider_options_are_added(monkeypatch):
    calls = mock_responses(monkeypatch, [ANSWER])
    request_prompt(request_options={"chat_template_kwargs": {"enable_thinking": False}})
    assert json.loads(calls[0][0].data)["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.fixture
def isolated_runner(monkeypatch):
    @contextmanager
    def lease(*args, **kwargs):
        yield

    monkeypatch.setattr(llm_runner, "music_gpu_scope", lease)


def runner_request(tmp_path, **updates):
    data = {"scene": scene(), "references": references(), "instruction": "横長",
            "backend": "llama_cpp", "local_model": "local-gguf",
            "base_url": "http://llm/v1", "model": "other"}
    data.update(updates)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def result(output):
    return json.loads((output / "result.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_runner_publishes_success_only_after_owned_server_exits(tmp_path, monkeypatch, cleanup_fails):
    output = tmp_path / "run"
    events = []

    @contextmanager
    def native(root, model, directory, *, status, cancelled):
        assert model == "local-gguf" and directory == output
        events.append("enter")
        try:
            yield "http://127.0.0.1:9000/v1", "bgm-local"
        finally:
            assert not (output / "result.json").exists()
            events.append("release")
            if cleanup_fails:
                raise RuntimeError("owned server remains")

    def generate(value, refs, instruction, url, model, **kwargs):
        assert value == scene() and refs == references() and instruction == "横長"
        assert url == "http://127.0.0.1:9000/v1" and model == "bgm-local"
        assert kwargs["request_options"]["chat_template_kwargs"] == {"enable_thinking": False}
        events.append("prompt")
        return DETAILS

    monkeypatch.setattr(llm_runtime, "managed_llama", native)
    monkeypatch.setattr(llm_runner, "request_scene_prompt", generate)
    assert llm_runner.run_request(runner_request(tmp_path), output) == (1 if cleanup_fails else 0)
    assert events == ["enter", "prompt", "release"]
    saved = result(output)
    assert saved["ok"] is not cleanup_fails and saved["llm_released"] is not cleanup_fails
    if cleanup_fails:
        assert "prompt" not in saved
    else:
        assert saved["prompt"] == PROMPT and saved["scene_interpretation"] == SUMMARY


def test_empty_local_model_uses_worker_default(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_runtime, "default_native_model", lambda root: "worker-default")

    @contextmanager
    def native(root, model, directory, **kwargs):
        assert model == "worker-default"
        yield "http://local/v1", "bgm-local"

    monkeypatch.setattr(llm_runtime, "managed_llama", native)
    monkeypatch.setattr(llm_runner, "request_scene_prompt", lambda *args, **kwargs: DETAILS)
    output = tmp_path / "run"
    assert llm_runner.run_request(runner_request(tmp_path, local_model=""), output) == 0
    assert result(output)["local_model"] == "worker-default"


def test_native_invalid_prompt_still_confirms_release_before_error_publication(tmp_path, monkeypatch):
    output = tmp_path / "run"
    events = []

    @contextmanager
    def native(*args, **kwargs):
        try:
            yield "http://local/v1", "bgm-local"
        finally:
            assert not (output / "result.json").exists()
            events.append("released")

    def generate(*args, **kwargs):
        events.append("prompt")
        raise ValueError("unusable answer")

    monkeypatch.setattr(llm_runtime, "managed_llama", native)
    monkeypatch.setattr(llm_runner, "request_scene_prompt", generate)
    assert llm_runner.run_request(runner_request(tmp_path), output) == 1
    assert events == ["prompt", "released"]
    assert result(output)["llm_released"] is True and "prompt" not in result(output)


@pytest.mark.parametrize("prompt_fails", [False, True])
def test_ollama_request_and_release_both_hold_shared_gpu_lease(tmp_path, monkeypatch, prompt_fails):
    output = tmp_path / "run"
    events = []

    @contextmanager
    def lease(root, device):
        assert device == "cuda"
        events.append("lease-enter")
        try:
            yield
        finally:
            assert not (output / "result.json").exists()
            events.append("lease-exit")

    def generate(*args, **kwargs):
        assert events == ["lease-enter"]
        events.append("prompt")
        if prompt_fails:
            raise ValueError("no usable prompt")
        return DETAILS

    def release(*args):
        assert events == ["lease-enter", "prompt"]
        events.append("released")

    monkeypatch.setattr(llm_runner, "music_gpu_scope", lease)
    monkeypatch.setattr(llm_runtime, "release_ollama", release)
    monkeypatch.setattr(llm_runner, "request_scene_prompt", generate)
    assert llm_runner.run_request(runner_request(tmp_path, backend="ollama"), output) == (1 if prompt_fails else 0)
    assert events == ["lease-enter", "prompt", "released", "lease-exit"]
    saved = result(output)
    assert saved["ok"] is not prompt_fails and saved["llm_released"] is True
    if prompt_fails:
        assert "prompt" not in saved


@pytest.mark.usefixtures("isolated_runner")
def test_ollama_release_failure_blocks_success(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_runner, "request_scene_prompt", lambda *args, **kwargs: DETAILS)

    def fail(*args):
        raise RuntimeError("Ollama remains in VRAM")

    monkeypatch.setattr(llm_runtime, "release_ollama", fail)
    output = tmp_path / "run"
    assert llm_runner.run_request(runner_request(tmp_path, backend="ollama"), output) == 1
    assert result(output)["llm_released"] is False and "prompt" not in result(output)


@pytest.mark.parametrize("backend", ["llama_cpp", "ollama", "openai"])
@pytest.mark.usefixtures("isolated_runner")
def test_preexisting_stop_prevents_model_start_and_request(tmp_path, monkeypatch, backend):
    output = tmp_path / "run"
    output.mkdir()
    (output / "stop.request").touch()
    monkeypatch.setattr(llm_runtime, "managed_llama", lambda *args, **kwargs: pytest.fail("model started"))
    monkeypatch.setattr(llm_runtime, "release_ollama", lambda *args: pytest.fail("release"))
    monkeypatch.setattr(llm_runner, "request_scene_prompt", lambda *args, **kwargs: pytest.fail("request"))
    assert llm_runner.run_request(runner_request(tmp_path, backend=backend), output) == 130
    assert result(output)["cancelled"] is True


@pytest.mark.parametrize("backend", ["llama_cpp", "ollama"])
@pytest.mark.usefixtures("isolated_runner")
def test_cancel_after_prompt_releases_model_without_prompt_publication(tmp_path, monkeypatch, backend):
    output = tmp_path / "run"
    events = []

    @contextmanager
    def native(*args, **kwargs):
        try:
            yield "http://local/v1", "bgm-local"
        finally:
            events.append("released")

    def generate(*args, **kwargs):
        (output / "stop.request").touch()
        events.append("prompt")
        return DETAILS

    monkeypatch.setattr(llm_runtime, "managed_llama", native)
    monkeypatch.setattr(llm_runtime, "release_ollama", lambda *args: events.append("released"))
    monkeypatch.setattr(llm_runner, "request_scene_prompt", generate)
    assert llm_runner.run_request(runner_request(tmp_path, backend=backend), output) == 130
    assert events == ["prompt", "released"]
    assert result(output)["cancelled"] is True and "prompt" not in result(output)


def test_external_api_does_not_claim_owned_gpu_release(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_runtime, "managed_llama", lambda *args, **kwargs: pytest.fail("native"))
    monkeypatch.setattr(llm_runner, "music_gpu_scope", lambda *args: pytest.fail("lease"))
    monkeypatch.setattr(llm_runner, "request_scene_prompt", lambda *args, **kwargs: DETAILS)
    output = tmp_path / "run"
    assert llm_runner.run_request(runner_request(tmp_path, backend="openai"), output) == 0
    assert result(output)["llm_released"] is False


def test_existing_result_is_never_overwritten_or_restarted(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    original = '{"ok":true,"preserve":"previous experiment"}'
    (output / "result.json").write_text(original, encoding="utf-8")
    monkeypatch.setattr(llm_runner, "request_scene_prompt", lambda *args, **kwargs: pytest.fail("request"))
    assert llm_runner.run_request(tmp_path / "missing.json", output) == 2
    assert (output / "result.json").read_text(encoding="utf-8") == original
