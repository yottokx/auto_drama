"""Keep model ownership separate from STEP3's per-job LLM state and deadlines."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.worker.generation import llm as llm_module
from services.worker.generation import llm_session
from services.worker.generation.cancellation import GenerationCancelled, cancellation_scope
from services.worker.generation.llm import LocalLLM
from services.worker.generation.pipeline import ROOT, load_config


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    model, server = tmp_path / "model.gguf", tmp_path / "server.exe"
    model.write_bytes(b"GGUF")
    server.touch()
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"model": {"relative_path": str(model), "size_bytes": 4},
                                "server": {"executable": str(server)}}), encoding="utf-8")
    config = load_config()
    config["llm_config"] = str(base)
    config["llm"].update(total_timeout_seconds=10, max_context_size=32768,
                         model_context_size=32768)
    probe = SimpleNamespace(config=config, now=0.0, processes=[], starts=[], events=[], requests=[], held=False)
    clock = SimpleNamespace(monotonic=lambda: probe.now, sleep=lambda _: None)
    monkeypatch.setattr(llm_module, "time", clock)
    monkeypatch.setattr(llm_session, "time", clock)

    class Process:
        def __init__(self):
            self.stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def kill(self):
            self.stopped = True
            probe.events.append("cancel")

    @contextmanager
    def owner(command, _log, **options):
        process = Process()
        probe.starts.append((command, options))
        probe.processes.append(process)
        probe.events.append("start")
        try:
            yield process
        finally:
            process.stopped = True
            probe.events.append("stop")

    class HTTP:
        def open(self, request, timeout):
            path = request.full_url.rsplit("/", 1)[-1]
            if path == "health":
                result = {"status": "ok"}
            elif path == "props":
                result = {"default_generation_settings": {"n_ctx": 32768}}
            elif path == "apply-template":
                result = {"prompt": "formatted source"}
            elif path == "tokenize":
                result = {"tokens": [1, 2]}
            else:
                assert path == "completions"
                probe.requests.append(json.loads(request.data))
                result = {"choices": [{"finish_reason": "stop", "message": {"content": "accepted"}}]}
            response = BytesIO(json.dumps(result).encode())
            response.status = 200
            return response

    @contextmanager
    def lease():
        assert not probe.held
        probe.held = True
        probe.events.append("acquire")
        try:
            yield
        finally:
            assert all(process.poll() is not None for process in probe.processes)
            probe.events.append("release")
            probe.held = False

    def client(name, *, seed=1, profile=None, config=None):
        instance = LocalLLM(ROOT, config or probe.config,
                            {"seed": seed, "profile": profile or {}}, tmp_path / name)
        instance.http = HTTP()
        return instance

    probe.lease, probe.client, probe.http_factory = lease, client, HTTP
    monkeypatch.setattr(llm_module, "owned_process", owner)
    return probe


def test_character_relationship_and_conversion_clients_reuse_weights_with_separate_job_state(runtime):
    with llm_session.reuse_llm_runtime() as session:
        clients = []
        for number, kind in enumerate(("m2_character", "m2_relationships", "m2_image"), 1):
            llm_session.prepare_job(kind)
            with session.gpu_scope(runtime.lease()), runtime.client(f"job-{number}", seed=number) as client:
                client.chat("source", [{"role": "user", "content": f"character {number}"}])
                assert client.requests == 1
                assert client.deadline == runtime.now + 10
                clients.append(client)
            runtime.now += 4
        assert len(runtime.starts) == 1
        assert len(runtime.requests) == 3
        assert len({client.process for client in clients}) == 1
        assert [request["seed"] for request in runtime.requests] == [2, 3, 4]
        assert all(list(client.output.glob("01-source-*.json")) for client in clients)
        assert all(client.tools is not clients[0].tools for client in clients[1:])
        assert clients[1].trace[0]["type"] == "model_reuse"
        assert runtime.events == ["acquire", "start"]
        assert runtime.starts[0][1]["timeout"] is None
        assert runtime.starts[0][1]["cancellable"] is False
    assert runtime.events == ["acquire", "start", "stop", "release"]
    assert llm_session.current_session() is None


def test_idle_retains_server_until_five_minutes_from_load_without_refresh_on_reuse(runtime):
    with llm_session.reuse_llm_runtime() as session:
        with session.gpu_scope(runtime.lease()), runtime.client("first"):
            pass
        runtime.now = 299
        assert not session.expire_if_needed()
        with session.gpu_scope(runtime.lease()), runtime.client("retake") as client:
            assert client.deadline == 309
            assert client.request("/health")["status"] == "ok"
        assert session._loaded_at == 0
        assert len(runtime.starts) == 1
        runtime.now = 300
        assert session.expire_if_needed()
        assert not runtime.held
        assert runtime.processes[0].poll() is not None
        with session.gpu_scope(runtime.lease()), runtime.client("after-timeout"):
            pass
        assert len(runtime.starts) == 2
        assert session._loaded_at == 300


def test_ttl_finishes_active_inference_and_then_releases_server(runtime):
    with llm_session.reuse_llm_runtime() as session:
        with session.gpu_scope(runtime.lease()), runtime.client("active"):
            runtime.now = 301
            assert not session.expire_if_needed()
            assert runtime.processes[0].poll() is None
        assert session._backend is session._lease is None
        assert not runtime.held


@pytest.mark.parametrize("change", ["context", "reasoning", "model"])
def test_launch_configuration_change_replaces_weights_under_same_gpu_lease(runtime, change):
    with llm_session.reuse_llm_runtime() as session:
        with session.gpu_scope(runtime.lease()), runtime.client("first"):
            pass
        config, profile = copy.deepcopy(runtime.config), {}
        if change == "context":
            profile["context_size"] = 32768
        elif change == "reasoning":
            config["llm"].update(reasoning_template="qwen3")
            profile["reasoning_budget_tokens"] = 2048
        else:
            base = json.loads(Path(config["llm_config"]).read_text())
            model = Path(config["llm_config"]).parent / "another.gguf"
            model.write_bytes(b"GGUF")
            base["model"]["relative_path"] = str(model)
            new_base = model.parent / "another-base.json"
            new_base.write_text(json.dumps(base))
            config["llm_config"] = str(new_base)
        with session.gpu_scope(runtime.lease()), runtime.client("changed", profile=profile, config=config):
            pass
        assert len(runtime.starts) == 2
        assert runtime.processes[0].poll() is not None
        assert runtime.processes[1].poll() is None
        assert runtime.events == ["acquire", "start", "stop", "start"]


def test_sampling_profile_changes_do_not_reload_model(runtime):
    with llm_session.reuse_llm_runtime() as session:
        with session.gpu_scope(runtime.lease()), runtime.client("first"):
            pass
        with session.gpu_scope(runtime.lease()), runtime.client("second", profile={
                "temperature": 0.4, "top_p": 0.8, "max_tokens": 1024}):
            pass
        assert len(runtime.starts) == 1


def test_old_job_cancellation_cannot_kill_reused_model_and_active_cancellation_closes_lease(runtime):
    with llm_session.reuse_llm_runtime() as session:
        with cancellation_scope() as old, session.gpu_scope(runtime.lease()), runtime.client("first"):
            pass
        with (pytest.raises(GenerationCancelled), cancellation_scope() as current,
              session.gpu_scope(runtime.lease()), runtime.client("next") as client):
            old.cancel()
            assert runtime.processes[0].poll() is None
            assert len(runtime.starts) == 1
            current.cancel()
            client.request("/health")
        assert runtime.processes[0].poll() is not None
        assert session._backend is session._lease is None
        assert not runtime.held


def test_other_job_kind_closes_llm_and_restores_ephemeral_lifetime(runtime):
    with llm_session.reuse_llm_runtime() as session:
        with session.gpu_scope(runtime.lease()), runtime.client("first"):
            pass
        llm_session.prepare_job("m3_background")
        assert session._backend is session._lease is None
        assert not session.enabled
        with runtime.lease(), runtime.client("media-conversion"):
            pass
        assert runtime.starts[-1][1]["timeout"] == 10
        assert "cancellable" not in runtime.starts[-1][1]


def test_cancellation_during_reuse_entry_allows_next_job_to_load(runtime, monkeypatch):
    with llm_session.reuse_llm_runtime() as session:
        with session.gpu_scope(runtime.lease()), runtime.client("first"):
            pass
        attach = session.attach
        with pytest.raises(GenerationCancelled), cancellation_scope() as token:
            def attach_then_cancel(client):
                attach(client)
                token.cancel()

            monkeypatch.setattr(session, "attach", attach_then_cancel)
            with session.gpu_scope(runtime.lease()), runtime.client("cancelled-entry"):
                pytest.fail("A cancelled runtime entry must not return a client")
        assert session._active_client is session._backend is session._lease is None
        monkeypatch.setattr(session, "attach", attach)
        with cancellation_scope(), session.gpu_scope(runtime.lease()), runtime.client("next"):
            pass
        assert len(runtime.starts) == 2


def test_failed_generation_closes_resident_server_and_gpu_lease(runtime):
    with llm_session.reuse_llm_runtime() as session:
        with (pytest.raises(ValueError, match="invalid response"),
              session.gpu_scope(runtime.lease()), runtime.client("failed")):
            raise ValueError("invalid response")
        assert session._backend is session._lease is None
        assert not runtime.held


def test_pipeline_text_and_portrait_batch_share_llm_then_release_before_image(runtime, tmp_path, monkeypatch):
    from services.worker.generation import pipeline
    from tests.unit.test_image_prompt_retry import _job

    monkeypatch.setattr(pipeline, "load_config", lambda: copy.deepcopy(runtime.config))
    monkeypatch.setattr(pipeline, "gpu_lock", lambda *_args: runtime.lease())
    monkeypatch.setattr(llm_module.urllib.request, "build_opener", lambda *_args: runtime.http_factory())

    def text(kind, payload, client):
        client.chat(kind, [{"role": "user", "content": payload["character_id"]}])
        return {"id": payload["character_id"]}

    def relationships(_payload, client):
        client.chat("relationships", [{"role": "user", "content": "relationships"}])
        return {"pairs": []}

    def prompt(payload, client):
        client.chat("conversion", [{"role": "user", "content": payload["character_id"]}])
        client.trace.append({"type": "image_prompt", "prompt_version": 1,
                             "visual_features": {}, "rendering_instruction": "",
                             "composition_prompt": "full body"})
        return "A visible traveller, full body"

    def image(*_args):
        assert runtime.held
        assert all(process.poll() is not None for process in runtime.processes)
        runtime.events.append("image")
        return b"image", {}

    monkeypatch.setattr(pipeline, "generate_text", text)
    monkeypatch.setattr(pipeline, "generate_relationships", relationships)
    monkeypatch.setattr(pipeline, "image_prompt", prompt)
    monkeypatch.setattr(pipeline, "generate_image", image)
    first, second = _job("m2_image"), _job("m2_image")
    second["payload"]["character_id"] = "guide"
    second["payload"]["character_result"]["id"] = "guide"
    batch = [{name: item["payload"][name] for name in ("character_id", "character_result")}
             for item in (first, second)]
    first["payload"]["portrait_prompt_batch"] = second["payload"]["portrait_prompt_batch"] = batch
    with llm_session.reuse_llm_runtime() as session:
        for number, identifier in enumerate(("receptionist", "guide")):
            pipeline.generate_job({"kind": "m2_character", "payload": {
                "schema_version": 1, "seed": number, "character_id": identifier}}, tmp_path / f"character-{number}")
        pipeline.generate_job({"kind": "m2_relationships", "payload": {
            "schema_version": 1, "seed": 9}}, tmp_path / "relationships")
        assert len(runtime.starts) == 1
        assert session._lease is not None
        pipeline.generate_job(first, tmp_path / "portrait-first")
        pipeline.generate_job(second, tmp_path / "portrait-second")
        assert len(runtime.starts) == 1
        assert len(runtime.requests) == 5
        assert session._backend is session._lease is None
    assert runtime.events == ["acquire", "start", "stop", "release",
                              "acquire", "image", "release", "acquire", "image", "release"]


def test_long_chapter_keeps_runtime_for_music_then_expires_from_last_completion(runtime):
    with llm_session.reuse_llm_runtime() as session:
        llm_session.prepare_job("m3_narrative")
        with llm_session.gpu_scope("m3_narrative", runtime.lease()), runtime.client("chapter"):
            runtime.now = 620
            assert not session.expire_if_needed()
        assert session._loaded_at == 0 and session._idle_since == 620
        assert runtime.held, "a chapter exceeding 300 seconds must retain its completed reviewer"
        llm_session.prepare_job("m3_music_plan")
        with llm_session.gpu_scope("m3_music_plan", runtime.lease()), runtime.client("music-plan") as client:
            assert client.trace[0]["type"] == "model_reuse"
            runtime.now = 650
        assert len(runtime.starts) == 1 and session._loaded_at == 0
        assert session._idle_since == 650
        runtime.now = 949
        llm_session.prepare_job("m3_music_plan")
        assert not session.expire_if_needed()
        assert session._idle_since == 650, "claiming/polling alone must not extend the idle deadline"
        runtime.now = 950
        assert session.expire_if_needed() and not runtime.held


def test_returning_from_long_m3_to_m2_preserves_original_m2_load_deadline(runtime):
    with llm_session.reuse_llm_runtime():
        llm_session.prepare_job("m3_narrative")
        with llm_session.gpu_scope("m3_narrative", runtime.lease()), runtime.client("chapter"):
            runtime.now = 620
        llm_session.prepare_job("m2_character")
        with llm_session.gpu_scope("m2_character", runtime.lease()), runtime.client("character"):
            pass
        assert len(runtime.starts) == 2
        assert runtime.events == ["acquire", "start", "stop", "release", "acquire", "start"]


@pytest.mark.parametrize("kind", ["m3_background", "m3_image", "m3_voice", "m3_voice_clone", "m3_music"])
def test_worker_preparation_releases_long_m3_llm_before_every_media_kind(runtime, kind):
    from services.worker.__main__ import prepare_media_job

    with llm_session.reuse_llm_runtime() as session:
        llm_session.prepare_job("m3_narrative")
        with llm_session.gpu_scope("m3_narrative", runtime.lease()), runtime.client("chapter"):
            runtime.now = 620
        prepare_media_job(kind)
        assert not session.enabled and session._backend is session._lease is None
        assert runtime.events == ["acquire", "start", "stop", "release"]
        with runtime.lease():
            assert all(process.poll() is not None for process in runtime.processes)


@pytest.fixture
def m3_chain(runtime, tmp_path, monkeypatch):
    from services.worker.generation import m3_pipeline, model_routing, music_pipeline, pipeline

    runtime.config["model_routing"] = {}
    runtime.config["llm"]["total_timeout_seconds"] = 1800
    runtime.config["music"] = {"python": "fake/python.exe"}
    monkeypatch.setattr(pipeline, "load_config", lambda: copy.deepcopy(runtime.config))
    monkeypatch.setattr(m3_pipeline, "gpu_lock", lambda *_args: runtime.lease())
    monkeypatch.setattr(music_pipeline, "gpu_lock", lambda *_args: runtime.lease())
    monkeypatch.setattr(llm_module.urllib.request, "build_opener", lambda *_args: runtime.http_factory())
    monkeypatch.setattr(model_routing, "time", SimpleNamespace(monotonic=lambda: runtime.now))
    from services.worker import model_config

    monkeypatch.setattr(model_config, "select_config", lambda config, *_args: config)
    profile = {"common_settings_version": 1, "model_id": runtime.config["llm"]["model_id"],
               "context_size": 16384, "temperature": .65, "max_tokens": 3072, "reasoning_level": "none"}
    frozen = {"approval_snapshot": {}, "narrative": {"scenes": [{"id": "s1"}]}}

    def narrative(payload, client):
        if hasattr(client, "select_purpose"):
            client.select_purpose("quality_review")
        client.chat("content-consistency", [{"role": "user", "content": "Validate finalized text."}])
        runtime.now = 620
        return frozen["narrative"]

    def plan(payload, client):
        client.chat("music-boundaries", [{"role": "user", "content": "Choose continuous scene music."}])
        client.chat("group-caption", [{"role": "user", "content": "Caption the whole group."}])
        return {"scenes": [{"scene_id": "s1", "prompt": "Instrumental pop.", "interpretation": "明るい場面"}]}

    monkeypatch.setattr(m3_pipeline, "generate_narrative", narrative)
    monkeypatch.setattr(music_pipeline, "plan_music", plan)
    return SimpleNamespace(runtime=runtime, pipeline=m3_pipeline, music=music_pipeline,
        narrative={"kind": "m3_narrative", "payload": {"schema_version": 1, "seed": 10, "profile": profile}},
        plan={"kind": "m3_music_plan", "payload": {"schema_version": 1, "seed": 20, "profile": profile, "context": frozen}},
        work=tmp_path)


@pytest.mark.parametrize("causal", [False, True])
def test_m3_pipeline_long_narrative_and_music_planning_share_one_actual_runtime(m3_chain, causal):
    from zipfile import ZipFile

    chain, probe = m3_chain, m3_chain.runtime
    if causal:
        chain.narrative["payload"]["story_workflow_version"] = 2
    with llm_session.reuse_llm_runtime() as session:
        chain.pipeline.generate_job(chain.narrative, chain.work / "chapter")
        assert session._backend is not None and probe.held
        plan = chain.pipeline.generate_job(chain.plan, chain.work / "music-plan")
        assert len(probe.starts) == 1 and len(probe.requests) == 3
        with ZipFile(BytesIO(plan)) as archive:
            trace = json.loads(archive.read("result.json"))["trace"]
        assert any(row["type"] == "model_reuse" for row in trace)
        assert session._idle_since == 620
    assert probe.events == ["acquire", "start", "stop", "release"]


@pytest.mark.parametrize("failure", ["validation", "cancel"])
def test_failed_or_cancelled_music_plan_closes_reused_chapter_llm_and_lease(m3_chain, monkeypatch, failure):
    chain, probe = m3_chain, m3_chain.runtime
    with llm_session.reuse_llm_runtime() as session:
        chain.pipeline.generate_job(chain.narrative, chain.work / "chapter")
        with cancellation_scope() as token:
            def rejected(payload, client):
                client.chat("music-plan", [{"role": "user", "content": "Create scene plan."}])
                if failure == "cancel":
                    token.cancel()
                    client.request("/health")
                raise ValueError("invalid scene plan")

            monkeypatch.setattr(chain.music, "plan_music", rejected)
            with pytest.raises(GenerationCancelled if failure == "cancel" else ValueError):
                chain.pipeline.generate_job(chain.plan, chain.work / "music-plan")
        assert len(probe.starts) == 1
        assert session._active_client is session._backend is session._lease is None
        assert not probe.held and probe.processes[0].poll() is not None


def test_actual_music_job_acquires_gpu_only_after_retained_planner_is_released(m3_chain, monkeypatch):
    chain, probe = m3_chain, m3_chain.runtime
    monkeypatch.setattr(chain.music, "music_settings", lambda *_args: {"model": "medium"})
    monkeypatch.setattr(chain.music, "model_identity", lambda *_args: "fake-music-model")

    class MusicRuntime:
        def gpu_scope(self, lease):
            return lease

        def run(self, request, **options):
            assert probe.held and all(process.poll() is not None for process in probe.processes)
            probe.events.append("music")
            output = Path(request["output_dir"])
            (output / "music.mp3").write_bytes(b"ID3fake-loop")
            (output / "source.mp3").write_bytes(b"ID3fake-source")
            return {"result": {"scene_id": "s1", "prompt": request["prompt"], "loop_start_seconds": 10,
                "loop_end_seconds": 70, "duration_seconds": 70, "source_duration_seconds": 120,
                "sample_rate": 44100, "quality": {}}, "provenance": {"model_reused": False}}

        def close(self):
            pass

    monkeypatch.setattr(chain.music.music_session, "MusicSession", MusicRuntime)
    with llm_session.reuse_llm_runtime() as session:
        chain.pipeline.generate_job(chain.narrative, chain.work / "chapter")
        chain.pipeline.generate_job(chain.plan, chain.work / "music-plan")
        chain.pipeline.generate_job({"kind": "m3_music", "payload": {"schema_version": 1,
            "seed": 30, "scene_id": "s1", "prompt": "Instrumental pop.", "context": chain.plan["payload"]["context"]}},
            chain.work / "music")
        assert session._backend is session._lease is None
        assert probe.events == ["acquire", "start", "stop", "release", "acquire", "music", "release"]


def test_music_planner_context_change_reloads_under_retained_exclusive_lease(runtime):
    with llm_session.reuse_llm_runtime() as session:
        llm_session.prepare_job("m3_narrative")
        with llm_session.gpu_scope("m3_narrative", runtime.lease()), runtime.client("chapter"):
            runtime.now = 620
        llm_session.prepare_job("m3_music_plan")
        with llm_session.gpu_scope("m3_music_plan", runtime.lease()), runtime.client("larger-planner", profile={"context_size": 32768}):
            assert len(runtime.starts) == 2
            assert runtime.processes[0].poll() is not None
            assert runtime.processes[1].poll() is None
            assert runtime.events == ["acquire", "start", "stop", "start"]
        assert session._loaded_at == 620
