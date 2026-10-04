"""Frozen BGM jobs and captions without loading an inference model."""
from __future__ import annotations

import io
import json
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from services.worker.generation import music_pipeline as music

PROMPT = "Genre: Bright modern pop. Instruments: electric piano, guitar and bass. A playful instrumental melody."


def context():
    return {"approval_snapshot": {"world": {"result": {"title": "深夜の喜劇", "genre": "現代コメディ"}}},
        "world": {"title": "深夜の喜劇", "genre": "現代コメディ"}, "chapter_number": 1,
        "overall_plot": {"ending": "魔王が接客を覚えて仲良くなる", "chapters": [{"number": 1,
            "role": "日常と尊大な人物の落差を笑う", "summary": "コンビニで働く魔王"}]},
        "narrative": {"title": "最初の勤務", "chapter_number": 1, "locations": [], "scenes": [
            {"id": "comic", "plan": {"atmosphere": "ギャグ", "location_id": "shop"},
             "raw_text": "魔王がお釣りを渡せず尊大に言い訳する。", "utterances": []},
            {"id": "farewell", "plan": {"atmosphere": "本物の悲しみ"},
             "raw_text": "夜勤仲間との別れ。", "utterances": []}]}}


class FakeLLM:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.messages = []

    def structured(self, stage, messages, schema):
        self.messages.append(json.loads(json.dumps(messages)))
        assert schema["additionalProperties"] is False
        return next(self.answers)


def answer(prompt=PROMPT):
    return {"scene_interpretation": "作品の現代コメディを踏まえ、軽快なポップで場面の笑いを表現します。", "english_prompt": prompt}


def test_caption_batch_keeps_story_identity_and_scene_order():
    llm = FakeLLM([answer(), answer("Genre: Reflective piano. Instruments: piano and cello. A wistful instrumental melody.")])
    result = music.plan_music({"context": context()}, llm)
    assert [row["scene_id"] for row in result["scenes"]] == ["comic", "farewell"]
    assert all(row["prompt"].endswith("TrackType: Music, VocalType: Instrumental.") for row in result["scenes"])
    source = json.loads(llm.messages[0][1]["content"])
    assert source["scene"]["story_context"]["brief"]["genre"] == "現代コメディ"
    assert source["scene"]["story_context"]["outline"]["ending"] == "魔王が接客を覚えて仲良くなる"
    assert source["scene"]["story_context"]["chapter"]["role"] == "日常と尊大な人物の落差を笑う"
    assert "本物の悲しみ" in llm.messages[1][1]["content"]


def test_invalid_music_metadata_gets_one_corrective_retry():
    llm = FakeLLM([answer("An instrumental musical background."), answer()])
    result = music.plan_music({"context": {**context(), "narrative": {
        **context()["narrative"], "scenes": context()["narrative"]["scenes"][:1]}}}, llm)
    assert len(result["scenes"]) == 1
    assert len(llm.messages) == 2
    assert "Correct the final JSON response" in llm.messages[1][-1]["content"]
    llm = FakeLLM([answer("missing labels"), answer("still missing labels")])
    with pytest.raises(ValueError, match="music caption is invalid"):
        music.plan_music({"context": context()}, llm)
    assert len(llm.messages) == 2


def test_explicit_source_prompt_skips_llm_and_adjustment_is_authorized():
    source = PROMPT + " TrackType: Music, VocalType: Instrumental."
    result = music.plan_music({"context": context(), "source_prompt": source}, None)
    assert result["scenes"][0]["prompt"] == source
    llm = FakeLLM([answer(), answer()])
    music.plan_music({"context": context(), "source_prompt": source, "instruction": "明るいファンクへ変更"}, llm)
    assert json.loads(llm.messages[0][1]["content"])["creative_instruction"] == "明るいファンクへ変更"
    assert "top-level creative_instruction is the user's" in llm.messages[0][0]["content"]


def boundaries(*actions):
    return {"scenes": [{"scene_id": scene["id"], "action": action,
                        "transition": {"visual": "dissolve"}, "reason": "同じ対話の流れを保ちます。"}
                       for scene, action in zip(context()["narrative"]["scenes"], actions, strict=True)]}


def test_chapter_continuity_caption_contains_late_group_member_and_preserves_frozen_text():
    frozen = context()
    frozen["narrative"]["scenes"][1]["raw_text"] = "導入" * 3000 + "終盤で真相が明らかになる。"
    before = json.dumps(frozen, ensure_ascii=False, sort_keys=True)
    llm = FakeLLM([boundaries("play", "continue"), answer()])
    result = music.plan_music({"context": frozen, "planning_version": 2}, llm)
    assert len(llm.messages) == 2, "one chapter decision then one caption for the whole continuous group"
    assert result["planning_version"] == 2
    assert result["scenes"][1]["action"] == "continue" and "prompt" not in result["scenes"][1]
    material = json.loads(llm.messages[1][1]["content"])
    assert [row["scene_id"] for row in material["continuity_group"]] == ["comic", "farewell"]
    assert "終盤で真相" in material["continuity_group"][1]["raw_text"]
    assert result["scenes"][0]["transition"]["duration_ms"] == 500
    assert result["scenes"][0]["transition"]["music_fade_out_ms"] == 0
    assert result["scenes"][1]["transition"]["music_fade_out_ms"] == 0
    assert result["scenes"][1]["transition"]["music_fade_in_ms"] == 0
    assert json.dumps(frozen, ensure_ascii=False, sort_keys=True) == before


def test_chapter_stop_has_no_caption_and_invalid_continue_gets_one_retry():
    llm = FakeLLM([boundaries("stop", "continue"), boundaries("play", "stop"), answer()])
    result = music.plan_music({"context": context(), "planning_version": 2}, llm)
    assert len(llm.messages) == 3
    assert "Correct the boundary JSON" in llm.messages[1][-1]["content"]
    assert result["scenes"][1]["action"] == "stop" and "prompt" not in result["scenes"][1]


def test_manual_version_two_never_uses_chapter_continuity_or_changes_explicit_prompt():
    frozen = context()
    frozen["narrative"]["scenes"] = frozen["narrative"]["scenes"][:1]
    source = PROMPT + " TrackType: Music, VocalType: Instrumental."
    result = music.plan_music({"context": frozen, "planning_version": 2, "planning_scope": "single_scene",
                               "source_prompt": source}, None)
    assert result["scenes"][0].get("action", "play") == "play"
    assert result["scenes"][0]["prompt"] == source


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    events, requests = [], []
    config = {"music": {"enabled": True, "python": "runtime/python.exe", "models": {"medium": "model"}},
        "gpu_lock_timeout_seconds": 5, "max_zip_bytes": 1024 * 1024}
    monkeypatch.setattr(music.pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(music.pipeline, "load_config", lambda: config)
    monkeypatch.setattr(music, "model_identity", lambda *args: "frozen-model-v1")

    @contextmanager
    def lease(*args):
        events.append("acquire")
        try:
            yield
        finally:
            events.append("release")

    monkeypatch.setattr(music, "gpu_lock", lease)

    class Session:
        def gpu_scope(self, lock):
            return lock

        def close(self):
            events.append("close")

        def run(self, request, **kwargs):
            requests.append(request)
            folder = music.Path(request["output_dir"])
            (folder / "source.mp3").write_bytes(b"ID3source")
            (folder / "music.mp3").write_bytes(b"ID3loop")
            return {"result": {"scene_id": request["scene_id"], "prompt": request["prompt"],
                "loop_start_seconds": 10, "loop_end_seconds": 70, "duration_seconds": 70,
                "source_duration_seconds": request["duration_seconds"], "sample_rate": 44100,
                "quality": {"needs_review": False}}, "provenance": {"backend": "stable_audio3", "model": "medium"}}

    monkeypatch.setattr(music.music_session, "MusicSession", Session)
    return SimpleNamespace(root=tmp_path, requests=requests, events=events)


def job(seed=2**40 + 7, retry=0):
    return {"kind": "m3_music", "retry_generation": retry, "payload": {"schema_version": 1,
        "seed": seed, "backend": "stable_audio3", "model": "medium", "scene_id": "comic",
        "prompt": PROMPT, "context": context()}}


def unpack(content):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert set(archive.namelist()) == {"result.json", "source.mp3", "music.mp3"}
        return json.loads(archive.read("result.json"))


def test_music_job_seed_mapping_complete_bundle_and_transport_replay(jobs):
    content = music.generate_job(job(), jobs.root / "first")
    report = unpack(content)
    assert jobs.requests[0]["duration_seconds"] == 120
    assert jobs.requests[0]["settings"]["dtype"] == "float32"
    assert jobs.requests[0]["seed"] == report["provenance"]["effective_seed"] == 7
    assert report["provenance"]["job_seed"] == 2**40 + 7
    assert report["provenance"]["retry_generation"] == 0
    assert music.generate_job(job(), jobs.root / "first") == content
    assert len(jobs.requests) == 1
    assert jobs.events == ["acquire", "release", "close"]


def test_manual_retry_changes_seed_but_lease_retry_repeats_it(jobs):
    base = unpack(music.generate_job(job(), jobs.root / "attempt-a"))
    lease_retry = unpack(music.generate_job(job(), jobs.root / "attempt-b"))
    manual = unpack(music.generate_job(job(retry=1), jobs.root / "manual"))
    assert base["provenance"]["effective_seed"] == lease_retry["provenance"]["effective_seed"] == 7
    assert manual["provenance"]["effective_seed"] == 8
    assert manual["provenance"]["retry_generation"] == 1


def test_failed_manual_retry_uses_same_frozen_directory_and_new_seed(jobs, monkeypatch):
    original = music.music_session.MusicSession.run

    def fail_first(self, request, **kwargs):
        if request["seed"] == 7:
            jobs.requests.append(request)
            raise ValueError("Music quality check failed: long_internal_silence")
        return original(self, request, **kwargs)

    monkeypatch.setattr(music.music_session.MusicSession, "run", fail_first)
    with pytest.raises(ValueError, match="quality check failed"):
        music.generate_job(job(), jobs.root / "failed")
    assert not (jobs.root / "failed/result.zip").exists()
    report = unpack(music.generate_job(job(retry=1), jobs.root / "failed"))
    assert [request["seed"] for request in jobs.requests] == [7, 8]
    assert report["provenance"]["effective_seed"] == 8


def test_mismatched_child_result_is_never_cached(jobs, monkeypatch):
    original = music.music_session.MusicSession.run

    def wrong(self, request, **kwargs):
        report = original(self, request, **kwargs)
        report["result"]["scene_id"] = "farewell"
        return report

    monkeypatch.setattr(music.music_session.MusicSession, "run", wrong)
    with pytest.raises(ValueError, match="frozen request"):
        music.generate_job(job(), jobs.root / "wrong")
    assert not (jobs.root / "wrong/result.zip").exists()
    assert jobs.events[-1] == "close"


def test_changed_checkpoint_or_input_cannot_reuse_saved_result(jobs, monkeypatch):
    music.generate_job(job(), jobs.root / "saved")
    with pytest.raises(ValueError, match="another frozen request"):
        music.generate_job(job(seed=44), jobs.root / "saved")
    monkeypatch.setattr(music, "model_identity", lambda *args: "replacement-model")
    with pytest.raises(ValueError, match="configuration changed"):
        music.generate_job(job(), jobs.root / "saved")


@pytest.mark.parametrize("override", [{"backend": "ace_step15"}, {"model": "ace15_turbo"}, {"seed": -1}, {"seed": True}])
def test_main_music_interface_rejects_unsupported_requests(jobs, override):
    request = job()
    request["payload"].update(override)
    with pytest.raises(ValueError):
        music.generate_job(request, jobs.root / "invalid")
    assert not jobs.requests


def test_readiness_is_lightweight_and_requires_every_prepared_weight_component(tmp_path, monkeypatch):
    root = tmp_path
    model = root / "model"
    model.mkdir()
    (root / "python.exe").write_bytes(b"runtime")
    (model / "model_index.json").write_text(json.dumps({"_class_name": "StableAudio3Pipeline"}))
    for component in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler", "duration_embedder"):
        (model / component).mkdir()
    (model / "transformer/config.json").write_text(json.dumps({"embed_dim": 1536, "depth": 24,
        "num_heads": 24, "use_differential_attention": True}))
    for component in ("transformer", "vae", "text_encoder", "duration_embedder"):
        (model / component / "model.safetensors").write_bytes(b"prepared component")
    monkeypatch.setattr(music.audio_files, "find_ffmpeg", lambda: root / "ffmpeg")
    monkeypatch.setattr(music.audio_files, "find_ffprobe", lambda: root / "ffprobe")
    config = {"music": {"enabled": True, "python": "python.exe", "models": {"medium": "model"}}}
    assert music.check_readiness(config, root) == {"ready": True, "errors": []}
    (model / "vae/model.safetensors").unlink()
    status = music.check_readiness(config, root)
    assert not status["ready"]
    assert "vae weights are missing" in status["errors"][0]
    assert music.check_readiness({"music": {"enabled": False}}, root) == {"ready": False, "errors": []}
