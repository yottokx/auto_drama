"""Optional CG planning, input identity, and resumable individual images."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from packages.contracts.event_cg import (
    BUDGET_ALLOCATION_VERSION,
    EventCgProfile,
    image_input_sha256,
)
from services.worker.client import WorkerClient
from services.worker.generation import event_cg_pipeline as cg
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.event_cg_session import EventCgGenerationError

PROMPT = ("Reference image 1 stands beside a window in a quiet room at dusk. "
    "Preserve the referenced character's facial features, hairstyle, eye color, clothing and proportions. "
    "A warm side light illuminates the face while cool evening light fills the room. "
    "Use a medium shot with the face above the lower message area, looking at an open book. "
    "Create one coherent full scene without text, lettering, captions, speech bubbles or watermarks.")


def payload():
    return {"schema_version": 1, "seed": 1, "policy": {"max_cgs": 2, "max_variants_per_cg": 2},
        "chapter_budget": 2, "references": [{"character_id": "alice", "name": "アリス", "outfit_id": "default"}],
        "context": {"source_sha256": "a" * 64, "overall_plot": {"chapters": [
            {"number": 1, "summary": "はじまり"}, {"number": 2, "summary": "結末"}]},
            "narrative": {"scenes": [{"id": "scene1", "plan": {"character_ids": ["alice"]},
                "utterances": [{"id": f"u{i}", "speaker_id": "alice", "display_text": f"台詞{i}"}
                    for i in range(1, 5)]}]}}}


def selection():
    return {"cgs": [{"scene_id": "scene1", "start_utterance_id": "u1", "last_utterance_id": "u3",
        "character_ids": ["alice"], "interpretation": "窓際で本を開く瞬間。",
        "variants": [{"start_utterance_id": "u2", "interpretation": "本を見て微笑む。"}]}]}


def visual_person(position="left of the window", action="holding an open book"):
    return {"position": position, "action": action, "expression": "a restrained smile",
        "gaze_target": "scene", "gaze_detail": "at the open book"}


def prompts():
    return {"images": [{"id": identifier, "interpretation": "窓際で本を開く瞬間。",
        "environment": "a quiet library at dusk", "lighting": "soft blue light from the window",
        "framing": "a medium shot", "characters": {"alice": visual_person()}}
        for identifier in ("cg_001", "cg_001_v01")]}


class FakeLLM:
    def __init__(self, answers):
        self.answers, self.messages, self.trace = iter(answers), [], []

    def structured(self, stage, messages, schema):
        self.messages.append((stage, copy.deepcopy(messages), schema))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return copy.deepcopy(answer)


def test_budget_is_bounded_and_repaired_once():
    llm = FakeLLM([{"chapters": [{"chapter_number": 1, "limit": 3, "reason": "多すぎる"}]},
        {"chapters": [{"chapter_number": 1, "limit": 1, "reason": "出会い"},
            {"chapter_number": 2, "limit": 1, "reason": "再会"}]}])
    result = cg.plan_budget(payload(), llm)
    assert sum(row["limit"] for row in result["chapters"]) == 2
    assert result["source_sha256"] == "a" * 64
    assert len(llm.messages) == 2


def test_invalid_budget_becomes_explicit_zero_allocation():
    result = cg.plan_budget(payload(), FakeLLM([{}, {}]))
    assert [row["limit"] for row in result["chapters"]] == [0, 0]
    assert result["omission_reason"]


@pytest.mark.parametrize("limits", [[0, 0], [1, 0]])
@pytest.mark.parametrize("version", [None, BUDGET_ALLOCATION_VERSION])
def test_budget_repairs_unused_slots_even_for_legacy_pending_requests(limits, version):
    source = payload()
    if version is not None:
        source["budget_allocation_version"] = version
    original = copy.deepcopy(source)
    answer = {"chapters": [{"chapter_number": number, "limit": limit, "reason": "後半に温存"}
        for number, limit in enumerate(limits, 1)]}
    repaired = {"chapters": [{"chapter_number": 1, "limit": 0, "reason": "導入"},
        {"chapter_number": 2, "limit": 2, "reason": "終幕の再会に配分"}]}
    llm = FakeLLM([answer, repaired])
    result = cg.plan_budget(source, llm)
    assert source == original
    assert [row["limit"] for row in result["chapters"]] == [0, 2]
    assert result["omission_reason"] == ""
    assert len(llm.messages) == 2
    correction = llm.messages[1][1][-1]["content"]
    assert f"total {sum(limits)}; allocate all 2" in correction


def test_repeated_zero_budget_is_an_explicit_failure_instead_of_normal_selection():
    answer = {"chapters": [{"chapter_number": number, "limit": 0, "reason": "後半に温存"}
        for number in (1, 2)]}
    llm = FakeLLM([answer, answer])
    result = cg.plan_budget(payload(), llm)
    assert [row["limit"] for row in result["chapters"]] == [0, 0]
    assert "total 0; allocate all 2" in result["omission_reason"]
    assert len(llm.messages) == 2


def test_positive_chapter_allowance_does_not_force_image_selection():
    llm = FakeLLM([{"cgs": []}])
    result = cg.plan_cgs(payload(), llm)
    assert result["cgs"] == []
    assert result["omission_reason"] == ""
    assert len(llm.messages) == 1


def test_interval_translation_prompts_and_role_numbering_leave_source_intact():
    source = payload()
    original = copy.deepcopy(source)
    llm = FakeLLM([selection(), prompts()])
    result = cg.plan_cgs(source, llm)
    assert source == original
    assert result["cgs"][0]["end_utterance_id"] == "u4"
    assert result["cgs"][0]["variants"][0]["start_utterance_id"] == "u2"
    visual_source = json.loads(llm.messages[1][1][1]["content"])
    assert visual_source["characters"][0]["character_id"] == "alice"
    assert "reference_index" not in visual_source["characters"][0]
    assert "Reference image 1 character:" in result["cgs"][0]["prompt"]
    assert "Reference image 2 character:" in result["cgs"][0]["variants"][0]["prompt"]
    assert visual_source["scene"]["utterances"][0]["speaker_id"] == "alice"
    assert "台詞" not in result["cgs"][0]["prompt"]


@pytest.mark.parametrize("change", ["unknown_character", "invalid_span", "bad_variant"])
def test_bad_selection_is_omitted_after_one_repair(change):
    answer = selection()
    row = answer["cgs"][0]
    if change == "unknown_character":
        row["character_ids"] = ["intruder"]
    elif change == "invalid_span":
        row["last_utterance_id"] = "u0"
    else:
        row["variants"][0]["start_utterance_id"] = "u1"
    llm = FakeLLM([answer, answer])
    result = cg.plan_cgs(payload(), llm)
    assert result["cgs"] == [] and result["omission_reason"]
    assert len(llm.messages) == 2


def test_prompt_failure_does_not_fall_back_to_raw_script():
    llm = FakeLLM([selection(), {"images": []}, {"images": []}])
    result = cg.plan_cgs(payload(), llm)
    assert result["cgs"] == []
    assert "台詞" not in result["omission_reason"]
    assert len(llm.messages) == 3


def test_llm_transport_failure_and_cancellation_are_not_omitted():
    for error in (RuntimeError("Local LLM HTTP 503"), GenerationCancelled()):
        with pytest.raises(type(error)):
            cg.plan_cgs(payload(), FakeLLM([error]))


def test_context_overflow_splits_by_scene_without_clipping_utterances():
    source = payload()
    second = copy.deepcopy(source["context"]["narrative"]["scenes"][0])
    second["id"] = "scene2"
    for row in second["utterances"]:
        row["id"] += "b"
    source["context"]["narrative"]["scenes"].append(second)
    llm = FakeLLM([cg.ContextBudgetError("too long"), {"cgs": []}, {"cgs": []}])
    assert cg.plan_cgs(source, llm)["cgs"] == []
    assert len(llm.messages) == 3
    for index in (1, 2):
        body = json.loads(llm.messages[index][1][1]["content"])
        assert len(body["scenes"]) == 1 and len(body["scenes"][0]["utterances"]) == 4


def image_payload(work):
    image = work / "cg-reference-01.png"
    Image.new("RGBA", (32, 64), "red").save(image)
    value = {"schema_version": 1, "seed": 1, "cg_id": "cg_001", "variant_id": None,
        "prompt": PROMPT, "cg_profile": EventCgProfile().model_dump(mode="json"),
        "context": {"source_sha256": "a" * 64}, "references": [{"reference_index": 1,
            "role": "character", "character_id": "alice", "outfit_id": "default",
            "artifact_id": "portrait", "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]}
    value["input_sha256"] = image_input_sha256(value)
    return value


@pytest.fixture
def image_runtime(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cg.pipeline, "load_config", lambda: {"gpu_lock_timeout_seconds": 10,
        "max_zip_bytes": 1000000})
    monkeypatch.setattr(cg, "model_identity", lambda *_: "model-identity")
    monkeypatch.setattr(cg, "gpu_lock", lambda *_: nullcontext())
    state = SimpleNamespace(error=None)

    class Owner:
        def gpu_scope(self, lease):
            return lease

        def close(self):
            pass

        def run(self, request, **kwargs):
            calls.append(copy.deepcopy(request))
            if state.error:
                raise state.error
            output = Path(request["output_dir"])
            Image.new("RGB", (960, 640), "blue").save(output / "image.png")
            Image.new("RGBA", (960, 640), (0, 0, 255, 250)).save(output / "original.png")
            report = {"ok": True, "provenance": {name + "_sha256": hashlib.sha256(
                (output / (name + ".png")).read_bytes()).hexdigest() for name in ("image", "original")}}
            (output / "result.json").write_text(json.dumps(report), encoding="utf-8")
            return report

    monkeypatch.setattr(cg.event_cg_session, "EventCgSession", Owner)
    return calls, state


def unpack(content):
    with zipfile.ZipFile(io.BytesIO(content)) as bundle:
        return json.loads(bundle.read("result.json")), bundle.namelist()


def test_budget_artifact_records_allocation_rules_version(tmp_path, monkeypatch):
    from services.worker import model_config

    source = payload()
    llm = FakeLLM([{"chapters": [{"chapter_number": 1, "limit": 0, "reason": "導入"},
        {"chapter_number": 2, "limit": 2, "reason": "終幕"}]}])
    llm.requests = 1
    monkeypatch.setattr(cg.pipeline, "load_config", lambda: {"gpu_lock_timeout_seconds": 10,
        "max_zip_bytes": 1000000, "llm": {"model_id": "test-model"}})
    monkeypatch.setattr(model_config, "select_config", lambda config, *_: config)
    monkeypatch.setattr(cg, "model_configuration_identity", lambda *_: "model-identity")
    monkeypatch.setattr(cg, "gpu_lock", lambda *_: nullcontext())
    monkeypatch.setattr(cg, "LocalLLM", lambda *_: nullcontext(llm))
    content = cg.generate_job({"kind": "m3_event_cg_budget", "payload": source}, tmp_path)
    envelope, names = unpack(content)
    assert envelope["provenance"]["budget_allocation_version"] == BUDGET_ALLOCATION_VERSION
    assert sum(row["limit"] for row in envelope["result"]["chapters"]) == source["policy"]["max_cgs"]
    assert names == ["result.json"]


def test_completed_image_is_cached_and_reuploaded_without_runtime_checks(tmp_path, monkeypatch, image_runtime):
    job = {"kind": "m3_event_cg", "payload": image_payload(tmp_path)}
    first = cg.generate_job(job, tmp_path)
    result, names = unpack(first)
    assert result["result"]["status"] == "complete"
    assert "image.png" in names and "original.png" in names
    monkeypatch.setattr(cg.pipeline, "load_config", lambda: pytest.fail("Cached result must not load models"))
    assert cg.generate_job(job, tmp_path) == first
    assert len(image_runtime[0]) == 1


def test_image_complete_before_bundle_failure_is_reused(tmp_path, monkeypatch, image_runtime):
    job = {"kind": "m3_event_cg", "payload": image_payload(tmp_path)}
    real_bundle = cg.pipeline._bundle
    monkeypatch.setattr(cg.pipeline, "_bundle", lambda *_: (_ for _ in ()).throw(OSError("upload staging failed")))
    with pytest.raises(OSError):
        cg.generate_job(job, tmp_path)
    monkeypatch.setattr(cg.pipeline, "_bundle", real_bundle)
    assert unpack(cg.generate_job(job, tmp_path))[0]["result"]["status"] == "complete"
    assert len(image_runtime[0]) == 1


def test_model_failure_is_an_omission_but_identity_and_cancel_are_not(tmp_path, image_runtime):
    _calls, state = image_runtime
    job = {"kind": "m3_event_cg", "payload": image_payload(tmp_path)}
    state.error = EventCgGenerationError("CUDA out of memory")
    result, names = unpack(cg.generate_job(job, tmp_path))
    assert result["result"]["status"] == "omitted"
    assert "CUDA out of memory" in result["result"]["reason"]
    assert names == ["result.json"]
    for index, error in enumerate((ValueError("identity mismatch"), GenerationCancelled())):
        folder = tmp_path / str(index)
        folder.mkdir()
        state.error = error
        with pytest.raises(type(error)):
            cg.generate_job({"kind": "m3_event_cg", "payload": image_payload(folder)}, folder)
        assert not (folder / "result.zip").exists()


def test_reference_tampering_rejected_before_qwen(tmp_path, image_runtime):
    value = image_payload(tmp_path)
    (tmp_path / "cg-reference-01.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="reference hash"):
        cg.generate_job({"kind": "m3_event_cg", "payload": value}, tmp_path)
    assert not image_runtime[0]


def test_worker_fetches_only_artifact_ids_and_keeps_original_bytes(tmp_path):
    value = image_payload(tmp_path)
    content = (tmp_path / "cg-reference-01.png").read_bytes()
    (tmp_path / "cg-reference-01.png").unlink()
    worker = object.__new__(WorkerClient)
    called = []
    worker._artifact = lambda identifier: called.append(identifier) or content
    heartbeat = SimpleNamespace(check=lambda: None)
    worker._prepare_cg_references({"payload": value}, tmp_path, heartbeat)
    worker._prepare_cg_references({"payload": value}, tmp_path, heartbeat)
    assert called == ["portrait"]
    assert (tmp_path / "cg-reference-01.png").read_bytes() == content


def test_missing_optional_qwen_does_not_import_gpu(tmp_path):
    assert cg.check_readiness({}, tmp_path) == {"ready": False, "errors": []}


def test_zero_allowance_does_not_query_llm():
    source = payload()
    source["policy"]["max_cgs"] = 0
    assert cg.plan_cgs(source, None)["cgs"] == []
    assert all(row["limit"] == 0 for row in cg.plan_budget(source, None)["chapters"])
    source["policy"]["max_cgs"] = 2
    source["chapter_budget"] = 0
    assert cg.plan_cgs(source, None)["cgs"] == []


def test_selection_and_drawing_keep_physical_location_and_speaker():
    source = payload()
    source["context"]["narrative"]["locations"] = [{"id": "room", "name": "月光の図書館",
        "description": "石造りの円形の部屋", "time_of_day": "夜"}]
    source["context"]["narrative"]["scenes"][0]["plan"]["location_id"] = "room"
    source["context"]["world"] = {"genre": "魔法少女", "setting": "夜の図書館", "private_notes": "不要"}
    llm = FakeLLM([selection(), prompts()])
    result = cg.plan_cgs(source, llm)
    assert len(result["cgs"]) == 1
    for index, key in ((0, "scenes"), (1, "scene")):
        body = json.loads(llm.messages[index][1][1]["content"])
        scene = body[key][0] if index == 0 else body[key]
        assert scene["location"]["name"] == "月光の図書館"
        assert scene["utterances"][0]["speaker_id"] == "alice"
        assert "private_notes" not in body["world"]


def test_partial_prompt_failure_is_visible_in_plan_result():
    source = payload()
    answer = selection()
    answer["cgs"][0]["last_utterance_id"] = "u2"
    second = copy.deepcopy(answer["cgs"][0])
    second.update(start_utterance_id="u3", last_utterance_id="u4", variants=[])
    answer["cgs"].append(second)
    llm = FakeLLM([answer, prompts(), {"images": []}, {"images": []}])
    result = cg.plan_cgs(source, llm)
    assert len(result["cgs"]) == 1
    assert result["omission_reason"] == ""
    assert result["prompt_omissions"][0]["cg_id"] == "cg_002"
    assert result["prompt_omissions"][0]["reason"]



def test_legacy_freeform_numbering_retries_to_character_keyed_directions():
    legacy = {"images": [{"id": "cg_001", "interpretation": "窓際で本を開く。", "prompt": PROMPT},
        {"id": "cg_001_v01", "interpretation": "同じ構図で微笑む。", "prompt": PROMPT}]}
    llm = FakeLLM([selection(), legacy, prompts()])
    result = cg.plan_cgs(payload(), llm)
    assert len(llm.messages) == 3
    assert "Reference image 2 character:" in result["cgs"][0]["variants"][0]["prompt"]
    assert llm.trace[-1]["type"] == "event_cg_visual_directions"
    assert llm.trace[-1]["version"] == 2



def test_non_object_visual_image_gets_one_repair_then_explicit_omission():
    llm = FakeLLM([selection(), {"images": [None]}, {"images": [None]}])
    result = cg.plan_cgs(payload(), llm)
    assert len(llm.messages) == 3
    assert result["cgs"] == []
    assert result["prompt_omissions"][0]["cg_id"] == "cg_001"
    assert "exact supplied order" in result["prompt_omissions"][0]["reason"]
