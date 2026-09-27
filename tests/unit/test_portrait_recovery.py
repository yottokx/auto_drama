import copy
import json
from contextlib import contextmanager

import pytest

from services.worker.generation import portrait_recovery as recovery

FAILURE = RuntimeError("Invalid background removal output: alpha=(0, 234)")
CHARACTER = {"id": "observer", "appearance": "姿は見えず、声としてのみ現れる。"}


def diagnosis(action="omit_portrait", quote=None):
    return {"action": action, "reason": "外見設定と人物画像の要求が一致しません。",
            "source_field": "appearance", "source_quote": quote or CHARACTER["appearance"],
            "revised_prompt": "a woman, isolated on a plain background" if action == "repair_prompt" else ""}


def harness(monkeypatch, tmp_path, outputs, answer=None):
    calls, requests = [], []

    def generate(payload, prompt, work, config):
        calls.append(prompt)
        value = outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    @contextmanager
    def llm(*args):
        class Stub:
            def __init__(self):
                self.trace = []

            def structured(self, stage, messages, schema):
                requests.append(messages)
                return answer or diagnosis()
        yield Stub()

    monkeypatch.setattr(recovery, "LocalLLM", llm)
    payload = {"seed": 1, "character_id": "observer", "character_result": copy.deepcopy(CHARACTER)}

    def run():
        return recovery.generate_with_recovery(payload, "invisible, full body", tmp_path, {},
                                              root=tmp_path, generate=generate)
    return run, calls, requests, payload


def test_success_does_not_diagnose(monkeypatch, tmp_path):
    run, calls, requests, _ = harness(monkeypatch, tmp_path, [(b"png", {})])
    assert run() == (b"png", {})
    assert len(calls) == 1 and requests == []
    assert not (tmp_path / "portrait-recovery").exists()


def test_omission_keeps_source_and_reuses_diagnosis(monkeypatch, tmp_path):
    run, calls, requests, payload = harness(monkeypatch, tmp_path, [FAILURE])
    before = copy.deepcopy(payload)
    for _ in range(2):
        data, metadata = run()
        assert data is None
        assert metadata["omission"]["diagnosis"]["source_quote"] == CHARACTER["appearance"]
    assert len(calls) == len(requests) == 1
    assert payload == before
    supplied = json.loads(requests[0][1]["content"])
    assert supplied["failed_stage"] == "background_removal"


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"), TimeoutError("timed out"),
                                  OSError("missing model"), RuntimeError("unknown runtime failure")])
def test_infrastructure_and_unknown_errors_do_not_change_character(monkeypatch, tmp_path, error):
    run, _, requests, _ = harness(monkeypatch, tmp_path, [error])
    with pytest.raises(type(error)):
        run()
    assert requests == []


def test_invented_evidence_cannot_omit(monkeypatch, tmp_path):
    run, _, _, _ = harness(monkeypatch, tmp_path, [FAILURE], diagnosis(quote="invented setting"))
    with pytest.raises(ValueError, match="exact quotation"):
        run()


def test_repair_runs_once_and_is_persisted(monkeypatch, tmp_path):
    run, calls, requests, _ = harness(monkeypatch, tmp_path, [FAILURE, (b"fixed", {})],
                                    diagnosis("repair_prompt"))
    assert run()[0] == b"fixed"
    assert run()[0] == b"fixed"
    assert len(calls) == 2 and len(requests) == 1
    assert calls[1] == diagnosis("repair_prompt")["revised_prompt"]


def test_failed_repair_cannot_loop_on_job_retry(monkeypatch, tmp_path):
    run, calls, requests, _ = harness(monkeypatch, tmp_path, [FAILURE, FAILURE], diagnosis("repair_prompt"))
    for _ in range(3):
        with pytest.raises(RuntimeError, match="after one correction"):
            run()
    assert len(calls) == 2 and len(requests) == 1


def test_inconclusive_diagnosis_stops_without_retry(monkeypatch, tmp_path):
    run, calls, requests, _ = harness(monkeypatch, tmp_path, [FAILURE], diagnosis("stop"))
    with pytest.raises(RuntimeError, match="stopped"):
        run()
    assert len(calls) == len(requests) == 1
