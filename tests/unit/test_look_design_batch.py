"""Looks that wait are put into words in one text-model session, then drawn without reloading."""
from contextlib import contextmanager, nullcontext

import pytest

from services.worker.generation import outfit_image


class Resident:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


@pytest.fixture
def designer(monkeypatch, tmp_path):
    sessions, resident = [], Resident()

    class Text:
        def __init__(self):
            self.requests = 0

        def retry_failed_from(self, number):
            self.requests = number - 1

    @contextmanager
    def local_llm(_root, _config, _payload, _directory):
        sessions.append(Text())
        yield sessions[-1]

    def design_outfit(payload, llm):
        llm.requests += 1
        if payload["outfit"]["name"] == "broken":
            raise ValueError("no words")
        return {"outfit_en": payload["outfit"]["name"] + " in words", "summary_ja": "要約", "occasion_ja": ""}

    monkeypatch.setattr(outfit_image, "LocalLLM", local_llm)
    monkeypatch.setattr(outfit_image, "gpu_lock", lambda *_args: nullcontext())
    monkeypatch.setattr(outfit_image, "design_outfit", design_outfit)
    monkeypatch.setattr(outfit_image.event_cg_session, "current_session", lambda: resident)
    monkeypatch.setattr("services.worker.model_config.select_config",
                        lambda config, _payload, _root: {"llm": {"model_id": "text-model"}})
    config = {"gpu_lock_timeout_seconds": 1}
    lock = tmp_path / "gpu.lock"

    def design(payload, job):
        work = tmp_path / "jobs" / job
        work.mkdir(parents=True, exist_ok=True)
        return outfit_image._design(payload, work, config, lock)

    return design, sessions, resident


def look(name, **extra):
    return {"character_id": "character-1", "character_result": {"id": "character-1", "name": "ミナ"},
            "outfit": {"id": name, "name": name, "description": name}, **extra}


def request(name, *waiting, **extra):
    return {"schema_version": 1, "seed": 1, "world_result": {"title": "世界"}, "profile": {"model_id": "text-model"},
            "reference_image": {"sha256": "0" * 64}, **look(name), **extra,
            **({"look_design_batch": list(waiting)} if waiting else {})}


def test_the_first_look_designs_those_that_wait_and_they_keep_the_editing_model(designer):
    design, sessions, resident = designer
    first = design(request("yukata", look("formal"), look("armor")), "a")
    assert first["outfit_en"] == "yukata in words" and first["requests"] == 1
    assert len(sessions) == 1 and sessions[0].requests == 3
    # The text model needed the GPU once.
    assert resident.closed == 1
    second = design(request("formal", look("armor")), "b")
    third = design(request("armor"), "c")
    assert (second["outfit_en"], third["outfit_en"]) == ("formal in words", "armor in words")
    assert len(sessions) == 1 and resident.closed == 1


def test_a_changed_request_is_put_into_words_again(designer):
    design, sessions, resident = designer
    design(request("yukata", look("formal")), "a")
    changed = request("formal")
    changed["outfit"]["description"] = "another wish"
    design(changed, "b")
    assert len(sessions) == 2 and resident.closed == 2


def test_a_waiting_look_that_fails_is_left_to_its_own_job(designer):
    design, sessions, _resident = designer
    assert design(request("yukata", look("broken"), look("formal")), "a")["outfit_en"] == "yukata in words"
    # The failed attempt is not counted, and the look behind it is still designed.
    assert sessions[0].requests == 2
    design(request("formal"), "c")
    assert len(sessions) == 1
    with pytest.raises(ValueError, match="no words"):
        design(request("broken"), "b")


def test_requests_that_need_no_words_are_not_given_to_the_text_model():
    fixed = outfit_image._fixed_design
    material = {"outfit_en": "uniform", "summary_ja": "制服"}
    assert fixed({**look("default"), "material_image": material})["outfit_en"] == "uniform"
    assert fixed({**look("yukata"), "edit": {"instruction": "  "}})["instruction_en"] == ""
    assert fixed({**look("yukata"), "expression": {"id": "smile"}})["requests"] == 0
    assert fixed(look("yukata")) is None and fixed({**look("yukata"), "edit": {"instruction": "袖を短く"}}) is None


def test_a_batch_that_is_not_a_list_of_looks_is_refused(designer):
    design, _sessions, _resident = designer
    with pytest.raises(ValueError, match="batch"):
        design({**request("yukata"), "look_design_batch": [{"outfit": {"id": "x"}, "seed": 3}]}, "a")


def test_words_made_ahead_serve_one_picture_and_tell_later_looks_about_earlier_ones(designer, monkeypatch, tmp_path):
    design, sessions, resident = designer
    seen = []
    original = outfit_image.design_outfit

    def recording(payload, llm):
        seen.append((payload["outfit"]["name"], [row["look"] for row in payload.get("other_looks") or []]))
        return original(payload, llm)

    monkeypatch.setattr(outfit_image, "design_outfit", recording)
    monkeypatch.setattr(outfit_image.pipeline, "load_config", lambda: {"gpu_lock_timeout_seconds": 1})
    work = tmp_path / "jobs" / "portrait"
    work.mkdir(parents=True)
    # A portrait job only announces the looks; it is not one itself.
    portrait = {key: value for key, value in request("unused", look("yukata"), look("formal")).items()
                if key not in {"outfit", "reference_image"}}
    outfit_image.design_ahead(portrait, work, lambda _config: nullcontext())
    assert seen == [("yukata", []), ("formal", ["yukata"])]
    assert len(sessions) == 1 and resident.closed == 0
    # Whatever the work wears by the time the look is drawn, the words made ahead are found.
    drawn = design(request("formal", other_looks=[{"worn_by": "ミナ", "look": "yukata", "outfit": "要約"}]), "b")
    assert drawn["outfit_en"] == "formal in words" and len(sessions) == 1 and resident.closed == 0
    # Asked for again, the look is designed afresh.
    design(request("formal"), "c")
    assert len(sessions) == 2 and resident.closed == 1
    # Nothing left to put into words: the text model is not loaded at all.
    outfit_image.design_ahead({**portrait, "look_design_batch": [look("yukata")]}, work, lambda _config: 1 / 0)


def test_a_file_that_is_briefly_held_does_not_fail_the_job(tmp_path, monkeypatch):
    from pathlib import Path

    from services.worker.generation import llm

    replace, refused = Path.replace, []

    def held(self, target):
        if len(refused) < 3:
            refused.append(target)
            raise PermissionError(5, "Access is denied")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", held)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    llm.write_json(tmp_path / "design.json", {"summary_ja": "要約"})
    assert len(refused) == 3 and (tmp_path / "design.json").read_text(encoding="utf-8").count("要約") == 1
    assert not (tmp_path / "design.json.tmp").exists()
