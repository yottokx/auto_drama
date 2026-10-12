"""Shared materials are put into words while the text model is loaded, then only drawn."""
import json

import pytest

from services.worker.generation import material_image


class Text:
    def __init__(self):
        self.requests = 0

    def retry_failed_from(self, number):
        self.requests = number - 1


@pytest.fixture
def worded(monkeypatch):
    def design_material(payload, llm):
        llm.requests += 1
        sheet = payload["material_sheet"]
        if sheet["name"] == "broken":
            raise ValueError("no words")
        return {"category": "outfit", "outfit_en": sheet["name"] + " for " + payload["world_result"]["genre"],
                "summary_ja": "要約", "change_en": ""}

    monkeypatch.setattr(material_image, "design_material", design_material)


def sheet(name, **extra):
    return {"mode": "text", "name": name, "label": "", "wish": "", **extra}


def test_the_world_job_words_the_sheets_that_follow_it(worded, tmp_path):
    work, llm, world = tmp_path / "jobs" / "world", Text(), {"genre": "SF", "mood": "静か", "title": "題"}
    work.mkdir(parents=True)
    material_image.design_waiting([sheet("uniform"), sheet("broken"), sheet("armband")], world, work, llm, "text-model")
    # The failed attempt is not counted, and the sheet behind it is still put into words.
    assert llm.requests == 2
    # The sheet job is asked with the adopted world and may bring an outdated design along.
    asked = {**sheet("uniform"), "design": {"version": 0, "outfit_en": "old"}}
    path = material_image._design_path(tmp_path / "jobs" / "sheet", asked, {**world, "summary": "あらすじ"})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "category": "outfit", "outfit_en": "uniform for SF", "summary_ja": "要約", "change_en": "",
        "model_id": "text-model", "requests": 1}
    assert not material_image._design_path(work, sheet("broken"), world).exists()
    # Already put into words, nothing is asked again.
    material_image.design_waiting([sheet("uniform"), sheet("armband")], world, work, llm, "text-model")
    assert llm.requests == 2


def test_other_words_or_another_world_are_not_mistaken_for_it(worded, tmp_path):
    work, world = tmp_path / "jobs" / "world", {"genre": "SF"}
    work.mkdir(parents=True)
    material_image.design_waiting([sheet("uniform")], world, work, Text(), "text-model")
    assert material_image._design_path(work, sheet("uniform"), world).is_file()
    for other, place in ((sheet("uniform", wish="赤く"), world), (sheet("uniform", label="夏服"), world),
                         (sheet("uniform"), {"genre": "時代劇"})):
        assert not material_image._design_path(work, other, place).exists()


def test_a_batch_that_is_not_a_list_of_sheets_is_refused(worded, tmp_path):
    with pytest.raises(ValueError, match="batch"):
        material_image.design_waiting([{"name": "uniform"}], {"genre": "SF"}, tmp_path, Text(), "text-model")
    material_image.design_waiting(None, {"genre": "SF"}, tmp_path, Text(), "text-model")


def test_a_sheet_drawn_from_words_keeps_the_portrait_model_loaded():
    from services.worker.generation import image_session

    assert material_image.KIND in image_session.IMAGE_KINDS
    # A sheet made by editing a picture is the editing model's work and releases the portrait model.
    assert material_image.EDIT_KIND not in image_session.IMAGE_KINDS
