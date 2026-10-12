"""An expression picture is its portrait with only the face replaced."""

import hashlib
import io
import json

import pytest
from PIL import Image, ImageChops

from packages.contracts.expressions import EXPRESSIONS
from services.worker.generation import outfit_image, pipeline, portrait_face


def png(colour, size=(200, 300), hole=True):
    image = Image.new("RGBA", size, colour)
    if hole:
        # A transparent strip, as around a portrait's figure.
        image.paste((0, 0, 0, 0), (0, 0, 40, size[1]))
    stream = io.BytesIO()
    image.save(stream, "PNG")
    return stream.getvalue()


def test_only_the_face_is_taken_from_the_edit():
    base, edit = png((200, 50, 50, 255)), png((50, 50, 200, 255), hole=False)
    box = portrait_face.area([80, 60, 140, 130], (200, 300), "smile")
    result = Image.open(io.BytesIO(portrait_face.composite(base, edit, box))).convert("RGBA")
    source = Image.open(io.BytesIO(base)).convert("RGBA")
    assert result.size == source.size
    changed = ImageChops.difference(result, source).getbbox(alpha_only=False)
    # The blend reaches a little past the box and no further; the silhouette is the portrait's own.
    assert changed is not None and all(abs(a - b) <= 20 for a, b in zip(changed, box, strict=True))
    assert result.getpixel((110, 95)) == (50, 50, 200, 255)
    assert result.getpixel((10, 150)) == (0, 0, 0, 0) and result.getpixel((190, 290)) == (200, 50, 50, 255)
    # An edit drawn at another size is brought to the portrait's.
    smaller = png((50, 50, 200, 255), size=(100, 150), hole=False)
    assert Image.open(io.BytesIO(portrait_face.composite(base, smaller, box))).size == (200, 300)


def test_the_area_leaves_room_around_the_face_and_more_below_for_tears():
    face = [100, 100, 200, 200]
    assert portrait_face.area(face, (400, 400), "smile") == (88, 95, 212, 208)
    assert portrait_face.area(face, (400, 400), "cry") == (88, 95, 212, 230)
    # It never leaves the picture.
    assert portrait_face.area([0, 0, 100, 100], (105, 104), "cry") == (0, 0, 105, 104)
    with pytest.raises(ValueError):
        portrait_face.area([10, 10, 10, 40], (100, 100), "smile")


def test_a_portrait_is_searched_once_and_a_missing_face_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    reference = tmp_path / "portrait.png"
    reference.write_bytes(png((1, 2, 3, 255)))
    runs, faces = [], [{"box": [10, 20, 30, 40], "score": 0.9}, {"box": [1, 1, 5, 5], "score": 0.3}]

    def run(command, log, *, cwd, timeout):
        runs.append(command)
        output = command[command.index("--output") + 1]
        with open(output, "w", encoding="utf-8") as stream:
            json.dump({"results": [{"faces": faces}]}, stream)

    monkeypatch.setattr(portrait_face, "run_process", run)
    config = {"image": {"python": "python.exe"}}
    assert portrait_face.detect(reference, config) == [10, 20, 30, 40]
    assert portrait_face.detect(reference, config) == [10, 20, 30, 40]
    # Found on the CPU by the portrait runtime, once per picture.
    assert len(runs) == 1 and runs[0][1].endswith("face_detect.py") and str(reference) in runs[0]
    other = tmp_path / "other.png"
    other.write_bytes(png((9, 9, 9, 255)))
    faces.clear()
    with pytest.raises(portrait_face.FaceNotFound):
        portrait_face.detect(other, config)


def test_every_expression_but_the_ordinary_one_has_its_instruction():
    assert set(outfit_image.EXPRESSION_INSTRUCTIONS) == set(EXPRESSIONS) - {"neutral"}
    for instruction in outfit_image.EXPRESSION_INSTRUCTIONS.values():
        assert instruction.startswith("Change only the facial expression")


def job(tmp_path, **changes):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    reference = png((200, 50, 50, 255), size=(256, 320))
    (work / outfit_image.REFERENCE).write_bytes(reference)
    payload = {"schema_version": 1, "seed": 7, "outfit": {"id": "default", "name": "通常の姿"},
               "reference_image": {"artifact_id": "portrait", "sha256": hashlib.sha256(reference).hexdigest()},
               "expression": {"id": "smile"}, **changes}
    return {"kind": outfit_image.KIND, "payload": payload}, work


def test_an_expression_job_edits_without_the_text_model_and_keeps_only_the_face(tmp_path, monkeypatch):
    request, work = job(tmp_path)
    calls = []

    def render(work, payload, config, prompt, references, width, height, fingerprint, *, resident=False):
        calls.append((prompt, resident, width, height))
        return png((50, 50, 200, 255), size=(256, 320), hole=False), {"provenance": {
            "width": width, "height": height, "inference_seconds": 1.0}}, "model"

    monkeypatch.setattr(outfit_image, "render", render)
    monkeypatch.setattr(outfit_image.pipeline, "load_config", lambda: {"image": {}})
    monkeypatch.setattr(portrait_face, "detect", lambda reference, config: [100, 80, 160, 150])
    monkeypatch.setattr(outfit_image, "LocalLLM", lambda *args, **kwargs: pytest.fail("no design is needed"))
    envelope, files = pipeline_bundle(outfit_image.generate_job(request, work))
    (prompt, resident, _width, _height), = calls
    assert outfit_image.EXPRESSION_INSTRUCTIONS["smile"] in prompt and resident is True
    made = envelope["provenance"]["outfit"]
    assert made["id"] == "default" and made["reference_sha256"] == request["payload"]["reference_image"]["sha256"]
    assert made["expression"] == {"id": "smile", "version": outfit_image.EXPRESSION_VERSION,
                                  "face_box": [100, 80, 160, 150], "area": [93, 76, 167, 156],
                                  "face_marked": False}
    image = Image.open(io.BytesIO(files["image.png"])).convert("RGBA")
    assert image.getpixel((130, 115)) == (50, 50, 200, 255) and image.getpixel((230, 300)) == (200, 50, 50, 255)
    assert image.getpixel((10, 10)) == (0, 0, 0, 0)


def test_a_face_marked_by_hand_is_used_instead_of_the_detector(tmp_path, monkeypatch):
    request, work = job(tmp_path, expression={"id": "cry", "face_box": [0.25, 0.25, 0.5, 0.5]})
    monkeypatch.setattr(outfit_image, "render", lambda *args, **kwargs: (
        png((50, 50, 200, 255), size=(256, 320), hole=False),
        {"provenance": {"width": 256, "height": 320, "inference_seconds": 1.0}}, "model"))
    monkeypatch.setattr(outfit_image.pipeline, "load_config", lambda: {"image": {}})
    monkeypatch.setattr(portrait_face, "detect", lambda *args: pytest.fail("the face was marked"))
    envelope, _ = pipeline_bundle(outfit_image.generate_job(request, work))
    made = envelope["provenance"]["outfit"]["expression"]
    assert made["face_box"] == [64, 80, 128, 160] and made["face_marked"] is True


@pytest.mark.parametrize("changes", [
    {"expression": {"id": "neutral"}}, {"expression": {"id": "wink"}}, {"expression": "smile"},
    {"edit": {"instruction": "髪を短く"}},
])
def test_only_a_fixed_expression_of_an_existing_portrait_is_accepted(tmp_path, changes):
    request, work = job(tmp_path, **changes)
    with pytest.raises(ValueError, match="fixed expressions"):
        outfit_image.generate_job(request, work)


def pipeline_bundle(data):
    import zipfile

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return json.loads(archive.read("result.json")), {name: archive.read(name) for name in archive.namelist()}
