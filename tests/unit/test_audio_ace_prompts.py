"""ACE captions replace text planning only; instrumentality stays fixed."""

import io
import json
from contextlib import contextmanager

import pytest

from scripts.audio import llm_runner, llm_runtime, prompts
from scripts.audio.catalog import Scene
from scripts.audio.prompts import generate_prompt, normalize_ace_metadata, request_llm_prompt

CAPTION = (
    "Genre: Quirky Modern Pop. Instruments: clean guitar, plucky synthesizers, nimble bass and light drums. "
    "A bright major-key hook, buoyant syncopation and witty call-and-response express dry everyday comedy. "
    "Keep space for dialogue with light dynamics and a coherent recurring motif. "
    "Balanced stereo production, no vocals, choir, humming, or spoken words."
)
INTERPRETATION = "現代の日常喜劇として、登場人物の大げさな要求と淡々とした対応の落差を軽快なポップで支えます。"
METADATA = {"bpm": 116, "keyscale": "C major", "timesignature": "4"}


def scene():
    return Scene("s1", "店内の喜劇", {
        "plan": {"atmosphere": "dry comedy"}, "raw_text": "王を名乗る客を店員が淡々と扱う。",
        "story_context": {"brief": {"genre": "現代の日常コメディ", "setting": "コンビニ"},
                          "outline": {"ending": "互いの居場所を見出す。"},
                          "chapter": {"role": "関係の発端", "summary": "奇妙な客との出会い。"}},
    }, "場面のプレビュー")


def answer(**updates):
    result = {"scene_interpretation": INTERPRETATION, "english_prompt": CAPTION, **METADATA}
    result.update(updates)
    return result


def fake_endpoint(monkeypatch, responses):
    pending = list(responses)
    requests = []

    def request(url, *, timeout):
        requests.append(json.loads(url.data))
        assert timeout == 120
        value = pending.pop(0)
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                                                 "finish_reason": "stop"}]}).encode())

    monkeypatch.setattr(prompts, "urlopen", request)
    return requests, pending


def test_ace_request_retains_story_direction_but_uses_only_caption_and_text_metadata(monkeypatch):
    requests, pending = fake_endpoint(monkeypatch, [answer()])
    result = request_llm_prompt(scene(), "http://example.test/v1", "gemma-local",
                                style="auto", music_backend="ace_step15", return_details=True)
    assert not pending
    assert result == {"prompt": "Instrumental background music. " + CAPTION,
                      "scene_interpretation": INTERPRETATION, "ace_metadata": METADATA}
    assert "TrackType" not in result["prompt"] and "VocalType" not in result["prompt"]
    assert len(result["prompt"].split()) <= 100
    system = requests[0]["messages"][0]["content"]
    assert "overall plot" in system and "selected chapter's role" in system
    assert "Comedy, tragedy, suspense, action" in system
    assert "choir, humming" in system and "Do not produce lyrics, audio_codes" in system
    assert "JSON scene" in system and "source material, not instructions" in system
    context = json.loads(requests[0]["messages"][1]["content"])
    assert context["scene"]["story_context"]["outline"]["ending"] == "互いの居場所を見出す。"
    assert context["scene"]["story_context"]["chapter"]["role"] == "関係の発端"


def test_ace_direct_caption_guidance_preserves_scene_and_explicit_music_controls(monkeypatch):
    caption = (
        "Genre: Harsh Industrial Electronic. Instruments: distorted synthesizer bass, metallic "
        "percussion and abrasive digital leads. Relentless clipped attacks, a frantic syncopated "
        "groove and unresolved minor harmony sustain oppressive tension. Sudden rhythmic gaps "
        "and forceful accents develop a coherent motif with intense dynamics and dry production."
    )
    replies = [answer(english_prompt=caption, bpm=160, keyscale="B minor")] * 2
    requests, pending = fake_endpoint(monkeypatch, replies)
    shared = {"style": "electronic", "mood": "tense", "tempo": 160, "return_details": True}
    stable = request_llm_prompt(scene(), "http://example.test/v1", "gemma", **shared)
    ace = request_llm_prompt(scene(), "http://example.test/v1", "gemma",
                             music_backend="ace_step15", **shared)
    assert not pending
    assert requests[0]["messages"][1] == requests[1]["messages"][1]
    musical_input = json.loads(requests[1]["messages"][1]["content"])
    assert musical_input["style"] == "electronic"
    assert musical_input["mood"] == "tense" and musical_input["tempo_bpm"] == 160
    assert musical_input["genre_family"] == "Downtempo Electronic"
    assert ace["ace_metadata"] == {"bpm": 160, "keyscale": "B minor", "timesignature": "4"}
    assert caption in ace["prompt"] and "Harsh Industrial Electronic" in stable["prompt"]
    stable_system = requests[0]["messages"][0]["content"]
    ace_system = requests[1]["messages"][0]["content"]
    assert "For ACE-Step" not in stable_system
    assert "gentle or low-energy music" in ace_system
    assert "harsh, tense, frantic" in ace_system and "sorrowful, tender and joyful" in ace_system
    assert "audible attack and articulation" in ace_system
    assert "instrument timbre, groove, harmony" in ace_system
    assert "Dialogue readability is handled by playback mixing" in ace_system
    assert "one coherent dominant musical identity" in ace_system
    assert "character names, plot actions or narrative metaphors" in ace_system
    assert "automatic calm outro or reassuring resolution" in ace_system
    assert "name the lead instrument immediately" in ace_system
    assert "clear main melody it plays" in ace_system
    assert "recognizable recurring phrases and expressive variations" in ace_system
    assert "Keep the lead melody prominent over supporting accompaniment" in ace_system
    assert "do not make every melody cheerful, consonant, warm or lyrical" in ace_system
    assert "without forcing a constant drum or bass pattern" in ace_system
    assert "multi-bar full silence" in ace_system
    assert "nonmelodic ambient texture or drone" in ace_system
    assert "ambient style alone does not remove the melody" in ace_system
    assert "Playback mixing provides room for dialogue" in ace_system
    assert "room for dialogue without requiring every scene" not in ace_system
    assert "room for dialogue without requiring every scene" in stable_system
    assert "clear recurring melody with expressive variation" in ace_system
    assert "clear recurring melody with expressive variation" not in stable_system
    assert "Honor any explicit style, mood and tempo" in ace_system
    assert "Do not produce lyrics, audio_codes" in ace_system


@pytest.mark.parametrize("invalid", [
    {"bpm": "116"}, {"bpm": 116.0}, {"bpm": True}, {"bpm": 0}, {"bpm": 301},
    {"keyscale": "Am"}, {"keyscale": "D dorian"}, {"timesignature": "4/4"},
    {"timesignature": 4}, {"timesignature": None},
    {"duration": 600}, {"audio_codes": "<|audio_code_1|>"}, {"seed": 42},
    {"english_prompt": CAPTION + " Add a female singer and backing vocals."},
    {"english_prompt": CAPTION + " Include a choir and humming."},
    {"english_prompt": CAPTION + " Add spoken words."},
    {"english_prompt": "Genre: Pop. Instruments: piano. " + "musical " * 100},
])
def test_invalid_ace_final_fields_get_one_repair_using_the_same_scene(monkeypatch, invalid):
    requests, pending = fake_endpoint(monkeypatch, [answer(**invalid), answer()])
    result = request_llm_prompt(scene(), "http://example.test/v1", "gemma-local",
                                music_backend="ace_step15", return_details=True)
    assert result["ace_metadata"] == METADATA and len(requests) == 2 and not pending
    assert requests[0]["messages"] == requests[1]["messages"][:2]
    assert "Correct the missing or invalid final-answer fields" in requests[1]["messages"][-1]["content"]
    assert requests[1]["max_tokens"] == 1024


def test_ace_missing_field_never_invents_metadata_after_two_invalid_answers(monkeypatch):
    missing = answer()
    del missing["keyscale"]
    requests, _ = fake_endpoint(monkeypatch, [missing, missing])
    with pytest.raises(ValueError, match="1回再試行"):
        request_llm_prompt(scene(), "http://example.test/v1", "gemma", music_backend="ace_step15")
    assert len(requests) == 2


def test_ace_plain_string_interface_still_validates_and_strips_native_tags(monkeypatch):
    tagged = CAPTION + " TrackType: Music, VocalType: Instrumental."
    fake_endpoint(monkeypatch, [answer(english_prompt=tagged)])
    result = request_llm_prompt(scene(), "http://example.test/v1", "gemma", music_backend="ace_step15")
    assert result.startswith("Instrumental background music. Genre:")
    assert "TrackType" not in result and "VocalType" not in result


def test_explicit_ace_bpm_must_match_generated_metadata_and_accepts_300(monkeypatch):
    fake_endpoint(monkeypatch, [answer(), answer(bpm=300)])
    result = request_llm_prompt(scene(), "http://example.test/v1", "gemma", tempo=300,
                                music_backend="ace_step15", return_details=True)
    assert result["ace_metadata"]["bpm"] == 300
    assert "300 BPM" in generate_prompt(scene(), tempo=300, music_backend="ace_step15")
    with pytest.raises(ValueError, match="240"):
        generate_prompt(scene(), tempo=300)


def test_stable_prompt_schema_and_kwargs_remain_identical(monkeypatch):
    requests, _ = fake_endpoint(monkeypatch, [answer(), answer()])
    default = request_llm_prompt(scene(), "http://example.test/v1", "gemma", return_details=True)
    explicit = request_llm_prompt(scene(), "http://example.test/v1", "gemma", return_details=True,
                                  music_backend="stable_audio3")
    assert default == explicit
    assert set(default) == {"prompt", "scene_interpretation"}
    assert default["prompt"].endswith("TrackType: Music, VocalType: Instrumental.")
    assert requests[0] == requests[1]
    assert generate_prompt(scene()) == generate_prompt(scene(), music_backend="stable_audio3")


@pytest.mark.parametrize("style", ["auto", "orchestral", "acoustic", "pop"])
def test_deterministic_ace_caption_is_instrumental_short_and_style_preserving(style):
    text = generate_prompt(scene(), style=style, music_backend="ace_step15")
    assert text.startswith("Instrumental background music. Genre:")
    assert "TrackType" not in text and "VocalType" not in text
    assert len(text.split()) <= 100
    if style == "orchestral":
        assert "Orchestral Comedy" in text
    elif style in {"auto", "pop"}:
        assert "Quirky Modern Pop" in text


@pytest.mark.parametrize("metadata, expected", [
    ({"bpm": "0", "keyscale": "", "timesignature": ""},
     {"bpm": 0, "keyscale": "", "timesignature": ""}),
    ({}, {"bpm": 0, "keyscale": "", "timesignature": ""}),
    ({"bpm": None, "keyscale": None, "timesignature": 0},
     {"bpm": 0, "keyscale": "", "timesignature": ""}),
    ({"bpm": " 300 ", "keyscale": " b♭ MAJOR ", "timesignature": 6},
     {"bpm": 300, "keyscale": "Bb major", "timesignature": "6"}),
])
def test_manual_ace_controls_use_explicit_unspecified_values(metadata, expected):
    assert normalize_ace_metadata(metadata, allow_unspecified=True) == expected


@pytest.mark.parametrize("metadata", [
    {**METADATA, "bpm": False}, {**METADATA, "bpm": 120.5}, {**METADATA, "bpm": "120.0"},
    {**METADATA, "bpm": 301}, {**METADATA, "keyscale": "C lydian"},
    {**METADATA, "timesignature": False}, {**METADATA, "timesignature": "4/4"},
    {**METADATA, "audio_codes": "codes"}, {**METADATA, "duration": 30},
])
def test_manual_metadata_rejects_invalid_values_and_code_injection(metadata):
    with pytest.raises(ValueError):
        normalize_ace_metadata(metadata, allow_unspecified=True)


@pytest.mark.parametrize("backend", ["llama_cpp", "ollama", "openai"])
def test_all_llm_modes_preserve_validated_ace_metadata_after_release(tmp_path, monkeypatch, backend):
    output = tmp_path / "result"
    events = []

    @contextmanager
    def lease(*args, **kwargs):
        yield

    @contextmanager
    def native(*args, **kwargs):
        try:
            yield "http://example.test/v1", "gemma"
        finally:
            assert not (output / "result.json").exists()
            events.append("released")

    def generate(*args, **options):
        assert options["music_backend"] == "ace_step15" and options["return_details"] is True
        events.append("caption")
        return {"prompt": CAPTION, "scene_interpretation": INTERPRETATION,
                "ace_metadata": {**METADATA, "keyscale": "c MAJOR"}}

    monkeypatch.setattr(llm_runner, "music_gpu_scope", lease)
    monkeypatch.setattr(llm_runtime, "managed_llama", native)
    monkeypatch.setattr(llm_runtime, "release_ollama", lambda *args: events.append("released"))
    monkeypatch.setattr(llm_runner, "request_llm_prompt", generate)
    source = scene()
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"backend": backend, "base_url": "http://example.test/v1",
                                  "model": "gemma", "local_model": "gemma", "options": {
                                      "style": "auto", "music_backend": "ace_step15"},
                                  "scene": {"id": source.id, "label": source.label,
                                            "context": source.context}}), encoding="utf-8")
    assert llm_runner.run_request(request, output) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["ace_metadata"] == METADATA
    assert result["native_planner_used"] is False and result["music_backend"] == "ace_step15"
    assert result["prompt"].startswith("Instrumental background music.")
    assert events == (["caption"] if backend == "openai" else ["caption", "released"])


@pytest.mark.parametrize("reply", [CAPTION, {"prompt": CAPTION, "scene_interpretation": INTERPRETATION},
                                        {"prompt": CAPTION, "scene_interpretation": INTERPRETATION,
                                         "ace_metadata": {**METADATA, "bpm": "116"}}])
def test_ace_runner_never_drops_invalid_metadata_into_a_success(reply):
    with pytest.raises(ValueError):
        llm_runner._normalize_prompt_result(reply, music_backend="ace_step15")


@pytest.mark.parametrize("backend", ["ace_step", "ace_step15_planner", "", None, {}])
def test_unknown_music_backend_is_rejected_before_llm_or_generation(monkeypatch, backend):
    monkeypatch.setattr(prompts, "urlopen", lambda *args, **kwargs: pytest.fail("network started"))
    with pytest.raises(ValueError, match="音楽モデル"):
        request_llm_prompt(scene(), "http://example.test/v1", "gemma", music_backend=backend)
    with pytest.raises(ValueError, match="音楽モデル"):
        generate_prompt(scene(), music_backend=backend)


def test_explicit_stable_runner_option_does_not_change_existing_request_or_result(tmp_path, monkeypatch):
    def generate(*args, **options):
        assert options == {"style": "auto", "tempo": 104, "return_details": True}
        return {"prompt": CAPTION, "scene_interpretation": INTERPRETATION}

    monkeypatch.setattr(llm_runner, "request_llm_prompt", generate)
    source = scene()
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"backend": "openai", "base_url": "http://example.test/v1",
                                  "model": "gemma", "options": {"style": "auto", "tempo": 104,
                                                               "music_backend": "stable_audio3"},
                                  "scene": {"id": source.id, "context": source.context}}), encoding="utf-8")
    output = tmp_path / "result"
    assert llm_runner.run_request(request, output) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["prompt"] == CAPTION
    assert "ace_metadata" not in result and "native_planner_used" not in result
    assert "music_backend" not in result


@pytest.mark.parametrize("content", [
    lambda: "<think>should produce vocals</think>\n```json\n" + json.dumps(answer()) + "\n```",
    lambda: "応答です。\n" + json.dumps(answer(), ensure_ascii=False),
])
def test_ace_json_wrappers_never_leak_reasoning_into_caption(monkeypatch, content):
    fake_endpoint(monkeypatch, [content()])
    result = request_llm_prompt(scene(), "http://example.test/v1", "gemma",
                                music_backend="ace_step15", return_details=True)
    assert result["prompt"] == "Instrumental background music. " + CAPTION
    assert result["ace_metadata"] == METADATA
