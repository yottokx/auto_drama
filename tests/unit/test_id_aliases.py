"""Long character IDs reach the model as short aliases and never leave the LLM layer as aliases."""

import json

import pytest

from services.worker.generation.id_aliases import IdAliases, label_length
from services.worker.generation.llm import LocalLLM
from services.worker.generation.pipeline import ROOT, load_config
from services.worker.generation.script_budget import SceneSize, SceneTokenPolicy, scene_output_budget

LONG_A = "character-cef9b8d4-0077-443b-a739-b782ddffebf0"
LONG_B = "character-64957cff-ac1c-489c-b371-d37fd4150638"


def test_aliases_skip_numbers_the_request_already_uses():
    aliases = IdAliases(f"character-1 {LONG_A} character-2 {LONG_B} {LONG_A}")
    assert aliases.forward == {LONG_A: "character-3", LONG_B: "character-4"}


def test_decoding_respects_id_boundaries():
    aliases = IdAliases(f"{LONG_A} character-1")
    assert aliases.forward == {LONG_A: "character-2"}
    # A real ID that merely starts with the alias, or a longer number, is someone else.
    text = "character-2: はい。character-29a1-x と character-20、character-2の声"
    assert aliases.decode(text) == f"{LONG_A}: はい。character-29a1-x と character-20、{LONG_A}の声"


def test_encoding_reaches_schemas_and_grammars():
    aliases = IdAliases.for_request([{"role": "user", "content": LONG_A}],
                                    {"grammar": f'root ::= "{LONG_A}: "',
                                     "response_format": {"enum": [LONG_A, "character-1"]}})
    encoded = aliases.encode({"grammar": f'root ::= "{LONG_A}: "', "response_format": {"enum": [LONG_A]}})
    assert encoded == {"grammar": 'root ::= "character-2: "', "response_format": {"enum": ["character-2"]}}


def test_ids_at_line_starts_and_in_keys_are_replaced():
    # Script lines start with the speaker ID right after a line break.
    messages = [{"role": "user", "content": f"台本:\n{LONG_A}: やあ\n{LONG_B}: うん"}]
    extra = {"schema": {"properties": {LONG_B: {"type": "string"}}}}
    aliases = IdAliases.for_request(messages, extra)
    assert aliases.forward == {LONG_A: "character-1", LONG_B: "character-2"}
    assert aliases.encode(messages)[0]["content"] == "台本:\ncharacter-1: やあ\ncharacter-2: うん"
    assert aliases.encode(extra) == {"schema": {"properties": {"character-2": {"type": "string"}}}}


def test_requests_without_long_ids_are_unchanged():
    aliases = IdAliases.for_request([{"role": "user", "content": "character-1: やあ"}], {})
    assert not aliases
    value = {"messages": ["character-1"]}
    assert aliases.encode(value) is value


def test_chat_sends_aliases_and_returns_and_stores_real_ids(tmp_path):
    llm = LocalLLM(ROOT, load_config(), {"seed": 1}, tmp_path)
    sent = []

    def request(path, value):
        sent.append(value)
        speaker = value["grammar"].split('"')[1].removesuffix(": ")
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": f"{speaker}: こんにちは\ncharacter-1: やあ",
            "tool_calls": [{"id": "t", "function": {"name": "pick", "arguments": json.dumps({"who": speaker})}}]}}]}

    llm.request = request
    messages = [{"role": "user", "content": f"話者: {LONG_A}, character-1"}]
    message = llm.chat("scene", messages, grammar=f'root ::= "{LONG_A}: "')
    assert LONG_A not in json.dumps(sent)
    assert sent[0]["messages"][0]["content"] == "話者: character-2, character-1"
    assert message["content"] == f"{LONG_A}: こんにちは\ncharacter-1: やあ"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"who": LONG_A}
    saved = json.loads(next(tmp_path.glob("01-scene-*.json")).read_text("utf-8"))
    assert saved["response"]["choices"][0]["message"]["content"].startswith(LONG_A)
    assert saved["raw_response"]["choices"][0]["message"]["content"].startswith("character-2")
    assert saved["id_aliases"] == {"character-2": LONG_A}
    # An exact replay from the saved cache returns the same real IDs without a request.
    replay = LocalLLM(ROOT, load_config(), {"seed": 1}, tmp_path / "new", replay_outputs=(tmp_path,))
    replay.request = lambda *_: pytest.fail("Cached replay must not contact the model")
    assert replay.chat("scene", messages, grammar=f'root ::= "{LONG_A}: "')["content"] == message["content"]


def test_scene_budget_counts_labels_as_the_model_writes_them():
    assert label_length(LONG_A) == len("character-00")
    assert label_length("sub-gunter") == len("sub-gunter")
    size = SceneSize(length_weight=1, body_characters=2000, dialogue_characters=1200)
    short = scene_output_budget(size, ["character-1", LONG_A], SceneTokenPolicy(), 8192)
    assert short["estimated_label_characters"] == scene_output_budget(
        size, ["character-1", "character-2"], SceneTokenPolicy(), 8192)["estimated_label_characters"] + 30
