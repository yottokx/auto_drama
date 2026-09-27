"""Main-character slots are bounded without restricting supporting cast in events."""

from copy import deepcopy

import pytest

from services.worker.generation.script_plot import (
    ChainCharacter,
    DetailedPlot,
    PlotCastError,
    PlotCore,
    StoryChainDraft,
    bind_core_cast,
    check_main_cast,
)


@pytest.mark.parametrize("model,core_name,character_name", [
    (StoryChainDraft, "ChainCore", "ChainCharacter"),
    (DetailedPlot, "PlotCore", "PlotCharacter"),
    (PlotCore, "PlotCore", "PlotCharacter"),
])
@pytest.mark.parametrize("main_count", [1, 2, 3])
def test_core_schema_has_exact_main_slots_and_preserves_event_cast(
        model, core_name, character_name, main_count):
    main_ids = {f"main-{number}" for number in range(main_count, 0, -1)}
    available_ids = sorted(main_ids | {"supporting-guide", "supporting-neighbor"})
    schema = model.model_json_schema()
    definitions = schema["$defs"]
    if "ChainStep" in definitions:
        definitions["ChainStep"]["properties"]["character_id"]["enum"] = available_ids
    original = deepcopy(schema)

    bind_core_cast(schema, main_ids, core=core_name, character=character_name)

    core = schema if model is PlotCore else definitions[core_name]
    slots = core["properties"]["characters"]
    assert slots["minItems"] == slots["maxItems"] == main_count
    assert slots["items"] == {"$ref": f"#/$defs/{character_name}"}
    assert definitions[character_name]["properties"]["character_id"]["enum"] == sorted(main_ids)
    assert "各IDを1件ずつ" in slots["description"]
    assert "サブキャラの変化はevents" in slots["description"]
    if "ChainStep" in definitions:
        assert definitions["ChainStep"] == original["$defs"]["ChainStep"]
        assert definitions["ChainStep"]["properties"]["character_id"]["enum"] == available_ids
    # Binding one approval must not alter a subsequently generated base schema.
    fresh = model.model_json_schema()
    fresh_core = fresh if model is PlotCore else fresh["$defs"][core_name]
    assert fresh_core["properties"]["characters"]["maxItems"] == 10
    assert "enum" not in fresh["$defs"][character_name]["properties"]["character_id"]


def character_rows(ids):
    return [ChainCharacter(character_id=character_id,
        initial_behavior="一人で作業する。", enduring_value="約束を守る。",
        final_behavior="仲間に役目を任せる。") for character_id in ids]


@pytest.mark.parametrize("main_count", [1, 2, 3])
def test_main_cast_accepts_each_approved_id_once_in_any_order(main_count):
    ids = [f"main-{number}" for number in range(1, main_count + 1)]
    check_main_cast(character_rows(list(reversed(ids))), set(ids))


@pytest.mark.parametrize("actual,expected,missing,duplicate,unexpected", [
    (["main-a"], {"main-a", "main-b"}, ["main-b"], [], []),
    ([], {"main-a"}, ["main-a"], [], []),
    (["main-a", "main-a"], {"main-a"}, [], ["main-a"], []),
    (["main-a", "supporting-guide"], {"main-a"}, [], [], ["supporting-guide"]),
    (["main-a", "main-a"], {"main-a", "main-b"}, ["main-b"], ["main-a"], []),
    (["main-a", "main-a", "supporting-guide"], {"main-a", "main-b"},
     ["main-b"], ["main-a"], ["supporting-guide"]),
])
def test_main_cast_error_identifies_missing_duplicate_and_unknown_ids(
        actual, expected, missing, duplicate, unexpected):
    rows = character_rows(actual)
    before = [row.model_dump() for row in rows]
    with pytest.raises(PlotCastError) as caught:
        check_main_cast(rows, expected)

    message = str(caught.value)
    assert isinstance(caught.value, ValueError)
    assert message.startswith("Plot core must identify the approved main cast exactly once.")
    assert f"Expected IDs (one row each): {sorted(expected)}" in message
    assert f"received: {actual}" in message
    assert f"missing: {missing}" in message
    assert f"duplicate: {duplicate}" in message
    assert f"unexpected: {unexpected}" in message
    assert "サブキャラの変化をメインのIDへ割り当てない" in message
    assert [row.model_dump() for row in rows] == before
