from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from contextlib import nullcontext

import pytest

from services.worker.generation import pipeline
from services.worker.generation.llm import LocalLLM
from services.worker.generation.processes import run_process
from services.worker.generation.random_tools import RandomTools
from services.worker.generation.schemas import (
    preserve_character,
    validate_relationships,
    validate_result,
    validate_schema,
)
from services.worker.generation.voice_runner import validate_request


def world():
    return {
        "title": "九つの庭",
        "prompt": "菌類文明の法廷劇",
        "genre": "菌類SF法廷劇",
        "mood": "乾いたユーモア",
        "notes": "",
        "chapterCount": 3,
        "setting": "菌類だけが暮らす庭で、土壌の記憶を証拠にした裁判が開かれる。",
    }


def character():
    return {
        "id": "character-1",
        "name": "ソイル",
        "age": "生育5周期",
        "gender": "無性別",
        "role": "菌糸の弁護士",
        "freeform": "",
        "settings": "土壌の記憶に欠落を見つけた。",
        "appearance": "白い菌糸で編んだ身体と紫の傘。",
        "voice": "低く静かな声。",
        "selfIntroduction": "私はソイル。土壌の記憶を読み、失われた声を法廷へ届ける菌糸の弁護士です。",
        "sampleLines": [
            "地面は、忘れていません。",
            "記憶にも空白はあります。",
            "まずは胞子を落ち着けましょう。",
        ],
        "locked": {"settings": False, "appearance": False, "voice": False},
    }


def job(kind="m2_world"):
    return {
        "id": "job-1",
        "kind": kind,
        "payload": {
            "schema_version": 1,
            "seed": 99,
            "character_contract_version": 2,
            "world_input": world(),
            "world_result": world(),
            "character_id": "character-1",
            "character_input": character(),
            "character_result": character(),
            "scope": "all",
            "instruction": "",
            "locked": character()["locked"],
            "base_revision": 1,
            "profile": {"provider": "local", "model_id": "gemma-4-31B-it-UD-Q4_K_XL"},
        },
    }


def test_four_random_tools_are_reproducible_and_trace_every_call():
    calls = [
        ("random_integer", {"minimum": -4, "maximum": 9}),
        ("random_choice", {"items": ["a", "b", "c"]}),
        ("random_weighted", {"items": ["a", "b", "c"], "weights": [0, 1, 0]}),
        ("random_sample", {"items": ["a", "b", "c"], "count": 2}),
    ]
    first, retry = RandomTools(551), RandomTools(551)
    for name, arguments in calls:
        assert first.execute(name, arguments) == retry.execute(name, arguments)
    assert first.trace == retry.trace
    assert first.trace[2]["result"] == {"value": "b"}
    assert len(set(first.trace[3]["result"]["value"])) == 2
    assert all(record["seed"] == 551 and "call_seed" in record for record in first.trace)


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("shell", {}),
        ("random_integer", {"minimum": True, "maximum": 5}),
        ("random_integer", {"minimum": 5, "maximum": 1}),
        ("random_choice", {"items": []}),
        ("random_choice", {"items": ["a", "a"]}),
        ("random_weighted", {"items": ["a"], "weights": [float("nan")]}),
        ("random_weighted", {"items": ["a"], "weights": [0]}),
        ("random_weighted", {"items": ["a"], "weights": [-1]}),
        ("random_sample", {"items": ["a"], "count": 2}),
        ("random_sample", {"items": ["a"], "count": False}),
        ("random_choice", {"items": ["a"], "code": "ignored"}),
    ],
)
def test_random_tools_reject_bad_calls(name, arguments):
    with pytest.raises(ValueError):
        RandomTools(2).execute(name, arguments)


def test_random_tool_limit_rejects_excess_without_consuming_draws():
    tools = RandomTools(5, limit=1)
    tools.execute("random_choice", {"items": ["a", "b"]})
    with pytest.raises(ValueError, match="limit"):
        tools.execute("random_choice", {"items": ["a", "b"]})
    assert len(tools.trace) == 1


def test_partial_revision_and_locks_protect_every_related_field():
    payload = job("m2_character")["payload"]
    generated = {
        key: "変更" if isinstance(value, str) else value for key, value in character().items()
    }
    generated["id"] = payload["character_id"]
    payload["scope"] = "voice"
    actual = preserve_character(generated, payload)
    assert actual["voice"] == "変更"
    assert {key: actual[key] for key in character() if key != "voice"} == {
        key: value for key, value in character().items() if key != "voice"
    }
    payload["scope"] = "all"
    payload["locked"] = {"settings": True, "appearance": True, "voice": False}
    actual = preserve_character(generated, payload)
    assert actual["name"] == character()["name"]
    assert actual["gender"] == character()["gender"]
    assert actual["appearance"] == character()["appearance"]
    assert actual["voice"] == "変更"
    assert actual["id"] == "character-1"
    assert actual["locked"] == payload["locked"]


@pytest.mark.parametrize(
    "field,bad", [("chapterCount", True), ("title", ""), ("setting", " "), ("genre", None)]
)
def test_incomplete_world_is_rejected(field, bad):
    value = world()
    value[field] = bad
    with pytest.raises(ValueError):
        validate_result("m2_world", value)


def test_freely_described_nonhuman_character_is_valid():
    assert validate_result("m2_character", character()) == character()


@pytest.mark.parametrize(
    "field,value",
    [("selfIntroduction", " "), ("sampleLines", ["一言"]), ("sampleLines", ["一言", "", "三言"])],
)
def test_new_character_requires_complete_spoken_lines(field, value):
    result = character()
    result[field] = value
    with pytest.raises(ValueError):
        validate_result("m2_character", result)


def test_voice_lock_preserves_its_spoken_introduction_during_settings_revision():
    payload = job("m2_character")["payload"]
    payload["locked"]["voice"] = True
    generated = character()
    generated["selfIntroduction"] = "変更された自己紹介です。"
    generated["sampleLines"] = ["新しい一言。", "新しい二言。", "新しい三言。"]
    result = preserve_character(generated, payload)
    assert result["selfIntroduction"] == character()["selfIntroduction"]
    assert result["sampleLines"] == generated["sampleLines"]


def test_old_completed_character_can_still_validate_under_its_original_contract():
    legacy = character()
    del legacy["selfIntroduction"], legacy["sampleLines"]
    assert validate_result("m2_character", legacy, contract_version=1) == legacy
    with pytest.raises(ValueError):
        validate_result("m2_character", legacy)


def cast(count=3):
    return [
        {**character(), "id": f"character-{index}", "name": f"ソイル{index}"}
        for index in range(1, count + 1)
    ]


def relationships(characters):
    ids = sorted(item["id"] for item in characters)
    return {
        "pairs": [
            {
                "characterIds": [first, second],
                "summary": "土壌の記録を調査する仕事仲間。",
                "firstToSecond": "慎重さを頼りにしている。",
                "secondToFirst": "行動力を認めている。",
            }
            for index, first in enumerate(ids)
            for second in ids[index + 1 :]
        ]
    }


def relationship_response(result, messages, schema):
    assignments, _ = json.JSONDecoder().raw_decode(
        messages[-1]["content"].split("出力キーごとの確定済み人物対応: ", 1)[1]
    )
    response = {}
    for key, pair in assignments.items():
        if "pairs" in result:
            ids = [pair["first"]["id"], pair["second"]["id"]]
            description = copy.deepcopy(next(
                item for item in result["pairs"] if item["characterIds"] == sorted(ids)
            ))
            if ids != sorted(ids):
                description["firstToSecond"], description["secondToFirst"] = (
                    description["secondToFirst"], description["firstToSecond"]
                )
        else:
            description = result[key]
        response[key] = {
            field: description[field.split("（", 1)[0]]
            for field in schema["properties"][key]["properties"]
        }
    return response


@pytest.mark.parametrize("count", [2, 3])
def test_relationships_require_every_unordered_pair_with_both_directions(count):
    people = cast(count)
    assert validate_relationships(relationships(people), people) == relationships(people)
    incomplete = relationships(people)
    incomplete["pairs"].pop()
    with pytest.raises(ValueError):
        validate_relationships(incomplete, people)
    duplicated = relationships(people)
    duplicated["pairs"].append(copy.deepcopy(duplicated["pairs"][0]))
    with pytest.raises(ValueError):
        validate_relationships(duplicated, people)
    reversed_pair = relationships(people)
    reversed_pair["pairs"][0]["characterIds"].reverse()
    unchanged = copy.deepcopy(reversed_pair)
    normalized = validate_relationships(reversed_pair, people)
    assert normalized["pairs"][0]["characterIds"] == sorted(
        reversed_pair["pairs"][0]["characterIds"]
    )
    assert normalized["pairs"][0]["firstToSecond"] == reversed_pair["pairs"][0]["secondToFirst"]
    assert normalized["pairs"][0]["secondToFirst"] == reversed_pair["pairs"][0]["firstToSecond"]
    assert reversed_pair == unchanged


def test_relationships_reject_reversed_duplicate_pair_and_self_relationship():
    people = cast()
    duplicated = relationships(people)
    duplicated["pairs"][2] = copy.deepcopy(duplicated["pairs"][0])
    duplicated["pairs"][2]["characterIds"].reverse()
    with pytest.raises(ValueError, match="every distinct character pair"):
        validate_relationships(duplicated, people)
    self_pair = relationships(people)
    self_pair["pairs"][0]["characterIds"] = [people[0]["id"], people[0]["id"]]
    with pytest.raises(ValueError, match="every distinct character pair"):
        validate_relationships(self_pair, people)


def test_relationships_reject_unknown_identity_and_empty_perspective():
    people = cast()
    unknown = relationships(people)
    unknown["pairs"][0]["characterIds"][1] = "not-a-character"
    with pytest.raises(ValueError):
        validate_relationships(unknown, people)
    empty = relationships(people)
    empty["pairs"][0]["secondToFirst"] = " "
    with pytest.raises(ValueError):
        validate_relationships(empty, people)


def image_features(**overrides):
    return {
        "subject": "nonhuman purple mushroom character",
        "body": "A body woven from white mycelium with a purple mushroom cap.",
        "skin": "",
        "hair": "",
        "eyes": "",
        "clothing": "",
        "accessories": "",
        "other_features": "",
        "rendering": "",
        **overrides,
    }


class ImageResponseLLM:
    def __init__(self, response):
        self.response = response
        self.trace = []

    def structured(self, stage, messages, schema):
        assert stage == "image-prompt"
        return copy.deepcopy(self.response)


class FakeLLM:
    entered = 0
    exited = 0

    def __init__(self, root, config, payload, output):
        self.trace = []
        self.base = {"model": {"revision": "pinned", "publisher_sha256": "hash"}}
        self.profile = {"temperature": 0.5, "max_tokens": 2048}
        self.requests = 0
        self.payload = payload
        self.calls = []

    def __enter__(self):
        type(self).entered += 1
        return self

    def __exit__(self, *args):
        type(self).exited += 1

    def retry_failed_from(self, request_index):
        self.retry_request_index = request_index

    def structured(self, stage, messages, schema):
        self.requests += 1
        self.calls.append({"stage": stage, "messages": copy.deepcopy(messages),
                           "schema": copy.deepcopy(schema)})
        if stage == "image-prompt":
            assert self.payload["character_result"]["appearance"] in messages[-1]["content"]
        else:
            assert "菌類" in messages[-1]["content"]
        if "-requirements-review-" in stage:
            return {"issues": []}
        if stage.endswith("candidates"):
            return {
                "candidates": [
                    {"id": key, "concept": "菌類の庭" + key, "tension": "土壌の断絶"}
                    for key in "ABC"
                ]
            }
        if stage.endswith("selected"):
            return {"selected_id": "B", "reason": "指定の菌類文明を深めるため。"}
        if stage == "image-prompt":
            return image_features(
                rendering="Front view with visible hands." if self.payload["instruction"] else ""
            )
        if stage == "relationships-final":
            return relationship_response(relationships(self.payload["cast_results"]), messages, schema)
        if stage == "final" and "id" in schema["properties"] and (
            "selfIntroduction" not in schema["properties"]
        ):
            return {
                key: value
                for key, value in {**character(), "height_cm": None, "body_type": "unknown"}.items()
                if key not in ("selfIntroduction", "sampleLines")
            }
        return world() if schema == pipeline.WORLD_SCHEMA else {
            **character(), "height_cm": None, "body_type": "unknown",
        }

    def random_context(self, stage, messages):
        tools = RandomTools(self.payload["seed"])
        tools.execute("random_choice", {"items": ["A", "B", "C"]})
        self.trace.append({**tools.trace[-1], "stage": stage})
        return [*messages, {"role": "assistant", "content": "Bを採用します。"}]


@pytest.fixture
def fake_runtime(monkeypatch):
    FakeLLM.entered = FakeLLM.exited = 0
    monkeypatch.setattr(pipeline, "LocalLLM", FakeLLM)
    monkeypatch.setattr(pipeline, "gpu_lock", lambda *args: nullcontext())


def unpack(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return json.loads(archive.read("result.json")), set(archive.namelist())


def test_blank_world_brief_uses_hierarchy_and_retry_reuses_completed_bytes(tmp_path, fake_runtime):
    request = job()
    request["payload"]["world_input"].update(prompt="", setting="", notes="")
    data = pipeline.generate_job(request, tmp_path)
    envelope, names = unpack(data)
    assert names == {"result.json"}
    assert envelope["result"] == {**world(), "prompt": ""}
    assert [entry["type"] for entry in envelope["trace"]].count("adoption") == 2
    assert [entry["type"] for entry in envelope["trace"]].count("candidates") == 2
    assert [entry["type"] for entry in envelope["trace"]].count("random_tool") == 2
    assert pipeline.generate_job(request, tmp_path) == data
    assert FakeLLM.entered == FakeLLM.exited == 1
    different = copy.deepcopy(request)
    different["payload"]["seed"] += 1
    with pytest.raises(ValueError, match="another input"):
        pipeline.generate_job(different, tmp_path)


@pytest.mark.parametrize("brief_field", ["prompt", "setting", "notes"])
def test_authored_world_brief_goes_directly_to_generation_without_stale_cast_results(brief_field):
    payload = job()["payload"]
    payload["world_input"].update(prompt="", setting="", notes="")
    payload["world_input"][brief_field] = "菌類だけが暮らす庭で、土壌の記憶を証拠にした裁判を開く。"
    payload.update(
        cast_inputs=[{**character(), "name": "指定した菌類の司書"}],
        relationship_inputs=[{"characterIds": ["one", "two"], "instruction": "親子ではなく師弟"}],
        cast_results=[{"name": "旧世界で生成された別人"}],
        character_result={"name": "旧世界で生成された別人"},
        relationships_result={"summary": "旧世界の古い関係性"},
    )
    original = copy.deepcopy(payload)
    captured = []

    class CombinedBriefLLM(FakeLLM):
        def capture(self, stage, messages):
            data, _ = json.JSONDecoder().raw_decode(messages[-1]["content"])
            assert data["world_input"] == payload["world_input"]
            assert data["cast_inputs"] == payload["cast_inputs"]
            assert data["relationship_inputs"] == payload["relationship_inputs"]
            assert "cast_results" not in data and "character_result" not in data
            assert "relationships_result" not in data
            assert "旧世界で生成された別人" not in messages[-1]["content"]
            assert "旧世界の古い関係性" not in messages[-1]["content"]
            captured.append(stage)

        def structured(self, stage, messages, schema):
            self.capture(stage, messages)
            return super().structured(stage, messages, schema)

        def random_context(self, stage, messages):
            self.capture(stage, messages)
            return super().random_context(stage, messages)

    llm = CombinedBriefLLM(None, None, payload, None)
    pipeline.generate_text("m2_world", payload, llm)
    assert captured == ["final", "m2_world-requirements-review-1"]
    assert not any(entry["type"] in {"candidates", "adoption", "random_tool"} for entry in llm.trace)
    assert payload == original


def test_character_generation_uses_original_brief_directly_with_confirmed_world():
    payload = job("m2_character")["payload"]
    payload["world_input"]["setting"] = "確認前の世界観"
    payload["world_result"] = {**world(), "setting": "確認して確定した菌類の世界観"}
    payload["cast_inputs"] = [payload["character_input"]]
    original_schema = copy.deepcopy(pipeline.CHARACTER_SCHEMA)
    captured = []

    class ConfirmedWorldLLM(FakeLLM):
        def structured(self, stage, messages, schema):
            data, _ = json.JSONDecoder().raw_decode(messages[-1]["content"])
            assert data["world_result"]["setting"] == "確認して確定した菌類の世界観"
            assert data["character_input"] == payload["character_input"]
            assert data["cast_inputs"] == payload["cast_inputs"]
            captured.append(stage)
            return super().structured(stage, messages, schema)

    llm = ConfirmedWorldLLM(None, None, payload, None)
    pipeline.generate_text("m2_character", payload, llm)
    assert captured == ["final", "m2_character-requirements-review-1"]
    assert not any(entry["type"] in {"candidates", "adoption", "random_tool"} for entry in llm.trace)
    generated_schema = llm.calls[0]["schema"]
    order = list(generated_schema["properties"])
    assert generated_schema["required"] == order
    assert set(order) == set(original_schema["properties"])
    assert order.index("selfIntroduction") < order.index("appearance")
    assert order.index("sampleLines") < order.index("appearance")
    assert pipeline.CHARACTER_SCHEMA == original_schema


def _brief_requirement_case(kind):
    request = job(kind)
    payload = request["payload"]
    people = cast(2)
    payload.update(cast_inputs=copy.deepcopy(people), relationship_inputs=[])
    if kind == "m2_world":
        requirement = "菌類だけが暮らし、人間は存在しない世界"
        payload["world_input"]["prompt"] = requirement
        rejected = {**world(), "setting": "人間の王が菌類の市民を統治する庭。",
                    "prompt": "勝手に変更された入力", "chapterCount": 7}
        repaired = {**world(), "setting": "菌類だけが暮らす庭。人間は存在せず、胞子の議会が裁判を開く。",
                    "prompt": "修正時にも変更された入力", "chapterCount": 8}
        marker = "setting"
    elif kind == "m2_character":
        requirement = "生まれつき目が見えず、視覚能力や視力回復はない菌類の弁護士"
        payload["character_input"]["freeform"] = requirement
        payload["cast_inputs"][0]["freeform"] = requirement
        payload["character_result"] = None
        rejected = {**character(), "settings": "土壌の発光を鋭い目で見抜く菌類の弁護士。",
                    "height_cm": None, "body_type": "unknown"}
        repaired = {**rejected, "settings": "生まれつき目が見えず、視覚能力はない。土壌の振動を証拠にする弁護士。"}
        marker = "settings"
    else:
        requirement = "二人は血のつながった兄弟であり、師弟ではない"
        payload.update(
            character_id=None, cast_results=people, relationships_result=None,
            relationship_inputs=[{"characterIds": [person["id"] for person in people],
                                  "instruction": requirement}],
            scope="relationships",
        )
        rejected = {"pair_1": {
            "summary": "血縁のない菌類の師弟。", "firstToSecond": "弟子の成長を喜ぶ。",
            "secondToFirst": "師を尊敬する。",
        }}
        repaired = {"pair_1": {
            "summary": "血のつながった菌類の兄弟で、法廷で協力する。", "firstToSecond": "弟を頼りにする。",
            "secondToFirst": "兄の慎重さを認める。",
        }}
        marker = "summary"
    issue = {"requirement": requirement, "source_quote": requirement,
             "problem": "生成した設定がこの明示的な指定と矛盾している。"}
    return request, rejected, repaired, marker, issue


def _reviewed_value(result, marker):
    if "pair_1" in result:
        return result["pair_1"][marker]
    return result["pairs"][0][marker] if "pairs" in result else result[marker]


@pytest.mark.parametrize("kind", ["m2_world", "m2_character", "m2_relationships"])
def test_final_brief_review_repairs_contradictions_before_adopting_result(kind):
    request, rejected, repaired, marker, issue = _brief_requirement_case(kind)
    payload = request["payload"]
    original = copy.deepcopy(payload)
    reviews = []

    class RepairingLLM(FakeLLM):
        def structured(self, stage, messages, schema):
            response = super().structured(stage, messages, schema)
            content = messages[-1]["content"]
            if stage in {"final", "relationships-final"}:
                response = copy.deepcopy(rejected)
                if kind == "m2_relationships":
                    response = relationship_response(response, messages, schema)
            elif "-requirements-review-" in stage:
                assert not any(item["type"] == "complete_result" for item in self.trace)
                context, _ = json.JSONDecoder().raw_decode(content)
                assert context["world_input"] == payload["world_input"]
                assert context["cast_inputs"] == (
                    [payload["cast_inputs"][0]] if kind == "m2_character" else payload["cast_inputs"]
                )
                assert context["relationship_inputs"] == payload["relationship_inputs"]
                sources, _ = json.JSONDecoder().raw_decode(
                    content.split("\noriginal_user_inputs: ", 1)[1]
                )
                assert sources["world_input"] == payload["world_input"]
                assert sources["cast_inputs"] == context["cast_inputs"]
                assert sources["relationship_inputs"] == payload["relationship_inputs"]
                assert not {"world_result", "cast_results", "character_result"} & sources.keys()
                reviewed_result = json.loads(content.split("\nresult: ", 1)[1])
                expected = rejected if not reviews else repaired
                assert _reviewed_value(reviewed_result, marker) == _reviewed_value(expected, marker)
                if kind == "m2_world":
                    assert reviewed_result["prompt"] == payload["world_input"]["prompt"]
                    assert reviewed_result["chapterCount"] == payload["world_input"]["chapterCount"]
                reviews.append(stage)
                response = {"issues": [issue] if len(reviews) == 1 else []}
            elif "-requirements-repair-" in stage:
                assert not any(item["type"] == "complete_result" for item in self.trace)
                assert issue["source_quote"] in content
                if kind == "m2_character":
                    assert pipeline.VOICE_DESIGN_RULES in content
                    assert "地の文・役名ラベル・括弧による演技説明を含めません。" in content
                    assert "sampleLinesも実際に発声する本文だけ" in content
                    assert "実際に出す鳴き声だけをselfIntroduction/sampleLinesに入れます。" in content
                elif kind == "m2_world":
                    assert "一つの完全な世界設定を日本語JSONで出力" in content
                    assert "未確定の候補や質問は残しません" in content
                    assert "promptは元の入力をそのまま保持" in content
                    assert "chapterCountは入力の章数" in content
                response = copy.deepcopy(repaired)
                if kind == "m2_relationships":
                    response = relationship_response(response, messages, schema)
            validate_schema(response, schema)
            return response

    llm = RepairingLLM(None, None, payload, None)
    result = pipeline.generate_relationships(payload, llm) if kind == "m2_relationships" else (
        pipeline.generate_text(kind, payload, llm)
    )

    if kind == "m2_relationships":
        expected = {"pairs": [{"characterIds": ["character-1", "character-2"],
                               **repaired["pair_1"]}]}
    elif kind == "m2_world":
        expected = {**repaired, "prompt": payload["world_input"]["prompt"],
                    "chapterCount": payload["world_input"]["chapterCount"]}
    else:
        expected = repaired
    assert result == expected
    assert payload == original
    assert reviews == [f"{kind}-requirements-review-1", f"{kind}-requirements-review-2"]
    assert [call["stage"] for call in llm.calls][-3:] == [
        f"{kind}-requirements-review-1", f"{kind}-requirements-repair-1",
        f"{kind}-requirements-review-2",
    ]
    assert sum(item["type"] == "complete_result" for item in llm.trace) == 1


@pytest.mark.parametrize("kind", ["m2_world", "m2_character", "m2_relationships"])
def test_accepted_instruction_history_reaches_generation_review_and_repairs(kind):
    request, rejected, repaired, _, _ = _brief_requirement_case(kind)
    payload = request["payload"]
    if kind == "m2_world":
        field, requirement = "title", "胞子の約束"
        repaired[field] = requirement
        changes = {field: requirement}
        scope = "world"
    elif kind == "m2_character":
        field, requirement = "name", "コケ"
        repaired[field] = requirement
        changes = {field: requirement}
        scope = "settings"
    else:
        field, requirement = "summary", "二人は同じ住居で暮らしている"
        repaired["pair_1"][field] += requirement + "。"
        changes = {"pairs": [{"characterIds": ["character-1", "character-2"],
                               **repaired["pair_1"]}]}
        scope = "relationships"
    history = [
        {"kind": "m2_world", "character_id": None, "scope": "world",
         "instruction": "菌類の社会は平和的で、対立は裁判で解決する。"},
        {"kind": kind, "character_id": "character-1" if kind == "m2_character" else None,
         "scope": scope, "changes": changes},
    ]
    payload["applied_instructions"] = copy.deepcopy(history)
    original = copy.deepcopy(payload)
    stages = []
    reviews = []

    class HistoricalBriefLLM(FakeLLM):
        def capture(self, stage, messages):
            context, _ = json.JSONDecoder().raw_decode(messages[-1]["content"])
            assert context["applied_instructions"] == history
            assert context["world_input"] == original["world_input"]
            assert context["relationship_inputs"] == original["relationship_inputs"]
            if kind == "m2_character":
                assert context["character_input"] == original["character_input"]
            stages.append(stage)

        def random_context(self, stage, messages):
            self.capture(stage, messages)
            return super().random_context(stage, messages)

        def structured(self, stage, messages, schema):
            self.capture(stage, messages)
            response = super().structured(stage, messages, schema)
            if stage in {"final", "relationships-final"}:
                response = copy.deepcopy(rejected)
                if kind == "m2_relationships":
                    response = relationship_response(response, messages, schema)
            elif "-requirements-review-" in stage:
                sources, _ = json.JSONDecoder().raw_decode(
                    messages[-1]["content"].split("\noriginal_user_inputs: ", 1)[1]
                )
                assert sources["applied_instructions"] == history
                assert sources["world_input"] == original["world_input"]
                assert sources["relationship_inputs"] == original["relationship_inputs"]
                reviews.append(stage)
                response = {"issues": [{"requirement": requirement,
                                        "source_quote": requirement,
                                        "problem": "適用済みの変更が設定結果に反映されていない。"}]
                            if len(reviews) == 1 else []}
            elif "-requirements-repair-" in stage:
                response = copy.deepcopy(repaired)
                if kind == "m2_relationships":
                    response = relationship_response(response, messages, schema)
            validate_schema(response, schema)
            return response

    llm = HistoricalBriefLLM(None, None, payload, None)
    result = pipeline.generate_relationships(payload, llm) if kind == "m2_relationships" else (
        pipeline.generate_text(kind, payload, llm)
    )

    assert requirement in _reviewed_value(result, field)
    assert payload == original
    assert stages == ["relationships-final" if kind == "m2_relationships" else "final",
                      f"{kind}-requirements-review-1", f"{kind}-requirements-repair-1",
                      f"{kind}-requirements-review-2"]
    assert llm.trace[-1]["type"] == "complete_result"


@pytest.mark.parametrize("kind", ["m2_world", "m2_character", "m2_relationships"])
def test_persistently_violated_brief_never_becomes_a_completed_bundle(
    kind, tmp_path, fake_runtime, monkeypatch,
):
    request, rejected, _, _, issue = _brief_requirement_case(kind)
    instances = []

    class RejectingLLM(FakeLLM):
        def __init__(self, *args):
            super().__init__(*args)
            instances.append(self)

        def structured(self, stage, messages, schema):
            response = super().structured(stage, messages, schema)
            if "-requirements-review-" in stage:
                response = {"issues": [issue]}
            elif stage in {"final", "relationships-final"} or "-requirements-repair-" in stage:
                response = copy.deepcopy(rejected)
                if kind == "m2_relationships":
                    response = relationship_response(response, messages, schema)
            validate_schema(response, schema)
            return response

    monkeypatch.setattr(pipeline, "LocalLLM", RejectingLLM)
    with pytest.raises(ValueError, match="STEP1"):
        pipeline.generate_job(request, tmp_path)

    assert not (tmp_path / "result.zip").exists()
    assert not any(item["type"] == "complete_result" for item in instances[0].trace)
    assert instances[0].retry_request_index == 1
    reviews = [call["stage"] for call in instances[0].calls if "-requirements-review-" in call["stage"]]
    repairs = [call["stage"] for call in instances[0].calls if "-requirements-repair-" in call["stage"]]
    assert reviews == [f"{kind}-requirements-review-{attempt}" for attempt in range(1, 4)]
    assert repairs == [f"{kind}-requirements-repair-{attempt}" for attempt in range(1, 3)]


def test_candidate_selection_schema_excludes_character_ids_even_after_distracted_consideration():
    class DistractedLLM(FakeLLM):
        def random_context(self, stage, messages):
            return [
                *messages,
                {
                    "role": "assistant",
                    "content": 'Aを採用します。人物: {"id":"character-1","name":"ソイル"}',
                },
            ]

        def structured(self, stage, messages, schema):
            if stage.endswith("selected"):
                selected_id_schema = schema["properties"]["selected_id"]
                assert selected_id_schema["enum"] == ["A", "B", "C"]
                with pytest.raises(ValueError, match="permitted"):
                    validate_schema("character-1", selected_id_schema)
                return {"selected_id": "A", "reason": "菌類文明に合うため。"}
            return super().structured(stage, messages, schema)

    llm = DistractedLLM(None, None, job()["payload"], None)
    selected = pipeline._select(llm, "details", "菌類文明", "本人の背景を具体化")
    assert selected["id"] == "A"


def test_media_starts_after_llm_exit_and_does_not_change_settings(
    tmp_path, fake_runtime, monkeypatch
):
    def image(payload, prompt, work, config):
        assert FakeLLM.entered == FakeLLM.exited == 1
        assert "nonhuman" in prompt
        return b"png-fixture", {"alpha_range": [0, 255]}

    monkeypatch.setattr(pipeline, "generate_image", image)
    source = job("m2_image")
    unchanged = copy.deepcopy(source)
    envelope, names = unpack(pipeline.generate_job(source, tmp_path))
    assert source == unchanged
    assert names == {"result.json", "image.png"}
    assert envelope["result"] == {}


@pytest.mark.parametrize("target_index", [0, 1, 2])
def test_image_translation_only_receives_the_selected_completed_appearance(target_index):
    people = [
        {**character(), "id": "character-1", "name": "記録係", "age": "17",
         "appearance": "細身、黒髪、紺色の目。古びた革表紙の日記帳を持つ。"},
        {**character(), "id": "character-958ce9a3-0243-4df2-9951-b5b07d639436",
         "name": "案内役", "appearance": "銀白色の長髪、白いリボン、琥珀色の目。"},
        {**character(), "id": "character-a002fb97-86e8-4c36-96de-4b72301e001d",
         "name": "競争相手", "age": "18",
         "appearance": "185cmの強靭な体格。逆立ったプラチナブロンド、深紅の目。黒い宝石の金の指輪。"},
    ]
    selected = people[target_index]
    payload = job("m2_image")["payload"]
    payload.update(
        character_id=selected["id"], character_result=selected,
        character_input={"id": selected["id"], "appearance": "変更前の古い入力"},
        cast_inputs=people[target_index:] + people[:target_index],
        cast_results=list(reversed(people)),
        world_result={"setting": "無関係な人物が登場する世界設定"},
        relationship_inputs=[{"instruction": "無関係な人物間の関係"}],
        relationships_result={"summary": "別の人物の特徴"},
        instruction="正面向きに、手を見せる構図で",
    )
    original = copy.deepcopy(payload)

    class CaptureImageLLM:
        def __init__(self):
            self.trace = []

        def structured(self, stage, messages, schema):
            assert stage == "image-prompt"
            data, _ = json.JSONDecoder().raw_decode(messages[-1]["content"])
            assert data == {
                "target_character": {
                    key: selected[key] for key in ("id", "name", "age", "gender", "appearance")
                },
                "retake_instruction": payload["instruction"],
            }
            sent = json.dumps(messages, ensure_ascii=False)
            for other in people:
                if other["id"] != selected["id"]:
                    assert other["appearance"] not in sent
                    assert other["id"] not in sent
            for unrelated in ("変更前の古い入力", "無関係な人物", "別の人物の特徴"):
                assert unrelated not in sent
            return image_features(
                subject="the selected character",
                body="A detailed depiction of the selected character's physical features.",
                rendering="Front view with visible hands.",
            )

    llm = CaptureImageLLM()
    pipeline.image_prompt(payload, llm)
    assert payload == original
    assert llm.trace[-1]["character_id"] == selected["id"]
    assert llm.trace[-1]["source_appearance"] == selected["appearance"]


@pytest.mark.parametrize(
    "response",
    [
        {key: "" for key in image_features()},
        {key: " \n " for key in image_features()},
        {
            **{key: "" for key in image_features()},
            "rendering": "Front view, dramatic lighting and detailed illustration.",
        },
    ],
)
def test_image_translation_requires_subject_features_before_adding_composition(response):
    llm = ImageResponseLLM(response)
    with pytest.raises(ValueError):
        pipeline.image_prompt(job("m2_image")["payload"], llm)
    assert llm.trace == []


@pytest.mark.parametrize(
    "response",
    [
        {key: value for key, value in image_features().items() if key != "skin"},
        image_features(skin=None),
        image_features(eyes=["purple"]),
        image_features(unsupported="extra field"),
        image_features(body="白い菌糸で編まれた身体。"),
        image_features(skin="fair skin、紫色の斑点"),
        image_features(rendering="正面から描く。"),
        image_features(skin="123"),
        image_features(body="large body " * 120),
        {key: "descriptive text " * 70 for key in image_features()},
    ],
)
def test_image_translation_rejects_malformed_or_untranslated_features(response):
    llm = ImageResponseLLM(response)
    with pytest.raises(ValueError):
        pipeline.image_prompt(job("m2_image")["payload"], llm)
    assert llm.trace == []


@pytest.mark.parametrize("bad_accessories", [
    'a handmade amulet hanging over armor with the text "目標達成まであと3％！"',
    "a handmade motivational amulet hanging over armor " * 30,
])
def test_image_translation_repairs_quoted_japanese_and_excessive_text_without_losing_features(
    bad_accessories,
):
    payload = job("m2_image")["payload"]
    payload["character_result"]["appearance"] = (
        "赤褐色の肌に折れた角を持つ屈強なオーク系の魔族。重厚な鎧の上に"
        "『目標達成まであと3％！』と書かれた手作りのお守り。充血した目と疲労困憊の表情。"
    )
    unchanged = copy.deepcopy(payload)
    original = image_features(
        subject="a male orc-like demon", body="strong build", skin="reddish-brown skin",
        eyes="bloodshot eyes", clothing="heavy armor", accessories=bad_accessories,
        other_features="broken horns, exhausted facial expression",
    )
    corrected = {**original, "accessories": (
        "a handmade amulet hanging over the armor with a motivational inscription "
        "meaning only 3% left to reach the target"
    )}
    calls = []

    class RepairLLM(ImageResponseLLM):
        def structured(self, stage, messages, schema):
            calls.append(copy.deepcopy(messages))
            return copy.deepcopy(original if len(calls) == 1 else corrected)

    llm = RepairLLM(original)
    prompt = pipeline.image_prompt(payload, llm)
    assert len(calls) == 2
    assert calls[1][:2] == calls[0]
    assert json.loads(calls[1][-2]["content"]) == original
    assert "accessories:" in calls[1][-1]["content"]
    for feature in corrected.values():
        if feature:
            assert feature in prompt
    assert "目標" not in prompt
    assert payload == unchanged
    assert llm.trace[0]["type"] == "image_prompt_repair"
    assert llm.trace[-1]["source_appearance"] == payload["character_result"]["appearance"]


def test_image_translation_stops_after_one_unsuccessful_repair():
    calls = []

    class InvalidLLM(ImageResponseLLM):
        def structured(self, stage, messages, schema):
            calls.append(copy.deepcopy(messages))
            return super().structured(stage, messages, schema)

    llm = InvalidLLM(image_features(accessories='an amulet reading "目標達成まであと3％！"'))
    with pytest.raises(ValueError, match="accessories: contains untranslated"):
        pipeline.image_prompt(job("m2_image")["payload"], llm)
    assert len(calls) == 2
    assert llm.trace == []


def test_image_translation_preserves_explicit_nonhuman_materials_and_part_colors():
    payload = job("m2_image")["payload"]
    payload["character_result"]["appearance"] = (
        "半透明の青いスライムの身体。長い銀髪と紫色の瞳。白いドレスを着ている。"
    )
    features = image_features(
        subject="a slime character",
        body="A translucent gelatinous body.",
        skin="blue skin",
        hair="long silver hair",
        eyes="purple irises",
        clothing="a white dress",
        rendering="Front view with visible hands.",
    )
    llm = ImageResponseLLM(features)
    unchanged = copy.deepcopy(payload)
    prompt = pipeline.image_prompt(payload, llm)
    for description in features.values():
        if description:
            assert description in prompt
    assert "human skin" not in prompt.lower()
    assert "fair skin" not in prompt.lower()
    assert payload == unchanged
    trace = llm.trace[-1]
    assert trace["prompt"] == prompt
    assert trace["prompt_version"] == 5
    assert trace["visual_features"] == {
        key: value for key, value in features.items() if key != "rendering"
    }
    assert trace["rendering_instruction"] == features["rendering"]
    assert trace["composition_prompt"] in prompt


def test_image_translation_does_not_fill_unspecified_features_with_human_defaults():
    features = image_features(subject="a floating geometric being", body="a silver cube")
    llm = ImageResponseLLM(features)
    prompt = pipeline.image_prompt(job("m2_image")["payload"], llm)
    for unrequested in ("human", "skin", "hair", "irises", "dress", "healthy"):
        assert unrequested not in prompt.lower()
    assert llm.trace[-1]["visual_features"]["skin"] == ""
    assert llm.trace[-1]["visual_features"]["clothing"] == ""


def test_image_translation_rejects_another_characters_result_before_calling_llm():
    payload = job("m2_image")["payload"]
    payload["character_id"] = "another-character"
    with pytest.raises(ValueError, match="target ID"):
        pipeline.image_prompt(payload, None)


@pytest.fixture
def recorded_image_runtime(monkeypatch):
    calls = []

    def run(command, *_args, **_kwargs):
        from pathlib import Path

        assert FakeLLM.entered == FakeLLM.exited
        options = dict(zip(command[2::2], command[3::2], strict=True))
        calls.append(options)
        output = Path(options["--output-dir"])
        (output / "character.png").write_bytes(b"image-fixture")
        report = {
            "prompt": options["--prompt"],
            "negative_prompt": options["--negative-prompt"],
            "guidance_scale": float(options["--guidance-scale"]),
            "official_diffusers_revision": "test-revision",
            "runtime_commit": "test-runtime",
            "versions": {},
            "width": int(options["--width"]),
            "height": int(options["--height"]),
            "steps": int(options["--steps"]),
            "alpha_range": [0, 255],
            "foreground_bbox": [1, 2, 760, 1000],
            "anchor_bottom_center": [380, 1000],
            "background_model_sha256": "a" * 64,
        }
        (output / "result.json").write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(pipeline, "run_process", run)
    return calls


def test_image_model_prompts_reach_runtime_and_provenance(
    tmp_path, fake_runtime, recorded_image_runtime
):
    profile = pipeline.load_config()["image"]
    source = job("m2_image")
    source["payload"]["instruction"] = "正面向きで、手が見えるように"
    envelope, _ = unpack(pipeline.generate_job(source, tmp_path))
    sent = recorded_image_runtime[0]
    used = envelope["provenance"]["image"]
    assert sent["--prompt"].startswith("masterpiece, best quality, score_7, safe, ")
    assert "nonhuman purple mushroom" in sent["--prompt"]
    assert sent["--negative-prompt"] == profile["negative_prompt"]
    assert "chromatic aberration" in sent["--negative-prompt"]
    assert used["prompt"] == sent["--prompt"]
    assert used["negative_prompt"] == sent["--negative-prompt"]
    assert used["guidance_scale"] == profile["guidance_scale"]
    assert used["character_id"] == job("m2_image")["payload"]["character_id"]
    assert used["source_appearance"] == character()["appearance"]
    translation = next(item for item in envelope["trace"] if item["type"] == "image_prompt")
    assert used["description_prompt"] == translation["prompt"]
    assert used["prompt_version"] == 5
    assert used["visual_features"] == translation["visual_features"]
    assert used["rendering_instruction"] == translation["rendering_instruction"]
    assert used["rendering_instruction"] == "Front view with visible hands."
    assert used["rendering_instruction"] in sent["--prompt"]
    assert used["composition_prompt"] == translation["composition_prompt"]
    assert used["composition_prompt"] in sent["--prompt"]
    for description in used["visual_features"].values():
        if description:
            assert description in sent["--prompt"]
    saved = json.loads((tmp_path / "job-request.json").read_text(encoding="utf-8"))
    assert saved["generation_config"]["image"] == profile


@pytest.mark.parametrize("prefix,negative", [("custom model quality", "custom negative"), ("", "")])
def test_image_prompt_settings_can_be_replaced_or_disabled(
    tmp_path, fake_runtime, recorded_image_runtime, monkeypatch, prefix, negative
):
    config = pipeline.load_config()
    config["image"].update(positive_prompt_prefix=prefix, negative_prompt=negative)
    monkeypatch.setattr(pipeline, "load_config", lambda: config)
    envelope, _ = unpack(pipeline.generate_job(job("m2_image"), tmp_path))
    used = envelope["provenance"]["image"]
    assert used["prompt"] == (prefix + ", " if prefix else "") + used["description_prompt"]
    assert recorded_image_runtime[0]["--negative-prompt"] == negative
    assert used["positive_prompt_prefix"] == prefix


def test_image_retry_preserves_legacy_job_prompt_snapshot(
    tmp_path, fake_runtime, recorded_image_runtime, monkeypatch
):
    current = pipeline.load_config()
    legacy = copy.deepcopy(current)
    legacy["image"].pop("positive_prompt_prefix")
    legacy["image"].pop("negative_prompt")
    monkeypatch.setattr(pipeline, "load_config", lambda: legacy)
    runtime = pipeline.run_process

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted before inference")

    monkeypatch.setattr(pipeline, "run_process", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.generate_job(job("m2_image"), tmp_path)
    monkeypatch.setattr(pipeline, "load_config", lambda: current)
    monkeypatch.setattr(pipeline, "run_process", runtime)
    envelope, _ = unpack(pipeline.generate_job(job("m2_image"), tmp_path))
    used = envelope["provenance"]["image"]
    assert used["positive_prompt_prefix"] == ""
    assert used["prompt"] == used["description_prompt"]
    assert used["negative_prompt"] == (
        "worst quality, low quality, score_1, score_2, score_3, artist name, "
        "blurry, jpeg artifacts, cropped, text"
    )


def test_relationship_generation_runs_without_optional_instructions(tmp_path, fake_runtime):
    request = job("m2_relationships")
    request["payload"].update(
        cast_inputs=cast(),
        cast_results=cast(),
        relationship_inputs=[],
        relationships_result=None,
        character_id=None,
        scope="relationships",
    )
    envelope, names = unpack(pipeline.generate_job(request, tmp_path))
    assert names == {"result.json"}
    assert envelope["result"] == relationships(cast())
    assert not any(item["type"] in {"candidates", "adoption", "random_tool"}
                   for item in envelope["trace"])
    assert envelope["trace"][-1]["type"] == "complete_result"


def test_relationship_generation_retains_every_persons_established_settings():
    people = cast(2)
    for index, person in enumerate(people):
        person["settings"] = f"菌類の人物{index}の固有の性格・背景・葛藤。"
    payload = job("m2_relationships")["payload"]
    payload.update(character_id=None, cast_results=people, relationship_inputs=[])
    stages = []

    class FullCastLLM(FakeLLM):
        def structured(self, stage, messages, schema):
            for person in people:
                assert person["settings"] in messages[-1]["content"]
            stages.append(stage)
            return super().structured(stage, messages, schema)

        def random_context(self, stage, messages):
            for person in people:
                assert person["settings"] in messages[-1]["content"]
            stages.append(stage)
            return super().random_context(stage, messages)

    result = pipeline.generate_relationships(payload, FullCastLLM(None, None, payload, None))
    assert result == relationships(people)
    assert stages == ["relationships-final", "m2_relationships-requirements-review-1"]


@pytest.mark.parametrize("count", [2, 3])
def test_relationship_generation_schema_is_bound_to_actual_cast_and_pair_count(count):
    class CapturingLLM(FakeLLM):
        final_schema = None
        final_output = None

        def structured(self, stage, messages, schema):
            output = super().structured(stage, messages, schema)
            if stage == "relationships-final":
                self.final_schema = schema
                self.final_output = output
            return output

    payload = job("m2_relationships")["payload"]
    payload.update(cast_results=list(reversed(cast(count))), relationship_inputs=[])
    llm = CapturingLLM(None, None, payload, None)
    original_schema = copy.deepcopy(pipeline.RELATIONSHIPS_SCHEMA)
    result = pipeline.generate_relationships(payload, llm)
    schema = llm.final_schema
    keys = [f"pair_{index}" for index in range(1, count * (count - 1) // 2 + 1)]
    assert list(schema["properties"]) == schema["required"] == keys
    assert all("characterIds" not in pair["properties"] for pair in schema["properties"].values())
    validate_schema(llm.final_output, schema)
    unknown = copy.deepcopy(llm.final_output)
    unknown["pair_99"] = unknown["pair_1"]
    with pytest.raises(ValueError, match="missing or unexpected fields"):
        validate_schema(unknown, schema)
    incomplete = copy.deepcopy(llm.final_output)
    incomplete.pop("pair_1")
    with pytest.raises(ValueError, match="missing or unexpected fields"):
        validate_schema(incomplete, schema)
    assert result == relationships(cast(count))
    assert pipeline.RELATIONSHIPS_SCHEMA == original_schema


@pytest.mark.parametrize("duplicate_names", [False, True])
def test_relationship_generation_assigns_uuid_pairs_and_perspectives_independent_of_cast_order(
    duplicate_names,
):
    people = [
        {**character(), "id": "character-1", "name": "主人公"},
        {
            **character(),
            "id": "character-97480dad-d560-446f-bde5-44babc28bb5c",
            "name": "ヒロイン",
            "role": "勇者",
        },
        {
            **character(),
            "id": "character-1def5441-dbdc-4382-9a26-e2a06b675027",
            "name": "ライバル",
            "role": "魔王",
        },
    ]
    if duplicate_names:
        people[2]["name"] = people[1]["name"]
    payload = job("m2_relationships")["payload"]
    payload.update(cast_results=people, relationship_inputs=[])
    by_id = {person["id"]: person for person in people}

    def perspective(first, second):
        return f"{first['name']}（{first['role']}）から{second['name']}（{second['role']}）への認識。"

    class AssignedLLM(FakeLLM):
        def structured(self, stage, messages, schema):
            if stage != "relationships-final":
                return super().structured(stage, messages, schema)
            content = messages[-1]["content"]
            assignments = json.loads(
                content.split("出力キーごとの確定済み人物対応: ", 1)[1].split("\n", 1)[0]
            )
            first_name, second_name = people[1]["name"], people[2]["name"]
            assert assignments["pair_3"]["first"] == {
                "id": people[1]["id"], "name": first_name,
            }
            assert assignments["pair_3"]["second"] == {
                "id": people[2]["id"], "name": second_name,
            }
            forward = f"firstToSecond（{first_name}→{second_name}）"
            reverse = f"secondToFirst（{second_name}→{first_name}）"
            assert assignments["pair_3"][forward] == {
                "認識する人物": {"name": first_name, "role": "勇者"},
                "認識される相手": {"name": second_name, "role": "魔王"},
            }
            assert assignments["pair_3"][reverse] == {
                "認識する人物": {"name": second_name, "role": "魔王"},
                "認識される相手": {"name": first_name, "role": "勇者"},
            }
            assert list(schema["properties"]["pair_3"]["properties"]) == ["summary", forward, reverse]
            result = {
                key: {
                    "summary": f"{pair['first']['name']}と{pair['second']['name']}の関係。",
                    f"firstToSecond（{pair['first']['name']}→{pair['second']['name']}）": perspective(
                        by_id[pair["first"]["id"]], by_id[pair["second"]["id"]],
                    ),
                    f"secondToFirst（{pair['second']['name']}→{pair['first']['name']}）": perspective(
                        by_id[pair["second"]["id"]], by_id[pair["first"]["id"]],
                    ),
                }
                for key, pair in assignments.items()
            }
            validate_schema(result, schema)
            return result

    result = pipeline.generate_relationships(payload, AssignedLLM(None, None, payload, None))
    assert [pair["characterIds"] for pair in result["pairs"]] == [
        [people[0]["id"], people[2]["id"]],
        [people[0]["id"], people[1]["id"]],
        [people[2]["id"], people[1]["id"]],
    ]
    assert result["pairs"][2]["firstToSecond"] == perspective(people[2], people[1])
    assert result["pairs"][2]["secondToFirst"] == perspective(people[1], people[2])


def test_queued_legacy_character_job_keeps_its_original_result_schema(tmp_path, fake_runtime):
    request = job("m2_character")
    request["payload"].pop("character_contract_version")
    request["payload"]["character_result"] = None
    envelope, _ = unpack(pipeline.generate_job(request, tmp_path))
    assert "selfIntroduction" not in envelope["result"]
    assert "sampleLines" not in envelope["result"]


@pytest.fixture
def recorded_voice_runtime(monkeypatch):
    requests = []

    def run(command, log, **kwargs):
        output = log.parent
        request = json.loads((output / "request.json").read_text(encoding="utf-8"))
        requests.append(request)
        validate_request(request, output)
        (output / "voice.wav").write_bytes(b"validated-audio")
        (output / "result.json").write_text(
            json.dumps(
                {
                    "decoded_and_non_silent": True,
                    "reference_text": request.get("reference_text", request["text"]),
                    "text": request["text"],
                    "watermarked": True,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(pipeline, "run_process", run)
    return requests


def test_reference_voice_speaks_generated_introduction(tmp_path, recorded_voice_runtime):
    payload = job("m2_voice")["payload"]
    _, report = pipeline.generate_voice(payload, tmp_path, pipeline.load_config())
    assert report["reference_text"] == character()["selfIntroduction"]
    assert recorded_voice_runtime[0]["text"] == character()["selfIntroduction"]
    assert recorded_voice_runtime[0]["mode"] == "design"


def test_new_voice_cannot_silently_fall_back_to_generic_legacy_greeting(
    tmp_path, recorded_voice_runtime
):
    payload = job("m2_voice")["payload"]
    payload["character_result"].pop("selfIntroduction")
    with pytest.raises(ValueError, match="self-introduction"):
        pipeline.generate_voice(payload, tmp_path, pipeline.load_config())
    assert recorded_voice_runtime == []
    payload.pop("character_contract_version")
    pipeline.generate_voice(payload, tmp_path, pipeline.load_config())
    assert recorded_voice_runtime[0]["text"] == pipeline.load_config()["voice"]["reference_text"]


def test_clone_uses_immutable_reference_and_original_transcript_not_current_intro(
    tmp_path, recorded_voice_runtime, fake_runtime
):
    request = job("m2_voice_clone")
    source = b"original-verified-reference-audio"
    (tmp_path / "reference-voice.wav").write_bytes(source)
    request["payload"].update(
        reference_voice={
            "artifact_id": "reference-1",
            "sha256": hashlib.sha256(source).hexdigest(),
            "text": "以前の正確な自己紹介です。",
        },
        dialogue_text="今日は土壌の記録を調べましょう。",
    )
    unchanged = copy.deepcopy(request)
    envelope, names = unpack(pipeline.generate_job(request, tmp_path))
    assert request == unchanged
    assert names == {"result.json", "voice.wav"}
    assert envelope["result"] == {}
    assert envelope["provenance"]["voice"]["reference_text"] == "以前の正確な自己紹介です。"
    assert recorded_voice_runtime[0]["text"] == request["payload"]["dialogue_text"]
    assert recorded_voice_runtime[0]["mode"] == "clone"
    assert recorded_voice_runtime[0]["reference_artifact_id"] == "reference-1"
    assert FakeLLM.entered == 0


def test_clone_rejects_reference_tampering_before_model_start(tmp_path, recorded_voice_runtime):
    payload = job("m2_voice_clone")["payload"]
    (tmp_path / "reference-voice.wav").write_bytes(b"tampered")
    payload.update(
        reference_voice={"artifact_id": "reference-1", "sha256": "0" * 64, "text": "元の台詞"},
        dialogue_text="新しい台詞",
    )
    with pytest.raises(ValueError, match="SHA256"):
        pipeline.generate_voice_clone(payload, tmp_path, pipeline.load_config())
    assert recorded_voice_runtime == []


def test_failed_generation_never_creates_success_bundle(tmp_path, fake_runtime, monkeypatch):
    def fail(*args):
        raise ValueError("Invalid transparent foreground")

    monkeypatch.setattr(pipeline, "generate_image", fail)
    with pytest.raises(ValueError):
        pipeline.generate_job(job("m2_image"), tmp_path)
    assert not (tmp_path / "result.zip").exists()
    assert FakeLLM.entered == FakeLLM.exited


def test_external_provider_is_not_silently_sent_anywhere(tmp_path):
    request = job("m2_voice")
    request["payload"]["profile"]["provider"] = "external"
    with pytest.raises(ValueError, match="local provider"):
        pipeline.generate_job(request, tmp_path)


def test_missing_readiness_manifest_is_reported_without_starting_models(monkeypatch):
    monkeypatch.setattr(pipeline, "CONFIG_PATH", pipeline.ROOT / "config/nonexistent-m2.json")
    assert pipeline.check_readiness()["ready"] is False
    assert pipeline.available_job_kinds() == []


def test_truncated_llm_completion_is_rejected(tmp_path):
    client = LocalLLM(pipeline.ROOT, pipeline.load_config(), job()["payload"], tmp_path)
    client.request = lambda *args: {
        "choices": [{"finish_reason": "length", "message": {"content": "{}"}}]
    }
    with pytest.raises(ValueError, match="truncated"):
        client.chat("final", [{"role": "user", "content": "x"}])


def test_valid_llm_response_cache_avoids_repeated_requests(tmp_path):
    response = {"choices": [{"finish_reason": "stop", "message": {"content": "答え"}}]}
    first = LocalLLM(pipeline.ROOT, pipeline.load_config(), job()["payload"], tmp_path)
    first.request = lambda *args: response
    assert first.chat("first", [{"role": "user", "content": "x"}])["content"] == "答え"
    retry = LocalLLM(pipeline.ROOT, pipeline.load_config(), job()["payload"], tmp_path)
    retry.request = lambda *args: pytest.fail("cache should avoid HTTP request")
    assert retry.chat("first", [{"role": "user", "content": "x"}])["content"] == "答え"


def test_owned_process_timeout_stops_only_spawned_process(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        start = time.monotonic()
        with pytest.raises((TimeoutError, RuntimeError)):
            run_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                tmp_path / "child.log",
                cwd=tmp_path,
                timeout=0.15,
            )
        assert time.monotonic() - start < 10
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)
