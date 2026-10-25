"""Live OpenAI / Anthropic calls through LocalLLM. Costs real money (well under one cent a run).

Needs the provider keys in the environment and the models in models.local.json's api_models.
"""

import json
import os

import pytest

from packages.contracts.llm_settings import LLMSettings
from packages.narrative.validation import parse_scene_text
from services.worker.generation.llm import LocalLLM
from services.worker.generation.pipeline import ROOT, load_config
from services.worker.model_config import select_config

pytestmark = pytest.mark.skipif(os.environ.get("AUTO_DRAMA_API_SMOKE") != "1",
                                reason="Set AUTO_DRAMA_API_SMOKE=1 for live external LLM calls")
MODELS = ["openai/gpt-6-luna", "anthropic/claude-haiku-5-5"]
# Bounds each API cannot enforce itself; the reply must still meet them all.
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["names", "age"],
          "properties": {"names": {"type": "array", "minItems": 2, "maxItems": 3, "uniqueItems": True,
                                   "items": {"type": "string"}},
                         "age": {"type": "integer", "minimum": 10, "maximum": 20},
                         "note": {"type": "string"}}}


def client(model, tmp_path):
    profile = LLMSettings(model=model, temperature=0.65, top_p=0.95, reasoning_effort="none",
                          ctx_size=16384).profile()
    config = select_config(load_config(), {"profile": profile}, ROOT)
    return LocalLLM(ROOT, config, {"seed": 1, "profile": profile}, tmp_path / "llm")


def keys():
    return [os.environ.get(name, "") for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")]


@pytest.mark.parametrize("model", MODELS)
def test_structured_scene_and_tools_through_the_external_api(model, tmp_path):
    with client(model, tmp_path) as llm:
        result = llm.structured("smoke-structured", [
            {"role": "system", "content": "指定のJSONだけを返します。"},
            {"role": "user", "content": "高校の同級生2人か3人の名前（重複なし）と、10〜20歳の年齢を1つ。"}], SCHEMA)
        scene = llm.chat("smoke-scene", [
            {"role": "system", "content": "あなたは日本語の脚本家です。"},
            {"role": "user", "content": "駅で再会した二人の短い場面を書いてください。各行は `NARRATOR: 地の文`、"
                                        "`character-1: 台詞`、`character-2: 台詞` のどれかで、6行ほど。"
                                        "前置きや見出しは書かない。"}], allow_truncated=True)
        history = llm.random_context("smoke-tools", [
            {"role": "system", "content": "乱数が必要なら道具を使います。"},
            {"role": "user", "content": "random_integer で1から6を1回引き、その値を一文で伝えてください。"}])
    assert len(result["names"]) == len(set(result["names"])) and 10 <= result["age"] <= 20
    assert scene["_finish_reason"] == "stop"
    assert len(parse_scene_text(scene["content"], "s", {"character-1", "character-2"})) >= 4
    assert any(message["role"] == "tool" for message in history)
    budgets = [row for row in llm.trace if row["type"] == "context_budget"]
    assert budgets and all(row["fits"] for row in budgets)
    saved = "".join(path.read_text("utf-8") for path in (tmp_path / "llm").glob("*.json"))
    assert all(key and key not in saved for key in keys())
    usage = [json.loads(path.read_text("utf-8"))["response"]["usage"]
             for path in (tmp_path / "llm").glob("0*.json")]
    assert all(row["completion_tokens_details"]["reasoning_tokens"] == 0 for row in usage)
