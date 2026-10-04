"""The ACE planner must preserve scene intent and generate only musical codes."""

import json
import weakref
from types import SimpleNamespace

import pytest

from scripts.audio import ace_planner as planner


def request(**changes):
    return {"prompt": "Instrumental industrial metal with hostile angular riffs.", "duration": 30,
            "bpm": 160, "keyscale": "D minor", "timesignature": "4", "seed": 123,
            "device": "cpu", **changes}


@pytest.mark.parametrize("duration,count", [(10, 50), (30, 150), (120, 600), (10.01, 51), (600, 3000)])
def test_duration_produces_complete_five_hz_codes(duration, count):
    normalized = planner.validate_request(request(duration=duration))
    assert normalized["code_count"] == count
    assert normalized["metadata"]["duration"] == duration


@pytest.mark.parametrize("field,value", [
    ("duration", True), ("duration", float("nan")), ("duration", 9), ("duration", 601),
    ("seed", -1), ("seed", True), ("seed", 2**32), ("device", "mps"),
    ("lyrics", "sing this"), ("vocal_language", "en"), ("bpm", 29),
    ("keyscale", "nonsense"), ("timesignature", "5"), ("prompt", ""),
])
def test_invalid_or_vocal_requests_are_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        planner.validate_request(request(**{field: value}))


def test_unspecified_metadata_is_explicit_and_never_invented():
    normalized = planner.validate_request(request(bpm=0, keyscale="", timesignature=""))
    assert normalized["metadata"] == {
        "bpm": None, "keyscale": None, "timesignature": None, "duration": 30.0, "language": "unknown",
    }


def test_prompt_matches_open_official_chat_format_and_preserves_gemma_caption():
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)) or "<assistant>\n")
    normalized = planner.validate_request(request(prompt="Sharp attacks: \"no sentimental pads\"."))
    formatted = planner.build_prompt(tokenizer, normalized["prompt"], normalized["metadata"])
    assert calls[0][1] == {"tokenize": False, "add_generation_prompt": True}
    assert calls[0][0][0]["content"] == "# Instruction\n" + planner.LM_INSTRUCTION + "\n\n"
    assert calls[0][0][1]["content"].endswith("# Lyric\n[Instrumental]\n")
    assert calls[0][0][1]["content"].startswith("# Caption\n" + normalized["prompt"])
    assert formatted.startswith("<assistant>\n<think>\nbpm: 160\ncaption: ")
    assert '\\"no sentimental pads\\"' in formatted
    assert 'keyscale: "D minor"\nlanguage: "unknown"\ntimesignature: 4' in formatted
    assert formatted.endswith("\n</think>\n\n")
    assert "<|im_end|>" not in formatted


def vocabulary():
    return {f"<|audio_code_{index}|>": index + 5 for index in range(64000)}


def test_audio_token_allowlist_uses_real_vocabulary_and_rejects_out_of_range():
    vocab = vocabulary()
    vocab.update({"plain text": 0, "<|audio_code_64000|>": 64005, "<|audio_code_3|> extra": 2})
    result = planner.audio_token_map(vocab)
    assert len(result) == 64000
    assert result[5] == 0 and result[64004] == 63999
    assert 0 not in result and 64005 not in result and 2 not in result


def test_incomplete_audio_vocabulary_is_rejected():
    vocab = vocabulary()
    del vocab["<|audio_code_17|>"]
    with pytest.raises(ValueError, match="完全"):
        planner.audio_token_map(vocab)


def test_cpu_planning_uses_local_builtin_model_preserves_metadata_and_releases(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model_refs = []
    calls = {}

    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        pad_token_id = 1
        eos_token_id = 2

        def get_vocab(self):
            return vocabulary()

        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            return "<assistant>\n"

        def __call__(self, _prompt, **kwargs):
            assert kwargs == {"return_tensors": "pt", "add_special_tokens": False, "truncation": False}
            return Batch(input_ids=torch.tensor([[3, 4]]), attention_mask=torch.ones((1, 2), dtype=torch.long))

    class Model:
        config = SimpleNamespace(vocab_size=64006, max_position_embeddings=40960)

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, **kwargs):
            calls["unconditional"] = (input_ids, kwargs)
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], self.config.vocab_size)),
                                   get=lambda _key, default=None: default)

        def generate(self, **kwargs):
            count = kwargs["max_new_tokens"]
            assert count == kwargs["min_new_tokens"] == 150
            assert kwargs["temperature"] == 0.85 and kwargs["top_p"] == 0.9 and kwargs["top_k"] == 0
            assert kwargs["guidance_scale"] == 1.0  # Our native fp32 processor performs CFG exactly once.
            scores = torch.zeros((1, self.config.vocab_size))
            masked = kwargs["logits_processor"](kwargs["input_ids"], scores)
            assert torch.isneginf(masked[0, :5]).all()
            assert masked[0, 5:64005].eq(0).all()
            tokens = torch.full((1, count), 22, dtype=torch.long)
            result = torch.cat((kwargs["input_ids"], tokens), dim=1)
            kwargs["stopping_criteria"](result, None)
            return result

    def load_model(path, **kwargs):
        calls["model"] = (path, kwargs)
        model = Model()
        model_refs.append(weakref.ref(model))
        return model

    def load_tokenizer(path, **kwargs):
        calls["tokenizer"] = (path, kwargs)
        return Tokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(planner, "validate_model", lambda _path: {"repo_revision": "pinned"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("explicit CPU cannot consult or allocate GPU"))
    normalized = planner.validate_request(request())
    result = planner.generate_codes(normalized, tmp_path)
    assert result["audio_codes"] == "<|audio_code_17|>" * 150
    assert result["code_count"] == 150 and result["planner_released"]
    assert result["actual_device"] == "cpu" and result["metadata"] == normalized["metadata"]
    assert result["planner_metadata"]["caption"] == normalized["prompt"]
    assert result["planner_metadata"]["cot_caption_rewrite"] is False
    assert result["planner_metadata"]["dtype"] == "float32"
    assert result["planner_metadata"]["cfg_scale"] == 2.0
    assert result["planner_metadata"]["cfg_negative_prompt"] == "NO USER INPUT"
    assert calls["model"][1]["trust_remote_code"] is False
    assert calls["model"][1]["local_files_only"] and calls["model"][1]["use_safetensors"]
    assert all(ref() is None for ref in model_refs)


def test_cfg_unconditional_prompt_uses_training_dropout_format():
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)) or "<assistant>\n")
    formatted = planner.build_unconditional_prompt(tokenizer)
    assert calls[0][0][1] == {"role": "user", "content": "NO USER INPUT"}
    assert calls[0][1] == {"tokenize": False, "add_generation_prompt": True}
    assert formatted == "<assistant>\n<think>\n\n</think>\n\n"
    assert "# Caption" not in formatted and "# Lyric" not in formatted


def test_native_cfg_matches_probability_formula_and_reuses_unconditional_cache():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    unconditional = torch.tensor([[[3.0, -1.0, 4.0]]], dtype=torch.bfloat16)
    calls = []
    cache = object()

    class Model:
        def __call__(self, input_ids, **kwargs):
            calls.append((input_ids.clone(), kwargs))
            return SimpleNamespace(logits=unconditional, get=lambda key, default=None: cache if key == "past_key_values" else default)

    processor = planner.make_cfg_processor(torch, transformers, Model(), {
        "input_ids": torch.tensor([[8, 9]]), "attention_mask": torch.ones((1, 2), dtype=torch.long),
    })
    conditional = torch.tensor([[1.0, 2.0, -3.0]], dtype=torch.bfloat16)
    guided = processor(torch.tensor([[1, 2]]), conditional)
    expected = unconditional[:, -1].float() + 2.0 * (conditional.float() - unconditional[:, -1].float())
    assert torch.equal(guided, expected) and guided.dtype == torch.float32
    log_probs = 2.0 * (conditional.float().log_softmax(-1) - unconditional[:, -1].float().log_softmax(-1)) + unconditional[:, -1].float().log_softmax(-1)
    assert torch.allclose(guided.softmax(-1), log_probs.softmax(-1), atol=1e-7)
    assert calls[0][0].tolist() == [[8, 9]] and calls[0][1]["past_key_values"] is None
    processor(torch.tensor([[1, 2, 17]]), conditional)
    assert calls[1][0].tolist() == [[17]] and calls[1][1]["past_key_values"] is cache
    assert calls[1][1]["attention_mask"].tolist() == [[1, 1, 1]]


def test_cancellation_before_loading_cannot_return_success(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    (tmp_path / "stop.request").touch()
    monkeypatch.setattr(planner, "validate_model", lambda _path: pytest.fail("cancelled load"))
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request()), encoding="utf-8")
    assert planner.run_request(path, tmp_path) == 130
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert not result["ok"] and result["cancelled"]


def test_existing_results_are_preserved(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"keep":true}', encoding="utf-8")
    assert planner.run_request(tmp_path / "request.json", tmp_path) == 2
    assert json.loads(path.read_text()) == {"keep": True}


def test_invalid_request_is_reported_without_loading(tmp_path, monkeypatch):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request(duration=1)), encoding="utf-8")
    monkeypatch.setattr(planner, "generate_codes", lambda *args: pytest.fail("invalid request"))
    assert planner.run_request(path, tmp_path) == 1
    assert not json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["ok"]
