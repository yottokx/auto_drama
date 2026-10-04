"""Model cache isolation must not discard or copy existing HF credentials."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts.audio import hub_auth


def token_file(home: Path):
    home.mkdir(parents=True)
    (home / "token").touch()
    return home / "token"


def test_default_login_is_reused_without_copying(monkeypatch, tmp_path):
    original = token_file(tmp_path / ".cache/huggingface")
    monkeypatch.setattr(hub_auth.Path, "home", classmethod(lambda cls: tmp_path))
    inherited = {"OTHER": "unchanged"}
    cache = tmp_path / "audio cache"
    environment = hub_auth.hub_environment(cache, inherited)
    assert environment["HF_HOME"] == str(cache)
    assert environment["HF_TOKEN_PATH"] == str(original)
    assert inherited == {"OTHER": "unchanged"}
    assert not (cache / "token").exists()


@pytest.mark.parametrize("key", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
def test_explicit_token_and_token_path_are_preserved(tmp_path, key):
    inherited = {key: "test-credential", "HF_TOKEN_PATH": str(tmp_path / "explicit/token")}
    environment = hub_auth.hub_environment(tmp_path / "new-cache", inherited)
    assert environment[key] == inherited[key]
    assert environment["HF_TOKEN_PATH"] == inherited["HF_TOKEN_PATH"]


def test_existing_runtime_login_has_priority(tmp_path):
    original = token_file(tmp_path / "old-cache")
    runtime = tmp_path / "runtime-cache"
    active = token_file(runtime)
    environment = hub_auth.hub_environment(runtime, {"HF_HOME": str(original.parent)})
    assert environment["HF_TOKEN_PATH"] == str(active)


def test_custom_expanded_home_is_respected(monkeypatch, tmp_path):
    original = token_file(tmp_path / "user-cache/huggingface")
    monkeypatch.setenv("AUDIO_AUTH_TEST_CACHE", str(tmp_path / "user-cache"))
    inherited = {"XDG_CACHE_HOME": "$AUDIO_AUTH_TEST_CACHE"}
    environment = hub_auth.hub_environment(tmp_path / "audio-cache", inherited)
    assert environment["HF_TOKEN_PATH"] == str(original)


def test_no_login_does_not_invent_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(hub_auth.Path, "home", classmethod(lambda cls: tmp_path))
    environment = hub_auth.hub_environment(tmp_path / "cache", {})
    assert "HF_TOKEN_PATH" not in environment
    assert "HF_TOKEN" not in environment


@pytest.fixture
def fake_hub(monkeypatch):
    class HubError(Exception):
        def __init__(self, status):
            self.response = SimpleNamespace(status_code=status)
    class NoToken(Exception):
        pass
    calls = []
    api = SimpleNamespace(whoami=lambda: {"name": "private-account", "token": "private-token"})
    hub = SimpleNamespace(HfApi=lambda: api,
                          hf_hub_download=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors",
                        SimpleNamespace(HfHubHTTPError=HubError, LocalTokenNotFoundError=NoToken))
    return hub, api, HubError, calls


def test_access_checks_only_config_and_does_not_disclose_profile(fake_hub, tmp_path):
    _hub, _api, _error, calls = fake_hub
    result = hub_auth.check_access("small", tmp_path)
    assert result["ok"]
    assert calls[0][0] == ("stabilityai/stable-audio-3-small-music", "model_config.json")
    assert calls[0][1]["token"] is True
    serialized = json.dumps(result)
    assert "private-account" not in serialized
    assert "private-token" not in serialized


def test_gate_denial_distinguishes_valid_login(fake_hub, tmp_path):
    hub, _api, error, _calls = fake_hub
    def denied(*args, **kwargs):
        raise error(403)
    hub.hf_hub_download = denied
    result = hub_auth.check_access("small", tmp_path)
    assert not result["ok"]
    assert result["authenticated"]
    assert "利用条件" in result["error"]
    assert result["model_url"].endswith("stable-audio-3-small-music")


def test_network_error_does_not_request_new_login(fake_hub, tmp_path):
    _hub, api, _error, _calls = fake_hub
    def offline():
        raise httpx.ConnectError("network down")
    api.whoami = offline
    result = hub_auth.check_access("small", tmp_path)
    assert "ネットワーク" in result["error"]
    assert "hf auth login" not in result["error"]


def test_public_ace_checkpoint_never_requires_login(fake_hub, tmp_path):
    from scripts.audio import ace_prepare

    _hub, api, _error, calls = fake_hub
    api.whoami = lambda: pytest.fail("public checkpoint must not inspect login")
    result = hub_auth.check_access("ace15_turbo", tmp_path)
    assert result["ok"] and result["authentication_required"] is False
    assert result["authenticated"] is False
    assert calls[0][0] == (ace_prepare.REPO_ID, "acestep-v15-turbo/config.json")
    assert calls[0][1]["revision"] == ace_prepare.REPO_REVISION
    assert calls[0][1]["token"] is False


def test_public_ace_network_failure_never_requests_terms_or_login(fake_hub, tmp_path):
    hub, api, error, _calls = fake_hub
    api.whoami = lambda: pytest.fail("public checkpoint must not inspect login")

    def denied(*args, **kwargs):
        raise error(403)

    hub.hf_hub_download = denied
    result = hub_auth.check_access("ace15_turbo", tmp_path)
    assert not result["ok"] and result["authentication_required"] is False
    assert "利用条件" not in result["error"] and "ログイン" not in result["error"]
