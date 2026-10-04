"""Separate model caches from the user's existing Hugging Face login."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

try:
    from .engine import MODEL_SPECS, write_json
except ImportError:
    from engine import MODEL_SPECS, write_json


def hub_environment(cache_dir: Path, inherited: Mapping[str, str]) -> dict[str, str]:
    """Keep explicit credentials and reuse a login without copying its token."""
    environment = dict(inherited)
    original_home = environment.get("HF_HOME")
    if not original_home:
        original_home = str(Path(environment.get("XDG_CACHE_HOME") or Path.home() / ".cache")
                            / "huggingface")
    if not environment.get("HF_TOKEN_PATH"):
        original_home = Path(os.path.expandvars(original_home)).expanduser()
        candidates = [cache_dir / "token", original_home / "token"]
        for path in candidates:
            if path.is_file():
                environment["HF_TOKEN_PATH"] = str(path)
                break
    environment["HF_HOME"] = str(cache_dir)
    return environment


def check_access(model: str, cache: Path) -> dict:
    """Validate login and one gated config file, without exposing account data."""
    from httpx import HTTPError
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

    if model == "ace15_turbo":
        try:
            from .ace_prepare import DIT_CONFIG, REPO_ID, REPO_REVISION
        except ImportError:
            from ace_prepare import DIT_CONFIG, REPO_ID, REPO_REVISION
        url = f"https://huggingface.co/{REPO_ID}"
        try:
            # ACE's official MIT checkpoint is public. Explicitly avoid an
            # unrelated expired login and never request authentication for it.
            hf_hub_download(REPO_ID, f"{DIT_CONFIG}/config.json", revision=REPO_REVISION,
                            cache_dir=str(cache), token=False)
        except (HfHubHTTPError, HTTPError, OSError):
            return {"ok": False, "model_url": url, "authentication_required": False,
                    "error": "公開 ACE-Step モデルの設定を取得できません。ネットワーク接続を確認してください。"}
        return {"ok": True, "authenticated": False, "authentication_required": False,
                "model_url": url, "message": "公開 ACE-Step 1.5 Turbo のアクセスを確認しました。ログイン不要で取得・変換できます。"}

    repo = MODEL_SPECS[model]["repo_id"]
    url = f"https://huggingface.co/{repo}"
    try:
        HfApi().whoami()
    except (LocalTokenNotFoundError, HfHubHTTPError):
        return {"ok": False, "error": "Hugging Faceのログインを確認できません。hf auth loginでログインしてください。",
                "authenticated": False, "model_url": url}
    except (HTTPError, OSError):
        return {"ok": False, "error": "Hugging Faceに接続できません。ネットワーク接続を確認してください。"}
    try:
        hf_hub_download(repo, "model_config.json", cache_dir=str(cache), token=True)
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in {401, 403}:
            return {"ok": False, "authenticated": True, "model_url": url,
                    "error": "Hugging Faceにはログイン済みですが、このモデルへのアクセスが拒否されました。\n"
                    "モデルのページで利用条件への同意・承認を確認してください。"
                    "制限付きトークンの場合は、このモデルを読み取る権限も必要です。\n" + url}
        return {"ok": False, "authenticated": True, "error": f"モデル設定を確認できません（HTTP {status}）。"}
    except (HTTPError, OSError):
        return {"ok": False, "error": "モデル設定の取得に失敗しました。ネットワーク接続を確認してください。"}
    return {"ok": True, "authenticated": True, "model_url": url,
            "message": f"{MODEL_SPECS[model]['label']}のアクセスを確認しました。取得・変換に進めます。"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hugging Faceモデルへのアクセス確認")
    parser.add_argument("--model", choices=MODEL_SPECS, default="small")
    parser.add_argument("--status-dir", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    cache = root / "services/worker/runtimes/stable_audio3/cache/huggingface"
    os.environ.update(hub_environment(cache, os.environ))
    # Import the hub inside check_access, after token/cache paths are configured.
    result = check_access(args.model, cache)
    message = result.get("message") or result["error"]
    print(message, flush=True)
    if args.status_dir:
        write_json(args.status_dir / "result.json", result)
        write_json(args.status_dir / "status.json", {"phase": "done" if result["ok"] else "error", "message": message})
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
