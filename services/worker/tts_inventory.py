"""CPU-only TTS file receipts and local runtime support for worker registration."""

from __future__ import annotations

import json
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path

from packages.contracts.tts_catalog import catalog, resolve_bundle
from services.worker.tts_downloads import _identity

_runtime_cache: dict[Path, tuple[float, dict]] = {}


def _command(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", timeout=15, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return result.stdout.strip()


def runtime_support(root: Path) -> dict:
    """Inspect installed packages and CUDA hardware without importing GPU libraries."""
    root = root.resolve()
    cached = _runtime_cache.get(root)
    if cached and time.monotonic() - cached[0] < 60:
        return deepcopy(cached[1])
    result = {"ready": False, "precisions": [], "source_commit": None}
    runtime = root / "services/worker/runtimes/irodori"
    python = runtime / ".venv/Scripts/python.exe"
    try:
        if not python.is_file():
            return result
        source_commit = _command(["git", "-C", str(runtime / "source"), "rev-parse", "HEAD"], cwd=root)
        code = ("import importlib.metadata as m,json; names=['torch','transformers','soundfile',"
                "'silentcipher','safetensors']; versions={n:m.version(n) for n in names}; "
                "versions['torchao']=next((d.version for d in m.distributions() "
                "if d.metadata['Name'].lower()=='torchao'),None); print(json.dumps(versions))")
        versions = json.loads(_command([str(python), "-I", "-X", "utf8", "-c", code], cwd=root))
        capabilities = _command(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], cwd=root)
        # The existing runtime uses CUDA device 0. Other GPUs may differ.
        major, _ = capabilities.splitlines()[0].strip().split(".")
        if "+cu" not in versions["torch"]:
            return result
        precisions = ["fp32"]
        if int(major) >= 8:
            precisions.append("bf16")
            if (versions.get("torchao") or "").startswith("0.16."):
                precisions.extend(("int8", "int4"))
        result = {"ready": True, "precisions": precisions, "source_commit": source_commit}
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
        pass
    finally:
        _runtime_cache[root] = (time.monotonic(), result)
    return deepcopy(result)


def downloaded_inventory(root: Path) -> list[dict]:
    """Reuse durable verification receipts; never hash or download a model at startup."""
    root = root.resolve()
    receipt_path = root / "services/worker/runtimes/irodori/models/.tts-downloads/verified-files.json"
    try:
        receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    result = []
    for model in catalog():
        for precision in model["precisions"]:
            bundle = resolve_bundle(model["model_id"], precision)
            valid = True
            for file in bundle["files"]:
                path = (root / file["local_dir"] / file["path"]).resolve()
                if not path.is_relative_to(root):
                    valid = False
                    break
                try:
                    stat = path.stat()
                except OSError:
                    valid = False
                    break
                receipt = receipts.get(str(path.relative_to(root)), {})
                if (stat.st_size != file["size"] or receipt.get("mtime_ns") != stat.st_mtime_ns
                        or receipt.get("identity") != _identity(file)):
                    valid = False
                    break
            if valid:
                result.append({"model_id": model["model_id"], "precision": precision,
                               "manifest_id": bundle["manifest_id"], "file_download_ready": True})
    return result


def generation_inventory(root: Path, models: list[dict]) -> list[dict]:
    result = [{**model, "generation_ready": False, "generation_purposes": []} for model in models]
    valid = []
    for model in result:
        try:
            bundle = resolve_bundle(model["model_id"], model["precision"])
            if model.get("file_download_ready") and model["manifest_id"] == bundle["manifest_id"]:
                valid.append((model, bundle))
        except (ValueError, KeyError):
            continue
    if not valid:
        return result
    support = runtime_support(root)
    for model, bundle in valid:
        if (support["ready"] and support["source_commit"] == bundle["source_commit"]
                and model["precision"] in support["precisions"]):
            model.update(generation_ready=True, generation_purposes=["voice_design", "voice_clone"])
    return result
