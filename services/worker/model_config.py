"""Resolve model IDs on this worker. Local paths are never sent to coordinator."""
import copy
import json
from pathlib import Path

from packages.contracts.llm_settings import LLMCapability

REGISTRY = Path("services/worker/config/models.local.json")


def entries(root: Path) -> dict:
    path = root / REGISTRY
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("models", {}), dict):
        raise ValueError("Invalid worker models.local.json")
    directories = value.get("model_dirs", [])
    if not isinstance(directories, list) or any(not isinstance(item, str) or not item for item in directories):
        raise ValueError("model_dirs must be a list of directory paths")
    registry = {}
    known = {}
    for metadata_path in (root / "config").glob("*llm.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "filename" in metadata.get("model", {}):
            known[metadata["model"]["filename"]] = metadata
    profiles = {}
    generation_path = root / "config/m2-generation.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        for profile in [generation.get("llm", {}), *[route.get("llm", {}) for route in generation.get("model_routing", {}).values()]]:
            if profile.get("model_id"):
                profiles[profile["model_id"]] = profile
    for directory in directories:
        folder = (root / directory).resolve(strict=True)
        if not folder.is_dir():
            raise ValueError(f"Model directory is not a directory: {folder}")
        for file in sorted(folder.rglob("*")):
            if not file.is_file() or file.suffix.lower() != ".gguf":
                continue
            model_id = file.stem
            if model_id in registry and Path(registry[model_id]["path"]) != file:
                raise ValueError(f"Duplicate model filename in model_dirs: {file.name}")
            metadata = copy.deepcopy(known.get(file.name, {
                "schema_version": 1,
                "model": {"filename": file.name, "size_bytes": file.stat().st_size,
                          "provenance": "Worker directory discovery; inference not verified."},
                "server": {"executable": value.get("server_executable",
                    "services/worker/runtimes/llama_cpp/bin/llama-server.exe")}}))
            if value.get("server_executable"):
                metadata["server"]["executable"] = value["server_executable"]
            profile = profiles.get(model_id, {})
            registry[model_id] = {
                "path": str(file), "llm_base": metadata,
                "max_context_size": 16384,
                "reasoning_efforts": profile.get("reasoning_levels", ["none"]),
                "reasoning_template": profile.get("reasoning_template")}
    # Optional overrides are still supported for existing installations.
    registry.update(value.get("models", {}))
    return registry


def resolve_entry(root: Path, model_id: str, entry: dict) -> tuple[dict, dict]:
    base = (copy.deepcopy(entry["llm_base"]) if "llm_base" in entry
            else json.loads((root / entry["llm_config"]).read_text(encoding="utf-8")))
    path = (root / entry["path"]).resolve(strict=True)
    if path.stat().st_size != base["model"]["size_bytes"]:
        raise ValueError(f"Model size mismatch: {model_id}")
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError(f"Invalid GGUF: {model_id}")
    if not (root / base["server"]["executable"]).is_file():
        raise ValueError(f"Missing llama-server: {model_id}")
    maximum = entry["max_context_size"]
    capability = LLMCapability(model=model_id, max_context_size=maximum,
                               reasoning_efforts=entry.get("reasoning_efforts", ["none"]))
    template = entry.get("reasoning_template")
    if any(level != "none" for level in capability.reasoning_efforts) and template != "qwen3":
        raise ValueError(f"Thinking requires a verified reasoning template: {model_id}")
    base["model"]["relative_path"] = str(path)
    settings = {"model_id": model_id, "context_size": 16384,
                "max_context_size": maximum, "model_context_size": maximum,
                "reasoning_levels": capability.reasoning_efforts,
                "reasoning_template": template, "max_output_tokens": 8192}
    return base, settings


def catalog(root: Path) -> tuple[list[dict], list[str]]:
    models, errors = [], []
    try:
        registry = entries(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [], [str(exc)]
    for model_id, entry in registry.items():
        try:
            _, settings = resolve_entry(root, model_id, entry)
            models.append(LLMCapability(model=model_id,
                reasoning_efforts=settings["reasoning_levels"]).model_dump(exclude_none=True))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{model_id}: {exc}")
    return models, errors


def select_config(config: dict, payload: dict, root: Path) -> dict:
    profile = payload.get("profile", {})
    if not profile.get("common_settings_version"):
        return config
    model_id = profile["model_id"]
    registry = entries(root)
    if model_id not in registry:
        raise ValueError(f"Model is not configured on this worker: {model_id}")
    base, settings = resolve_entry(root, model_id, registry[model_id])
    requested_context = profile.get("context_size", 16384)
    if type(requested_context) is not int or requested_context < 16384:
        raise ValueError("Common context_size must be an integer of at least 16384.")
    # This is the user's requested runtime allocation, not an inferred model limit.
    # Let llama-server load this context and report actual capacity or a load error.
    settings.update(context_size=requested_context, max_context_size=requested_context,
                    model_context_size=requested_context, allow_context_expansion=False)
    result = copy.deepcopy(config)
    result["llm_base"] = base
    result["llm_config"] = registry[model_id].get("llm_config", "worker-discovered")
    result["llm"].update(settings)
    result["model_routing"] = {}
    return result


def model_base(root: Path, config: dict) -> dict:
    return (copy.deepcopy(config["llm_base"]) if "llm_base" in config
            else json.loads((root / config["llm_config"]).read_text(encoding="utf-8")))


def runtime_paths(root: Path, config: dict, base: dict) -> tuple[Path, Path]:
    """Resolve deployment paths at launch without rewriting saved generation identity."""
    saved_model = root / base["model"]["relative_path"]
    saved_server = root / base["server"]["executable"]
    if "llm_base" not in config and saved_model.is_file() and saved_server.is_file():
        return saved_model.resolve(strict=True), saved_server.resolve(strict=True)
    entry = entries(root).get(config["llm"]["model_id"])
    location = resolve_entry(root, config["llm"]["model_id"], entry)[0] if entry else base
    return ((root / location["model"]["relative_path"]).resolve(strict=True),
            (root / location["server"]["executable"]).resolve(strict=True))
