"""Resolve model IDs on this worker. Local paths are never sent to coordinator."""
import copy
import json
import os
import re
from pathlib import Path

from packages.contracts.llm_settings import LLMCapability

from .generation.llm_providers import (
    API_TYPES,
    DEFAULT_ENDPOINTS,
    PROTOCOL_VERSION,
    REASONING_EFFORTS,
    SAMPLING_EFFORTS,
)

REGISTRY = Path("services/worker/config/models.local.json")
PROVIDER_ID = re.compile(r"[A-Za-z0-9._-]+")
KEY_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
    providers = api_providers(value.get("api_providers", []))
    api_models = value.get("api_models", [])
    if not isinstance(api_models, list):
        raise TypeError("api_models must be a list of {provider, model, max_input_tokens} entries")
    for item in api_models:
        if not isinstance(item, dict) or item.get("provider") not in providers:
            raise ValueError(f"api_models entries name one of the api_providers {sorted(providers)}: {item}")
        provider = providers[item["provider"]]
        api_model = item.get("model")
        if not isinstance(api_model, str) or not api_model.strip():
            raise ValueError(f"api_models entries need the provider's model name: {item}")
        model_id = f"{item['provider']}/{api_model}"
        limit = item.get("max_input_tokens")
        if type(limit) is not int or limit < 16384:
            raise ValueError(f"{model_id}: max_input_tokens must be an integer of at least 16384")
        if model_id in registry:
            raise ValueError(f"Duplicate model ID in api_models: {model_id}")
        efforts = item.get("reasoning_efforts", REASONING_EFFORTS[provider["type"]])
        if (not isinstance(efforts, list) or "none" not in efforts
                or not all(isinstance(level, str) and level for level in efforts)):
            raise ValueError(f"{model_id}: reasoning_efforts must be a list of names including 'none'")
        registry[model_id] = {**provider, "provider": item["provider"], "api_model": api_model,
                              "max_input_tokens": limit, "reasoning_efforts": efforts}
    # Optional overrides are still supported for existing installations.
    registry.update(value.get("models", {}))
    return registry


def api_providers(rows) -> dict:
    """Where API models are reached: the request shape, the endpoint and the key's variable."""
    if not isinstance(rows, list):
        raise TypeError("api_providers must be a list of {id, type, endpoint, api_key_env} entries")
    providers = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not PROVIDER_ID.fullmatch(row["id"]):
            raise ValueError(f"api_providers entries need an id of letters, digits, '.', '_' or '-': {row}")
        name = row["id"]
        if name in providers or name == "local":
            raise ValueError(f"api_providers id is reserved or repeated: {name}")
        if row.get("type") not in API_TYPES:
            raise ValueError(f"{name}: type must be one of {API_TYPES}")
        endpoint = row.get("endpoint", DEFAULT_ENDPOINTS.get(row["type"]))
        if not isinstance(endpoint, str) or not endpoint.startswith(("https://", "http://")):
            raise ValueError(f"{name}: endpoint must be an http(s) URL; {row['type']} has no default")
        key = row.get("api_key_env")
        if key is not None and (not isinstance(key, str) or not KEY_VARIABLE.fullmatch(key)):
            raise ValueError(f"{name}: api_key_env names an environment variable, never the key itself")
        fields = row.get("request_fields", {})
        if not isinstance(fields, dict):
            raise TypeError(f"{name}: request_fields must be an object")
        providers[name] = {"type": row["type"], "endpoint": endpoint.rstrip("/"), "api_key_env": key,
                           "request_fields": fields}
    return providers


def _resolve_api_entry(model_id: str, entry: dict) -> tuple[dict, dict]:
    """An API model has no files to check; it is offered only while its key is set."""
    key = entry["api_key_env"]
    if key is not None and not os.environ.get(key):
        raise ValueError(f"{key} is not set for this worker")
    limit = entry["max_input_tokens"]
    # Everything that may change a reply joins the cached model identity; the key's variable does not.
    base = {"schema_version": 1, "model": {
        "provider": entry["provider"], "type": entry["type"], "endpoint": entry["endpoint"],
        "api_model": entry["api_model"], "protocol": PROTOCOL_VERSION}}
    if entry["request_fields"]:
        base["model"]["request_fields"] = copy.deepcopy(entry["request_fields"])
    settings = {"model_id": model_id, "provider": entry["provider"], "api_key_env": key,
                "context_size": 16384, "max_context_size": limit, "model_context_size": limit,
                "reasoning_levels": entry["reasoning_efforts"], "reasoning_template": None,
                "sampling_efforts": SAMPLING_EFFORTS[entry["type"]], "max_output_tokens": 8192}
    return base, settings


def resolve_entry(root: Path, model_id: str, entry: dict) -> tuple[dict, dict]:
    if "provider" in entry:
        return _resolve_api_entry(model_id, entry)
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
                reasoning_efforts=settings["reasoning_levels"],
                sampling_efforts=settings.get("sampling_efforts")).model_dump(exclude_none=True))
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
    if settings.get("provider", "local") != "local":
        # The tier sizes the work and its answers; whether the input fits is checked against
        # the model's own limit, since its tokenizer counts the same text differently.
        if requested_context > settings["max_context_size"]:
            raise ValueError(f"{model_id} takes at most {settings['max_context_size']} input tokens.")
        settings.update(context_size=requested_context, allow_context_expansion=True)
    else:
        # This is the user's requested runtime allocation, not an inferred model limit.
        # Let llama-server load this context and report actual capacity or a load error.
        settings.update(context_size=requested_context, max_context_size=requested_context,
                        model_context_size=requested_context, allow_context_expansion=False)
    result = copy.deepcopy(config)
    result["llm_base"] = base
    result["llm_config"] = registry[model_id].get(
        "llm_config", "worker-api" if "provider" in registry[model_id] else "worker-discovered")
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
