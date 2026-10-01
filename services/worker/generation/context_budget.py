"""Context budgets constrained by verified worker settings and the running server."""
from __future__ import annotations

from dataclasses import asdict, dataclass


def positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


@dataclass(frozen=True)
class OutputTokenPolicy:
    """Split common-setting context growth equally between input and generation.

    Profiles retain their 16k baseline values so saved model identities remain
    stable. The extension is applied once when constructing the request; explicit
    request limits (including frozen scene budgets) are already total limits.
    This policy is independent of the provider and reasoning-effort vocabulary.
    """

    base_limit: int
    extension_tokens: int = 0

    @classmethod
    def resolve(cls, settings: dict, profile: dict) -> OutputTokenPolicy:
        worker_limit = positive_integer(settings.get("max_output_tokens", 8192),
                                        "max_output_tokens", minimum=256)
        base_limit = positive_integer(profile.get("max_output_tokens", worker_limit),
                                      "profile.max_output_tokens", minimum=256)
        if base_limit > worker_limit:
            raise ValueError("Profile max_output_tokens exceeds the worker output allowance.")
        extension = 0
        if profile.get("common_settings_version"):
            context = positive_integer(profile.get("context_size", settings.get("context_size")),
                                       "context_size", minimum=16384)
            # Give an odd leftover token to input. Never round generation upward.
            extension = (context - 16384) // 2
        return cls(base_limit, extension)

    @property
    def limit(self) -> int:
        return self.base_limit + self.extension_tokens

    def tokens(self, baseline: int) -> int:
        if positive_integer(baseline, "baseline output tokens") > self.base_limit:
            raise ValueError("Baseline output tokens exceed the worker output allowance.")
        return baseline + self.extension_tokens


@dataclass(frozen=True)
class ContextPolicy:
    target: int
    permitted_maximum: int
    model_maximum: int
    margin: int
    allow_expansion: bool

    @classmethod
    def resolve(cls, settings: dict, profile: dict) -> ContextPolicy:
        # A configured target is a verified deployment setting. A larger automatic
        # range needs both a resource allowance and an explicitly verified model limit.
        configured = positive_integer(settings.get("context_size"), "context_size")
        permitted = positive_integer(settings.get("max_context_size", configured),
                                     "max_context_size")
        model = positive_integer(settings.get("model_context_size", configured),
                                 "model_context_size")
        if configured > min(permitted, model):
            raise ValueError("Configured context_size exceeds the verified context limits.")
        target = positive_integer(profile.get("context_size", configured), "profile.context_size")
        requested_maximum = positive_integer(profile.get("max_context_size", permitted),
                                             "profile.max_context_size")
        if requested_maximum > permitted:
            raise ValueError("Profile max_context_size exceeds the worker resource allowance.")
        maximum = min(permitted, model, requested_maximum)
        if target > maximum:
            raise ValueError("Profile context_size exceeds the verified context limits.")
        margin = positive_integer(profile.get("context_margin_tokens",
                                               settings.get("context_margin_tokens", 512)),
                                  "context_margin_tokens", minimum=0)
        worker_expansion = settings.get("allow_context_expansion", False)
        expansion = profile.get("allow_context_expansion", worker_expansion)
        if type(worker_expansion) is not bool or type(expansion) is not bool:
            raise ValueError("allow_context_expansion must be a boolean.")
        if expansion and not worker_expansion:
            raise ValueError("Automatic context expansion is not allowed by this worker.")
        return cls(target, maximum, model, margin, expansion)

    @property
    def startup_size(self) -> int:
        return self.permitted_maximum if self.allow_expansion else self.target

    def budget(self, prompt_tokens: int, output_tokens: int, server_context_size: int) -> dict:
        server = positive_integer(server_context_size, "server_context_size")
        prompt = positive_integer(prompt_tokens, "prompt_tokens", minimum=0)
        output = positive_integer(output_tokens, "output_tokens")
        available = min(server, self.permitted_maximum, self.model_maximum)
        target = min(self.target, available)
        required = prompt + output + self.margin
        context = available if self.allow_expansion and required > target else target
        return {"prompt_tokens": prompt, "output_tokens": output,
                "margin_tokens": self.margin, "context_size": context,
                "target_context_size": self.target, "server_context_size": server,
                "permitted_context_size": self.permitted_maximum,
                "model_context_size": self.model_maximum,
                "input_token_limit": max(0, context - output - self.margin),
                "required_tokens": required, "fits": required <= context,
                "expanded": context > self.target}

    def identity(self) -> dict:
        return asdict(self)


def server_context_from_properties(properties: dict) -> int | None:
    """Read per-slot context only; a model's training maximum is not runtime capacity."""
    defaults = properties.get("default_generation_settings", {})
    value = defaults.get("n_ctx") if isinstance(defaults, dict) else None
    if value is None:
        value = properties.get("n_ctx")
    if value is None:
        return None
    return positive_integer(value, "server n_ctx")
