"""Bounded deterministic randomness. Models choose when to call, Python draws."""
from __future__ import annotations

import hashlib
import json
import math
import random


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": required}


_items = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 100}
_integer = {"type": "integer"}
TOOL_DEFINITIONS = [
    {"type": "function", "function": {"name": name, "description": description,
     "parameters": _schema(properties, list(properties))}}
    for name, description, properties in [
        ("random_integer", "Draw one integer in an inclusive range.",
         {"minimum": _integer, "maximum": _integer}),
        ("random_choice", "Choose one item uniformly.", {"items": _items}),
        ("random_weighted", "Choose one item using nonnegative weights.",
         {"items": _items, "weights": {"type": "array", "items": {"type": "number"},
                                      "minItems": 1, "maxItems": 100}}),
        ("random_sample", "Choose count distinct items without replacement.",
         {"items": _items, "count": _integer}),
    ]
]


class RandomTools:
    def __init__(self, seed: int, limit: int = 8) -> None:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63).")
        self.seed = seed
        self.limit = limit
        self.trace: list[dict] = []

    def execute(self, name: str, arguments: dict) -> dict:
        if len(self.trace) >= self.limit:
            raise ValueError("Random tool call limit exceeded.")
        expected = {"random_integer": {"minimum", "maximum"}, "random_choice": {"items"},
                    "random_weighted": {"items", "weights"}, "random_sample": {"items", "count"}}
        if name not in expected or not isinstance(arguments, dict) or set(arguments) != expected[name]:
            raise ValueError("Unknown random tool or invalid argument keys.")
        if name == "random_integer":
            low, high = arguments["minimum"], arguments["maximum"]
            if (type(low) is not int or type(high) is not int
                    or not -1_000_000_000 <= low <= high <= 1_000_000_000):
                raise ValueError("Invalid random integer bounds.")
        else:
            items = arguments["items"]
            if (not isinstance(items, list) or not 1 <= len(items) <= 100
                    or any(not isinstance(item, str) or not item.strip() or len(item) > 2000
                           for item in items) or len(set(items)) != len(items)):
                raise ValueError("Tool items must be 1–100 unique nonempty strings.")
            if name == "random_sample":
                count = arguments["count"]
                if type(count) is not int or not 1 <= count <= len(items):
                    raise ValueError("Invalid random sample count.")
            if name == "random_weighted":
                weights = arguments["weights"]
                if (not isinstance(weights, list) or len(weights) != len(items)
                        or any(type(w) not in (int, float) or not math.isfinite(w)
                               or not 0 <= w <= 1e12 for w in weights) or sum(weights) <= 0):
                    raise ValueError("Weights must be finite, nonnegative, and have positive sum.")
        index = len(self.trace)
        material = json.dumps([self.seed, index, name, arguments], ensure_ascii=False,
                              sort_keys=True, separators=(",", ":"))
        call_seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
        rng = random.Random(call_seed)
        if name == "random_integer":
            value = rng.randint(low, high)
        elif name == "random_choice":
            value = rng.choice(items)
        elif name == "random_weighted":
            value = rng.choices(items, weights=weights, k=1)[0]
        else:
            value = rng.sample(items, count)
        result = {"value": value}
        self.trace.append({"type": "random_tool", "index": index, "name": name,
                           "seed": self.seed, "call_seed": call_seed,
                           "arguments": arguments, "result": result})
        return result
