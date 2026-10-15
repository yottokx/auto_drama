"""Short stand-ins for long character IDs inside one LLM request.

Characters added after the first get IDs like character-<uuid>. Each mention
costs about 35 tokens, and a script repeats the speaker ID on every line, so
the long form took about a quarter of a scene's output. Inside a request each
long ID becomes an unused character-N; the reply is mapped back before anyone
reads it, so saved stories, plans and caches only ever hold the real IDs.
"""

from __future__ import annotations

import re

_EDGE_BEFORE, _EDGE_AFTER = r"(?<![0-9A-Za-z_-])", r"(?![0-9A-Za-z_-])"
LONG_ID = re.compile(_EDGE_BEFORE + r"character-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
                     + _EDGE_AFTER)
SHORT_ID = re.compile(_EDGE_BEFORE + r"character-(\d+)" + _EDGE_AFTER)


def _strings(value):
    """Every string in JSON-compatible data, dictionary keys included.

    Matching runs on the strings themselves: in serialized JSON a line break
    before a speaker ID would read as the letter n and hide the ID's boundary.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _map_strings(value, change):
    if isinstance(value, str):
        return change(value)
    if isinstance(value, dict):
        return {_map_strings(key, change): _map_strings(item, change) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_map_strings(item, change) for item in value]
    return value


def label_length(character_id: str) -> int:
    """How long a speaker label is as the model writes it."""
    return len("character-00") if LONG_ID.fullmatch(character_id) else len(character_id)


class IdAliases:
    """A bijection between the long IDs of one request and numbers it does not use."""

    def __init__(self, value):
        texts = list(_strings(value))
        used = {int(number) for text in texts for number in SHORT_ID.findall(text)}
        self.forward: dict[str, str] = {}
        number = 1
        for real in dict.fromkeys(match for text in texts for match in LONG_ID.findall(text)):
            while number in used:
                number += 1
            used.add(number)
            self.forward[real] = f"character-{number}"
        self.backward = {alias: real for real, alias in self.forward.items()}
        self._alias = re.compile(_EDGE_BEFORE + "(" + "|".join(
            re.escape(alias) for alias in sorted(self.backward, key=len, reverse=True)) + ")" + _EDGE_AFTER)

    @classmethod
    def for_request(cls, messages: list[dict], extra: dict) -> IdAliases:
        return cls([messages, extra])

    def __bool__(self):
        return bool(self.forward)

    def encode(self, value):
        """Replace long IDs anywhere in JSON-compatible data (prompts, schemas, grammars)."""
        if not self:
            return value
        return _map_strings(value, lambda text: LONG_ID.sub(lambda match: self.forward[match.group()], text))

    def decode(self, text: str) -> str:
        if not self or not isinstance(text, str):
            return text
        return self._alias.sub(lambda match: self.backward[match.group()], text)

    def decode_response(self, response: dict) -> dict:
        """The same response with every text the model wrote mapped back to real IDs."""
        if not self or not isinstance(response.get("choices"), list):
            return response
        choices = []
        for choice in response["choices"]:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                choices.append(choice)
                continue
            message = {**message, **{key: self.decode(message[key])
                                     for key in ("content", "reasoning_content") if key in message}}
            if isinstance(message.get("tool_calls"), list):
                message["tool_calls"] = [
                    {**call, "function": {**call["function"], "arguments": self.decode(call["function"].get("arguments"))}}
                    if isinstance(call, dict) and isinstance(call.get("function"), dict) else call
                    for call in message["tool_calls"]]
            choices.append({**choice, "message": message})
        return {**response, "choices": choices}
