"""Estimate how much of the staging answers' output tokens go to focus annotations (read-only).

Usage: python -I scripts/story/focus_output_tokens.py outputs tmp

Reads the saved LLM answers (`*-staging-*.json`) under the given folders. Without the model's
tokenizer, tokens are fitted per answer from its ASCII and other characters against the
recorded completion tokens; focus annotations are ASCII, so their share follows from that fit.
"""
import collections
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
files = [p for root in sys.argv[1:] for p in Path(root).rglob("*-staging-*.json") if "runtimes" not in p.parts]
rows = []
for path in files:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        response = saved["response"]
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        answer = json.loads(content)
        tokens = response["usage"]["completion_tokens"]
    except (KeyError, ValueError, TypeError, IndexError):
        continue
    if message.get("reasoning_content") or "directions" not in answer:
        continue
    indent = 2 if "\n  " in content else None
    sep = (",", ": ") if indent else ((",", ":") if ", " not in content and ": " not in content else (", ", ": "))

    def size(value):
        text = json.dumps(value, ensure_ascii=False, indent=indent, separators=None if indent else sep)
        if indent:
            text = text.replace("\n", "\n" + " " * 4) + " " * 4 + ",\n"
        return text

    focus = [d for d in answer["directions"] if d.get("kind") == "focus"]
    others = [d for d in answer["directions"] if d.get("kind") != "focus"]
    ascii_all = sum(c.isascii() for c in content)
    rows.append({"path": str(path), "tokens": tokens, "ascii": ascii_all, "other": len(content) - ascii_all,
                 "focus_chars": sum(len(size(d)) for d in focus), "focus": len(focus),
                 "other_dirs": len(others), "emotions": len(answer.get("emotions", []))})
print("responses", len(rows))
# tokens ~ a*ascii + b*non-ascii, least squares without intercept
sxx = sum(r["ascii"] ** 2 for r in rows); syy = sum(r["other"] ** 2 for r in rows)
sxy = sum(r["ascii"] * r["other"] for r in rows)
sxt = sum(r["ascii"] * r["tokens"] for r in rows); syt = sum(r["other"] * r["tokens"] for r in rows)
det = sxx * syy - sxy ** 2
a = (sxt * syy - syt * sxy) / det; b = (syt * sxx - sxt * sxy) / det
err = sum(abs(a * r["ascii"] + b * r["other"] - r["tokens"]) / r["tokens"] for r in rows) / len(rows)
print(f"tokens per ascii char {a:.3f}, per other char {b:.3f}, mean abs error {err:.1%}")
total = sum(r["tokens"] for r in rows)
focus_tokens = sum(a * r["focus_chars"] for r in rows)
print(f"output tokens {total}, focus ~{focus_tokens:.0f} ({focus_tokens / total:.1%})")
print("focus entries", sum(r["focus"] for r in rows), "other directions", sum(r["other_dirs"] for r in rows),
      "emotions", sum(r["emotions"] for r in rows))
print(f"tokens per focus entry ~{focus_tokens / max(1, sum(r['focus'] for r in rows)):.0f}")
by_dir = collections.defaultdict(lambda: [0, 0])
for r in rows:
    key = Path(r["path"]).parts[1] if len(Path(r["path"]).parts) > 1 else r["path"]
    by_dir[key][0] += r["tokens"]; by_dir[key][1] += a * r["focus_chars"]
for key, (t, f) in sorted(by_dir.items(), key=lambda kv: -kv[1][0])[:12]:
    print(f"  {key}: {t} tokens, focus {f / t:.1%}")
