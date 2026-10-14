"""Summarize a probe_id_alias run: tokens, time, validity and what the outputs look like."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path


def pairs(rows, stage):
    table = {(row["job"], row["name"], row["arm"]): row for row in rows if row["stage"] == stage}
    keys = sorted({key[:2] for key in table})
    return [(table.get((*key, "current")), table.get((*key, "alias"))) for key in keys]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--texts", type=Path, help="write the scene texts side by side for reading")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rows = json.loads((args.output / "results.json").read_text(encoding="utf-8"))
    print("calls", len(rows), collections.Counter((row["stage"], row["arm"], row["status"]) for row in rows))
    for row in rows:
        if row["status"] != "valid":
            print("  not valid:", row["job"][:8], row["stage"], row["name"], row["arm"], row.get("error", "")[:240])
    for stage in ("plan", "text", "staging"):
        both = [(a, b) for a, b in pairs(rows, stage) if a and b and a["usage"] and b["usage"]]
        if not both:
            continue
        print(f"\n== {stage}: {len(both)} pairs ==")
        for label, index in (("current", 0), ("alias", 1)):
            use = [pair[index] for pair in both]
            prompt = [row["usage"]["prompt_tokens"] for row in use]
            reply = [row["usage"]["completion_tokens"] for row in use]
            seconds = [row["elapsed_seconds"] for row in use]
            print(f"  {label:8s} input mean {statistics.mean(prompt):6.0f}  output mean {statistics.mean(reply):5.0f}  "
                  f"seconds mean {statistics.mean(seconds):5.1f}  valid {sum(row['status'] == 'valid' for row in use)}")
        change = lambda key: statistics.mean(1 - b["usage"][key] / a["usage"][key] for a, b in both) * 100
        print(f"  mean reduction per pair: input {change('prompt_tokens'):.0f}%  output {change('completion_tokens'):.0f}%  "
              f"seconds {statistics.mean(1 - b['elapsed_seconds'] / a['elapsed_seconds'] for a, b in both) * 100:.0f}%")
        if stage == "text":
            for label, index in (("current", 0), ("alias", 1)):
                use = [pair[index] for pair in both if pair[0]["status"] == pair[1]["status"] == "valid"]
                body = [row["checked"]["body_characters"] for row in use]
                lines = [row["checked"]["utterances"] for row in use]
                per = [row["usage"]["completion_tokens"] / row["checked"]["body_characters"] for row in use]
                print(f"  {label:8s} body chars mean {statistics.mean(body):5.0f}  lines mean {statistics.mean(lines):4.1f}  "
                      f"output tokens per body char {statistics.mean(per):.2f}")
        if stage == "plan":
            for a, b in both:
                print("   ", a["job"][:8], "current", a.get("checked"), "| alias", b.get("checked"))
        if stage == "staging":
            same_voice = same_face = total = 0
            for a, b in both:
                if a["status"] != "valid" or b["status"] != "valid":
                    continue
                left = {item["utterance_id"]: item for item in json.loads(a["content"])["emotions"]}
                right = {item["utterance_id"]: item for item in json.loads(b["content"])["emotions"]}
                for line in a["lines"]:
                    if line["speaker"] is None:
                        continue
                    total += 1
                    same_voice += left[line["id"]]["voice_emotion"] == right[line["id"]]["voice_emotion"]
                    same_face += left[line["id"]].get("expression") == right[line["id"]].get("expression")
            print(f"  agreement on dialogue lines: {total} lines, voice {same_voice * 100 / total:.0f}%, "
                  f"face {same_face * 100 / total:.0f}%")
            for label, index in (("current", 0), ("alias", 1)):
                kinds = collections.Counter()
                for pair in both:
                    if pair[index]["status"] == "valid":
                        kinds.update(item["kind"] for item in json.loads(pair[index]["content"])["directions"])
                print(f"  {label:8s} directions {dict(kinds)}")
    leaked = [row for row in rows if row["arm"] == "alias" and row.get("content")
              and any(alias in row["content"] for alias in row["aliases"].values())]
    print("\nalias names left in decoded replies:", len(leaked))
    if args.texts:
        text = []
        for a, b in pairs(rows, "text"):
            for row in (a, b):
                if row and row.get("content"):
                    names = {real: alias for real, alias in (b or row)["aliases"].items()}
                    body = row["content"]
                    for real, alias in names.items():
                        body = body.replace(real, alias)
                    text.append(f"\n\n######## {row['job'][:8]} {row['name']} [{row['arm']}] {row['status']}\n{body}")
        args.texts.write_text("".join(text), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
