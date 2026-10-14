"""Summarize a probe_staging_context run: tokens, validity, and how annotations moved."""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

ARMS = ("current", "reseed", "trimmed", "preceding", "single", "blank", "minimal")
JAPANESE = re.compile(r"[ぁ-んァ-ヶ一-龠]")


def load(output):
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    return rows, {(row["title"], row["scene"], row["batch"], row["arm"]): row for row in rows}


def annotations(row):
    value = json.loads(row["content"])
    return {item["utterance_id"]: item for item in value["emotions"]}, value["directions"]


def foreign(text):
    return bool(text.strip()) and not JAPANESE.search(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blind", type=Path, help="write a shuffled sheet of the lines that moved")
    parser.add_argument("--blind-arms", nargs="+", default=["trimmed"])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rows, table = load(args.output)
    arms = [arm for arm in ARMS if any(row["arm"] == arm for row in rows)]
    keys = sorted({key[:3] for key in table})
    print("calls", len(rows))
    print("\n== validity ==")
    for arm in arms:
        mine = [row for row in rows if row["arm"] == arm]
        print(f"{arm:10s}", dict(collections.Counter(row["status"] for row in mine)),
              dict(collections.Counter(row.get("finish_reason") for row in mine)))
        for row in mine:
            if row["status"] != "valid":
                print("   ", row["title"], row["scene"], row["batch"], row.get("error", "")[:200])
    print("\n== tokens and time per call, against 'current' on the same batches ==")
    for arm in arms:
        common = [key for key in keys if (*key, arm) in table and (*key, "current") in table]
        base, mine = [table[(*key, "current")] for key in common], [table[(*key, arm)] for key in common]
        mean = lambda use, field: statistics.mean(row["usage"].get(field, 0) for row in use)
        seconds = lambda use: statistics.mean(row["elapsed_seconds"] for row in use)
        print(f"{arm:10s} n={len(common):2d}  input {mean(mine, 'prompt_tokens'):5.0f} (current {mean(base, 'prompt_tokens'):5.0f})  "
              f"output {mean(mine, 'completion_tokens'):5.0f} ({mean(base, 'completion_tokens'):5.0f})  "
              f"seconds {seconds(mine):4.1f} ({seconds(base):4.1f})")
    print("\n== annotation profile per arm (valid calls) ==")
    for arm in arms:
        voice, face, kinds = collections.Counter(), collections.Counter(), collections.Counter()
        spoken = delivered = inner = lines = unset = inner_foreign = delivery_foreign = 0
        batches = foreign_batches = 0
        for row in rows:
            if row["arm"] != arm or row["status"] != "valid":
                continue
            emotions, directions = annotations(row)
            speakers = {line["id"]: line["speaker"] for line in row["lines"]}
            batches += 1
            here = 0
            for uid, item in emotions.items():
                lines += 1
                unset += item["inner_emotion"] == "未指定"
                inner_foreign += foreign(item["inner_emotion"])
                here += foreign(item["inner_emotion"])
                if speakers[uid] is None:
                    continue
                spoken += 1
                voice[item["voice_emotion"]] += 1
                face[item.get("expression", "neutral")] += 1
                delivered += bool(item["delivery"].strip())
                delivery_foreign += foreign(item["delivery"])
                inner += len(item["inner_emotion"])
            foreign_batches += here * 2 >= len(emotions)
            kinds.update(item["kind"] for item in directions)
        if not spoken:
            continue
        print(f"{arm}: lines {lines}, dialogue {spoken}")
        print("   voice:", dict(voice.most_common()), f"| neutral {voice['neutral'] * 100 / spoken:.0f}%")
        print("   face :", dict(face.most_common()), f"| neutral {face['neutral'] * 100 / spoken:.0f}%")
        print(f"   inner left as '未指定' {unset} | inner not Japanese {inner_foreign}/{lines} | "
              f"delivery not Japanese {delivery_foreign}/{spoken} | batches mostly not Japanese {foreign_batches}/{batches}")
        print(f"   delivery given {delivered * 100 / spoken:.0f}% | inner emotion mean {inner / spoken:.1f} chars | "
              f"directions {dict(kinds)}")
    print("\n== agreement with 'current' on dialogue lines (reseed = noise floor) ==")
    for arm in arms[1:]:
        same_voice = same_face = total = 0
        for key in keys:
            a, b = table.get((*key, "current")), table.get((*key, arm))
            if not a or not b or a["status"] != "valid" or b["status"] != "valid":
                continue
            left, right = annotations(a)[0], annotations(b)[0]
            for line in a["lines"]:
                if line["speaker"] is None:
                    continue
                total += 1
                same_voice += left[line["id"]]["voice_emotion"] == right[line["id"]]["voice_emotion"]
                same_face += left[line["id"]].get("expression") == right[line["id"]].get("expression")
        if total:
            print(f"{arm:10s} lines {total:4d}  voice {same_voice * 100 / total:.0f}%  face {same_face * 100 / total:.0f}%")
    if args.blind:
        rng, answers, text = random.Random(20261013), [], []
        for arm in args.blind_arms:
            for key in keys:
                a, b = table.get((*key, "current")), table.get((*key, arm))
                if not a or not b or a["status"] != "valid" or b["status"] != "valid":
                    continue
                left, right = annotations(a)[0], annotations(b)[0]
                for index, line in enumerate(a["lines"]):
                    if line["speaker"] is None:
                        continue
                    one, two = left[line["id"]], right[line["id"]]
                    if one["voice_emotion"] == two["voice_emotion"] and one.get("expression") == two.get("expression"):
                        continue
                    pair = [("current", one), (arm, two)]
                    rng.shuffle(pair)
                    answers.append({"number": len(answers) + 1, "P": pair[0][0], "Q": pair[1][0]})
                    before = a["lines"][index - 1]["text"][:60] if index else ""
                    text.append(f"\n#{len(answers)} {key[0][:8]} {line['id']} (前: {before})\n  「{line['text'][:100]}」")
                    for label, (_, item) in zip("PQ", pair):
                        text.append(f"   {label}: 声={item['voice_emotion']} 顔={item.get('expression', 'neutral')} "
                                    f"内心={item['inner_emotion'][:24]} 出し方={item['delivery'][:30]}")
        args.blind.write_text("\n".join(text), encoding="utf-8")
        args.blind.with_suffix(".key.json").write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
        print("\nblind sheet:", len(answers), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
