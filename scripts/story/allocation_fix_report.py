"""Summarize probe_allocation_fix.py runs: outcome, cut, and each chapter's title and role.

Usage: allocation_fix_report.py ALLOC_DIR  (writes ALLOC_DIR/summary.md)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

FIXES = ("now", "pick", "code", "order")


def run_rows(folder: Path) -> dict:
    allocation = folder / "chapter-allocation.json"
    requests = sorted((folder / "requests").glob("chapter-allocation-*.json"))
    requests = [path for path in requests if not path.name.endswith(".validation.json")]
    prompt = []
    metrics = folder / "llm" / "llm-metrics.json"
    if metrics.exists():
        prompt = [row["prompt_tokens"] for row in json.loads(metrics.read_text(encoding="utf-8"))
                  if row.get("type") == "context_budget" and row.get("stage") == "script-allocation"]
    row = {"attempts": len(requests), "prompt_tokens": max(prompt) if prompt else None}
    if allocation.exists():
        value = json.loads(allocation.read_text(encoding="utf-8"))
        row.update(ok=True, ends=[c["last_event"] for c in value["chapters"]], chapters=value["chapters"])
    else:
        row.update(ok=False, ends=None, chapters=[])
    return row


def main() -> int:
    alloc = Path(sys.argv[1])
    runs = defaultdict(dict)
    for folder in sorted(path for path in alloc.iterdir() if path.is_dir()):
        match = re.fullmatch(r"(.+)-(now|pick|code|order)-s(\d+)", folder.name)
        if match:
            runs[match[1]][(match[2], int(match[3]))] = run_rows(folder)
    lines = ["# 章分けの直し方（鉄と血）", "", "| 連鎖 | seed | " + " | ".join(FIXES) + " |",
             "|---|---|" + "---|" * len(FIXES)]
    totals = defaultdict(lambda: [0, 0, 0])
    for chain, rows in runs.items():
        for seed in sorted({seed for _, seed in rows}):
            cells = []
            for fix in FIXES:
                row = rows.get((fix, seed))
                if row is None:
                    cells.append("-")
                    continue
                totals[fix][0] += 1
                totals[fix][1] += row["ok"]
                totals[fix][2] += row["attempts"] > 1
                mark = ",".join(map(str, row["ends"])) if row["ok"] else "失敗"
                cells.append(mark + (f"（{row['attempts']}回目）" if row["ok"] and row["attempts"] > 1 else ""))
            lines.append(f"| {chain} | {seed} | " + " | ".join(cells) + " |")
    lines += ["", "| 方式 | 成功/実行 | やり直し | 章分け入力の最大トークン |", "|---|---|---|---|"]
    for fix in FIXES:
        tokens = [row["prompt_tokens"] for rows in runs.values() for (f, _), row in rows.items()
                  if f == fix and row["prompt_tokens"]]
        done, ok, retried = totals[fix]
        lines.append(f"| {fix} | {ok}/{done} | {retried} | {max(tokens) if tokens else '-'} |")
    for chain, rows in runs.items():
        for (fix, seed), row in sorted(rows.items(), key=lambda item: (item[0][1], FIXES.index(item[0][0]))):
            if not row["ok"]:
                continue
            lines += ["", f"## {chain} / {fix} / seed {seed}（{','.join(map(str, row['ends']))}）", ""]
            for chapter in row["chapters"]:
                lines.append(f"- 第{chapter['number']}章〜出来事{chapter['last_event']}「{chapter['title']}」{chapter['role']}")
    (alloc / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(alloc / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
