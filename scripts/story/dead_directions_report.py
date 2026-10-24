"""Find directions whose removal changes nothing a viewer sees (read-only, counterfactual).

What is seen is sampled at two moments of every line: while its text shows (after `before`
and `start`) and after its `after` directions. Inside an event CG only the CG is seen.
A remote line also shows its speaker's current look in the frame. Time effects (pause,
blackout) are counted apart: they are dead only where the compiler drops them.

An expression is judged twice: with the face pictures the work has ("no_picture" when only
those are missing) and as if every face had one ("dead" when even that shows nothing).

Usage: python -I scripts/story/dead_directions_report.py data <per-work counts.json>
Reads every stored script.json through the database opened read-only.
"""
import collections
import json
import sqlite3
import sys
from pathlib import Path

STATE = {"background", "enter", "exit", "position", "outfit", "expression", "focus"}


def cg_lines(script):
    order = {u["id"]: i for i, u in enumerate(script["utterances"])}
    lines = {}
    for seg in script.get("event_cg_segments", []):
        start = order[seg["start_utterance_id"]]
        end = order[seg["end_utterance_id"]] if seg.get("end_utterance_id") else len(order)
        variants = {v["utterance_id"]: v for v in seg.get("variants", [])}
        asset = seg["base_asset_id"]
        for u in script["utterances"][start:end]:
            if u["id"] in variants:
                asset = variants[u["id"]]["asset_id"]
            lines[u["id"]] = asset
    return lines


def snapshots(script, skip, faces_exist, cgs, by_anchor):
    bg, cast, worn, faces, out = None, {}, {}, {}, []
    pictures = {c["id"] for c in script["characters"] if c.get("image_asset_id")}

    def picture(cid):
        outfit = worn.get(cid)
        outfit = None if outfit == "default" else outfit
        face = faces.get(cid)
        return (outfit, face if face and faces_exist(cid, outfit, face) else None)

    def view(u, lit):
        if u["id"] in cgs:
            return ("cg", cgs[u["id"]])
        people = tuple(sorted((cid, slot, picture(cid)) for cid, slot in cast.items()))
        frame = None
        if u.get("remote") and u.get("speaker_id"):
            o = worn.get(u["speaker_id"])
            frame = None if o in (None, "default") else o
        return (bg, people, lit if lit in cast else "", frame)

    def run(u, timing, lit):
        nonlocal bg
        for d in by_anchor.get((u["id"], timing), []):
            if d["id"] in skip:
                continue
            k, cid = d["kind"], d.get("character_id")
            if k == "background":
                bg = d["asset_id"]
            elif k == "enter":
                worn[cid] = d.get("outfit_id")
                faces.pop(cid, None)
                cast[cid] = d["position"]
            elif k == "exit":
                cast.pop(cid, None)
            elif k == "position":
                if cid in cast:
                    cast[cid] = d["position"]
            elif k == "outfit":
                worn[cid] = d["outfit_id"]
            elif k == "expression":
                faces[cid] = d["expression"]
            elif k == "focus":
                lit = cid
        return lit

    for u in script["utterances"]:
        lit = run(u, "before", "")
        speaker = u.get("speaker_id")
        lit = speaker if speaker in pictures and not u.get("remote") else ""
        lit = run(u, "start", lit)
        out.append(view(u, lit))
        lit = run(u, "after", lit)
        out.append(view(u, lit))
    return out


def analyse(script):
    cgs = cg_lines(script)
    by_anchor = collections.defaultdict(list)
    for d in script["directions"]:
        by_anchor[(d["utterance_id"], d["timing"])].append(d)
    real = {(c["id"], e.get("outfit_id"), e["expression"]) for c in script["characters"]
            for e in c.get("expressions", [])}
    has_real = lambda cid, outfit, face: (cid, outfit, face) in real
    has_all = lambda cid, outfit, face: face != "neutral"
    base = {name: snapshots(script, set(), fn, cgs, by_anchor) for name, fn in
            (("real", has_real), ("all", has_all))}
    entries = {t["utterance_id"] for t in script.get("scene_transitions", [])}
    speakers = {u["id"]: u.get("speaker_id") for u in script["utterances"]}
    result = []
    for d in script["directions"]:
        k = d["kind"]
        if k in STATE:
            dead_all = snapshots(script, {d["id"]}, has_all, cgs, by_anchor) == base["all"]
            dead_real = dead_all or snapshots(script, {d["id"]}, has_real, cgs, by_anchor) == base["real"]
            verdict = "dead" if dead_all else "no_picture" if dead_real else "live"
        elif k == "blackout":
            verdict = "dead" if d["utterance_id"] in entries and d["timing"] in {"before", "start"} else "live"
        else:
            verdict = "live" if d.get("duration_ms", 0) > 0 else "dead"
        detail = ""
        if k == "focus":
            detail = "speaker" if d["character_id"] == speakers[d["utterance_id"]] else "other"
        result.append((k, d["timing"], verdict, d["utterance_id"] in entries, detail))
    return result


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    sys.stdout.reconfigure(encoding="utf-8")
    db = sqlite3.connect(f"file:{data / 'auto_drama.db'}?mode=ro", uri=True)
    rows = db.execute("SELECT a.storage_key, p.title FROM artifact a LEFT JOIN project p ON p.id=a.project_id "
                      "WHERE a.kind='script' ORDER BY a.created_at").fetchall()
    seen, counts, works = set(), collections.Counter(), collections.defaultdict(collections.Counter)
    for key, title in rows:
        raw = (data / "artifacts" / key).read_bytes()
        if raw in seen:
            continue
        seen.add(raw)
        for k, timing, verdict, entry, detail in analyse(json.loads(raw)):
            counts[(k, verdict)] += 1
            counts[(k, timing, verdict, "entry" if entry else "mid", detail)] += 1
            works[title][(k, verdict)] += 1
    print("scripts", len(seen))
    for k in ["background", "enter", "exit", "position", "outfit", "expression", "focus", "pause", "blackout"]:
        total = sum(v for key, v in counts.items() if len(key) == 2 and key[0] == k)
        parts = {v: counts[(k, v)] for v in ("live", "no_picture", "dead")}
        print(f"{k:10} total {total:6}  " + "  ".join(f"{a} {b}" for a, b in parts.items()))
    print()
    for key, v in sorted(counts.items(), key=lambda kv: (str(kv[0][0]), -kv[1])):
        if len(key) == 5 and key[2] != "live":
            print(key, v)
    out.write_text(json.dumps({t: {"/".join(k): v for k, v in c.items()} for t, c in works.items()},
                              ensure_ascii=False, indent=1), encoding="utf-8")


main()
