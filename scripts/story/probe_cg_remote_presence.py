"""Tell the event CG prompt which referenced people are only heard, not present.

The CG prompt says every referenced character "stands on the same ground as their surroundings",
so a girl on a video call, planned "inside the smartphone screen", is often also drawn standing
in the boy's room. Staging already knows who is only heard in a scene (every line of theirs is
a remote voice; they never enter), and the CG plan input carries those annotations.

  current   the saved prompt, rebuilt from the saved visual directions
  presence  the same, except for people only heard in the CG's scene: their line says they are
            not physically there, and only the people who are present stand on the ground

compare (no GPU): rebuild every saved CG prompt both ways and list which ones change.
generate: render the changed CGs with the presence prompt at their saved seeds (and extra seeds,
both arms), plus unchanged CGs of other scenes at their saved seeds as a control.
Nothing in the product changes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from packages.contracts.event_cg import image_input_sha256
from services.worker.generation import event_cg_visuals
from services.worker.generation.llm import write_json

DATA = ROOT / "data"
GROUND = "The referenced characters stand on the same ground as their surroundings;"
PRESENT_GROUND = "The characters who are physically present stand on the same ground as their surroundings;"


def absent(reference: int) -> str:
    return (f"The character from Reference image {reference} is not physically present in this place: "
            "show them only as the directions place them, such as on a device screen, "
            "never as a person standing or sitting here.")


def connect():
    return sqlite3.connect(f"file:{DATA / 'auto_drama.db'}?mode=ro", uri=True)


def artifact(db, artifact_id) -> Path:
    key, = db.execute("select storage_key from artifact where id=?", (artifact_id,)).fetchone()
    return DATA / "artifacts" / key


def heard_only(scene: dict) -> set[str]:
    """Speakers whose every line in the scene is a remote voice: staging never brings them in."""
    lines: dict[str, list[bool]] = {}
    for row in scene["utterances"]:
        if row.get("speaker_id"):
            lines.setdefault(row["speaker_id"], []).append((row.get("remote") or {}).get("source") == "remote")
    return {cid for cid, remote in lines.items() if all(remote)}


def presence(prompt: str, references: dict[str, int], away: set[str]) -> str:
    if not away & set(references):
        return prompt
    assert prompt.count(GROUND) == 1
    out = []
    for line in prompt.split("\n"):
        for cid, number in references.items():
            if cid in away and line.startswith(f"Reference image {number} character:"):
                line = f"{line} {absent(number)}"
        out.append(line)
    return "\n".join(out).replace(GROUND, PRESENT_GROUND)


def saved_plans(db):
    """Every completed CG plan with its narrative, visual directions and saved prompts."""
    rows = db.execute("""select j.id, j.project_id, p.title, j.payload, a.storage_key from job j
        join project p on p.id = j.project_id join artifact a on a.id = j.result_artifact_id
        where j.kind = 'm3_event_cg_plan' and j.status = 'completed' order by j.created_at""").fetchall()
    for job_id, project_id, title, payload, key in rows:
        envelope = json.loads(zipfile.ZipFile(DATA / "artifacts" / key).read("result.json"))
        directions = {row["cg_id"]: row for row in envelope.get("trace", [])
                      if row.get("type") == "event_cg_visual_directions"}
        yield {"job_id": job_id, "project_id": project_id, "title": title, "payload": json.loads(payload),
               "cgs": envelope["result"]["cgs"], "directions": directions}


def compare(db) -> list[dict]:
    rows = []
    for plan in saved_plans(db):
        scenes = {row["id"]: row for row in plan["payload"]["context"]["narrative"]["scenes"]}
        for cg in plan["cgs"]:
            found = plan["directions"].get(cg["id"])
            if found is None:
                rows.append({"title": plan["title"], "plan": plan["job_id"], "cg": cg["id"], "skipped": "no directions"})
                continue
            ids = found["character_ids_in_reference_order"]
            expected = [cg["id"], *[row["id"] for row in cg["variants"]]]
            try:
                images = event_cg_visuals.compile_images({"images": found["images"]}, ids, expected)
            except ValueError as exc:
                rows.append({"title": plan["title"], "plan": plan["job_id"], "cg": cg["id"], "skipped": str(exc)[:120]})
                continue
            away = heard_only(scenes[cg["scene_id"]]) & set(ids)
            saved = [cg["prompt"], *[row["prompt"] for row in cg["variants"]]]
            for index, (image, old) in enumerate(zip(images, saved, strict=True)):
                references = {cid: number for number, cid in enumerate(ids, 2 if index else 1)}
                new = presence(image["prompt"], references, away)
                rows.append({"title": plan["title"], "project": plan["project_id"], "plan": plan["job_id"],
                             "chapter": plan["payload"]["chapter_number"], "scene": cg["scene_id"],
                             "cg": cg["id"], "image": image["id"], "characters": ids, "away": sorted(away),
                             "rebuilt_matches_saved": image["prompt"] == old, "changed": new != image["prompt"],
                             "current": image["prompt"], "presence": new})
    return rows


def image_jobs(db, project_id, chapter, cg_id):
    rows = db.execute("""select id, payload, result_artifact_id, created_at from job where project_id=? and
        kind='m3_event_cg' and status='completed' order by created_at""", (project_id,)).fetchall()
    out = []
    for job_id, payload, result_id, _ in rows:
        payload = json.loads(payload)
        if payload["chapter_number"] == chapter and payload["cg_id"] == cg_id and payload.get("variant_id") is None:
            out.append((job_id, payload, result_id))
    return out


def render(db, payload: dict, prompt: str, seed: int, work: Path) -> Path:
    """Run the product's CG image job on a copy of a saved request with another prompt and seed."""
    from services.worker.generation import event_cg_pipeline

    payload = copy.deepcopy(payload)
    payload.update(prompt=prompt, seed=seed)
    payload["input_sha256"] = image_input_sha256(payload)
    work.mkdir(parents=True, exist_ok=True)
    for row in payload["references"]:
        target = work / f"cg-reference-{row['reference_index']:02d}.png"
        if not target.is_file():
            data = artifact(db, row["artifact_id"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
            target.write_bytes(data)
    bundle = event_cg_pipeline.generate_job({"kind": "m3_event_cg", "payload": payload}, work)
    image = work / "image.png"
    image.write_bytes(zipfile.ZipFile(io.BytesIO(bundle)).read("image.png"))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--extra-seeds", type=int, default=2)
    parser.add_argument("--controls", type=int, default=3, help="unchanged CGs of other works to render")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    db = connect()
    rows = compare(db)
    write_json(output / "compare.json", rows)
    built = [row for row in rows if "skipped" not in row]
    changed = [row for row in built if row["changed"]]
    print(json.dumps({"saved_images": len(rows), "rebuilt": len(built),
                      "skipped": len(rows) - len(built),
                      "rebuilt_differs_from_saved": sum(not row["rebuilt_matches_saved"] for row in built),
                      "changed": [(row["title"], row["chapter"], row["scene"], row["image"], row["away"])
                                  for row in changed]}, ensure_ascii=False, indent=1))
    if not args.generate:
        return 0
    runs = output / "runs.json"
    done = json.loads(runs.read_text(encoding="utf-8")) if runs.is_file() else []
    keys = {(row["plan"], row["image"], row["arm"], row["seed"]) for row in done}
    tasks = []
    for row in changed:
        if row["image"] != row["cg"]:
            continue  # Variants edit their own base; the base CGs are enough here.
        jobs = image_jobs(db, row["project"], row["chapter"], row["cg"])
        match = [job for job in jobs if job[1]["prompt"] == row["current"]]
        if not match:
            print("no saved image job for", row["plan"], row["cg"], flush=True)
            continue
        _, payload, _ = match[0]
        tasks.append((row, payload, "presence", payload["seed"]))
        for offset in range(1, args.extra_seeds + 1):
            for arm in ("current", "presence"):
                tasks.append((row, payload, arm, payload["seed"] + offset))
    # Controls: CGs whose prompt the change leaves alone, rendered from their saved request as it
    # is. Older works were compiled by older prompt code, so the saved prompt is used, not a rebuild.
    unchanged = {(row["project"], row["chapter"], row["cg"]) for row in built
                 if not row["changed"] and row["image"] == row["cg"]}
    picked, seen = [], set()
    for project, chapter, cg_id in sorted(unchanged, key=lambda key: key[0]):
        title = next(row["title"] for row in built if row["project"] == project)
        target = title in {item["title"] for item in changed}
        if (title in seen or len(picked) >= args.controls) and not target:
            continue
        jobs = image_jobs(db, project, chapter, cg_id)
        if jobs:
            row = {"plan": jobs[-1][0], "image": cg_id, "title": title, "chapter": chapter, "changed": False,
                   "current": jobs[-1][1]["prompt"], "presence": jobs[-1][1]["prompt"]}
            picked.append((row, jobs[-1][1]))
            seen.add(title)
    for row, payload in picked:
        tasks.append((row, payload, "presence", payload["seed"]))
    print(f"{len(tasks)} renders", flush=True)
    from services.worker.generation import event_cg_session

    # Keep the image model loaded across renders, as the worker does.
    with event_cg_session.reuse_event_cg_runtime():
        for row, payload, arm, seed in tasks:
            key = (row["plan"], row["image"], arm, seed)
            if key in keys:
                continue
            name = f"{row['plan'][:8]}-{row['image']}-{arm}-{seed}"
            try:
                image = render(db, payload, row[arm], seed, output / "work" / name)
                record = {"plan": row["plan"], "image": row["image"], "arm": arm, "seed": seed,
                          "title": row["title"], "chapter": row["chapter"], "changed": row["changed"],
                          "saved_seed": seed == payload["seed"], "path": str(image)}
            except Exception as exc:  # noqa: BLE001 - a failed render is recorded and the rest continue
                record = {"plan": row["plan"], "image": row["image"], "arm": arm, "seed": seed,
                          "error": f"{type(exc).__name__}: {exc}"[:400]}
            done.append(record)
            write_json(runs, done)
            print(json.dumps({k: v for k, v in record.items() if k != "path"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
