"""Verify the extended M2 cast using the running API and actual local models."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists() and not args.resume:
        parser.error("Choose a new output directory or pass --resume.")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    began = time.monotonic()

    def event(stage, **fields):
        if stage != "failed":
            report.pop("error", None)
        report.update(stage=stage, **fields)
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)

    with httpx.Client(base_url=args.coordinator, timeout=45, trust_env=False) as client:

        def request(method, path, body=None):
            response = client.request(method, path, json=body)
            if response.is_error:
                raise RuntimeError(
                    f"{method} {path}: {response.status_code} {response.text[:1200]}"
                )
            return response.json()

        def detail():
            return request("GET", f"/api/m2/projects/{report['project_id']}")

        def action(name, **fields):
            current = detail()
            return request(
                "POST",
                f"/api/m2/projects/{report['project_id']}/actions",
                {
                    "expected_revision": current["draft"]["revision"],
                    "action": name,
                    **fields,
                },
            )

        def wait(stage):
            previous = None
            while time.monotonic() - began < 3600:
                current = detail()
                active = [job for job in current["jobs"] if job["status"] in {"running", "pending"}]
                ready = [
                    entry["id"]
                    for entry in current["draft"]["characters"]
                    if entry["result"] and entry["imageArtifactId"] and entry["voiceArtifactId"]
                ]
                summary = {
                    "active": [(j["id"], j["kind"], j["status"]) for j in active],
                    "ready_characters": ready,
                }
                if summary != previous:
                    event(stage, **summary)
                    previous = summary
                if not active:
                    failed = [
                        j
                        for j in current["jobs"]
                        if j["status"] == "failed"
                        and j["id"] == current["draft"].get("activeJobId")
                    ]
                    if failed:
                        raise RuntimeError(str(failed[0].get("error")))
                    return current
                time.sleep(2)
            raise TimeoutError(
                "Verification reached its one-hour limit; durable jobs remain available."
            )

        try:
            if not report.get("project_id"):
                world = {
                    "title": "検証｜潮待ち郵便局",
                    "chapterCount": 3,
                    "prompt": "海辺の町の郵便局を舞台に、届かなかった手紙と小さな誤解を会話で解く物語。殺人や戦闘に頼らず、日常の仕事と住民の温かさを描く。",
                    "genre": "お仕事もの × 書簡ミステリー",
                    "mood": "穏やかで少しおかしい",
                }
                characters = [
                    {
                        "id": "a-postmaster",
                        "name": "朝倉 澄",
                        "age": "40代",
                        "gender": "女性",
                        "role": "主人公・郵便局長",
                        "freeform": "実直で慎重。海鳥を観察するのが趣味。簡潔な丁寧語で話す。",
                    },
                    {
                        "id": "b-repairer",
                        "name": "三崎 灯",
                        "age": "30代",
                        "gender": "男性",
                        "role": "時計修理職人",
                        "freeform": "好奇心旺盛で手先が器用。作業中は無口だが普段は軽快に話す。",
                    },
                    {
                        "id": "c-courier",
                        "name": "冬野 凪",
                        "age": "20代",
                        "gender": "女性",
                        "role": "配達員",
                        "freeform": "明るく行動的で、雨でも散歩を楽しむ。率直で快活な話し方。",
                    },
                ]
                rejected = client.post(
                    "/api/m2/projects",
                    json={"world": world, "characters": [*characters, {"id": "fourth"}]},
                )
                assert rejected.status_code == 422, "Four-character limit was not enforced"
                created = request(
                    "POST", "/api/m2/projects", {"world": world, "characters": characters}
                )
                report["project_id"] = created["project"]["id"]
                inputs = [
                    {
                        "characterIds": list(pair),
                        "instruction": "仕事を通じて知り合った同業者。親族でも恋人でもなく、互いの専門性を尊重する仲。"
                        if pair == ("a-postmaster", "b-repairer")
                        else "",
                    }
                    for pair in itertools.combinations([c["id"] for c in characters], 2)
                ]
                # Save both kinds of instruction before world generation.
                report["relationship_inputs"] = inputs
                event("created", project_id=report["project_id"], limit_verified=True)
            current = wait("recovering")
            if not report.get("relations_saved"):
                current = action(
                    "save-brief", world=current["draft"]["worldInput"],
                    characters=[entry["input"] for entry in current["draft"]["characters"]],
                    relationshipInputs=report["relationship_inputs"],
                )
                event("combined-input-saved", relations_saved=True)
            if current["draft"]["worldResult"] is None or current["draft"]["worldPendingChanges"]:
                action("generate-world")
                current = wait("world")
            if not current["draft"]["worldConfirmed"]:
                action("confirm-world")
            current = wait("cast")
            if any(
                not c["result"] or not c["imageArtifactId"] or not c["voiceArtifactId"]
                for c in current["draft"]["characters"]
            ):
                action("generate-characters")
                current = wait("cast")
            if (
                current["draft"]["relationships"]["pendingChanges"]
                or not current["draft"]["relationships"]["result"]
            ):
                action("generate-relationships")
                current = wait("relationships")
            draft = current["draft"]
            for entry in draft["characters"]:
                result = entry["result"]
                assert result["selfIntroduction"].strip() and len(result["sampleLines"]) == 3
                assert entry["voiceReferenceText"] == result["selfIntroduction"], (
                    "Sample does not speak its introduction"
                )
                assert entry["imageArtifactId"] and entry["voiceArtifactId"]
            pairs = draft["relationships"]["result"]["pairs"]
            expected = {
                tuple(pair)
                for pair in itertools.combinations(sorted(c["id"] for c in draft["characters"]), 2)
            }
            assert {tuple(pair["characterIds"]) for pair in pairs} == expected
            assert all(
                pair["summary"].strip()
                and pair["firstToSecond"].strip()
                and pair["secondToFirst"].strip()
                for pair in pairs
            )
            if not draft["approved"]:
                current = action("approve")
            if not report.get("reference_voice"):
                first = current["draft"]["characters"][0]
                report.update(
                    reference_voice=first["voiceArtifactId"],
                    reference_approval=current["draft"]["approval"],
                    clone_text=first["result"]["sampleLines"][0],
                )
                event("cast-approved")
            first = current["draft"]["characters"][0]
            if not report.get("clone_requested"):
                action("clone-voice", character_id=first["id"], text=report["clone_text"])
                event("voice-clone-requested", clone_requested=True)
            current = wait("voice-clone")
            first = current["draft"]["characters"][0]
            assert first["voiceArtifactId"] == report["reference_voice"]
            assert (
                current["draft"]["approved"]
                and current["draft"]["approval"] == report["reference_approval"]
            )
            trial = first["voiceTests"][-1]
            assert (
                trial["text"] == report["clone_text"]
                and trial["sourceVoiceArtifactId"] == report["reference_voice"]
            )
            for entry in current["draft"]["characters"]:
                for label, artifact_id, ext in (
                    ("portrait", entry["imageArtifactId"], "png"),
                    ("introduction", entry["voiceArtifactId"], "wav"),
                ):
                    response = client.get(f"/api/artifacts/{artifact_id}/content")
                    response.raise_for_status()
                    (output / f"{entry['id']}-{label}.{ext}").write_bytes(response.content)
            clone = client.get(f"/api/artifacts/{trial['artifactId']}/content")
            clone.raise_for_status()
            (output / "voice-clone.wav").write_bytes(clone.content)
            (output / "final.json").write_text(
                json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            event(
                "completed",
                elapsed_seconds=round(time.monotonic() - began, 2),
                clone_sha256=hashlib.sha256(clone.content).hexdigest(),
                pair_count=len(pairs),
            )
            return 0
        except (
            AssertionError,
            RuntimeError,
            TimeoutError,
            ValueError,
            KeyError,
            OSError,
            httpx.HTTPError,
        ) as error:
            event("failed", error=f"{type(error).__name__}: {error}")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
