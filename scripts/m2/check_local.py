"""Exercise the running M2 API and local worker with actual model generation.

This creates a clearly labelled verification project and immutable artifacts.
It never substitutes fixtures for generated content. Use --resume to continue
the same verification report after diagnosing a failed job.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def write_report(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-revision", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists() and not args.resume:
        parser.error("Use a new output directory, or --resume to continue this verification.")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    started = time.monotonic()
    deadline = started + args.timeout_seconds

    def event(stage: str, **fields):
        report.update(
            stage=stage, updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields
        )
        write_report(report_path, report)
        print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)

    with httpx.Client(base_url=args.coordinator, timeout=45, trust_env=False) as client:

        def request(method, path, body=None):
            response = client.request(method, path, json=body)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"{method} {path}: {response.status_code} {response.text[:1500]}"
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

        def wait_for_jobs(stage):
            previous = None
            while time.monotonic() < deadline:
                current = detail()
                active = [j for j in current["jobs"] if j["status"] in {"pending", "running"}]
                summary = [(j["id"], j["kind"], j["status"], j["attempt_count"]) for j in active]
                if summary != previous:
                    event(stage, active_jobs=summary)
                    previous = summary
                if not active:
                    failed = [j for j in current["jobs"] if j["status"] == "failed"]
                    if failed and current["draft"].get("activeJobId"):
                        raise RuntimeError(f"Generation failed: {failed[-1].get('error')}")
                    return current
                time.sleep(2)
            raise TimeoutError(
                "M2 model verification exceeded its time budget; jobs remain durable."
            )

        try:
            if not report.get("project_id"):
                created = request(
                    "POST",
                    "/api/m2/projects",
                    {
                        "world": {
                            "title": "検証｜夜明けの図書列車",
                            "chapterCount": 3,
                            "prompt": "砂漠を巡る移動図書館の列車が舞台。土地ごとに異なる民話を集め、失われた駅の由来を調べる。戦いや殺人に頼らず、会話と小さな発見で進む物語。",
                            "genre": "旅情ミステリー × 民俗学 × 日常",
                            "mood": "乾いたユーモアと静かな余韻",
                            "notes": "人の善悪を単純化しない。魔法で問題を一括解決しない。",
                            "setting": "",
                        },
                        "characters": [
                            {
                                "id": "librarian",
                                "name": "",
                                "age": "40代",
                                "gender": "",
                                "role": "主人公・列車の司書",
                                "freeform": "旅人の話を聞きながら、民話を収集する司書。無口だが時折皮肉な冗談を言う。服は実用的な旅装。名前と性別は世界観に合うものを補完してください。",
                                "settings": "",
                                "appearance": "",
                                "voice": "",
                                "locked": {"settings": False, "appearance": False, "voice": False},
                            }
                        ],
                    },
                )
                report["project_id"] = created["project"]["id"]
                event("created", project_id=report["project_id"])
            current = wait_for_jobs("recovering")
            if current["draft"]["worldResult"] is None or current["draft"]["worldPendingChanges"]:
                action("generate-world")
                current = wait_for_jobs("world")
                if not current["draft"]["worldResult"]:
                    raise RuntimeError(
                        "World generation did not produce a result; inspect job history."
                    )
            if not current["draft"]["worldConfirmed"]:
                action("confirm-world")
            # Confirmation now starts the character pipeline automatically.
            current = wait_for_jobs("characters-and-assets")
            if any(
                not c["result"] or not c["imageArtifactId"] or not c["voiceArtifactId"]
                for c in current["draft"]["characters"]
            ):
                action("generate-characters")
                current = wait_for_jobs("characters-and-assets")
            character = current["draft"]["characters"][0]
            if (
                not character["result"]
                or not character["imageArtifactId"]
                or not character["voiceArtifactId"]
            ):
                raise RuntimeError(
                    "Character generation did not produce complete settings/image/voice."
                )
            if "initial_assets" not in report:
                report["initial_assets"] = {
                    "image": character["imageArtifactId"],
                    "voice": character["voiceArtifactId"],
                }
                event("initial-generation-complete")
            if not args.skip_revision and not report.get("revision_verified"):
                if not character["locked"]["appearance"]:
                    action("toggle-lock", character_id=character["id"], scope="appearance")
                if not report.get("revision_requested"):
                    action(
                        "revise-character",
                        character_id=character["id"],
                        scope="all",
                        instruction="外見は変えず、旅人の話に耳を傾ける慎重な人物に。声は落ち着いた低めの口調にし、語尾は柔らかくしてください。",
                    )
                    event("revision-requested", revision_requested=True)
                current = wait_for_jobs("locked-character-revision")
                character = current["draft"]["characters"][0]
                assert character["imageArtifactId"] == report["initial_assets"]["image"], (
                    "Locked portrait was replaced"
                )
                assert character["voiceArtifactId"] != report["initial_assets"]["voice"], (
                    "Voice revision was not generated"
                )
                event("revision-verified", revision_verified=True)
            current = action("approve") if not detail()["draft"]["approved"] else detail()
            assert current["draft"]["approved"] and current["draft"]["approval"]
            (output / "final.json").write_text(
                json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for character in current["draft"]["characters"]:
                for kind, key in (("portrait", "imageArtifactId"), ("voice", "voiceArtifactId")):
                    response = client.get(f"/api/artifacts/{character[key]}/content")
                    response.raise_for_status()
                    extension = "png" if kind == "portrait" else "wav"
                    (output / f"{character['id']}-{kind}.{extension}").write_bytes(response.content)
            event(
                "completed",
                approval=current["draft"]["approval"],
                elapsed_seconds=round(time.monotonic() - started, 2),
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
            event(
                "failed",
                error=f"{type(error).__name__}: {error}",
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
