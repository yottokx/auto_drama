"""Run M3 from a real approved M2 project through the HTTP worker protocol.

The coordinator and worker must already be running. This explicitly starts
production for the selected approval and saves the published source bundle.
No generated text, images or audio are substituted with fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", default="http://127.0.0.1:8000")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    report_path = output / "report.json"
    if report_path.exists() and not args.resume:
        parser.error("Use a new output directory or --resume.")
    output.mkdir(parents=True, exist_ok=True)
    report = {"project_id": args.project_id, "status": "starting"}
    started = time.monotonic()

    def save(**values):
        report.update(values, elapsed_seconds=round(time.monotonic() - started, 2))
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        print(json.dumps(values, ensure_ascii=False), flush=True)

    with httpx.Client(base_url=args.coordinator, timeout=120, trust_env=False) as client:
        endpoint = f"/api/m3/projects/{args.project_id}"
        try:
            response = client.post(endpoint + "/start", json={})
            response.raise_for_status()
            previous = None
            while time.monotonic() - started < args.timeout_seconds:
                response = client.get(endpoint)
                response.raise_for_status()
                production = response.json()["production"]
                summary = [(job["kind"], job["status"], job["attempt_count"])
                           for job in production["jobs"]]
                if summary != previous:
                    save(status=production["status"], stage=production["stage"], jobs=summary)
                    previous = summary
                if production["status"] == "failed":
                    raise RuntimeError(production.get("error") or "M3 generation failed")
                if production["status"] == "published":
                    bundle = client.get(production["export_url"])
                    bundle.raise_for_status()
                    (output / "tyrano-source.zip").write_bytes(bundle.content)
                    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
                        if archive.testzip() is not None:
                            raise RuntimeError("Corrupt source bundle")
                        files = archive.namelist()
                        if any(name.startswith("tyrano/") for name in files):
                            raise RuntimeError("Player engine must not be bundled")
                        for name in files:
                            if name.endswith((".json", ".txt")):
                                # Retain inspectable data without trusting arbitrary paths.
                                target = output / (name.replace("/", "__"))
                                target.write_bytes(archive.read(name))
                    player = client.get(production["player_url"])
                    player.raise_for_status()
                    save(status="published", build=production["build"],
                         player_url=str(client.base_url).rstrip("/") + production["player_url"],
                         bundle_sha256=hashlib.sha256(bundle.content).hexdigest(),
                         bundle_bytes=len(bundle.content), files=files,
                         browser_review="pending")
                    return 0
                time.sleep(3)
            raise TimeoutError("Generation is still durable; resume the check after completion.")
        except Exception as exc:
            save(status="check_failed", error=str(exc))
            raise


if __name__ == "__main__":
    raise SystemExit(main())
