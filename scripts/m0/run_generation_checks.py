"""Run M0 model checks sequentially; keep ordinary log files for long operations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIMES = ROOT / "services/worker/runtimes"


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def gpu_snapshot() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    return {"time": datetime.now(UTC).isoformat(), "gpu": result.stdout.strip(),
            "compute_processes": processes.stdout.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--download-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    downloaded = args.download_run.resolve()
    if not run.is_relative_to(ROOT / "private/m0"):
        raise ValueError("Output must be inside private/m0")
    if (run / "status.json").exists():
        raise FileExistsError("Choose a new run directory to preserve previous results")
    source_status = json.loads((downloaded / "status.json").read_text(encoding="utf-8-sig"))
    if any(source_status[key]["state"] != "completed" for key in ("downloads", "llm")):
        raise RuntimeError("The preceding downloads and GGUF validation must have succeeded")
    run.mkdir(parents=True, exist_ok=True)
    narrative = (downloaded / "llm/narrative.txt").read_text(encoding="utf-8").strip()
    scene = {"speaker": "ナレーター", "dialogue_text": " ".join(narrative.splitlines()),
             "display_text": narrative, "source": str(downloaded / "llm/narrative.txt")}
    write_json(run / "scene.json", scene)
    python = {
        "image": str(RUNTIMES / "diffusers/.venv/Scripts/python.exe"),
        "tts": str(RUNTIMES / "irodori/.venv/Scripts/python.exe"),
        "worker": str(ROOT / "services/worker/.venv/Scripts/python.exe"),
    }
    converted = RUNTIMES / "diffusers/models" / f"anima-converted-{run.name}"

    def command(runtime: str, script: str, *arguments) -> list[str]:
        return [python[runtime], "-I", "-u", "-X", "utf8", str(ROOT / "scripts/m0" / script),
                *map(str, arguments)]

    steps = [
        ("image", command("image", "image_smoke.py", "--mode", "character", "--output-dir", run / "image")),
        ("background", command("image", "image_smoke.py", "--mode", "background", "--width", 1024,
                               "--height", 768, "--output-dir", run / "background")),
        ("conversion", command("image", "convert_anima_smoke.py", "--output-dir", converted)),
        ("converted_image", command("image", "image_smoke.py", "--mode", "character", "--model-dir", converted,
                                    "--output-dir", run / "converted_image")),
        ("tts", command("tts", "tts_smoke.py", "--scene-json", run / "scene.json", "--output-dir", run / "tts")),
        ("llm_after_tts", command("worker", "llm_smoke.py", "--model", source_status["llm"]["model"],
                                  "--output-dir", run / "llm")),
        ("tyrano_build", command("worker", "build_tyrano_scene.py", "--output-dir", run / "player",
                                "--image-dir", run / "image", "--background-dir", run / "background",
                                "--tts-dir", run / "tts", "--llm-dir", downloaded / "llm")),
    ]
    status = {"state": "running", "pid": os.getpid(), "started_at": datetime.now(UTC).isoformat(),
              "run_dir": str(run), "download_run": str(downloaded), "steps": {},
              "gpu_before": gpu_snapshot(),
              "remaining_review": "画像の目視、音声試聴、ティラノ実再生の確認。外部APIは後で設定。"}
    write_json(run / "status.json", status)
    for name, argv in steps:
        process = None
        detail = {"state": "running", "command": argv, "gpu_before": gpu_snapshot()}
        status["steps"][name] = detail
        status["current_step"] = name
        write_json(run / "status.json", status)
        print(f"START {name} -> {run / (name + '.log')}", flush=True)
        start = time.monotonic()
        try:
            with (run / f"{name}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                detail["pid"] = process.pid
                while process.poll() is None:
                    detail["elapsed_seconds"] = round(time.monotonic() - start, 1)
                    write_json(run / "status.json", status)
                    if (run / "stop.request").exists() or time.monotonic() - start > 3600:
                        raise TimeoutError("Stop requested or stage exceeded one hour")
                    print(f"RUNNING {name}: {detail['elapsed_seconds']} sec", flush=True)
                    time.sleep(10)
            if process.returncode:
                raise RuntimeError(f"{name} exited with code {process.returncode}; see {name}.log")
            detail.update(state="completed", elapsed_seconds=round(time.monotonic() - start, 1))
            time.sleep(2)
            detail["gpu_after_process_exit"] = gpu_snapshot()
            print(f"DONE {name}: {detail['elapsed_seconds']} sec; {detail['gpu_after_process_exit']['gpu']}", flush=True)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            if process is not None and process.poll() is None:
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, check=False, timeout=30)
                process.wait(timeout=30)
            detail.update(state="failed", error=str(exc))
            status["state"] = "failed"
            print(f"FAILED {name}: {exc}", flush=True)
            write_json(run / "status.json", status)
            return 1
        write_json(run / "status.json", status)
    status.update(state="generation_completed_review_pending", finished_at=datetime.now(UTC).isoformat())
    write_json(run / "status.json", status)
    print(f"Generation checks completed. Visual/audio/player review remains: {run}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
