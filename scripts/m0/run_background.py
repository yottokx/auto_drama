"""Download pinned M0 models and verify the supplied LLM; record plain progress logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[2]
LOCK = threading.RLock()
STATE: dict = {}
RUN: Path


def now() -> str:
    return datetime.now(UTC).isoformat()


def update(section: str, **values) -> None:
    with LOCK:
        STATE[section].update(values)
        STATE["updated_at"] = now()
        temporary = RUN / "status.tmp"
        temporary.write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(RUN / "status.json")


def stopped() -> None:
    if (RUN / "stop.request").exists():
        raise RuntimeError("Stopped by user request; partial downloads are retained.")


def hashes(path: Path, callback=None) -> dict[str, str]:
    size = path.stat().st_size
    digests = {"sha256": hashlib.sha256(), "md5": hashlib.md5(), "git_blob_sha1": hashlib.sha1()}
    digests["git_blob_sha1"].update(f"blob {size}\0".encode())
    done, last = 0, 0.0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            stopped()
            for digest in digests.values():
                digest.update(block)
            done += len(block)
            if callback and time.monotonic() - last > 0.5:
                callback(done, size)
                last = time.monotonic()
    if callback:
        callback(done, size)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def verify(path: Path, item: dict, callback=None) -> dict:
    if path.stat().st_size != item["size"]:
        raise ValueError(f"Size mismatch: {path.name}")
    actual = hashes(path, callback)
    key = "sha256" if item.get("sha256") else "md5" if item.get("md5") else "git_blob_sha1"
    if item.get(key) and actual[key] != item[key]:
        raise ValueError(f"{key} mismatch: {path.name}; the existing file is preserved for inspection")
    return actual


def download_models() -> None:
    complete_bytes, count = 0, 0
    receipts = []
    try:
        models = []
        for name in ("m0-models-tts.json", "m0-models-image.json"):
            models.extend(json.loads((ROOT / "config" / name).read_text(encoding="utf-8-sig"))["models"])
        total = sum(f["size"] for m in models for f in m["files"])
        file_count = sum(len(m["files"]) for m in models)
        if shutil.disk_usage(ROOT).free < total + 2 * 1024**3:
            raise RuntimeError("Insufficient free disk space for model downloads")
        update("downloads", state="running", total_bytes=total, total_files=file_count)
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, connect=30),
                          headers={"Accept-Encoding": "identity", "User-Agent": "auto-drama-m0/0.1"}) as client:
            for model in models:
                base = (ROOT / model["local_dir"]).resolve()
                if not base.is_relative_to(ROOT / "services/worker"):
                    raise ValueError("Model destination must be inside the worker")
                for item in model["files"]:
                    stopped()
                    destination = (base / item["path"]).resolve()
                    if not destination.is_relative_to(base):
                        raise ValueError("Invalid model file path")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    partial = destination.with_name(destination.name + ".part")
                    update("downloads", model=model["id"], file=item["path"], phase="download",
                           file_bytes=0, file_total=item["size"], bytes_per_second=0, attempt=0)
                    if not destination.exists():
                        url = item.get("url") or (
                            f"https://huggingface.co/{model['repo_id']}/resolve/{model['revision']}/"
                            f"{quote(item['path'], safe='/')}?download=true"
                        )
                        for attempt in range(1, 5):
                            try:
                                offset = partial.stat().st_size if partial.exists() else 0
                                if offset > item["size"]:
                                    raise ValueError("Partial file is larger than expected")
                                if offset == item["size"]:
                                    break
                                update("downloads", attempt=attempt)
                                headers = {"Range": f"bytes={offset}-"} if offset else {}
                                with client.stream("GET", url, headers=headers) as response:
                                    response.raise_for_status()
                                    if offset and response.status_code == 200:
                                        offset = 0
                                    elif offset and not response.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                                        raise ValueError("Server returned an unexpected download range")
                                    received = offset
                                    started, last = time.monotonic(), 0.0
                                    with partial.open("ab" if offset else "wb") as stream:
                                        for block in response.iter_bytes(1024 * 1024):
                                            stopped()
                                            stream.write(block)
                                            received += len(block)
                                            if received > item["size"]:
                                                raise ValueError("Downloaded file is larger than the pinned size")
                                            if time.monotonic() - last > 0.5:
                                                update("downloads", done_bytes=complete_bytes + received,
                                                       file_bytes=received,
                                                       bytes_per_second=(received-offset) / max(time.monotonic()-started, 0.01))
                                                last = time.monotonic()
                                if received != item["size"]:
                                    raise OSError("Download ended before the expected file size")
                                break
                            except (httpx.HTTPError, OSError):
                                if attempt == 4:
                                    raise
                                time.sleep(attempt * 2)
                    update("downloads", phase="verify", bytes_per_second=0)
                    actual = verify(destination if destination.exists() else partial, item,
                                    lambda done, size: update("downloads", verify_bytes=done, verify_total=size))
                    if not destination.exists():
                        partial.replace(destination)
                    complete_bytes += item["size"]
                    count += 1
                    receipts.append({"model": model["id"], "file": str(destination.relative_to(ROOT)),
                                     "size": item["size"], **actual})
                    (RUN / "download-receipts.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
                    update("downloads", done_bytes=complete_bytes, done_files=count, file_bytes=item["size"])
        update("downloads", state="completed", phase="done", finished_at=now())
    except Exception as exc:  # noqa: BLE001 -- surface any background failure in the progress file
        update("downloads", state="stopped" if (RUN / "stop.request").exists() else "failed",
               error=str(exc), finished_at=now())


def kill_own_tree(process: subprocess.Popen) -> None:
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=False)
        else:
            process.terminate()
        process.wait(timeout=30)


def check_llm(model: Path) -> None:
    process = None
    try:
        update("llm", state="running", phase="hash", model=str(model), total_bytes=model.stat().st_size)
        expected = {"size": 18822970304,
                    "sha256": "9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575"}
        actual = verify(model, expected, lambda done, size: update("llm", done_bytes=done, total_bytes=size))
        update("llm", phase="inference", sha256=actual["sha256"])
        command = [str(ROOT / "services/worker/.venv/Scripts/python.exe"), "-I", "-u",
                   str(ROOT / "scripts/m0/llm_smoke.py"), "--model", str(model),
                   "--output-dir", str(RUN / "llm")]
        with (RUN / "llm.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            update("llm", pid=process.pid)
            start = time.monotonic()
            while process.poll() is None:
                stopped()
                if time.monotonic() - start > 1800:
                    raise TimeoutError("LLM verification exceeded 30 minutes")
                update("llm", elapsed_seconds=int(time.monotonic()-start))
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f"LLM verification exited with code {process.returncode}; see llm.log")
        update("llm", state="completed", phase="done", finished_at=now())
    except Exception as exc:  # noqa: BLE001 -- surface any background failure in the progress file
        update("llm", state="stopped" if (RUN / "stop.request").exists() else "failed",
               error=str(exc), finished_at=now())
    finally:
        if process is not None:
            kill_own_tree(process)


def main() -> None:
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    if not RUN.is_relative_to(ROOT / "private/m0"):
        raise ValueError("Run directory must be inside private/m0")
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "status.json").exists():
        raise ValueError("Use a new run directory; downloaded .part files resume automatically")
    STATE.update({"run_id": RUN.name, "pid": os.getpid(), "started_at": now(),
                  "run_dir": str(RUN),
                  "downloads": {"state": "queued", "done_bytes": 0, "done_files": 0},
                  "llm": {"state": "queued"},
                  "next": "画像生成・単体変換・声デザイン→クローン・ティラノ再生は次の検証で実施。外部APIは後で設定。"})
    update("downloads")
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(download_models), executor.submit(check_llm, args.model.resolve())]
        while True:
            with LOCK:
                d, l = dict(STATE["downloads"]), dict(STATE["llm"])
            print(f"[{now()}] download={d['state']} "
                  f"{d.get('done_bytes', 0)/1e9:.2f}/{d.get('total_bytes', 0)/1e9:.2f} GB "
                  f"files={d.get('done_files', 0)}/{d.get('total_files', 0)} "
                  f"{d.get('model', '')}/{d.get('file', '')} ({d.get('phase', '')}) "
                  f"| LLM={l['state']} {l.get('phase', '')} "
                  f"{l.get('done_bytes', 0)/1e9:.2f}/{l.get('total_bytes', 0)/1e9:.2f} GB",
                  flush=True)
            if all(job.done() for job in jobs):
                for job in jobs:
                    job.result()
                break
            time.sleep(5)
    print(json.dumps(STATE, ensure_ascii=False, indent=2), flush=True)
    if any(STATE[name]["state"] != "completed" for name in ("downloads", "llm")):
        sys.exit(1)


if __name__ == "__main__":
    main()
