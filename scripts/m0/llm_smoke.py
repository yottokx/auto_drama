"""Run bounded local Gemma M0 checks and stop only the llama-server started here.

No model download, copy, or whole-file hash is performed. The parent workflow can
verify the publisher SHA256 before invoking this script. All HTTP stays on loopback.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class WindowsChildJob:
    """Make Windows close the child if the Python supervisor is forcibly terminated."""

    def __init__(self) -> None:
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
        ]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process: subprocess.Popen) -> None:
        if self.handle and not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class SmokeRun:
    def __init__(self, config: dict, output: Path, deadline: float) -> None:
        self.config = config
        self.output = output
        self.deadline = deadline
        # An empty ProxyHandler avoids accidentally sending local requests to a proxy.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.base_url = ""
        self.results: dict[str, dict] = {}

    def event(self, stage: str, status: str, **detail: object) -> None:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "stage": stage, "status": status, **detail}
        print(json.dumps(record, ensure_ascii=False), flush=True)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(self.output / "status.json", record)

    def request(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The total M0 LLM time limit was reached.")
        request = urllib.request.Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        limit = timeout or self.config["server"]["request_timeout_seconds"]
        try:
            with self.http.open(request, timeout=min(remaining, limit)) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} at {path}: {body[:4000]}") from exc

    def chat(self, name: str, messages: list[dict], **extra: object) -> dict:
        payload = {"model": "m0-local-gemma", "messages": messages,
                   **self.config["generation"], **extra}
        write_json(self.output / f"{name}.request.json", payload)
        response = self.request("/v1/chat/completions", payload)
        write_json(self.output / f"{name}.response.json", response)
        choices = response.get("choices", [])
        if len(choices) != 1:
            raise ValueError("Expected exactly one completion choice.")
        if choices[0].get("finish_reason") == "length":
            raise ValueError("The completion was truncated by its token limit.")
        return choices[0]["message"]

    @staticmethod
    def content(message: dict) -> str:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("The assistant returned empty text.")
        if message.get("tool_calls"):
            raise ValueError("An unexpected tool call appeared in a plain text response.")
        return content.strip()

    def narrative(self) -> None:
        message = self.chat("narrative", [{"role": "user", "content":
            "夜の灯台でミナがユウを引き止める場面を、日本語の短い地の文と台詞で"
            "120字程度、4行以内で書いてください。両名の名前を出し、実際の会話と反応を"
            "描いてください。JSON、箇条書き、見出しは使わず、本文だけを返してください。"}])
        content = self.content(message)
        if not all(name in content for name in ("ミナ", "ユウ")) or len(content) < 40:
            raise ValueError("The narrative is too short or omits a requested character.")
        if content.startswith(("{", "[", "```")):
            raise ValueError("Expected plain narrative text, not JSON or a code block.")
        (self.output / "narrative.txt").write_text(content + "\n", encoding="utf-8")

    def structured(self) -> None:
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "location": {"type": "string", "enum": ["灯台"]},
                "characters": {"type": "array", "items": {"type": "string", "enum": ["ミナ", "ユウ"]},
                               "minItems": 2, "maxItems": 2},
                "summary": {"type": "string"},
            },
            "required": ["location", "characters", "summary"],
        }
        message = self.chat("structured", [{"role": "user", "content":
            "ミナが灯台でユウを引き止めた場面のデータをJSONで返してください。"
            "locationは灯台、charactersはミナとユウの2名、summaryは日本語の短い一文です。"}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "m0_scene", "strict": True, "schema": schema}})
        value = json.loads(self.content(message))
        # Check all constraints of this small, fixed schema without an extra package.
        valid = (isinstance(value, dict)
                 and set(value) == {"location", "characters", "summary"}
                 and value["location"] == "灯台"
                 and isinstance(value["characters"], list)
                 and sorted(value["characters"]) == ["ミナ", "ユウ"]
                 and isinstance(value["summary"], str) and bool(value["summary"].strip()))
        if not valid:
            raise ValueError("Generated JSON did not satisfy the expected scene schema/content.")
        write_json(self.output / "scene.json", value)

    def tools(self) -> None:
        tool = {"type": "function", "function": {
            "name": "random_integer", "description": "Draw one random integer in an inclusive range.",
            "parameters": {"type": "object", "additionalProperties": False,
                           "properties": {"minimum": {"type": "integer"},
                                          "maximum": {"type": "integer"}},
                           "required": ["minimum", "maximum"]}}}
        messages = [{"role": "user", "content":
            "random_integerツールをminimum=1、maximum=6で一度だけ呼び出してください。"
            "ツール結果を受け取ったら、そのvalueを使って「抽選結果はNです。」とだけ答えてください。"
            "値は自分で推測しないでください。"}]
        first = self.chat("tool-call", messages, tools=[tool], tool_choice="required",
                          parallel_tool_calls=False)
        calls = first.get("tool_calls", [])
        if len(calls) != 1:
            raise ValueError("Expected exactly one random_integer tool call.")
        call = calls[0]
        function = call.get("function", {})
        if function.get("name") != "random_integer" or not call.get("id"):
            raise ValueError("Unexpected function name or missing tool call ID.")
        arguments = json.loads(function["arguments"])
        if (arguments != {"minimum": 1, "maximum": 6}
                or any(type(value) is not int for value in arguments.values())):
            raise ValueError("The random tool arguments must be the requested integers 1 and 6.")
        seed = self.config["random_tool_seed"]
        value = random.Random(seed).randint(arguments["minimum"], arguments["maximum"])
        result = {"value": value}
        write_json(self.output / "random-tool-result.json", {
            "tool_call_id": call["id"], "seed": seed, "arguments": arguments, "result": result})
        assistant = {"role": "assistant", "content": first.get("content"), "tool_calls": calls}
        messages.extend([assistant, {"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(result)}])
        final = self.chat("tool-result", messages, tools=[tool], tool_choice="none")
        content = self.content(final)
        if not re.search(rf"(?<!\d){value}(?!\d)", content):
            raise ValueError("The assistant did not use the program-generated random result.")
        (self.output / "tool-answer.txt").write_text(content + "\n", encoding="utf-8")

    def reasoning_none(self) -> None:
        message = self.chat("reasoning-none", [{"role": "user", "content":
            "17+25の答えを、半角数字だけで答えてください。"}], reasoning_effort="none")
        content = self.content(message)
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning or content != "42":
            raise ValueError("reasoning_effort=none did not return the plain expected answer.")
        write_json(self.output / "reasoning-none-check.json", {
            "request_reasoning_effort": "none", "reasoning_output_empty": True,
            "answer": content, "scope": "Observed response behavior; no internal reasoning claim."})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "config/m0-llm.json").read_text(encoding="utf-8"))
    model = args.model.resolve(strict=True)
    if model.stat().st_size != config["model"]["size_bytes"]:
        parser.error("Model file size does not match the configured Gemma GGUF.")
    with model.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            parser.error("The model does not have a GGUF header.")
    server = (ROOT / config["server"]["executable"]).resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "events.jsonl").exists():
        parser.error("Output directory already contains an M0 LLM run; use a new directory.")
    started = time.monotonic()
    run = SmokeRun(config, output, started + config["server"]["total_timeout_seconds"])
    process = None
    job = None
    timer = None
    report = {"status": "failed", "model_path": str(model), "model_size": model.stat().st_size,
              "model_sha256_checked_here": False, "checks": run.results}

    def interrupted(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        write_json(output / "config-snapshot.json", config)
        manifest = json.loads((ROOT / config["server"]["manifest"]).read_text(encoding="utf-8"))
        write_json(output / "llama-runtime-manifest.json", manifest)
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        run.base_url = f"http://127.0.0.1:{port}"
        command = [str(server), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
                   "--ctx-size", str(config["server"]["context_size"]),
                   "--n-gpu-layers", config["server"]["gpu_layers"],
                   "--parallel", str(config["server"]["parallel"]),
                   "--jinja", "--reasoning-format", "deepseek", "--alias", "m0-local-gemma"]
        write_json(output / "server-command.json", command)
        run.event("server", "starting", port=port)
        job = WindowsChildJob()
        # Avoid inheriting global llama-server defaults, especially host/model/tool settings.
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("LLAMA_")}
        with (output / "llama-server.log").open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                command, cwd=server.parent, stdin=subprocess.DEVNULL, stdout=server_log,
                stderr=subprocess.STDOUT, env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            job.assign(process)

            def total_timeout() -> None:
                if process.poll() is None:
                    process.kill()

            timer = threading.Timer(config["server"]["total_timeout_seconds"], total_timeout)
            timer.daemon = True
            timer.start()
            report["server_pid"] = process.pid
            health_deadline = min(run.deadline, started + config["server"]["startup_timeout_seconds"])
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with {process.returncode}; see llama-server.log")
                if time.monotonic() >= health_deadline:
                    raise TimeoutError("llama-server did not become healthy within the startup time limit.")
                try:
                    health = run.request("/health", timeout=2)
                    if health.get("status") == "ok":
                        break
                except (OSError, RuntimeError, TimeoutError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            run.event("server", "ready", pid=process.pid)
            write_json(output / "server-models.json", run.request("/v1/models"))
            write_json(output / "server-props.json", run.request("/props"))
            for name, check in (("plain_narrative", run.narrative), ("json_schema", run.structured),
                                ("random_tool_roundtrip", run.tools), ("reasoning_none", run.reasoning_none)):
                run.event(name, "started")
                check_start = time.monotonic()
                try:
                    check()
                    run.results[name] = {"status": "passed"}
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    run.results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                run.results[name]["elapsed_seconds"] = round(time.monotonic() - check_start, 3)
                run.event(name, **run.results[name])
            report["status"] = "passed" if all(
                check["status"] == "passed" for check in run.results.values()) else "failed"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            subprocess.SubprocessError, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if timer:
            timer.cancel()
        try:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if job:
                job.close()
        report["server_stopped"] = process is None or process.poll() is not None
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(output / "summary.json", report)
        run.event("complete", report["status"], server_stopped=report["server_stopped"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
