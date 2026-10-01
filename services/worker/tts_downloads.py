"""CPU-only pinned TTS acquisition with durable progress and verified inventory."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import ssl
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from uuid import uuid4

import httpx

from packages.contracts.tts_catalog import catalog, manifest_fingerprint, resolve_bundle

logger = logging.getLogger(__name__)
ACTIVE = {"queued", "downloading", "verifying"}


class DownloadCancelled(Exception):
    pass


class DownloadError(Exception):
    """A safe, static error that may be shown by the coordinator."""

    def __init__(self, message: str, *, cause: Exception | None = None,
                 retryable: bool = False):
        super().__init__(message)
        self.safe_message = message
        self.retryable = retryable
        self.diagnostics = _safe_diagnostics(cause) if cause is not None else "type=DownloadError"


def _exception_chain(error: Exception) -> list[BaseException]:
    """Inspect nested transport errors without exposing their messages or URLs."""
    result, pending, seen = [], [error], set()
    while pending and len(result) < 12:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        pending.extend(cause for cause in (current.__cause__, current.__context__)
                       if cause is not None)
    return result


def _safe_diagnostics(error: Exception) -> str:
    details = []
    for item in _exception_chain(error):
        name = type(item).__name__
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", name):
            name = "Exception"
        values = [f"type={name}"]
        if isinstance(item, httpx.HTTPStatusError):
            values.append(f"http_status={item.response.status_code}")
        for attribute in ("errno", "winerror"):
            value = getattr(item, attribute, None)
            if isinstance(value, int):
                values.append(f"{attribute}={value}")
        details.append(",".join(values))
    return ";".join(dict.fromkeys(details))


def _classified_error(error: Exception, *, downloading: bool = False) -> DownloadError:
    if isinstance(error, DownloadError):
        return error
    chain = _exception_chain(error)
    # Some transport wrappers keep only the Windows error text, so recognize its
    # fixed numeric form as a fallback. The original message is never returned.
    if any(getattr(item, "winerror", None) == 10013
           or getattr(item, "errno", None) == 10013
           or re.search(r"\bWinError\s+10013\b", str(item)) for item in chain):
        return DownloadError("Network access blocked (WinError 10013)", cause=error)
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return DownloadError(f"HTTP {status} from model source", cause=error,
                             retryable=status not in (401, 403, 404))
    if isinstance(error, httpx.TimeoutException):
        return DownloadError("Model download timed out", cause=error, retryable=True)
    if any(isinstance(item, ssl.SSLError) for item in chain):
        return DownloadError("TLS verification failed", cause=error, retryable=True)
    if isinstance(error, PermissionError) or (isinstance(error, OSError)
                                             and error.errno in (errno.EACCES, errno.EPERM)):
        return DownloadError("Local storage permission denied", cause=error)
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return DownloadError("Insufficient free disk space for model download", cause=error)
    if isinstance(error, httpx.HTTPError) or downloading:
        return DownloadError("Network connection failed; retry to resume partial files",
                             cause=error, retryable=True)
    return DownloadError("Model download failed", cause=error)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(file: dict) -> str:
    return hashlib.sha256(json.dumps(file, sort_keys=True).encode()).hexdigest()


def _validated_manifest(manifest: dict, model_id: str, precision: str) -> dict:
    value = copy.deepcopy(manifest)
    if (value.get("model_id"), value.get("precision")) != (model_id, precision):
        raise ValueError("TTS manifest selection mismatch")
    expected = value.pop("manifest_id", None)
    actual = manifest_fingerprint(value)
    if expected != actual:
        raise ValueError("TTS manifest fingerprint mismatch")
    files = value.get("files", [])
    if not files or value.get("total_bytes") != sum(file["size"] for file in files):
        raise ValueError("Invalid TTS manifest size")
    for file in files:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", file["repo_id"]):
            raise ValueError("Invalid TTS source repository")
        if not re.fullmatch(r"[a-f0-9]{40}", file["revision"]):
            raise ValueError("TTS downloads require a pinned source revision")
        relative = PurePosixPath(file["path"])
        directory = PurePosixPath(file["local_dir"])
        if (relative.is_absolute() or ".." in relative.parts or "\\" in file["path"]
                or directory.is_absolute() or ".." in directory.parts or "\\" in file["local_dir"]
                or not file["local_dir"].startswith("services/worker/runtimes/irodori/models/")):
            raise ValueError("Invalid TTS destination")
        digest = file.get("sha256") or file.get("git_blob_sha1")
        length = 64 if file.get("sha256") else 40
        if (not isinstance(file["size"], int) or file["size"] < 1
                or not re.fullmatch(f"[a-f0-9]{{{length}}}", digest or "")):
            raise ValueError("Invalid TTS file integrity metadata")
    value["manifest_id"] = expected
    return value


class DownloadManager:
    """Own background downloads while the worker continues its normal poll loop.

    Only fingerprinted file descriptions are used: destinations are restricted
    to worker model storage and URLs are constructed from pinned Hugging Face refs.
    This class does not import torch, acquire a GPU, or change synthesis settings.
    """

    def __init__(self, root: Path, report_callback: Callable[[dict], None] | None = None,
                 *, client_factory: Callable | None = None,
                 bundle_resolver: Callable = resolve_bundle, scan_existing: bool = True,
                 reserve_bytes: int = 256 * 1024**2):
        self.root = Path(root).resolve()
        self.directory = self.root / "services/worker/runtimes/irodori/models/.tts-downloads"
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "operations").mkdir(exist_ok=True)
        (self.directory / "locks").mkdir(exist_ok=True)
        self.callback = report_callback
        self.resolve = bundle_resolver
        self.reserve_bytes = reserve_bytes
        self.client_factory = client_factory or (lambda: httpx.Client(
            follow_redirects=True, timeout=httpx.Timeout(60, connect=30), trust_env=False,
            headers={"Accept-Encoding": "identity", "User-Agent": "auto-drama-tts/0.1"}))
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._operations: dict[str, dict] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        try:
            self._receipts = json.loads((self.directory / "verified-files.json").read_text())
        except (FileNotFoundError, ValueError):
            self._receipts = {}
        for path in (self.directory / "operations").glob("*.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                if state.get("status") in ACTIVE:
                    state.update(status="cancelled", phase="interrupted", error=None)
                self._operations[state["operation_id"]] = state
            except (ValueError, KeyError):
                logger.warning("Ignoring unreadable TTS operation record")
        self._scanner = None
        if scan_existing:
            self._scanner = threading.Thread(target=self._scan_existing,
                                             name="tts-inventory", daemon=True)
            self._scanner.start()

    def start(self, operation: dict) -> dict:
        operation_id = operation.get("operation_id", operation.get("id"))
        if not isinstance(operation_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", operation_id):
            raise ValueError("Invalid TTS download operation ID")
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("TTS download manager is closed")
            if operation_id in self._threads and self._threads[operation_id].is_alive():
                return self._public(self._operations[operation_id])
            previous = self._operations.get(operation_id)
            manifest = operation.get("manifest") or (previous or {}).get("manifest")
            manifest = manifest or self.resolve(operation["model_id"], operation["precision"])
            manifest = _validated_manifest(manifest, operation["model_id"], operation["precision"])
            if previous and previous["manifest"]["manifest_id"] != manifest["manifest_id"]:
                raise ValueError("A resumed TTS operation cannot change its pinned manifest")
            state = {
                "operation_id": operation_id, "model_id": operation["model_id"],
                "precision": operation["precision"], "manifest": manifest,
                "manifest_id": manifest["manifest_id"], "status": "queued", "phase": "queued",
                "completed_bytes": 0, "total_bytes": manifest["total_bytes"],
                "current_file": None, "error": None, "prepared_only": True, "updated_at": _now(),
            }
            self._operations[operation_id] = state
            self._cancel[operation_id] = threading.Event()
            self._save(state)
            thread = threading.Thread(target=self._run, args=(operation_id,),
                                      name=f"tts-download-{operation_id[:12]}", daemon=True)
            self._threads[operation_id] = thread
            thread.start()
            return self._public(state)

    def cancel(self, operation_id: str) -> None:
        with self._lock:
            if operation_id in self._cancel:
                self._cancel[operation_id].set()

    def snapshots(self) -> list[dict]:
        with self._lock:
            return [self._public(state) for state in self._operations.values()]

    def inventory(self) -> list[dict]:
        result = []
        for model in catalog():
            for precision in model["precisions"]:
                bundle = self.resolve(model["model_id"], precision)
                if all(self._verified(file) for file in bundle["files"]):
                    result.append({"model_id": model["model_id"], "precision": precision,
                                   "manifest_id": bundle["manifest_id"],
                                   "file_download_ready": True, "prepared_only": True})
        return result

    def close(self, wait: bool = False) -> None:
        self._closed.set()
        with self._lock:
            threads = list(self._threads.values()) + ([self._scanner] if self._scanner else [])
            for event in self._cancel.values():
                event.set()
        if wait:
            for thread in threads:
                thread.join(timeout=65)

    def _public(self, state: dict) -> dict:
        value = copy.deepcopy({key: value for key, value in state.items() if key != "manifest"})
        value["downloaded_bytes"] = value["completed_bytes"]
        value["done_bytes"] = value["completed_bytes"]
        return value

    def _save(self, state: dict) -> None:
        _atomic_json(self.directory / "operations" / f"{state['operation_id']}.json", state)

    def _update(self, operation_id: str, **values) -> None:
        with self._lock:
            state = self._operations[operation_id]
            state.update(values, updated_at=_now())
            self._save(state)
            public = self._public(state)
        if self.callback:
            try:
                self.callback(public)
            except Exception:  # noqa: BLE001 -- reporting failures cannot discard downloaded bytes
                logger.warning("TTS progress callback failed", exc_info=False)

    def _check(self, operation_id: str | None) -> None:
        if self._closed.is_set() or (operation_id and self._cancel[operation_id].is_set()):
            raise DownloadCancelled()

    def _path(self, file: dict) -> Path:
        base = (self.root / file["local_dir"]).resolve()
        destination = (base / file["path"]).resolve()
        permitted = (self.root / "services/worker/runtimes/irodori/models").resolve()
        if not base.is_relative_to(permitted) or not destination.is_relative_to(base):
            raise DownloadError("Invalid model destination")
        return destination

    def _verified(self, file: dict) -> bool:
        path = self._path(file)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False
        with self._lock:
            receipt = self._receipts.get(str(path.relative_to(self.root)), {})
            return (receipt.get("identity") == _identity(file) and stat.st_size == file["size"]
                    and receipt.get("mtime_ns") == stat.st_mtime_ns)

    def _verify(self, path: Path, file: dict, operation_id: str | None = None) -> None:
        self._check(operation_id)
        if path.stat().st_size != file["size"]:
            raise DownloadError("File size mismatch; existing file was preserved")
        digest = hashlib.sha256() if file.get("sha256") else hashlib.sha1()
        if not file.get("sha256"):
            digest.update(f"blob {file['size']}\0".encode())
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024**2):
                self._check(operation_id)
                digest.update(chunk)
        if digest.hexdigest() != (file.get("sha256") or file["git_blob_sha1"]):
            raise DownloadError("File hash mismatch; existing file was preserved")

    def _receipt(self, file: dict) -> None:
        path = self._path(file)
        with self._lock, self._directory_lock("verified-file-receipts", None):
            receipt_path = self.directory / "verified-files.json"
            try:
                receipts = json.loads(receipt_path.read_text())
            except (FileNotFoundError, ValueError):
                receipts = {}
            receipts[str(path.relative_to(self.root))] = {
                "identity": _identity(file), "size": file["size"],
                "mtime_ns": path.stat().st_mtime_ns, "verified_at": _now(),
            }
            _atomic_json(receipt_path, receipts)
            self._receipts = receipts

    @contextmanager
    def _directory_lock(self, directory: str, operation_id: str | None):
        path = self.directory / "locks" / (hashlib.sha256(directory.encode()).hexdigest() + ".lock")
        with path.open("a+b") as stream:
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            while True:
                self._check(operation_id)
                try:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    self._closed.wait(0.2)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _scan_existing(self) -> None:
        seen = set()
        try:
            for model in catalog():
                for precision in model["precisions"]:
                    for file in self.resolve(model["model_id"], precision)["files"]:
                        identity = _identity(file)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        self._check(None)
                        if self._path(file).is_file() and not self._verified(file):
                            with self._directory_lock(file["local_dir"], None):
                                try:
                                    self._verify(self._path(file), file)
                                    self._receipt(file)
                                except DownloadError:
                                    logger.warning("Existing TTS file failed integrity verification")
        except DownloadCancelled:
            return
        except Exception:  # noqa: BLE001 -- inventory scan must not kill the worker
            logger.warning("TTS inventory scan failed", exc_info=False)

    def _download(self, client: httpx.Client, file: dict, operation_id: str,
                  complete: int) -> None:
        destination = self._path(file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        # Resolve partials as well to reject a preexisting symlink escaping storage.
        if not partial.resolve().is_relative_to(destination.parent.resolve()):
            raise DownloadError("Invalid partial-file destination")
        url = (f"https://huggingface.co/{file['repo_id']}/resolve/{file['revision']}/"
               f"{quote(file['path'], safe='/')}?download=true")
        for attempt in range(4):
            self._check(operation_id)
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > file["size"]:
                partial.replace(partial.with_name(partial.name + f".corrupt-{time.time_ns()}"))
                raise DownloadError("Partial file exceeds the pinned file size")
            if offset == file["size"]:
                return
            try:
                with client.stream("GET", url, headers={"Range": f"bytes={offset}-"}
                                   if offset else {}) as response:
                    response.raise_for_status()
                    if response.status_code == 200:
                        offset = 0
                    elif response.status_code == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                             response.headers.get("content-range", ""))
                        if (not match or int(match[1]) != offset
                                or int(match[3]) != file["size"]
                                or int(match[2]) != file["size"] - 1):
                            raise DownloadError("Server returned an unexpected download range")
                    else:
                        raise DownloadError("Server returned an unexpected download response")
                    received, last = offset, 0.0
                    with partial.open("ab" if offset else "wb") as stream:
                        for chunk in response.iter_bytes(1024**2):
                            self._check(operation_id)
                            if received + len(chunk) > file["size"]:
                                raise DownloadError("Downloaded file exceeds the pinned size")
                            stream.write(chunk)
                            received += len(chunk)
                            if time.monotonic() - last > 0.5:
                                self._update(operation_id, completed_bytes=complete + received)
                                last = time.monotonic()
                    if received != file["size"]:
                        raise OSError("Incomplete model download")
                return
            except (httpx.HTTPError, OSError) as error:
                failure = _classified_error(error, downloading=True)
                if not failure.retryable or attempt == 3:
                    raise failure from None
                if self._cancel[operation_id].wait(2 * (attempt + 1)):
                    raise DownloadCancelled()

    def _run(self, operation_id: str) -> None:
        bundle = self._operations[operation_id]["manifest"]
        try:
            with ExitStack() as stack:
                for directory in sorted({file["local_dir"] for file in bundle["files"]}):
                    stack.enter_context(self._directory_lock(directory, operation_id))
                remaining = 0
                for file in bundle["files"]:
                    destination = self._path(file)
                    if not destination.exists():
                        partial = destination.with_name(destination.name + ".part")
                        remaining += max(0, file["size"] - (partial.stat().st_size
                                                          if partial.exists() else 0))
                if shutil.disk_usage(self.root).free < remaining + self.reserve_bytes:
                    raise DownloadError("Insufficient free disk space for model download")
                complete = 0
                with self.client_factory() as client:
                    for file in bundle["files"]:
                        self._check(operation_id)
                        destination = self._path(file)
                        self._update(operation_id, current_file=file["path"],
                                     status="downloading", phase="download")
                        if not destination.exists():
                            self._download(client, file, operation_id, complete)
                        self._update(operation_id, status="verifying", phase="verify")
                        partial = destination.with_name(destination.name + ".part")
                        if not self._verified(file):
                            try:
                                self._verify(destination if destination.exists() else partial,
                                             file, operation_id)
                            except DownloadError:
                                corrupt = destination if destination.exists() else partial
                                if corrupt.exists():
                                    corrupt.replace(corrupt.with_name(
                                        corrupt.name + f".corrupt-{time.time_ns()}"))
                                raise
                            self._check(operation_id)
                            if not destination.exists():
                                partial.replace(destination)
                            self._receipt(file)
                        complete += file["size"]
                        self._update(operation_id, completed_bytes=complete)
            self._update(operation_id, status="completed", phase="done", current_file=None,
                         completed_bytes=bundle["total_bytes"], error=None)
        except DownloadCancelled:
            self._update(operation_id, status="cancelled", phase="cancelled", error=None)
        except Exception as error:  # noqa: BLE001 -- record every background failure durably
            failure = _classified_error(error)
            logger.error("TTS download failed operation_id=%s current_file=%s reason=%s %s",
                         operation_id, self._operations[operation_id].get("current_file"),
                         failure.safe_message, failure.diagnostics)
            self._update(operation_id, status="failed", phase="failed", error=failure.safe_message)
