"""Serve one generated M0 scene and the installed Tyrano engine on loopback only."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
ENGINE_EXTENSIONS = {".js", ".css", ".html", ".json", ".png", ".jpg", ".jpeg", ".gif",
                     ".svg", ".webp", ".ico", ".ttf", ".otf", ".woff", ".woff2",
                     ".wav", ".ogg", ".mp3", ".mp4", ".webm"}


def clean_path(request_path: str) -> str:
    url = urlsplit(request_path)
    if url.scheme or url.netloc:
        raise ValueError("Only relative request paths are supported.")
    decoded = unquote(url.path, errors="strict")
    if not decoded.startswith("/") or any(char in decoded for char in ("\\", "\x00", ":")):
        raise ValueError("Invalid request path.")
    parts = decoded[1:].split("/")
    if any(part in (".", "..") or part.startswith(".") for part in parts):
        raise ValueError("Hidden paths and traversal are not permitted.")
    return "index.html" if decoded == "/" else decoded[1:]


class SceneRoutes:
    def __init__(self, scene: Path, engine: Path) -> None:
        self.scene = scene.resolve(strict=True)
        self.engine = engine.resolve(strict=True)
        if not self.scene.is_relative_to((ROOT / "private").resolve()):
            raise ValueError("Only a generated scene under project/private can be served.")
        manifest = json.loads((self.scene / "scene-manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "built" or manifest.get("engine_included") is not False:
            raise ValueError("Expected a completed local scene manifest.")
        self.allowed = set(manifest["files"])
        for path in self.allowed:
            if clean_path("/" + path) != path or PurePosixPath(path).is_absolute():
                raise ValueError("Invalid manifest route.")

    def resolve(self, request_path: str) -> Path:
        relative = clean_path(request_path)
        if relative.startswith("tyrano/"):
            root, child = self.engine, relative[len("tyrano/"):]
            if Path(child).suffix.lower() not in ENGINE_EXTENSIONS:
                raise FileNotFoundError("Unsupported engine resource.")
        elif relative in self.allowed:
            root, child = self.scene, relative
        else:
            raise FileNotFoundError("This file is not part of the published scene.")
        target = (root / child).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise FileNotFoundError("The resource is outside the published directory.")
        return target


def byte_range(value: str | None, size: int) -> tuple[int, int, bool]:
    if value is None:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()):
        raise ValueError("Only one byte range is supported.")
    first, last = match.groups()
    if first:
        start, end = int(first), int(last) if last else size - 1
    else:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("Invalid suffix range.")
        start, end = max(0, size - suffix), size - 1
    if start >= size or start > end:
        raise ValueError("Range is not satisfiable.")
    return start, min(end, size - 1), True


def make_handler(routes: SceneRoutes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AutoDramaM0/1"

        def do_HEAD(self) -> None:
            self.deliver(head=True)

        def do_GET(self) -> None:
            self.deliver(head=False)

        def deliver(self, head: bool) -> None:
            port = self.server.server_port
            if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self.send_error(403, "Loopback host required")
                return
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                self.send_error(403, "Cross-origin requests are not allowed")
                return
            try:
                target = routes.resolve(self.path)
            except (OSError, ValueError, UnicodeError):
                self.send_error(404)
                return
            try:
                with target.open("rb") as stream:
                    size = target.stat().st_size
                    try:
                        start, end, partial = byte_range(self.headers.get("Range"), size)
                    except ValueError:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    mime = {".ks": "text/plain; charset=utf-8", ".tjs": "text/plain; charset=utf-8",
                            ".js": "text/javascript; charset=utf-8", ".wav": "audio/wav"}.get(
                                target.suffix.lower(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                    self.send_response(206 if partial else 200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(max(0, end - start + 1)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                    if partial:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    if not head:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining > 0:
                            block = stream.read(min(65536, remaining))
                            if not block:
                                break
                            self.wfile.write(block)
                            remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    routes = SceneRoutes(args.scene_dir, ROOT / "tyranoscript/tyrano")
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(routes)) as server:
        print(f"M0 scene: http://127.0.0.1:{args.port}/", flush=True)
        print("Scene assets and /tyrano/ only; Ctrl+C stops the server.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
