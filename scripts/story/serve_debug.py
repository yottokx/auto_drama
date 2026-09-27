"""Serve an exported debug player and the installed engine on loopback only."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.m0.serve_scene import clean_path as clean_request_path
from scripts.m0.serve_scene import make_handler
from services.coordinator.m3_player import InstalledEngine, clean_path
from services.coordinator.service import ServiceError


class DebugRoutes:
    """Read only manifest-declared export files and the separate engine tree."""

    def __init__(self, directory: Path, engine_dir: Path) -> None:
        self.directory = directory.resolve(strict=True)
        self.engine = InstalledEngine(engine_dir)
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("format") != "auto-drama.tyrano-source"
                or manifest.get("engine_included") is not False
                or not isinstance(manifest.get("files"), dict)
                or "index.html" not in manifest["files"]):
            raise ValueError("Expected an extracted compile_bundle player with its manifest.")
        self.allowed = set(manifest["files"]) | {"manifest.json"}
        try:
            for relative in self.allowed:
                clean_path(relative)
                if relative.startswith("tyrano/"):
                    raise ValueError("The debug export must not contain engine files.")
        except ServiceError as exc:
            raise ValueError("The export manifest contains an invalid file path.") from exc

    def resolve(self, request_path: str) -> Path:
        try:
            # Decode once, then use the shared player rule to reject residual
            # escapes, hidden names, separators, ADS syntax and control bytes.
            relative = clean_path(clean_request_path(request_path))
            if relative.startswith("tyrano/"):
                return self.engine.resolve(relative)
            if relative not in self.allowed:
                raise FileNotFoundError("The file is not part of this debug export.")
            target = (self.directory / relative).resolve(strict=True)
            if not target.is_relative_to(self.directory) or not target.is_file():
                raise FileNotFoundError("The file is outside this debug export.")
            return target
        except ServiceError as exc:
            raise FileNotFoundError(exc.detail) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True,
                        help="The extracted player/ directory produced by the debug export")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--engine-dir", type=Path, default=ROOT / "tyranoscript")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    try:
        routes = DebugRoutes(args.directory, args.engine_dir)
        routes.engine.validate()
    except (OSError, ValueError, ServiceError) as exc:
        parser.error(str(exc))
    handler = make_handler(routes)
    handler.server_version = "AutoDramaDebug/1"
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Debug player: http://127.0.0.1:{args.port}/", flush=True)
        print("Read-only export and /tyrano/ engine files; Ctrl+C stops the server.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
