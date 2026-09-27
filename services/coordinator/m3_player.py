"""Serve immutable chapter ZIPs over a separately installed, read-only Tyrano engine."""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response

from packages.tyrano_export.player import ENGINE_SCRIPTS, ENGINE_STYLES

from .service import Coordinator, ServiceError

ROOT = Path(__file__).resolve().parents[2]
ENGINE_EXTENSIONS = {
    ".js", ".css", ".html", ".json", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".ico", ".ttf", ".otf", ".woff", ".woff2", ".wav", ".ogg", ".mp3", ".mp4", ".webm",
}


def clean_path(path: str) -> str:
    """Paths have already been URL-decoded once by ASGI; reject residual escapes."""
    if not path or any(char in path for char in ("\\", ":", "%", "?", "#")):
        raise ServiceError(404, "公開されたファイルではありません。")
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise ServiceError(404, "公開されたファイルではありません。")
    if any(not part or part.startswith(".") for part in path.split("/")):
        raise ServiceError(404, "公開されたファイルではありません。")
    return path


class InstalledEngine:
    def __init__(self, directory: Path | str | None = None):
        root = Path(directory or os.environ.get("AUTO_DRAMA_TYRANO_DIR", ROOT / "tyranoscript"))
        self.root = (root if root.is_absolute() else ROOT / root).resolve()

    def resolve(self, relative: str) -> Path:
        relative = clean_path(relative)
        if not relative.startswith("tyrano/") or Path(relative).suffix.lower() not in ENGINE_EXTENSIONS:
            raise ServiceError(404, "公開されたエンジンファイルではありません。")
        target = (self.root / relative).resolve()
        # Do not follow a symlink or junction out of the engine's own directory.
        if not target.is_relative_to(self.root / "tyrano") or not target.is_file():
            raise ServiceError(404, "エンジンファイルが見つかりません。")
        return target

    def validate(self) -> None:
        try:
            source = self.resolve("tyrano/plugins/kag/kag.js").read_text("utf-8-sig")
            if not re.search(r"\bversion\s*:\s*520\b", source):
                raise ServiceError(503, "このプレイヤーはティラノスクリプト V520 に対応しています。")
            for path in (*ENGINE_SCRIPTS, *ENGINE_STYLES):
                self.resolve("tyrano/" + path)
        except (OSError, ServiceError) as exc:
            if isinstance(exc, ServiceError) and exc.status == 503:
                raise
            raise ServiceError(
                503,
                "ティラノスクリプト V520 の必要ファイルが見つかりません。制御PCの "
                "AUTO_DRAMA_TYRANO_DIR に本体のフォルダーを設定して再起動してください。"
                "生成済みの章は書き出せます。",
            ) from exc


def bundle_member(data: bytes, relative: str) -> bytes:
    """Read a manifest-declared member, never extract a ZIP into the filesystem."""
    clean_path(relative)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise ValueError("duplicate member")
            for entry in entries:
                clean_path(entry.filename)
                if entry.compress_type != zipfile.ZIP_STORED or entry.file_size > len(data):
                    raise ValueError("unsupported ZIP member")
            manifest = json.loads(archive.read("manifest.json"))
            if (
                manifest.get("engine_included") is not False
                or manifest.get("format") != "auto-drama.tyrano-source"
                or set(names) != set(manifest["files"]) | {"manifest.json"}
                or any(name.startswith("tyrano/") for name in names)
            ):
                raise ValueError("invalid manifest")
            if relative not in names:
                raise ServiceError(404, "このファイルは公開章に含まれません。")
            content = archive.read(relative)
            if relative != "manifest.json":
                expected = manifest["files"][relative]
                if expected != {
                    "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()
                }:
                    raise ValueError("invalid member hash")
            return content
    except ServiceError:
        raise
    except (ValueError, TypeError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ServiceError(503, "公開章の整合性を確認できません。保存済みの章を確認してください。") from exc


def byte_range(value: str | None, size: int) -> tuple[int, int, bool]:
    if value is None:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()):
        raise ValueError("Only one range is supported")
    first, last = match.groups()
    if first:
        start, end = int(first), int(last) if last else size - 1
    else:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("Invalid suffix")
        start, end = max(0, size - suffix), size - 1
    if start >= size or start > end:
        raise ValueError("Unsatisfiable range")
    return start, min(end, size - 1), True


def install_player_routes(
    app: FastAPI,
    coordinator: Callable[[], Coordinator],
    tyrano_dir: Path | str | None = None,
) -> None:
    engine = InstalledEngine(tyrano_dir)

    @app.api_route("/player/{build_id}/{resource:path}", methods=["GET", "HEAD"])
    @app.api_route("/player/{build_id}/", methods=["GET", "HEAD"])
    def player(request: Request, build_id: str, resource: str = "index.html"):
        from .m3_service import M3Service

        service = coordinator()
        build = M3Service(service).build(build_id)
        if build["status"] != "published":
            raise ServiceError(404, "この章はまだ公開されていません。")
        relative = clean_path(resource)
        is_engine = relative.startswith("tyrano/")
        if relative == "player-context.json":
            # This immutable identity is deliberately separate from the source
            # ZIP. An extracted ZIP has an explicit static end, while the live
            # player can query only this build's validated successor.
            content = json.dumps({
                "schema_version": 1, "mode": "live", "build_id": build["id"],
                "production_id": build.get("production_id"),
                "storyline_id": build.get("validation", {}).get("storyline_id", build.get("production_id")),
                "project_id": build.get("project_id"),
                "chapter_number": build.get("chapter_number", 1),
                "next_url": f"/api/m3/builds/{build['id']}/next",
            }, ensure_ascii=False, sort_keys=True).encode("utf-8")
        elif is_engine:
            content = engine.resolve(relative).read_bytes()
        else:
            if relative == "index.html":
                engine.validate()
            _, data = service.artifact(build["export_artifact_id"])
            content = bundle_member(data, relative)
        headers = {
            "Accept-Ranges": "bytes", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer", "Cross-Origin-Resource-Policy": "same-origin",
            "Cache-Control": "no-store" if is_engine else "private, max-age=31536000, immutable",
            "ETag": '"' + hashlib.sha256(content).hexdigest() + '"',
        }
        try:
            start, end, partial = byte_range(request.headers.get("range"), len(content))
        except ValueError:
            return Response(status_code=416, headers={
                **headers, "Content-Range": f"bytes */{len(content)}", "Content-Length": "0",
            })
        if partial:
            headers["Content-Range"] = f"bytes {start}-{end}/{len(content)}"
        headers["Content-Length"] = str(max(0, end - start + 1))
        media_type = {
            ".ks": "text/plain", ".tjs": "text/plain", ".js": "text/javascript",
            ".wav": "audio/wav",
        }.get(Path(relative).suffix.lower(), mimetypes.guess_type(relative)[0])
        return Response(
            b"" if request.method == "HEAD" else content[start:end + 1],
            status_code=206 if partial else 200,
            headers=headers,
            media_type=media_type or "application/octet-stream",
        )
