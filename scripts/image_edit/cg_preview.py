"""Play a published chapter in the Tyrano player with one experimental CG swapped in.

The chapter is read through the coordinator's existing player route and never
modified. Only the script, scenario, stage data and the CG are served from here;
voices, backgrounds, portraits and the engine are relayed unchanged.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

ASSET_ID = "cg_preview"
SEGMENT_ID = "cg_preview_segment"
MAX_SESSIONS = 8
# The published player switches to and from a CG with an immediate cut.
TRANSITIONS = ("cut", "dissolve", "fade")


class PreviewError(ValueError):
    pass


def player_url(server: str, build_id: str) -> str:
    parsed = urlparse(server)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PreviewError("作品サーバーのURLが正しくありません。")
    return f"{server.rstrip('/')}/player/{quote(build_id, safe='')}/"


def fetch(url: str, timeout: float = 30, headers: dict | None = None):
    """Return (status, headers, body) from the coordinator; a missing file is not an error."""
    try:
        with urlopen(Request(url, headers=headers or {}), timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()
    except (URLError, OSError) as exc:
        raise PreviewError(f"作品サーバーに接続できません: {exc}") from exc


def preview_interval(record: dict) -> tuple[str, str]:
    """First and last displayed utterance IDs of the proposal that produced this image."""
    context = record.get("context") or {}
    proposal = (context.get("cg_proposal") or {}).get("proposal") or {}
    identifiers = [row.get("id") for row in context.get("utterances") or [] if isinstance(row, dict)]
    first, last = proposal.get("display_from"), proposal.get("display_to")
    if (type(first) is not int or type(last) is not int
            or not 1 <= first <= last <= len(identifiers)):
        raise PreviewError("この結果には、CGを表示する発話の範囲が記録されていません。")
    return identifiers[first - 1], identifiers[last - 1]


def build_files(record: dict, image: bytes, read, transition=("cut", 0)) -> dict[str, bytes]:
    """Recompile the published script with the CG interval added; nothing is published.

    `read(path)` returns one file of the published chapter, or None when it is absent.
    `transition` is the effect and its milliseconds for entering, leaving and changing a CG.
    """
    visual, duration = transition
    if visual not in TRANSITIONS or type(duration) is not int or not 0 <= duration <= 3000:
        raise PreviewError("CGの切り替えの効果と時間（0〜3000ミリ秒）が正しくありません。")
    from packages.contracts import Script
    from packages.tyrano_export.compiler import ASSET_FOLDERS, canonical_json, player_overlay

    context = record.get("context") or {}
    first, last = preview_interval(record)
    scene_ids = [row["id"] for row in context["utterances"] if isinstance(row, dict)]
    published = read("script.json")
    if published is None:
        raise PreviewError("公開版の台本を取得できません。章が公開済みか確認してください。")
    value = json.loads(published)
    order = [row["id"] for row in value["utterances"]]
    positions = {identifier: index for index, identifier in enumerate(order)}
    if any(identifier not in positions for identifier in scene_ids):
        raise PreviewError("この画像を作った場面の本文が、現在の公開版と一致しません。"
                           "同じ制作版を選んでいるか確認してください。")
    start, end = positions[first], positions[last] + 1
    kept = []
    for segment in value.get("event_cg_segments") or []:
        begin = positions[segment["start_utterance_id"]]
        finish = (positions[segment["end_utterance_id"]]
                  if segment.get("end_utterance_id") else len(order))
        # The experimental CG replaces an adopted one only where they would overlap.
        if finish <= start or begin >= end:
            kept.append(segment)
    used = {asset["id"] for segment in kept
            for asset in ({"id": segment["base_asset_id"]},
                          *({"id": row["asset_id"]} for row in segment.get("variants") or []))}
    value["assets"] = [asset for asset in value["assets"]
                       if asset["kind"] != "event_cg" or asset["id"] in used]
    if any(asset["id"] == ASSET_ID or asset["filename"].casefold() == ASSET_ID + ".png"
           for asset in value["assets"]):
        raise PreviewError("公開版に実験用と同じ名前の素材があります。")
    value["assets"].append({"id": ASSET_ID, "kind": "event_cg", "artifact_id": ASSET_ID,
                            "filename": ASSET_ID + ".png",
                            "sha256": hashlib.sha256(image).hexdigest()})
    kept.append({"id": SEGMENT_ID, "start_utterance_id": first,
                 "end_utterance_id": order[end] if end < len(order) else None,
                 "base_asset_id": ASSET_ID})
    value["event_cg_segments"] = sorted(kept, key=lambda row: positions[row["start_utterance_id"]])
    script = Script.model_validate(value)
    images = {}
    for asset in script.assets:
        if asset.kind in {"background", "character"}:
            content = read(f"data/{ASSET_FOLDERS[asset.kind]}/{asset.filename}")
            if content is None:
                raise PreviewError(f"公開版の画像を取得できません: {asset.filename}")
            images[asset.id] = content
    script_bytes = canonical_json(script.model_dump(mode="json"))
    files = player_overlay(script, images, script_bytes)
    files["script.json"] = script_bytes
    if visual != "cut" and duration:
        # CG lines carry their own stage anchors. The player animates one only when the
        # picture actually changes, so lines inside an interval still pass instantly.
        stages = json.loads(files["data/others/auto_drama_stages.json"])
        for anchor in stages["scenes"]:
            if anchor["id"].startswith("cg:"):
                anchor.update(visual=visual, duration_ms=duration)
        files["data/others/auto_drama_stages.json"] = canonical_json(stages)
    files[f"data/{ASSET_FOLDERS['event_cg']}/{ASSET_ID}.png"] = image
    scene_start = positions[scene_ids[0]]
    if scene_start:
        # The player has no start-at-line entry. Its own "resume" does: record the scene's
        # first line as this page's resume point. Storage belongs to this preview origin.
        key = json.dumps(f"adn_auto_v1:{script.id}").replace("<", "\\u003c")
        saved = json.dumps(json.dumps({"id": scene_ids[0]})).replace("<", "\\u003c")
        seed = f"<head><script>try{{localStorage.setItem({key},{saved})}}catch(_){{}}</script>"
        files["index.html"] = files["index.html"].replace(b"<head>", seed.encode(), 1)
    return files


class PreviewServer:
    """Loopback-only pages; each preview keeps its own files and relays the rest."""

    def __init__(self):
        self.sessions: dict[str, dict] = {}
        self._server = None
        self._lock = threading.Lock()

    def add(self, upstream: str, files: dict[str, bytes]) -> str:
        with self._lock:
            if self._server is None:
                self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
                self._server.daemon_threads = True
                threading.Thread(target=self._server.serve_forever, daemon=True).start()
            while len(self.sessions) >= MAX_SESSIONS:
                self.sessions.pop(next(iter(self.sessions)))
            token = uuid.uuid4().hex
            self.sessions[token] = {"upstream": upstream, "files": files}
            return f"http://127.0.0.1:{self._server.server_port}/{token}/"

    def close(self):
        with self._lock:
            server, self._server = self._server, None
            self.sessions.clear()
        if server is not None:
            server.shutdown()
            server.server_close()

    def _handler(self):
        sessions = self.sessions

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                self._respond(True)

            def do_HEAD(self):
                self._respond(False)

            def _respond(self, body):
                token, _, relative = unquote(urlparse(self.path).path).lstrip("/").partition("/")
                session = sessions.get(token)
                relative = relative or "index.html"
                if session is None or "\\" in relative or any(
                        part in {"", ".", ".."} for part in relative.split("/")):
                    self._send(404, {}, b"", body)
                    return
                content = session["files"].get(relative)
                if content is not None:
                    kind = {".ks": "text/plain", ".tjs": "text/plain", ".js": "text/javascript"}.get(
                        relative[relative.rfind("."):].lower(), mimetypes.guess_type(relative)[0])
                    self._send(200, {"Content-Type": kind or "application/octet-stream",
                                     "Cache-Control": "no-store"}, content, body)
                    return
                forwarded = {"Range": self.headers["Range"]} if self.headers.get("Range") else {}
                try:
                    status, headers, content = fetch(
                        session["upstream"] + quote(relative), headers=forwarded)
                except PreviewError:
                    self._send(502, {}, b"", body)
                    return
                kept = {key: headers[key] for key in ("Content-Type", "Content-Range", "Accept-Ranges")
                        if key in headers}
                self._send(status, {**kept, "Cache-Control": "no-store"}, content, body)

            def _send(self, status, headers, content, body):
                try:
                    self.send_response(status)
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    if body:
                        self.wfile.write(content)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    pass

            def log_message(self, *_args):
                pass

        return Handler


def prepare(record: dict, image_path, server: str,
            transition=("cut", 0)) -> tuple[str, dict[str, bytes]]:
    """Network work for a background thread: returns the upstream URL and the local files."""
    from pathlib import Path

    build_id = (record.get("context") or {}).get("published_build_id")
    if not isinstance(build_id, str) or not build_id:
        raise PreviewError("この場面の章は公開されていないため、再生できません。")
    upstream = player_url(server, build_id)

    def read(path):
        status, _, content = fetch(upstream + quote(path))
        if status == 404:
            return None
        if status != 200:
            raise PreviewError(f"作品サーバーが {path} にHTTP {status}を返しました。")
        return content

    return upstream, build_files(record, Path(image_path).read_bytes(), read, transition)
