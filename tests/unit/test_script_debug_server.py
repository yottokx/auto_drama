"""The debug server exposes export members and engine assets, never parent files."""

import io
import json
import sys

import pytest

from scripts.story import serve_debug


@pytest.fixture
def routes(tmp_path):
    player = tmp_path / "player"
    player.mkdir()
    (player / "index.html").write_bytes(b"player")
    (player / "script.json").write_bytes(b"script")
    (player / "unlisted.txt").write_bytes(b"not public")
    (player / ".env").write_bytes(b"not public")
    (tmp_path / "private.txt").write_bytes(b"not public")
    (player / "manifest.json").write_text(json.dumps({
        "format": "auto-drama.tyrano-source", "engine_included": False,
        "files": {"index.html": {}, "script.json": {}}}), encoding="utf-8")
    engine = tmp_path / "engine"
    (engine / "tyrano").mkdir(parents=True)
    (engine / "tyrano/tyrano.js").write_bytes(b"engine")
    (engine / "tyrano/private.py").write_bytes(b"not public")
    (engine / "data").mkdir()
    (engine / "data/private.ks").write_bytes(b"not public")
    return serve_debug.DebugRoutes(player, engine)


def test_root_and_declared_files_use_export_while_only_tyrano_uses_engine(routes):
    assert routes.resolve("/").read_bytes() == b"player"
    assert routes.resolve("/script.json?cache=1").read_bytes() == b"script"
    assert routes.resolve("/tyrano/tyrano.js").read_bytes() == b"engine"
    assert routes.resolve("/manifest.json").is_file()


@pytest.mark.parametrize("path", [
    "/unlisted.txt", "/.env", "/../private.txt", "/%2e%2e/private.txt",
    "/%252e%252e/private.txt", "/tyrano/../private.txt", "/tyrano/%2e%2e/private.txt",
    "/tyrano/tyrano.js:private", "/tyrano/private.py", "/data/private.ks",
    "/tyrano\\tyrano.js", "/tyrano%5ctyrano.js", "/tyrano//tyrano.js",
    "https://example.test/tyrano/tyrano.js", "/tyrano/%00tyrano.js",
])
def test_unlisted_or_escaping_paths_are_not_served(routes, path):
    with pytest.raises((OSError, ValueError)):
        routes.resolve(path)


def test_declared_link_cannot_escape_export(routes, tmp_path):
    target = routes.directory / "linked.txt"
    try:
        target.symlink_to(tmp_path / "private.txt")
    except OSError:
        pytest.skip("Creating symbolic links is not permitted on this Windows account.")
    routes.allowed.add("linked.txt")
    with pytest.raises(FileNotFoundError, match="outside"):
        routes.resolve("/linked.txt")


def test_handler_serves_get_head_and_rejects_mutation_without_starting_server(routes):
    class Connection:
        def __init__(self, request):
            self.input = io.BytesIO(request)
            self.output = bytearray()

        def makefile(self, mode, buffering=-1):
            assert mode == "rb"
            return self.input

        def sendall(self, data):
            self.output.extend(data)

    class Server:
        server_port = 8766

    handler = serve_debug.make_handler(routes)

    def request(method):
        connection = Connection(
            f"{method} / HTTP/1.0\r\nHost: 127.0.0.1:8766\r\n\r\n".encode())
        handler(connection, ("127.0.0.1", 12345), Server())
        return bytes(connection.output).split(b"\r\n\r\n", 1)

    headers, body = request("GET")
    assert b" 200 " in headers and body == b"player"
    headers, body = request("HEAD")
    assert b" 200 " in headers and b"Content-Length: 6" in headers and body == b""
    headers, _ = request("POST")
    assert b" 501 " in headers


def test_cli_binds_only_loopback_with_requested_directory(monkeypatch, routes):
    calls = []

    class Server:
        def __init__(self, address, handler):
            calls.append(address)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def serve_forever(self):
            pass

    monkeypatch.setattr(serve_debug, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(serve_debug.InstalledEngine, "validate", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["serve_debug", "--directory", str(routes.directory),
        "--engine-dir", str(routes.engine.root)])
    assert serve_debug.main() == 0
    assert calls == [("127.0.0.1", 8766)]
