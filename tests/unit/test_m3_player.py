from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from packages.contracts import Script
from packages.tyrano_export import compile_bundle, compile_scenario, demo_content, validate_bundle
from packages.tyrano_export.player import (
    ENGINE_SCRIPTS,
    ENGINE_STYLES,
    player_config,
    player_files,
    player_html,
)
from services.coordinator.m3_player import (
    InstalledEngine,
    bundle_member,
    byte_range,
    clean_path,
    install_player_routes,
)
from services.coordinator.service import ServiceError
from services.coordinator.storage import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]


def engine_fixture(directory):
    for path in (*ENGINE_SCRIPTS, *ENGINE_STYLES):
        target = directory / "tyrano" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("/* fixture engine */", "utf-8")
    (directory / "tyrano/plugins/kag/kag.js").write_text("const kag = {version: 520};", "utf-8")
    (directory / "data/scenario").mkdir(parents=True)
    (directory / "data/scenario/private.ks").write_text("private sample", "utf-8")
    (directory / ".env").write_text("SECRET=private", "utf-8")
    return directory


LEGACY_FILES = {
    "index.html": b"<!doctype html><button id=\"ad-start-button\">old screen</button>",
    "data/scenario/first.ks": "_旧画面の本文[p]\n[s]\n".encode(),
    "data/others/auto_drama_player.js": b"// old player",
    "data/others/auto_drama_playback.js": b"// old playback helper",
}


def legacy_bundle(data: bytes) -> bytes:
    """A published ZIP as an earlier player wrote it: same script and assets, older screen."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = {name: archive.read(name) for name in archive.namelist() if name != "manifest.json"}
        manifest = json.loads(archive.read("manifest.json"))
    del files["data/others/auto_drama_states.json"]
    files.update(LEGACY_FILES)
    manifest["files"] = {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                         for name, content in sorted(files.items())}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
        archive.writestr("manifest.json", json.dumps(manifest))
    return output.getvalue()


@pytest.fixture
def player(tmp_path, monkeypatch, request):
    script, assets = demo_content()
    if getattr(request, "param", False) is True:
        music = b"ID3adopted MP3 route fixture"
        assets["scene_music"] = music
        value = script.model_dump(mode="json")
        value["assets"].append({"id": "scene_music", "kind": "music",
                                "artifact_id": "adopted-music", "filename": "scene_music.mp3",
                                "sha256": hashlib.sha256(music).hexdigest()})
        value["music_cues"] = [{"id": "music_start", "utterance_id": script.utterances[0].id,
                                "action": "play", "asset_id": "scene_music",
                                "loop_start_seconds": 10.0, "loop_end_seconds": 60.0}]
        script = Script.model_validate(value)
    data = compile_bundle(script, assets)
    if getattr(request, "param", False) == "legacy":
        data = legacy_bundle(data)
    store = ArtifactStore(tmp_path / "store")
    stored = store.put(data)
    record = vars(stored)
    build = {"id": "published", "status": "published", "export_artifact_id": "export",
             "production_id": "production", "project_id": "project", "chapter_number": 1}

    class Service:
        def artifact(self, identifier):
            assert identifier == "export"
            return record, store.read(record)

    class Builds:
        def __init__(self, service):
            pass

        def build(self, identifier):
            if identifier == "unpublished":
                return {**build, "status": "building"}
            if identifier != "published":
                raise ServiceError(404, "not found")
            return build.copy()

    # The serving boundary is tested independently of the generation state
    # machine, which has its own integration coverage.
    monkeypatch.setitem(sys.modules, "services.coordinator.m3_service", types.SimpleNamespace(
        M3Service=Builds,
    ))
    app = FastAPI()

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    @app.exception_handler(OSError)
    async def storage_error(request: Request, exc: OSError):
        return JSONResponse(status_code=503, content={"detail": "artifact integrity failure"})

    engine = engine_fixture(tmp_path / "engine")
    install_player_routes(app, Service, engine)
    with TestClient(app) as client:
        yield client, data, record, store, engine


def test_source_export_contains_owned_launcher_and_public_documents():
    script, assets = demo_content()
    documents = {"approval.json": b'{"world": "public"}', "narrative.json": b"{}",
                 "sources/scene_001.txt": "原文\n".encode()}
    data = compile_bundle(script, assets, documents=documents)
    manifest = validate_bundle(data, script, assets, documents=documents)
    assert manifest["engine_included"] is False
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.read("sources/scene_001.txt") == documents["sources/scene_001.txt"]
        assert "index.html" in manifest["files"]
        assert not any(name.startswith("tyrano/") for name in archive.namelist())
        assert b"auto_drama_player.js" in archive.read("index.html")
        assert b"auto_drama_playback.js" not in archive.read("index.html")
    assert data == compile_bundle(script, assets, documents=dict(reversed(list(documents.items()))))
    assert player_config(b"first") != player_config(b"second")


def test_export_assets_keep_the_loaded_launcher_revision_after_source_updates(monkeypatch):
    expected = player_files(b"chapter")
    original = Path.read_text

    def edited_source(path, *args, **kwargs):
        if path.parent == ROOT / "packages/tyrano_export" and path.suffix in {".js", ".css"}:
            return "new release with incompatible element IDs"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", edited_source)
    assert player_files(b"chapter") == expected


@pytest.mark.parametrize("path", ["../secret", "tyrano/a.js", "index.html", "sources/../x.txt",
                                  "sources/.hidden.txt", "sources/x.txt:other"])
def test_public_document_paths_are_closed(path):
    script, assets = demo_content()
    with pytest.raises(ValueError, match="document path"):
        compile_bundle(script, assets, documents={path: b"private"})


@pytest.mark.parametrize("path", ["../secret", "/secret", ".env", "tyrano/../secret",
                                  "data\\secret", "data/x:secret", "data//x", "data/%2e%2e",
                                  "data/x\x00", "data/x?foo", "data/x#foo"])
def test_path_traversal_and_hidden_files_are_rejected(path):
    with pytest.raises(ServiceError) as exc:
        clean_path(path)
    assert exc.value.status == 404


def test_player_routes_only_published_immutable_files(player):
    client, data, _, _, engine = player
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in engine.rglob("*") if p.is_file()}
    response = client.get("/player/published/")
    assert response.status_code == 200
    assert "再生する" not in response.text  # Only enabled by client after engine readiness.
    assert response.content == bundle_member(data, "index.html")
    assert response.headers["x-content-type-options"] == "nosniff"
    # The screen follows the running code; only script, assets and documents are immutable.
    assert response.headers["cache-control"] == "no-store"
    scenario = client.get("/player/published/data/scenario/first.ks")
    assert scenario.content == bundle_member(data, "data/scenario/first.ks")
    assert scenario.headers["cache-control"] == "no-store"
    assert client.get("/player/published/script.json").headers["cache-control"].endswith("immutable")
    assert client.get("/player/published/data/bgimage/station.png").headers["cache-control"].endswith("immutable")
    assert client.get("/player/published/tyrano/tyrano.js").status_code == 200
    assert client.get("/player/published/data/scenario/private.ks").status_code == 404
    assert client.get("/player/published/.env").status_code == 404
    assert client.get("/player/unpublished/").status_code == 404
    assert client.get("/player/missing/tyrano/tyrano.js").status_code == 404
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


@pytest.mark.parametrize("path", ["%2e%2e%2fsecret", "tyrano/%2e%2e%2f.env",
                                  "tyrano/%252e%252e/secret", "tyrano/a%5csecret.js",
                                  "data/a%3asecret", "%2fetc%2fpasswd"])
def test_encoded_traversal_cannot_escape_player(player, path):
    assert player[0].get("/player/published/" + path).status_code == 404


def test_range_and_head_allow_media_seeking(player):
    client, data, _, _, _ = player
    path = "data/bgimage/station.png"
    expected = bundle_member(data, path)
    response = client.get("/player/published/" + path, headers={"Range": "bytes=0-15"})
    assert response.status_code == 206
    assert response.content == expected[:16]
    assert response.headers["content-range"] == f"bytes 0-15/{len(expected)}"
    head = client.head("/player/published/" + path)
    assert head.status_code == 200 and head.content == b""
    assert int(head.headers["content-length"]) == len(expected)
    assert client.get("/player/published/" + path, headers={"Range": "bytes=0-1,4-5"}).status_code == 416
    assert byte_range("bytes=-8", 100) == (92, 99, True)


@pytest.mark.parametrize("player", [True], indirect=True)
def test_adopted_music_and_owned_helper_are_served_from_the_immutable_bundle(player):
    client, data, _, _, _ = player
    path = "data/bgm/scene_music.mp3"
    expected = bundle_member(data, path)
    response = client.get("/player/published/" + path)
    assert response.status_code == 200 and response.content == expected
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.headers["cache-control"].endswith("immutable")
    partial = client.get("/player/published/" + path, headers={"Range": "bytes=3-12"})
    assert partial.status_code == 206 and partial.content == expected[3:13]
    assert partial.headers["content-range"] == f"bytes 3-12/{len(expected)}"
    head = client.head("/player/published/" + path)
    assert head.status_code == 200 and head.content == b""
    assert int(head.headers["content-length"]) == len(expected)
    helper = "data/others/auto_drama_music.js"
    assert client.get("/player/published/" + helper).content == bundle_member(data, helper)
    assert client.get("/player/published/data/bgm/original.wav").status_code == 404
    assert client.get("/player/unpublished/" + path).status_code == 404


def test_live_player_identity_is_fixed_to_build_and_absent_from_static_bundle(player):
    client, data, _, _, _ = player
    response = client.get("/player/published/player-context.json")
    assert response.status_code == 200
    assert response.json() == {
        "schema_version": 1, "mode": "live", "build_id": "published",
        "production_id": "production", "project_id": "project", "chapter_number": 1,
        "storyline_id": "production",
        "next_url": "/api/m3/builds/published/next",
    }
    assert response.headers["cache-control"].endswith("immutable")
    assert client.get("/player/unpublished/player-context.json").status_code == 404
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert "player-context.json" not in archive.namelist()
        assert b"adn-end" in archive.read("data/others/auto_drama_player.js")
        assert b"*auto_drama_chapter_end" in archive.read("data/scenario/first.ks")


def test_corrupted_published_object_is_not_served(player):
    client, _, record, store, _ = player
    assert client.get("/player/published/script.json").status_code == 200
    (store.root / record["storage_key"]).write_bytes(b"partial")
    assert client.get("/player/published/script.json").status_code == 503


def test_missing_and_unsupported_engine_report_actionable_error(player):
    client, _, _, _, engine = player
    source = engine / "tyrano/plugins/kag/kag.js"
    source.write_text("const kag = {version: 999};", "utf-8")
    response = client.get("/player/published/")
    assert response.status_code == 503 and "V520" in response.json()["detail"]
    source.unlink()
    response = client.get("/player/published/")
    assert response.status_code == 503 and "AUTO_DRAMA_TYRANO_DIR" in response.json()["detail"]
    assert str(engine) not in response.text
    # Generated sources stay available without an engine installation.
    assert client.get("/player/published/script.json").status_code == 200


def test_manifest_hash_and_extra_members_are_verified():
    script, assets = demo_content()
    data = compile_bundle(script, assets)
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        files = {name: source.read(name) for name in source.namelist()}
    for mutation in ({"script.json": b"{}"}, {"extra.txt": b"unpublished"}):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for path, content in (files | mutation).items():
                archive.writestr(path, content)
        with pytest.raises(ServiceError) as exc:
            bundle_member(output.getvalue(), "script.json")
        assert exc.value.status == 503


def test_installed_engine_matches_owned_launcher():
    if not (ROOT / "tyranoscript").is_dir():
        pytest.skip("The user-supplied engine is not installed")
    InstalledEngine(ROOT / "tyranoscript").validate()


@pytest.mark.parametrize("player", ["legacy"], indirect=True)
def test_chapter_published_by_an_earlier_player_opens_in_the_current_screen(player):
    client, data, _, _, _ = player
    script, assets = demo_content()
    assert bundle_member(data, "index.html") == LEGACY_FILES["index.html"]  # The stored ZIP is untouched.
    assert client.get("/player/published/").content == player_html()
    scenario = client.get("/player/published/data/scenario/first.ks")
    assert scenario.content == compile_scenario(script, assets).encode("utf-8")
    assert scenario.headers["cache-control"] == "no-store"
    states = client.get("/player/published/data/others/auto_drama_states.json")
    assert states.status_code == 200
    assert list(states.json()["entries"]) == [line.id for line in script.utterances]
    current = player_files(bundle_member(data, "script.json"))
    for name in ("data/others/auto_drama_player.js", "data/others/auto_drama_player.css",
                 "data/system/Config.tjs", "data/system/KeyConfig.js"):
        assert client.get("/player/published/" + name).content == current[name]
    # Published content is still served as stored, hash-checked and immutable.
    art = client.get("/player/published/data/bgimage/station.png")
    assert art.content == assets["station"] and art.headers["cache-control"].endswith("immutable")
    assert client.get("/player/published/script.json").content == bundle_member(data, "script.json")


@pytest.mark.parametrize("player", ["legacy"], indirect=True)
def test_corrupted_earlier_chapter_is_refused_instead_of_recompiled(player):
    client, _, record, store, _ = player
    (store.root / record["storage_key"]).write_bytes(b"partial")
    assert client.get("/player/published/").status_code == 503
    assert client.get("/player/published/data/scenario/first.ks").status_code == 503


def test_player_javascript_parses_and_registers_its_tags_before_the_engine_starts():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the player script check")
    # Tags must exist when the engine copies its tag table at initialization, so they
    # are registered while the script loads, before any element or story data exists.
    source = r'''
const fs = require('fs'), vm = require('vm');
const element = () => new Proxy(function () {}, {get: (_, key) => key === 'classList' ?
  {add() {}, remove() {}, toggle() {}, contains: () => false} : key === 'style' ? {setProperty() {}} :
  key === 'dataset' ? {} : key === 'querySelectorAll' ? () => [] : key === 'querySelector' ? () => null :
  key === 'getBoundingClientRect' ? () => ({width: 0, height: 0, left: 0, top: 0}) : element(),
  apply: () => undefined, set: () => true});
const tags = {};
const context = {window: {tyrano: {plugin: {kag: {tag: tags}}}, location: {pathname: '/player/build/'},
    localStorage: {getItem: () => null, setItem() {}}, addEventListener() {}, innerWidth: 960, innerHeight: 640},
  document: {body: {insertAdjacentHTML() {}}, documentElement: element(), getElementById: element,
    querySelectorAll: () => [], addEventListener() {}, fonts: {ready: Promise.resolve()}},
  fetch: () => new Promise(() => {}), setInterval() {}, setTimeout() {}, clearTimeout() {},
  performance: {now: () => 0}, Promise, JSON, Math, Number, Object, Map, Date, Error, String};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
process.stdout.write(JSON.stringify({tags: Object.keys(tags).sort(),
  vital: Object.fromEntries(Object.entries(tags).map(([name, tag]) => [name, tag.vital])),
  state: context.window.AutoDramaPlayer.state()}));
'''
    result = subprocess.run([node, "-e", source, str(ROOT / "packages/tyrano_export/player.js")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["tags"] == ["ad_end", "ad_gate", "ad_line", "ad_music", "ad_pause", "ad_say",
                             "ad_transition", "ad_wait"]
    assert value["vital"]["ad_say"] == ["id"] and value["vital"]["ad_transition"] == ["cue"]
    assert value["state"]["started"] is False and value["state"]["tag"] == "idle"
