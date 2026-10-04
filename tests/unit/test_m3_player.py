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
from packages.tyrano_export import compile_bundle, demo_content, validate_bundle
from packages.tyrano_export.player import ENGINE_SCRIPTS, ENGINE_STYLES, player_config, player_files
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


@pytest.fixture
def player(tmp_path, monkeypatch, request):
    script, assets = demo_content()
    if getattr(request, "param", False):
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
        assert b"ad-start-button" in archive.read("index.html")
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
    assert response.headers["cache-control"].endswith("immutable")
    assert client.get("/player/published/data/scenario/first.ks").content == bundle_member(
        data, "data/scenario/first.ks"
    )
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
        assert b"ad-chapter-end" in archive.read("index.html")
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


@pytest.mark.parametrize("native_input", [False, True])
def test_player_javascript_parses_and_uses_safe_read_only_backlog(native_input):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the player control test")
    native_path = ROOT / "tyranoscript/tyrano/plugins/kag/kag.key_mouse.js"
    if native_input and not native_path.is_file():
        pytest.skip("The installed engine is needed for its real mouse handler regression")
    # Exercise our actual JS with a tiny DOM, a chapter containing malicious
    # markup, an unread future line, and native-shaped engine control methods.
    source = r'''
const fs = require('fs'), vm = require('vm');
class Element {
  constructor(tag = '') { this.tag = tag; this.children = []; this.listeners = {}; this.value = '24'; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  append(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  querySelectorAll(selector) {
    return selector === 'button, select, input' ? this.children :
      this.children.filter(c => c.tag === 'audio');
  }
  setAttribute(name, value) { this[name] = value; }
  showModal() { this.open = true; }
  close() { this.open = false; this.listeners.close(); }
  pause() { this.paused = true; }
  closest() { return this.control ? this : null; }
  getBoundingClientRect() { return {left:20, right:620, top:400, bottom:700, width:600, height:300}; }
}
const elements = {}, callbacks = {}, actions = [], nativeEvents = {};
const get = id => elements[id] ||= new Element();
get('ad-toolbar').children = ['ad-auto','ad-save','ad-load','ad-backlog','ad-font','ad-volume'].map(id => {
  const control = get(id); control.control = true; return control;
});
const document = {getElementById: get, createElement: tag => new Element(tag),
  addEventListener(name, fn) { callbacks[name] = fn; },
  querySelectorAll(selector) {return selector === '.message_outer' ? [get('message')] : []}};
const k = {stat: {current_scenario:'first.ks', font:{}, default_font:{}}, tmp:{},
  config:{projectID:'immutable'}, readyAudio() {actions.push('audio')}, on() {},
  setAuto(value) {this.stat.is_auto = value; actions.push(['setAuto',value])},
  key_mouse: {next() {actions.push('next')}, auto() {actions.push('native-auto'); return true},
    qsave() {return true}, qload() {},
    util:{canShowMenu() {return true}}},
  ftag:{array_tag:[{name:'label',pm:{label_name:'utterance_seen'}}, {name:'text'}, {name:'p'},
    {name:'label',pm:{label_name:'utterance_future'}}, {name:'text'}, {name:'p'}],current_order_index:2,
    nextOrder() {this.current_order_index++; actions.push('next-order')},
    startTag(name, params) {
      actions.push([name,params]); if (name === 'autostop') k.stat.is_auto = false;
    }}};
const script = {characters:[{id:'a',name:'<img onerror=bad>'}],assets:[{id:'audio',filename:'line.wav'}],
  utterances:[{id:'seen',speaker_id:'a',display_text:'<script>bad</script>',audio_asset_id:'audio'},
    {id:'future',display_text:'SPOILER'}]};
let tick, manualRequests = 0;
const context = {document, window:{TYRANO:{kag:k}}, tyrano:{plugin:{kag:{}}},
  $:() => ({0:{}, on(name, fn) {nativeEvents[name] = fn;}}),
  fetch:async() => ({ok:true, json:async() => script}), setInterval(fn) {tick=fn}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const makePlayback = context.window.AutoDramaPlayback;
context.window.AutoDramaPlayback = (...args) => {
  const control = makePlayback(...args), advance = control.advance;
  control.advance = () => {manualRequests++; return advance();};
  return control;
};
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
callbacks.DOMContentLoaded();
let nativeInterference = false;
if (process.argv[3]) {
  vm.runInContext(fs.readFileSync(process.argv[3],'utf8'),context);
  const native = context.tyrano.plugin.kag.key_mouse;
  native.kag = k; native.util.parent = native;
  native.util.refer(native.util); native.mouse.init(native);
  k.stat.is_auto = true; k.config.autoClickStop = 'true';
  nativeEvents.mousedown({button:0});
  nativeInterference = !k.stat.is_auto; // Prove the actual engine exhibits the original trigger.
  actions.length = 0;
}
function bubble(target, surface, type) {
  const event = {button:0, stopped:false, prevented:false,
    stopPropagation() {this.stopped=true}, preventDefault() {this.prevented=true}};
  if (target.listeners[type]) target.listeners[type](event);
  if (!event.stopped && surface.listeners[type]) surface.listeners[type](event);
  if (!event.stopped && nativeEvents[type]) nativeEvents[type](event);
  return {stopped:event.stopped, prevented:event.prevented};
}
setImmediate(() => {
  tick(); get('ad-start-button').listeners.click({stopPropagation(){}});
  k.ftag.current_order_index = 4; k.stat.is_adding_text = true;
  get('ad-backlog').listeners.click();
  const openedDuringTyping = get('ad-log').open;
  get('ad-volume').listeners.input({target:{value:'42'}});
  get('ad-font').listeners.change({target:{value:'28'}});
  get('ad-log-close').listeners.click();
  tick();
  const activeControls = ['ad-auto','ad-font','ad-volume'].map(id => get(id).disabled);
  const guardedControls = ['ad-save','ad-load','ad-backlog'].map(id => get(id).disabled);
  get('ad-auto').listeners.click();
  const autoArmed = k.stat.is_auto;
  const propagation = [];
  for (const surface of [get('ad-toolbar'),get('ad-log')]) {
    for (const type of ['pointerdown','mousedown','touchstart','keydown','keyup']) {
      propagation.push(bubble(new Element(),surface,type));
    }
  }
  bubble(get('ad-auto'),get('ad-toolbar'),'mousedown');
  bubble(get('ad-auto'),get('ad-toolbar'),'mouseup');
  bubble(get('ad-auto'),get('ad-toolbar'),'click');
  function input(type, data) {
    const event = {button:0, target:get('message'), clientX:200, clientY:500, key:'Enter',
      repeat:false, stopped:false, prevented:false, ...data,
      stopPropagation() {this.stopped=true}, preventDefault() {this.prevented=true}};
    callbacks[type](event);
    return {stopped:event.stopped, prevented:event.prevented, requests:manualRequests};
  }
  const messageInput = {
    outside: input('click', {clientY:100}),
    toolbar: input('click', {target:get('ad-auto')}),
    message: input('click', {}),
    controlEnter: input('keydown', {target:get('ad-volume')}),
    enter: input('keydown', {}),
    repeatedEnter: input('keydown', {repeat:true}),
  };
  const row = get('ad-log-items').children[0];
  process.stdout.write(JSON.stringify({actions, rows:get('ad-log-items').children.length,
    name:row.children[0].textContent, text:row.children[1].textContent,
    audio:row.children[2].src, font:k.stat.default_font.size, gate:get('ad-start').hidden,
    activeControls, guardedControls, autoArmed, autoStopped:!k.stat.is_auto,
    propagation, nativeInterference, openedDuringTyping, messageInput,
    revealRequested:!!k.stat.is_click_text}));
});
'''
    result = subprocess.run([node, "-e", source, str(ROOT / "packages/tyrano_export/player.js"),
                             str(ROOT / "packages/tyrano_export/playback.js"),
                             str(native_path) if native_input else ""],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["rows"] == 1
    assert value["name"] == "<img onerror=bad>"
    assert value["text"] == "<script>bad</script>"
    assert value["audio"] == "./data/sound/line.wav"
    assert value["font"] == "28" and value["gate"] is True
    assert value["activeControls"] == [False, False, False]
    assert value["guardedControls"] == [True, True, False]
    assert value["openedDuringTyping"] is True
    assert value["autoArmed"] is True and value["autoStopped"] is True
    assert value["propagation"] == [{"stopped": True, "prevented": False}] * 10
    if native_input:
        assert value["nativeInterference"] is True
    assert value["messageInput"] == {
        "outside": {"stopped": False, "prevented": False, "requests": 0},
        "toolbar": {"stopped": False, "prevented": False, "requests": 0},
        "message": {"stopped": True, "prevented": True, "requests": 1},
        "controlEnter": {"stopped": False, "prevented": False, "requests": 1},
        "enter": {"stopped": True, "prevented": True, "requests": 2},
        "repeatedEnter": {"stopped": True, "prevented": True, "requests": 2},
    }
    assert value["revealRequested"] is True
    assert value["actions"] == ["audio", "next", ["seopt", {"volume": "42", "next": "false"}],
                                ["setAuto", True], ["autostop", {"next": "false"}]]
