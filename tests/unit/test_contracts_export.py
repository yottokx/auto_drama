from __future__ import annotations

import hashlib
import io
import json
import shutil
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.contracts import Script
from packages.tyrano_export import compile_bundle, compile_scenario, demo_content, validate_bundle

ROOT = Path(__file__).resolve().parents[2]


def test_export_reproducible_and_complete():
    script, assets = demo_content()
    content = compile_bundle(script, assets)
    assert content == compile_bundle(Script.model_validate_json(script.model_dump_json()), assets)
    assert content == compile_bundle(script, dict(reversed(list(assets.items()))))
    manifest = validate_bundle(content, script, assets)
    assert manifest["engine_included"] is False
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert not any(name.startswith("tyrano/") for name in archive.namelist())
        assert Script.model_validate_json(archive.read("script.json")) == script
        assert json.loads(archive.read("manifest.json")) == manifest
        assert set(archive.namelist()) == set(manifest["files"]) | {"manifest.json"}
        for name, record in manifest["files"].items():
            data = archive.read(name)
            assert record == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for entry in archive.infolist():
            assert entry.date_time == (1980, 1, 1, 0, 0, 0)
    assert script == demo_content()[0]
    assert assets == demo_content()[1]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda content: content[:-10],
        lambda content: content + b"uncommitted tail",
        lambda content: content.replace(b"first.ks", b"other.ks"),
    ],
)
def test_reject_unexpected_or_incomplete_worker_result(mutation):
    script, assets = demo_content()
    with pytest.raises(ValueError, match="deterministic bundle"):
        validate_bundle(mutation(compile_bundle(script, assets)), script, assets)


def test_reject_missing_extra_and_corrupted_assets():
    script, assets = demo_content()
    for supplied in ({}, {**assets, "unrequested": b"extra"}, {**assets, "station": b"partial"}):
        with pytest.raises(ValueError):
            compile_bundle(script, supplied)


@pytest.mark.parametrize(
    "filename",
    [
        "../scene.png",
        "..\\scene.png",
        "/tmp/scene.png",
        "C:\\scene.png",
        "\\\\host\\a.png",
        "scene.png:secret",
        "scene.png?x=1",
        "CON.png",
        "LPT1.png",
        "nul.PNG",
        "scene.png ",
        'a"].png',
        "scene.svg",
        "&alert(1).png",
        "a%2F.png",
    ],
)
def test_reject_unsafe_or_unsupported_asset_filename(filename):
    script, _ = demo_content()
    value = script.model_dump(mode="json")
    value["assets"][0]["filename"] = filename
    with pytest.raises(ValidationError):
        Script.model_validate(value)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), 2),
        (("schema_version",), True),
        (("utterances", 0, "speaker_id"), "missing"),
        (("utterances", 0, "audio_asset_id"), "station"),
        (("characters", 0, "image_asset_id"), "missing"),
        (("directions", 0, "utterance_id"), "missing"),
        (("directions", 0, "asset_id"), "aki_sprite"),
        (("directions", 1, "character_id"), "missing"),
        (("directions", 0, "kind"), "iscript"),
        (("directions", 3, "duration_ms"), -1),
        (("directions", 3, "duration_ms"), True),
        (("utterances", 1, "id"), "line_001"),
        (("title",), "text\x00hidden"),
        (("title",), "text\ud800"),
    ],
)
def test_invalid_script_references_and_content(path, value):
    script, _ = demo_content()
    data = script.model_dump(mode="json")
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        Script.model_validate(data)


def test_reject_case_insensitive_output_collision_and_extra_code():
    script, _ = demo_content()
    data = script.model_dump(mode="json")
    data["assets"][2]["filename"] = "AKI_SPRITE.PNG"
    with pytest.raises(ValidationError, match="collide"):
        Script.model_validate(data)
    data = script.model_dump(mode="json")
    data["directions"][0]["javascript"] = "alert(1)"
    with pytest.raises(ValidationError):
        Script.model_validate(data)


def test_all_directions_and_audio_compile():
    script, assets = demo_content()
    data = script.model_dump(mode="json")
    assets["voice"] = b"RIFF\x24\x00\x00\x00WAVEfmt "
    data["assets"].append(
        {
            "id": "voice",
            "kind": "audio",
            "artifact_id": "audio-artifact",
            "filename": "voice.wav",
            "sha256": hashlib.sha256(assets["voice"]).hexdigest(),
        }
    )
    data["utterances"][1]["audio_asset_id"] = "voice"
    data["directions"].extend(
        [
            {
                "id": "move",
                "kind": "position",
                "utterance_id": "line_002",
                "character_id": "ren",
                "position": "center",
            },
            {
                "id": "focus",
                "kind": "focus",
                "utterance_id": "line_002",
                "timing": "start",
                "character_id": "ren",
            },
            {"id": "dark", "kind": "blackout", "utterance_id": "line_004", "timing": "after"},
            {
                "id": "exit",
                "kind": "exit",
                "utterance_id": "line_004",
                "timing": "after",
                "character_id": "aki",
            },
        ]
    )
    script = Script.model_validate(data)
    scenario = compile_scenario(script, assets)
    for tag in (
        "[chara_move ",
        "[chara_ptext ",
        "[chara_hide ",
        "[mask ",
        "[mask_off ",
        "[ad_pause ",
        '[ad_say id="line_002"]',
        "[ad_wait]",
    ):
        assert tag in scenario
    # The viewing screen plays voices itself and reads prose from script.json by ID.
    assert "[playse " not in scenario and "[wse]" not in scenario and "[wait " not in scenario
    assert all(line.display_text not in scenario for line in script.utterances)
    # Directions after a line run between its text and its click wait.
    last = scenario.index(f'[ad_say id="{script.utterances[-1].id}"]')
    assert last < scenario.index("[mask ") < scenario.index("[ad_wait]", last)
    validate_bundle(compile_bundle(script, assets), script, assets)


def test_schema_and_fixture_are_current():
    schema = json.loads((ROOT / "packages/contracts/script.schema.json").read_text("utf-8"))
    assert schema == Script.model_json_schema()
    fixture = Script.model_validate_json((ROOT / "tests/fixtures/m1-script.json").read_bytes())
    assert fixture == demo_content()[0]


def test_fixture_pngs_decode_without_external_libraries():
    _, assets = demo_content()
    for content in assets.values():
        assert content[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", content[16:24])
        offset = 8
        compressed = b""
        while offset < len(content):
            length = struct.unpack(">I", content[offset : offset + 4])[0]
            kind = content[offset + 4 : offset + 8]
            payload = content[offset + 8 : offset + 8 + length]
            crc = struct.unpack(">I", content[offset + 8 + length : offset + 12 + length])[0]
            assert crc == zlib.crc32(kind + payload) & 0xFFFFFFFF
            if kind == b"IDAT":
                compressed += payload
            offset += 12 + length
        assert len(zlib.decompress(compressed)) == height * (1 + width * 4)


def _node(source: str, value: dict, *args: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to verify the installed Tyrano JavaScript parser")
    result = subprocess.run(
        [node, "-e", source, *args],
        input=json.dumps(value),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_native_tyrano_parser_never_sees_author_text():
    parser_path = ROOT / "tyranoscript/tyrano/plugins/kag/kag.parser.js"
    if not parser_path.is_file():
        pytest.skip(
            "Install the separately supplied Tyrano engine for parser compatibility testing"
        )
    title = '@jump storage="evil.ks" [iscript]alert(1)[endscript] '
    name = '#actor" ` \\ <img src=x onerror=alert(1)> [s]'
    body = (
        "_下線 [eval exp='evil']\\末尾\\  \n;コメント\n*ラベル\n@jump\n#話者\n/*\n*/\n\n&式 <tag>"
    )
    script = Script.model_validate(
        {
            "id": "escaping",
            "title": title,
            "characters": [{"id": "actor", "name": name}],
            "utterances": [
                {
                    "id": "line",
                    "speaker_id": "actor",
                    "display_text": body,
                    "spoken_text": "読み上げる本文",
                }
            ],
        }
    )
    scenario = compile_scenario(script, {})
    parsed = _node(
        r"""
const fs = require('fs');
const vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = {tyrano: {plugin: {kag: {}}}, $: {trim: value => String(value).trim()}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const parser = context.tyrano.plugin.kag.parser;
parser.kag = {config: {KeepSpaceInParameterValue: '3'}, stat: {current_scenario: 'first.ks'},
  convertLang() {}, warning(message) {throw Error(message)}, error(message) {throw Error(message)}};
process.stdout.write(JSON.stringify(parser.parseScenario(input.scenario).array_s));
""",
        {"scenario": scenario},
        str(parser_path),
    )
    # Title, speaker name and prose stay in script.json; the scenario carries IDs only.
    assert [item["name"] for item in parsed] == [
        "eval", "chara_config", "layopt", "ad_gate", "label", "ad_line", "chara_ptext",
        "ad_say", "ad_wait", "label", "ad_end", "s",
    ]
    by_name = {item["name"]: item["pm"] for item in parsed}
    assert by_name["ad_line"]["id"] == by_name["ad_say"]["id"] == "line"
    assert by_name["chara_ptext"]["name"] == ""  # No portrait: nobody is focused.
    # The only script in the scenario is the compiler's fixed reload guard for stale launchers.
    assert by_name["eval"]["exp"].startswith("window.AutoDramaPlayer||") and "reload()" in by_name["eval"]["exp"]
    for fragment in (title, name, *[line for line in body.split("\n") if line], "evil", "iscript"):
        assert fragment not in scenario


def test_native_tyrano_demo_tag_and_asset_smoke():
    """Exercise actual native control handlers; DOM rendering is outside this smoke test."""
    engine = ROOT / "tyranoscript/tyrano/plugins/kag"
    if not (engine / "kag.parser.js").is_file():
        pytest.skip("Install the separately supplied Tyrano engine for native tag testing")
    script, assets = demo_content()
    with zipfile.ZipFile(io.BytesIO(compile_bundle(script, assets))) as archive:
        paths = archive.namelist()
    result = _node(
        r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const state = {charas: {}, jcharas: {}, current_speaker: '', chara_talk_anim: 'none',
  chara_talk_focus: 'none', chara_brightness_value: '60', log_join: 'false'};
const log = [], focus = [], clicks = [], timers = [], preloads = [];  // log/clicks/timers must stay unused
let next = 0, strong = false, weak = false;
const query = value => ({
  find() {return this}, css() {return this}, updatePText() {return this},
  get() {return typeof value === 'object' ? value : null},
});
query.trim = value => String(value).trim();
query.isHTTP = value => /^https?:/.test(value);
const context = {tyrano: {plugin: {kag: {tag: {}}}}, TYRANO: {kag: {}}, $: query,
  setTimeout(callback, milliseconds) {timers.push(Number(milliseconds)); callback(); return 1}};
vm.createContext(context);
for (const file of ['kag.parser.js', 'kag.tag.js', 'kag.tag_ext.js', 'kag.tag_audio.js']) {
  vm.runInContext(fs.readFileSync(path.join(process.argv[1], file), 'utf8'), context);
}
const tags = context.tyrano.plugin.kag.tag;
const kag = {stat: state, config: {KeepSpaceInParameterValue: '3'}, tmp: {},
  ftag: {nextOrder() {next++}, showNextImg() {}},
  weaklyStop() {weak = true}, cancelWeakStop() {weak = false},
  stronglyStop() {strong = true}, cancelStrongStop() {strong = false},
  waitClick(kind) {clicks.push(kind)}, pushBackLog(html) {log.push(html)},
  convertLang() {}, error(message) {throw Error(message)}, warning(message) {throw Error(message)},
  preload(url, callback) {
    if (!input.paths.includes(url.replace(/^\.\//, ''))) throw Error('Missing preload: ' + url);
    preloads.push(url); callback({width: 28, height: 42});
  },
  chara: {
    getCharaNameArea() {return query()}, getCharaContainer(name) {return {name}},
    setNotSpeakerStyle() {}, setSpeakerStyle(container) {focus.push(container.name)},
  },
};
const parser = context.tyrano.plugin.kag.parser;
parser.kag = kag;
const parsed = parser.parseScenario(input.scenario).array_s;
const instances = Object.fromEntries(Object.entries(tags).map(([name, definition]) =>
  [name, Object.assign(Object.create(definition), {kag})]));
// Tags owned by the viewing screen are registered by its script, not by the engine.
const owned = {ad_gate: 0, ad_line: 0, ad_say: 0, ad_wait: 0, ad_end: 0, ad_pause: 0};
const shown = [], pauses = [], guards = [];
for (const tag of parsed) {
  if (tag.name === 'eval') { guards.push(tag.pm.exp); continue; }  // kag.tag_system.js is not loaded here
  if (tag.name in owned) {
    owned[tag.name]++;
    if (tag.name === 'ad_pause') pauses.push(Number(tag.pm.time));
    if (['ad_line', 'ad_say'].includes(tag.name) && !input.lines.includes(tag.pm.id)) throw Error('Unknown line');
    continue;
  }
  if (!instances[tag.name]) throw Error('Unknown tag: ' + tag.name);
  const handler = instances[tag.name];
  const pm = {...JSON.parse(JSON.stringify(handler.pm || {})), ...tag.pm};
  for (const required of handler.vital || []) {
    if (!pm[required]) throw Error('Missing required parameter: ' + tag.name + '.' + required);
  }
  if (['chara_config', 'chara_new', 'chara_ptext', 's'].includes(tag.name)) {
    handler.start(pm);
  } else if (tag.name === 'chara_show') {
    if (!state.charas[pm.name]) throw Error('Undefined character');
    // Transparent padding and lower body may extend beyond the stage by design.
    if (+pm.width <= 0 || +pm.height <= 0 || +pm.top >= 440)
      throw Error('Character cannot be seen above the message window');
    shown.push(pm.name);
  } else if (tag.name === 'bg' && !input.paths.includes('data/bgimage/' + pm.storage)) {
    throw Error('Missing background');
  } else if (tag.name === 'layopt' && (pm.layer !== 'message0' || pm.visible !== 'false')) {
    throw Error('The engine message layer must stay hidden');
  }
}
process.stdout.write(JSON.stringify({owned, pauses, guards, focus, shown, preloads, strong, weak,
  names: parsed.map(tag => tag.name)}));
""",
        {"scenario": compile_scenario(script, assets), "paths": paths,
         "lines": [line.id for line in script.utterances]},
        str(engine),
    )
    count = len(script.utterances)
    assert result["owned"] == {"ad_gate": 1, "ad_line": count, "ad_say": count, "ad_wait": count,
                               "ad_end": 1, "ad_pause": 1}
    assert result["pauses"] == [450]
    assert len(result["guards"]) == 1 and result["names"][0] == "eval"
    assert result["shown"] == ["ad_aki", "ad_ren"]
    assert result["focus"] == ["ad_ren", "ad_aki", "ad_ren"]
    assert len(result["preloads"]) == 2
    assert result["strong"] and result["weak"]  # The explicit final [s] stops playback.
    # Nothing reaches the engine's text renderer, click waits or backlog.
    assert not {"text", "p", "l", "r", "cm", "playse", "wse", "wait", "loadjs"} & set(result["names"])
