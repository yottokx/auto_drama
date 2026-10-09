"""The scene preview recompiles a published script locally and relays everything else."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from packages.contracts import Script
from packages.tyrano_export import demo_content
from packages.tyrano_export.compiler import canonical_json
from packages.tyrano_export.demo import _png
from scripts.image_edit import cg_preview

IMAGE = _png(96, 64, lambda x, y: (200, 40, 40, 255))


def published(extra=None):
    """A chapter as the coordinator serves it, plus the files a reader can ask for."""
    script, assets = demo_content()
    value = script.model_dump(mode="json")
    value["utterances"].extend({"id": f"line_{index:03}", "display_text": str(index),
                                "spoken_text": str(index)} for index in (5, 6, 7, 8))
    value.update(extra or {})
    script = Script.model_validate(value)
    files = {"script.json": canonical_json(script.model_dump(mode="json"))}
    for asset in script.assets:
        folder = {"background": "bgimage", "character": "fgimage", "event_cg": "cgimage"}[asset.kind]
        files[f"data/{folder}/{asset.filename}"] = assets.get(asset.id, IMAGE)
    return script, files


def record(scene_ids, first, last):
    return {"context": {"published_build_id": "build-1", "scene_id": "s2",
                        "utterances": [{"id": value} for value in scene_ids],
                        "cg_proposal": {"proposal": {"display_from": first, "display_to": last}}}}


def test_interval_is_added_inside_the_scene_and_everything_else_is_unchanged():
    script, files = published()
    scene = ["line_003", "line_004", "line_005", "line_006"]
    result = cg_preview.build_files(record(scene, 2, 3), IMAGE, files.get)
    value = json.loads(result["script.json"])
    assert value["event_cg_segments"] == [{
        "id": "cg_preview_segment", "start_utterance_id": "line_004",
        "end_utterance_id": "line_006", "base_asset_id": "cg_preview", "variants": []}]
    assert value["utterances"] == script.model_dump(mode="json")["utterances"]
    assert value["directions"] == script.model_dump(mode="json")["directions"]
    assert result["data/cgimage/cg_preview.png"] == IMAGE
    states = json.loads(result["data/others/auto_drama_states.json"])["entries"]
    shown = [key for key, entry in states.items() if entry["event_cg"]]
    # A line's entry state is the picture left by the previous line.
    assert shown == ["line_005", "line_006"]
    assert states["line_005"]["event_cg"]["storage"] == "cg_preview.png"
    assert '[ad_line id="line_004"]' in result["data/scenario/first.ks"].decode()
    # The scene does not open the chapter, so the page records its first line for "resume".
    html = result["index.html"].decode()
    assert 'localStorage.setItem("adn_auto_v1:demo_chapter_01","{\\"id\\": \\"line_003\\"}")' in html
    assert html.index("localStorage.setItem") < html.index("auto_drama_player.js")


def test_transition_applies_to_cg_anchors_only_and_the_default_stays_a_cut():
    extra = {"scene_transitions": [{"id": "entry", "utterance_id": "line_006", "visual": "fade",
                                    "duration_ms": 900, "music_fade_out_ms": 0, "music_fade_in_ms": 0}]}
    _, files = published(extra)
    scene = ["line_003", "line_004", "line_005", "line_006", "line_007"]

    def anchors(**options):
        result = cg_preview.build_files(record(scene, 2, 3), IMAGE, files.get, **options)
        stages = json.loads(result["data/others/auto_drama_stages.json"])["scenes"]
        return {row["id"]: (row["visual"], row["duration_ms"]) for row in stages}

    assert anchors() == {"cg:line_004": ("cut", 0), "cg:line_005": ("cut", 0), "entry": ("fade", 900)}
    assert anchors(transition=("dissolve", 500)) == {
        "cg:line_004": ("dissolve", 500), "cg:line_005": ("dissolve", 500), "entry": ("fade", 900)}
    for bad in (("wipe", 500), ("fade", 5000), ("fade", "500")):
        with pytest.raises(cg_preview.PreviewError, match="効果と時間"):
            anchors(transition=bad)


def test_chapter_opening_scene_needs_no_resume_point_and_a_last_line_keeps_the_cg_to_the_end():
    _, files = published()
    order = [row["id"] for row in json.loads(files["script.json"])["utterances"]]
    result = cg_preview.build_files(record(order, len(order) - 1, len(order)), IMAGE, files.get)
    assert b"localStorage" not in result["index.html"]
    assert json.loads(result["script.json"])["event_cg_segments"][0]["end_utterance_id"] is None


def test_only_an_overlapping_adopted_cg_is_replaced():
    adopted = {"assets": None, "event_cg_segments": [
        {"id": "early", "start_utterance_id": "line_001", "end_utterance_id": "line_003",
         "base_asset_id": "cg_early"},
        {"id": "clash", "start_utterance_id": "line_004", "end_utterance_id": "line_006",
         "base_asset_id": "cg_clash"}]}
    script, _ = demo_content()
    adopted["assets"] = script.model_dump(mode="json")["assets"] + [
        {"id": name, "kind": "event_cg", "artifact_id": name, "filename": name + ".png",
         "sha256": "0" * 64} for name in ("cg_early", "cg_clash")]
    _, files = published(adopted)
    result = cg_preview.build_files(
        record(["line_003", "line_004", "line_005", "line_006"], 3, 4), IMAGE, files.get)
    value = json.loads(result["script.json"])
    assert [row["id"] for row in value["event_cg_segments"]] == ["early", "cg_preview_segment"]
    assert {row["id"] for row in value["assets"] if row["kind"] == "event_cg"} == {
        "cg_early", "cg_preview"}


@pytest.mark.parametrize("change, message", [
    (lambda value: value["context"]["cg_proposal"]["proposal"].update(display_to=9), "範囲"),
    (lambda value: value["context"]["cg_proposal"].clear(), "範囲"),
    (lambda value: value["context"]["utterances"].append({"id": "other-u1"}), "一致しません"),
])
def test_unusable_records_are_explained(change, message):
    _, files = published()
    value = record(["line_003", "line_004"], 1, 2)
    change(value)
    with pytest.raises(cg_preview.PreviewError, match=message):
        cg_preview.build_files(value, IMAGE, files.get)


def test_unpublished_chapter_and_missing_script_are_explained(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(IMAGE)
    value = record(["line_001"], 1, 1)
    value["context"]["published_build_id"] = None
    with pytest.raises(cg_preview.PreviewError, match="公開されていない"):
        cg_preview.prepare(value, path, "http://127.0.0.1:1")
    with pytest.raises(cg_preview.PreviewError, match="台本を取得できません"):
        cg_preview.build_files(record(["line_001"], 1, 1), IMAGE, lambda path: None)


@pytest.fixture
def upstream():
    """Stands in for the coordinator's /player/{build}/ route."""
    _, files = published()
    files["data/sound/voice.wav"] = bytes(range(100))
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Range")))
            content = files.get(self.path.removeprefix("/player/build-1/"))
            if content is None:
                self.send_error(404)
                return
            partial = self.headers.get("Range") == "bytes=10-19"
            self.send_response(206 if partial else 200)
            if partial:
                content = content[10:20]
                self.send_header("Content-Range", "bytes 10-19/100")
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()


def test_server_serves_local_files_and_relays_the_rest_with_ranges(upstream, tmp_path):
    base, requests = upstream
    path = tmp_path / "image.png"
    path.write_bytes(IMAGE)
    source, files = cg_preview.prepare(
        record(["line_003", "line_004", "line_005"], 1, 2), path, base)
    assert source == base + "/player/build-1/"
    assert all(item[0] != "/player/build-1/data/sound/voice.wav" for item in requests)
    server = cg_preview.PreviewServer()
    try:
        url = server.add(source, files)
        assert url.startswith("http://127.0.0.1:")
        with urlopen(url) as response:
            assert b"auto_drama_player.js" in response.read()
            assert response.headers["Cache-Control"] == "no-store"
        with urlopen(url + "data/cgimage/cg_preview.png") as response:
            assert response.read() == IMAGE and response.headers["Content-Type"] == "image/png"
        with urlopen(url + "script.json") as response:
            assert json.loads(response.read())["event_cg_segments"][0]["id"] == "cg_preview_segment"
        with urlopen(Request(url + "data/sound/voice.wav", headers={"Range": "bytes=10-19"})) as response:
            assert response.status == 206 and response.read() == bytes(range(10, 20))
            assert response.headers["Content-Range"] == "bytes 10-19/100"
        for missing in (url + "data/sound/none.wav", url.rsplit("/", 2)[0] + "/unknown/index.html",
                        url + "..%2Fscript.json"):
            with pytest.raises(Exception) as error:
                urlopen(missing)
            assert getattr(error.value, "code", None) == 404
    finally:
        server.close()
