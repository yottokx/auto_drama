"""Build one local M0 scene from generated assets, without copying Tyrano's engine."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import struct
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "tyranoscript"
TITLE = "灯台で、ひと呼吸"
VOICES = ("clone_neutral", "clone_happy", "clone_worried")


def write_text(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def encode_text(text: str) -> str:
    """Keep user text out of Tyrano tag/label/comment syntax; [r] is our own tag."""
    if "\x00" in text:
        raise ValueError("NUL is not supported in scene text.")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    escaped = [line.replace("\\", "\\\\").replace("[", "\\[") for line in lines]
    # A leading underscore is Tyrano's literal-text prefix. The trailing [r]
    # also keeps trailing whitespace from being trimmed by its line parser.
    return "\n".join("_" + line + ("[r]" if i < len(lines) - 1 else "")
                     for i, line in enumerate(escaped))


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG image: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if not width or not height:
        raise ValueError(f"Empty PNG dimensions: {path}")
    return width, height


def config_text(project_id: str) -> str:
    # Retain the installed engine's configuration keys, but no engine or sample code.
    values = {}
    for line in (ENGINE / "data/system/Config.tjs").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith(";") and "=" in line:
            key, value = line.split("//", 1)[0].replace(";", "").replace('"', "").split("=", 1)
            values[key.strip()] = value.strip()
    values.update({
        "System.title": TITLE, "projectID": project_id, "scWidth": "1280", "scHeight": "720",
        "ScreenRatio": "fix", "ScreenCentering": "true", "patch_apply_auto": "false",
        "use3D": "false", "useCamera": "false", "useGamepad": "false",
        "debugMenu.visible": "false", "configVisible": "false", "configThumbnail": "false",
        "configSave": "webstorage", "configSaveSlotNum": "3", "maxBackLogNum": "80",
        "chSpeed": "18", "defaultFontSize": "26", "defaultLineSpacing": "7",
        "defaultAutoReturn": "false", "autoSpeed": "700", "autoSpeedWithText": "0",
        "autoClickStop": "true", "mediaFormatDefault": "wav", "useCloseConfirm": "false",
        "ml": "20", "mt": "446", "mw": "1240", "mh": "254",
        "marginL": "28", "marginR": "28", "marginT": "50", "marginB": "18",
        "frameColor": "0x08131F", "frameOpacity": "220", "defaultChColor": "0xF4F0E7",
    })
    return "// Local M0 scene configuration\n" + "\n".join(
        f";{key} = {value};" for key, value in values.items()) + "\n"


KEY_CONFIG = """window.__tyrano_key_config = {
  system_key_event: "true", system_mouse_event: "false",
  key: {Enter:"ok -a", Escape:"cancel -a", " ":"hidemessage", a:"auto", b:"backlog", s:"save", l:"load"},
  mouse: {right:"hidemessage", center:"menu", wheel_up:"backlog", wheel_down:"next"},
  gesture: {}, gamepad: {button:{}, stick:{}}
};
"""

PLAYER_JS = """"use strict";
// User text is already escaped for .ks syntax. Escape the separate HTML backlog
// sink as well, without turning & < > into visible entity strings in the story.
const m0Text = tyrano.plugin.kag.tag.text;
const m0PushLog = m0Text.pushTextToBackLog;
const m0Escape = text => String(text).replace(/[&<>"']/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
m0Text.pushTextToBackLog = function(name, text) {
  return m0PushLog.call(this, m0Escape(name), m0Escape(text));
};
document.addEventListener("DOMContentLoaded", () => {
  const buttons = document.querySelectorAll("#m0-toolbar button[data-action]");
  buttons.forEach(button => button.addEventListener("click", event => {
    event.stopPropagation();
    const kag = window.TYRANO && TYRANO.kag;
    if (kag && kag.key_mouse) kag.key_mouse.doAction(button.dataset.action, event);
  }));
  document.addEventListener("play", event => {
    if (event.target.tagName !== "AUDIO") return;
    document.querySelectorAll("audio").forEach(audio => {
      if (audio !== event.target) audio.pause();
    });
  }, true);
  document.addEventListener("click", event => {
    if (event.target.closest(".menu_close")) {
      document.querySelectorAll("audio").forEach(audio => audio.pause());
    }
  }, true);
  setInterval(() => {
    const kag = window.TYRANO && TYRANO.kag;
    const ready = !!(kag && kag.key_mouse && kag.stat && kag.stat.current_scenario);
    buttons.forEach(button => { button.disabled = !ready; });
    const auto = document.querySelector('[data-action="auto"]');
    if (auto && ready) auto.setAttribute("aria-pressed", String(!!kag.stat.is_auto));
    // The voice replay controls are for reviewing a finished line: do not mix
    // browser audio with an active Tyrano voice (which may be waiting at [wse]).
    document.querySelectorAll(".m0-replay audio").forEach(audio => {
      const playing = !!(kag && kag.tmp && kag.tmp.is_vo_play);
      audio.style.pointerEvents = playing ? "none" : "auto";
      audio.style.opacity = playing ? ".45" : "1";
      if (playing) audio.pause();
    });
  }, 200);
});
"""


def index_html() -> str:
    source = (ENGINE / "index.html").read_text(encoding="utf-8-sig")
    dependencies = []
    pattern = r'<script\b[^>]*\bsrc="([^"]+)"[^>]*></script>|<link\b[^>]*\bhref="([^"]+)"[^>]*>'
    for match in re.finditer(pattern, source, flags=re.IGNORECASE):
        script, stylesheet = match.groups()
        reference = script or stylesheet
        if reference == "./data/system/KeyConfig.js":
            dependencies.append('<script src="/data/system/KeyConfig.js"></script>')
            continue
        if not reference.startswith("./tyrano/") or ".." in reference.split("/"):
            continue
        if not (ENGINE / reference[2:]).is_file():
            raise FileNotFoundError(f"Missing installed engine dependency: {reference}")
        path = html.escape(reference[1:], quote=True)
        dependencies.append(f'<script src="{path}"></script>' if script else
                            f'<link href="{path}" rel="stylesheet">')
    if not dependencies:
        raise ValueError("Could not find Tyrano dependencies in the installed index.html.")
    assets = "\n".join(dependencies)
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>{TITLE}</title>
{assets}
<style>
html,body{{margin:0;background:#08131f;color:#f4f0e7;font-family:sans-serif}}
#m0-toolbar{{position:fixed;top:12px;left:16px;right:16px;z-index:999999;display:flex;align-items:center;gap:8px;pointer-events:none}}
#m0-toolbar strong{{margin-right:auto;padding:9px 14px;border-radius:8px;background:#08131fcc;letter-spacing:.08em}}
#m0-toolbar button{{pointer-events:auto;border:1px solid #adbec955;border-radius:8px;background:#08131fcc;color:#f4f0e7;padding:9px 13px;cursor:pointer}}
#m0-toolbar button[aria-pressed=true]{{background:#315865}}
#m0-toolbar button:disabled{{opacity:.45}}
button:focus-visible,a:focus-visible{{outline:3px solid #e1c581}}
@media(max-width:650px){{#m0-toolbar{{gap:4px;top:6px;left:6px;right:6px}}#m0-toolbar strong{{font-size:12px;padding:7px}}#m0-toolbar button{{font-size:11px;padding:7px}}}}
</style><script src="/data/ui/player.js"></script></head>
<body oncontextmenu="return false">
<nav id="m0-toolbar" aria-label="鑑賞操作"><strong>{TITLE}</strong>
<button data-action="auto" aria-pressed="false" disabled>自動送り</button>
<button data-action="backlog" disabled>本文・音声</button>
<button data-action="save" disabled>保存</button><button data-action="load" disabled>読込</button></nav>
<div id="tyrano_base" class="tyrano_base" style="overflow:hidden" unselectable="on"></div>
<div id="vchat_base" class="vchat_base" style="overflow:hidden"></div>
<div class="remodal-bg"></div><div class="remodal" data-remodal-id="modal"
data-remodal-options="hashTracking:false,closeOnEscape:false,closeOnOutsideClick:false">
<h1 class="remodal_title"></h1><p class="remodal_txt"></p>
<button data-remodal-action="cancel" id="remodal-cancel" class="remodal-cancel">戻る</button>
<button data-remodal-action="confirm" id="remodal-confirm" class="remodal-confirm">OK</button></div>
</body></html>
"""


def backlog_html() -> str:
    players = "".join(f'<label style="display:block;margin:12px 0">{label}<br>'
                      f'<audio controls preload="none" src="/data/sound/{voice}.wav" '
                      'style="width:100%"></audio></label>'
                      for voice, label in zip(VOICES, ("通常の語り", "感情比較：嬉しい", "感情比較：心配")))
    return f"""<section id="menu_backlog_wrapper" style="position:absolute;inset:0;background:#08131ff5;color:#f4f0e7;padding:76px 36px 28px;box-sizing:border-box">
<button class="menu_close" style="float:right;padding:12px 24px">閉じる</button>
<h1 style="margin:0 0 24px">本文・音声</h1>
<div style="display:grid;grid-template-columns:2fr 1fr;gap:28px;height:510px">
<div class="log_body" style="position:static;box-sizing:border-box;width:auto;height:auto;overflow:auto;line-height:1.9;font-size:24px;padding:20px;background:#18304080"></div>
<aside class="m0-replay" style="overflow:auto"><h2 style="font-size:22px">語りを聴く</h2>
<p style="font-size:16px;line-height:1.7">本編の語りが終わると再生できます。感情比較では、同じ本文を異なる声の調子で試聴できます。</p>{players}</aside>
</div></section>
"""


def scenario(text: str, sprite_size: tuple[int, int]) -> str:
    sprite_width = min(500, round(610 * sprite_size[0] / sprite_size[1]))
    passage = encode_text(text)
    return f"""; Generated M0 scene. All prose is escaped as literal Tyrano text.
[title name="{TITLE}"]
[hidemenubutton]
[sysview type="backlog" storage="./data/ui/backlog.html"]
[bg storage="lighthouse.png" time=0]
[chara_new name="mina" storage="character.png" jname="ミナ"]
[chara_show name="mina" left=700 top=40 width={sprite_width} height=610 time=0]
[ptext name="speaker_name" layer=message0 x=52 y=458 size=23 color=0xE1C581 text=""]
[chara_config ptext="speaker_name" talk_focus="none"]
[position layer=message0 left=20 top=446 width=1240 height=254 margint=50 marginl=28 marginr=28 marginb=18]
[layopt layer=message0 visible=true]
[deffont size=26 color=0xF4F0E7]
_波音の届く灯台で、二人の足が止まる。[r]
_クリックまたは Enter で進みます。音声が終わるまで次の場面には進みません。[r]
[link target="scene"]【はじめから】[endlink]
[if exp="sf.system.autosave === true"]
_　[link target="resume"]【続きから】[endlink]
[endif]
[s]

*resume
[autoload]
[s]

*scene
[cm]
[vostop]
[chara_ptext name=""]
[autosave title="灯台・語りの直前"]
[voconfig sebuf=1 name="ナレーター" vostorage="clone_neutral.wav" number=1 waittime=500 preload=false]
[vostart]
[chara_ptext name="ナレーター"]
{passage}
[wse]
[p]
[vostop]
[cm]
[chara_ptext name=""]
_灯台の夜に、ひと呼吸。[r]
_一場面の鑑賞が終わりました。「本文・音声」から語りを聴き直せます。[r]
[link target="scene"]【もう一度】[endlink]
_　[link target="compare"]【感情の違いを聴く】[endlink]
[s]

*compare
[cm]
[chara_ptext name=""]
_ここからは、同じ本文を使った感情比較です。[r]
[link target="happy"]【嬉しそうな語り】[endlink]
_　[link target="worried"]【心配そうな語り】[endlink]
_　[link target="scene"]【通常の語り】[endlink]
[s]
*happy
[voconfig sebuf=1 name="ナレーター" vostorage="clone_happy.wav" number=1 preload=false]
[jump target="compare_play"]
*worried
[voconfig sebuf=1 name="ナレーター" vostorage="clone_worried.wav" number=1 preload=false]
[jump target="compare_play"]
*compare_play
[cm]
[vostart]
[chara_ptext name="ナレーター"]
{passage}
[wse]
[p]
[vostop]
[jump target="compare"]
"""


def build(output: Path, image_dir: Path, background_dir: Path, tts_dir: Path, llm_dir: Path) -> dict:
    output = output.resolve()
    if not output.is_relative_to((ROOT / "private").resolve()) or output == ROOT / "private":
        raise ValueError("M0 scene output must be a dedicated subdirectory of project/private.")
    if output.exists():
        raise FileExistsError(f"Use a new scene output directory: {output}")
    original = (llm_dir / "narrative.txt").read_text(encoding="utf-8-sig")
    tts = json.loads((tts_dir / "tts-result.json").read_text(encoding="utf-8-sig"))
    if tts.get("status") != "passed":
        raise ValueError("TTS generation must pass before building a playable scene.")
    samples = {sample["id"]: sample for sample in tts["samples"]}
    display = samples["clone_neutral"]["display_text"]
    if not isinstance(display, str) or not display.strip():
        raise ValueError("TTS display_text is empty.")
    normalize = lambda value: " ".join(value.split())
    if normalize(display) != normalize(original):
        raise ValueError("The narrator's TTS display_text must match the source LLM narrative.")
    if any(normalize(samples[voice]["display_text"]) != normalize(display) for voice in VOICES):
        raise ValueError("The emotion comparisons must use the same display text.")
    assets = {"data/fgimage/character.png": image_dir / "character.png",
              "data/bgimage/lighthouse.png": background_dir / "image.png"}
    dimensions = png_size(assets["data/fgimage/character.png"])
    png_size(assets["data/bgimage/lighthouse.png"])
    for voice in VOICES:
        source = tts_dir / f"{voice}.wav"
        with source.open("rb") as stream:
            header = stream.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"Not a RIFF WAVE file: {source}")
        assets[f"data/sound/{voice}.wav"] = source
    output.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256((display + str(output)).encode("utf-8")).hexdigest()[:16]
    private_root = (ROOT / "private").resolve()
    staging = (output.parent / f".m0-scene-{uuid4().hex}").resolve()
    if not staging.is_relative_to(private_root) or staging.parent != output.parent:
        raise ValueError("Invalid staging directory.")
    # TemporaryDirectory uses a private Windows ACL. Normal mkdir preserves the
    # parent's ACL so the user's separate preview process can read the result.
    staging.mkdir()
    published = False
    try:
        for relative, source in assets.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        write_text(staging, "index.html", index_html())
        write_text(staging, "data/system/Config.tjs", config_text("auto_drama_m0_" + fingerprint))
        write_text(staging, "data/system/KeyConfig.js", KEY_CONFIG)
        write_text(staging, "data/scenario/first.ks", scenario(display, dimensions))
        write_text(staging, "data/scenario/make.ks", "[return]\n")
        write_text(staging, "data/ui/player.js", PLAYER_JS)
        write_text(staging, "data/ui/backlog.html", backlog_html())
        write_text(staging, "narrative-source.txt", original)
        files = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "narrative-source.txt":
                files[path.relative_to(staging).as_posix()] = {
                    "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest = {
            "schema_version": 1, "status": "built", "title": TITLE,
            "engine_included": False, "engine_route": "/tyrano/", "engine_install": "tyranoscript/tyrano",
            "local_only": True, "files": files, "display_text": display, "speaker": "ナレーター",
            "tts_samples": [{"id": voice, "display_text": samples[voice]["display_text"],
                             "voice_emotion": samples[voice]["voice_emotion"]} for voice in VOICES],
            "checks": {"narrative_tts_match": True, "input_headers": True,
                       "browser_playback_verified": False},
            "serve_command": f'python scripts/m0/serve_scene.py --scene-dir "{output}" --port 8090',
        }
        write_text(staging, "scene-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        # Atomic publish: the server never needs to observe a half-built scene.
        staging.rename(output)
        published = True
    finally:
        if not published and staging.exists():
            resolved_staging = staging.resolve(strict=True)
            if resolved_staging != staging or not resolved_staging.is_relative_to(private_root):
                raise ValueError("Refusing to remove a staging directory outside project/private.")
            shutil.rmtree(staging)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("output-dir", "image-dir", "background-dir", "tts-dir", "llm-dir"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output_dir, args.image_dir, args.background_dir, args.tts_dir, args.llm_dir)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir.resolve()),
                      "engine_included": False, "files": len(result["files"])}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
