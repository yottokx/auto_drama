"""Application-owned Tyrano V520 launcher; no engine files are redistributed."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .presentation import MESSAGE_WINDOW, STAGE

# Order follows the separately installed V520 entry point. These are references,
# not copies of the runtime or the sample game's configuration/scenarios.
ENGINE_SCRIPTS = (
    "libs/jquery-3.6.0.min.js", "libs/jquery-migrate-1.4.1.js",
    "libs/jquery-ui/jquery-ui.min.js", "libs/jquery.a3d.js", "libs/jsrender.min.js",
    "libs/alertify/alertify.min.js", "libs/remodal/remodal.js", "libs/html2canvas.js",
    "lang.js", "libs.js", "tyrano.js", "tyrano.base.js",
    "plugins/kag/kag.js", "plugins/kag/kag.event.js", "plugins/kag/kag.key_mouse.js",
    "plugins/kag/kag.layer.js", "plugins/kag/kag.menu.js", "plugins/kag/kag.parser.js",
    "plugins/kag/kag.rider.js", "plugins/kag/kag.studio_v6.js",
    "plugins/kag/kag.tag_audio.js", "plugins/kag/kag.tag_camera.js",
    "plugins/kag/kag.tag_ext.js", "plugins/kag/kag.tag_system.js",
    "plugins/kag/kag.tag_vchat.js", "plugins/kag/kag.tag_ar.js",
    "plugins/kag/kag.tag_three.js", "plugins/kag/kag.tag.js",
    "libs/textillate/assets/jquery.lettering.js", "libs/textillate/jquery.textillate.js",
    "libs/jquery.touchSwipe.min.js", "libs/howler.js", "libs/jsQR.js",
    "libs/anime.min.js", "libs/lz-string.min.js",
)
ENGINE_STYLES = (
    "tyrano.css", "libs/jquery-ui/jquery-ui.css", "libs/alertify/alertify.core.css",
    "libs/alertify/alertify.default.css", "libs/remodal/remodal.css",
    "libs/remodal/remodal-default-theme.css", "libs/textillate/assets/animate.css",
)
KEY_CONFIG = b'''window.__tyrano_key_config = {
  system_key_event: "true", system_mouse_event: "false",
  key: {Enter: "next", Escape: "cancel -a", a: "auto"},
  mouse: {}, gesture: {}, gamepad: {button: {}, stick: {}}
};
'''

# Keep the launcher and its assets on the same loaded code revision. Reading
# assets at export time lets a still-running pre-update server combine its old
# Python-generated HTML with newly edited JavaScript, producing an unusable ZIP.
_DIRECTORY = Path(__file__).parent
_PLAYER_ASSETS = {
    "data/others/auto_drama_player.js": (_DIRECTORY / "player.js").read_text("utf-8").encode(),
    "data/others/auto_drama_playback.js": (_DIRECTORY / "playback.js").read_text("utf-8").encode(),
    "data/others/auto_drama_player.css": (_DIRECTORY / "player.css").read_text("utf-8").encode(),
}


def player_config(script_bytes: bytes) -> bytes:
    # A separate storage namespace per adopted script prevents cross-project or
    # cross-revision saves from resuming a different generated chapter.
    # Saved engine state includes portrait coordinates. A presentation revision
    # must not restore coordinates from an earlier portrait layout.
    fingerprint = hashlib.sha256(b"portrait-layout-v3-player-m4\0" + script_bytes).hexdigest()
    window, padding = MESSAGE_WINDOW, MESSAGE_WINDOW["padding"]
    values = {
        "global.config_version": "5.00", "System.title": "AIオートドラマ 鑑賞",
        "projectID": f"auto_drama_{fingerprint}", "game_version": "1.0",
        "ScreenRatio": "fix", "ScreenCentering": "true", "patch_apply_auto": "false",
        "useCamera": "false", "use3D": "false", "KeepSpaceInParameterValue": "3",
        "scWidth": str(STAGE["width"]), "scHeight": str(STAGE["height"]), "vchat": "false",
        "vchatMenuVisible": "false", "chSpeed": "20", "defaultChEffect": "none",
        "defaultChEffectSpeed": "0.2s", "skipSpeed": "30", "skipEffectIgnore": "true",
        "autoSpeed": "1200", "autoSpeedWithText": "40", "autoClickStop": "true",
        "cursorDefault": "default", "mediaFormatDefault": "wav", "defaultBgmVolume": "100",
        "defaultSeVolume": "100", "defaultMovieVolume": "100", "defaultBgmSlotNum": "1",
        "defaultSoundSlotNum": "3", "configVisible": "false", "configLeft": "-1",
        "configTop": "-1", "configSave": "webstorage", "configThumbnail": "false",
        "configThumbnailQuality": "middle", "configThumbnailScale": "0.125",
        "configSaveSlotNum": "1", "configSaveDateFormat": "yyyy/M/d h:mm:ss",
        "configSaveOverwrite": "false", "maxBackLogNum": "500", "autoRecordLabel": "false",
        "unReadTextSkip": "false", "alreadyReadTextColor": "0xffffff",
        "numCharacterLayers": "3", "numMessageLayers": "2",
        "initialMessageLayerVisible": "true", "marginL": str(padding["left"]), "marginT": str(padding["top"]),
        "marginR": str(padding["right"]), "marginB": str(padding["bottom"]), "ml": str(window["left"]),
        "mt": str(window["top"]), "mw": str(window["width"]), "mh": str(window["height"]),
        "debugMenu.visible": "false", "frameColor": "0x" + window["background"][1:],
        "frameOpacity": str(round(window["opacity"] * 255)), "defaultAutoReturn": "true", "marginRCh": "1",
        "defaultFontSize": str(window["font_size"]), "defaultLineSpacing": str(window["line_spacing"]), "defaultPitch": "0",
        "userFace": "sans-serif", "defaultChColor": "0x" + window["color"][1:], "defaultBold": "false",
        "defaultRubySize": "10", "defaultRubyOffset": "-2", "defaultAntialiased": "1",
        "defaultShadow": "false", "defaultShadowColor": "0x000000", "defaultEdge": "false",
        "defaultEdgeColor": "0x000000", "defaultLinkColor": "0x95cabc",
        "defaultLinkOpacity": "64", "vertical": "false", "useKeyFocus": "false",
        "useGamepad": "false", "useCloseConfirm": "false", "offscreenClickable": "false",
    }
    return ("// Auto Drama player settings\n" + "\n".join(
        f";{key} = {value};" for key, value in values.items()
    ) + "\n").encode("utf-8")


def player_html() -> bytes:
    dependencies = [f'<link rel="stylesheet" href="./tyrano/{path}">' for path in ENGINE_STYLES]
    dependencies.append('<script src="./data/system/KeyConfig.js"></script>')
    dependencies.extend(f'<script src="./tyrano/{path}"></script>' for path in ENGINE_SCRIPTS)
    return ('''<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>AIオートドラマ 鑑賞</title>
''' + "\n".join(dependencies) + '''
<link rel="stylesheet" href="./data/others/auto_drama_player.css">
<script defer src="./data/others/auto_drama_playback.js"></script>
<script defer src="./data/others/auto_drama_player.js"></script></head>
<body>
<nav id="ad-toolbar" aria-label="鑑賞操作">
<button id="ad-auto" aria-pressed="false" disabled>オート</button>
<button id="ad-save" disabled>途中保存</button><button id="ad-load" disabled>続きから</button>
<button id="ad-backlog" disabled>バックログ</button>
<label>文字 <select id="ad-font" disabled><option value="20">小</option>
<option value="24" selected>標準</option><option value="28">大</option></select></label>
<label>音量 <input id="ad-volume" type="range" min="0" max="100" value="100" disabled></label>
<span id="ad-status" role="status"></span></nav>
<div id="tyrano_base" class="tyrano_base" style="overflow:hidden"></div>
<div id="vchat_base" class="vchat_base" style="overflow:hidden"></div>
<section id="ad-start" aria-label="再生開始"><div><h1>鑑賞をはじめる</h1>
<p>音声が流れます。メッセージ欄のクリックまたは Enter で全文表示、もう一度で次へ進みます。</p>
<button id="ad-start-button" disabled>読み込み中…</button>
<button id="ad-resume-button" hidden disabled>保存した位置から再開</button>
<p>「途中保存」はこのブラウザーに保存します。</p></div></section>
<section id="ad-chapter-end" aria-label="章の終了" hidden>
<p id="ad-next-status" role="status"></p>
<button id="ad-next-button" hidden>次の章へ進む</button>
</section>
<dialog id="ad-log" aria-labelledby="ad-log-title"><button id="ad-log-close">閉じる</button>
<h2 id="ad-log-title">バックログ</h2><p>表示済みの本文を確認し、台詞を聴き直せます。閉じると再開します。</p><div id="ad-log-items"></div></dialog>
<div class="remodal-bg"></div><div class="remodal" data-remodal-id="modal"
data-remodal-options="hashTracking:false,closeOnEscape:false,closeOnOutsideClick:false">
<h1 class="remodal_title"></h1><p class="remodal_txt"></p>
<button data-remodal-action="cancel" id="remodal-cancel" class="remodal-cancel">戻る</button>
<button data-remodal-action="confirm" id="remodal-confirm" class="remodal-confirm">OK</button>
</div></body></html>
''').encode("utf-8")


def player_files(script_bytes: bytes) -> dict[str, bytes]:
    return {
        "index.html": player_html(),
        "data/system/Config.tjs": player_config(script_bytes),
        "data/system/KeyConfig.js": KEY_CONFIG,
        **_PLAYER_ASSETS,
    }
