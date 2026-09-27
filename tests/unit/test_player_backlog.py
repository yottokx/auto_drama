"""The owned backlog pauses playback and exposes only already displayed story text."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "packages/tyrano_export/playback.js"

HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const program = JSON.parse(fs.readFileSync(0, 'utf8'));
const ctx = {window: {}};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), ctx);
vm.runInContext(`
const tag = (name, pm = {}) => ({name, pm});
const tags = [
  tag('text', {val: '作品タイトル'}), tag('p'),
  tag('label', {label_name: 'utterance_first'}), tag('cm'),
  tag('chara_ptext', {name: 'actor'}), tag('text', {val: '話者名'}), tag('r'),
  tag('playse', {buf: '1'}), tag('text', {val: '台詞の一行目'}), tag('r'),
  tag('text', {val: 'まだ表示していない二行目'}), tag('r'), tag('wse'),
  tag('wait', {time: '100'}), tag('p'),
  tag('label', {label_name: 'utterance_second'}), tag('cm'),
  tag('text', {val: '次の地の文'}), tag('r'), tag('p'),
  tag('cm'), tag('text', {val: 'この章はここまでです。'}), tag('s'),
];
const story = {
  utterances: [
    {id: 'first', speaker_id: 'actor', display_text: '台詞の一行目\\nまだ表示していない二行目'},
    {id: 'second', speaker_id: null, display_text: '次の地の文'},
  ],
  characters: [{id: 'actor', name: '話者名'}], assets: [],
};
const calls = {next: 0, pause: 0, play: 0, auto: 0, skip: 0};
let playing = true, canMenu = false;
const sound = {
  playing() {return playing;},
  pause() {calls.pause++; playing = false;},
  play() {calls.play++; playing = true;},
};
const k = {
  stat: {current_scenario: 'first.ks', is_auto: true, is_skip: true,
    is_adding_text: false, is_wait: false, is_strong_stop: false},
  tmp: {is_se_play: true, is_se_play_wait: true, map_se: {'1': sound}},
  ftag: {
    array_tag: tags, current_order_index: 12,
    nextOrder() {calls.next++; this.current_order_index++;},
    startTag(name, pm) {
      if (name !== 'autostop' || pm.next !== 'false') throw Error('Unexpected advancing tag');
      calls.auto++; k.stat.is_auto = false;
    },
  },
  setSkip(value) {calls.skip++; this.stat.is_skip = value;},
  key_mouse: {
    util: {canShowMenu() {return canMenu;}},
    next() {k.ftag.nextOrder(); return true;},
  },
};
const control = window.AutoDramaPlayback(k, story);
const ids = () => control.displayedLines().map(line => line.id);
`, ctx);
process.stdout.write(JSON.stringify(vm.runInContext(program, ctx)));
"""


def run(program):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to execute the player controller")
    result = subprocess.run(
        [node, "-e", HARNESS, str(SOURCE)], input=json.dumps(program), text=True,
        encoding="utf-8", capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("index,adding,expected", [
    (0, False, []),                   # Title, before any utterance.
    (5, False, []),                   # The speaker name is not the current dialogue.
    (8, False, []),                   # First body line finished; second remains undisplayed.
    (10, True, []),                   # Last body line is still typing.
    (10, False, ["first"]),          # Text complete, nextOrder may currently be suspended.
    (12, False, ["first"]),          # Text complete while the voice continues at wse.
    (13, False, ["first"]),          # A post-dialogue effect is still running.
    (14, False, ["first"]),
    (17, True, ["first"]),
    (17, False, ["first", "second"]),
    (19, False, ["first", "second"]),
    (21, True, ["first", "second"]),  # Chapter-closing text is not an utterance boundary.
    (22, False, ["first", "second"]),
])
def test_only_fully_displayed_dialogue_is_listed(index, adding, expected):
    result = run(f"k.ftag.current_order_index={index}; k.stat.is_adding_text={str(adding).lower()}; ids()")
    assert result == expected


def test_rewinding_reconstructs_history_instead_of_retaining_future_lines():
    result = run("""
k.ftag.current_order_index = 22;
const completed = ids();
k.ftag.current_order_index = 14;
const earlier = ids();
k.ftag.current_order_index = 8;
const partial = ids();
({completed, earlier, partial});
""")
    assert result == {"completed": ["first", "second"], "earlier": ["first"], "partial": []}


def test_audio_wait_opens_without_native_menu_and_pauses_only_the_playing_voice():
    result = run("""
const allowed = control.canOpenLog();
const opened = control.openLog();
const index = k.ftag.current_order_index;
const lines = ids();
({allowed, opened, index, lines, playing, auto: k.stat.is_auto, skip: k.stat.is_skip, calls});
""")
    assert result == {
        "allowed": True, "opened": True, "index": 12, "lines": ["first"],
        "playing": False, "auto": False, "skip": False,
        "calls": {"next": 0, "pause": 1, "play": 0, "auto": 1, "skip": 1},
    }


def test_typing_may_finish_without_revealing_future_lines_or_advancing_behind_log():
    result = run("""
k.ftag.current_order_index = 10; k.stat.is_adding_text = true; canMenu = true;
const opened = control.openLog();
const before = ids();
k.stat.is_adding_text = false;
k.ftag.nextOrder();
const after = ids(), heldIndex = k.ftag.current_order_index, heldNext = calls.next;
control.closeLog();
({opened, before, after, heldIndex, heldNext, index: k.ftag.current_order_index, calls});
""")
    assert result["opened"] is True
    assert result["before"] == [] and result["after"] == ["first"]
    assert result["heldIndex"] == 10 and result["heldNext"] == 0
    assert result["index"] == 11
    assert result["calls"]["next"] == result["calls"]["pause"] == result["calls"]["play"] == 1


def test_pending_advance_is_resumed_once_and_open_close_are_idempotent():
    result = run("""
control.openLog(); control.openLog();
k.ftag.nextOrder(); k.ftag.nextOrder();
const blocked = calls.next;
control.closeLog(); control.closeLog();
({blocked, index: k.ftag.current_order_index, auto: k.stat.is_auto, calls});
""")
    assert result["blocked"] == 0 and result["index"] == 13
    assert result["auto"] is False
    assert result["calls"]["pause"] == result["calls"]["play"] == result["calls"]["next"] == 1


def test_closing_audio_wait_resumes_voice_without_skipping_the_current_line():
    result = run("""
control.openLog(); control.closeLog();
const resumed = {index: k.ftag.current_order_index, next: calls.next, playing};
// The resumed Howl eventually ends, and native wse calls nextOrder itself.
playing = false; k.tmp.is_se_play = false; k.ftag.nextOrder();
({resumed, index: k.ftag.current_order_index, calls});
""")
    assert result["resumed"] == {"index": 12, "next": 0, "playing": True}
    assert result["index"] == 13 and result["calls"]["next"] == 1


def test_each_reopening_pauses_and_resumes_the_same_voice_once():
    result = run("""
control.openLog(); control.closeLog(); control.openLog(); control.closeLog();
({index: k.ftag.current_order_index, calls});
""")
    assert result["index"] == 12
    assert result["calls"]["pause"] == result["calls"]["play"] == 2
    assert result["calls"]["next"] == 0


def test_already_stopped_or_absent_voice_is_never_started_by_closing_backlog():
    result = run("""
playing = false; k.tmp.is_se_play = false;
control.openLog(); control.closeLog();
delete k.tmp.map_se['1'];
control.openLog(); control.closeLog();
({playing, calls});
""")
    assert result["playing"] is False
    assert result["calls"]["pause"] == result["calls"]["play"] == result["calls"]["next"] == 0


@pytest.mark.parametrize("change", [
    "k.stat = {...k.stat}",
    "k.ftag.array_tag = [...k.ftag.array_tag]",
    "k.stat.current_scenario = 'other.ks'",
])
def test_loading_or_replacing_story_state_discards_stale_audio_and_queued_advance(change):
    result = run(f"""
control.openLog(); k.ftag.nextOrder();
{change};
control.closeLog();
({{calls, playing}});
""")
    assert result["calls"]["play"] == result["calls"]["next"] == 0
    assert result["playing"] is False


def test_reset_discards_audio_and_pending_advance_but_releases_the_progress_guard():
    result = run("""
control.openLog(); k.ftag.nextOrder();
control.reset(); control.closeLog();
const discarded = {next: calls.next, play: calls.play, playing};
k.ftag.nextOrder();
({discarded, index: k.ftag.current_order_index, calls});
""")
    assert result["discarded"] == {"next": 0, "play": 0, "playing": False}
    assert result["index"] == 13 and result["calls"]["next"] == 1


def test_replaced_or_ended_voice_is_not_resurrected_on_close():
    result = run("""
control.openLog();
k.tmp.map_se['1'] = {playing() {return false;}, pause() {}, play() {throw Error('New voice resumed');}};
control.closeLog();
k.tmp.map_se['1'] = sound; playing = true; k.tmp.is_se_play = true;
control.openLog(); k.tmp.is_se_play = false; control.closeLog();
({calls, playing});
""")
    assert result["calls"]["play"] == result["calls"]["next"] == 0
    assert result["calls"]["pause"] == 2


@pytest.mark.parametrize("index", [7, 13])
def test_sound_loading_and_effect_waits_are_not_interrupted(index):
    result = run(f"""
k.ftag.current_order_index = {index};
k.stat.is_wait = {str(index == 13).lower()};
const allowed = control.canOpenLog(), opened = control.openLog();
({{allowed, opened, calls}});
""")
    assert result["allowed"] is False and result["opened"] is False
    assert result["calls"] == {"next": 0, "pause": 0, "play": 0, "auto": 0, "skip": 0}
