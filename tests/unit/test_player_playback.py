"""Playback interaction regressions using native Tyrano timing/tag handlers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "tyranoscript/tyrano/plugins/kag"


HARNESS = r"""
const fs = require('fs'), path = require('path'), vm = require('vm');
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
let now = 0, timerId = 0;
const timers = [], visited = [], stopCalls = [], effects = [], listeners = new Map();
const schedule = (fn, delay = 0) => {
  const item = {id: ++timerId, due: now + Number(delay), fn, cancelled: false};
  timers.push(item); return item.id;
};
const clearTimer = id => { const timer = timers.find(item => item.id === id);
  if (timer) timer.cancelled = true; };
const tick = () => {
  timers.sort((a, b) => a.due - b.due || a.id - b.id);
  const item = timers.shift();
  if (!item) throw Error('No pending timer');
  if (!item.cancelled) { now = item.due; item.fn(); }
};
function until(predicate) {
  for (let i = 0; i < 5000; i++) { if (predicate()) return; tick(); }
  throw Error('Timer loop did not reach the expected state');
}
const query = () => ({find() {return this}, setStyleMap() {return this},
  css() {return this}, addClass() {return this}, removeClass() {return this}});
query.setTimeout = schedule;
query.trim = value => String(value).trim();
let shown = [], eventLayer = false;
const context = {console, $: query, setTimeout: schedule, clearTimeout: clearTimer,
  performance: {now: () => now}, tyrano: {plugin: {kag: {tag: {}}}}, TYRANO: {kag: {}}};
context.window = context;
vm.createContext(context);
for (const file of ['kag.tag.js', 'kag.tag_audio.js'])
  vm.runInContext(fs.readFileSync(path.join(process.argv[1], file), 'utf8'), context);
const definitions = context.tyrano.plugin.kag.tag;
const tag = (name, pm = {}) => ({name, pm});
const lineTags = (id, {voice = false, name = false, multi = false, afterWait = false} = {}) => [
  tag('label', {label_name: 'utterance_' + id}), tag('cm'),
  ...(name ? [tag('text', {val: '父親'}), tag('r')] : []),
  ...(voice ? [tag('playse', {buf: '1', storage: id + '.wav'})] : []),
  tag('text', {val: id === 'one' ? '今日も一緒に朝ご飯を食べよう。' : '次の地の文が続く。'}), tag('r'),
  ...(multi ? [tag('text', {val: 'まだ表示していない二行目。'}), tag('r')] : []),
  ...(voice ? [tag('wse')] : []),
  ...(afterWait ? [tag('wait', {time: 100}), tag('effect', {value: 'after'})] : []),
  tag('p'),
];
const firstOptions = {voice: request.voice !== false, name: !!request.name,
  multi: !!request.multi, afterWait: !!request.afterWait};
const array = [...lineTags('one', firstOptions), ...lineTags('two', {voice: !!request.secondVoice}),
  ...lineTags('three'), tag('s')];
const script = {utterances: [
  {id: 'one', speaker_id: 'father', display_text: '今日も一緒に朝ご飯を食べよう。' +
    (request.multi ? '\nまだ表示していない二行目。' : ''), audio_asset_id: 'voice'},
  {id: 'two', speaker_id: null, display_text: '次の地の文が続く。'},
  {id: 'three', speaker_id: null, display_text: '次の地の文が続く。'},
], characters: [{id: 'father', name: '父親'}], assets: []};
const k = {
  stat: {current_scenario: 'first.ks', is_adding_text: false, is_click_text: false,
    is_auto: false, is_skip: false, is_wait: false, is_stop: false, is_strong_stop: false,
    is_hide_message: false, is_wait_auto: false, flag_ref_page: false, current_se: {},
    fuki: {active: false}, font: {}, current_message_str: ''},
  tmp: {is_se_play: false, is_se_play_wait: false, is_vo_play: false, is_vo_play_wait: false,
    map_se: {}, map_bgm: {}, popopo: {key: null}, ready_audio: true},
  config: {autoSpeed: '1000', autoSpeedWithText: '0', autoClickStop: 'true', skipSpeed: '30'},
  on(event, fn) { if (!listeners.has(event)) listeners.set(event, []);
    listeners.get(event).push(fn); },
  off(event, fn) { listeners.set(event, (listeners.get(event) || []).filter(x => x !== fn)); },
  once(event, fn) { const wrapped = (...args) => {this.off(event, wrapped); fn(...args)};
    this.on(event, wrapped); },
  trigger(event, args) { for (const fn of [...(listeners.get(event) || [])]) fn(args); },
  weaklyStop() { this.stat.is_stop = true; },
  cancelWeakStop() { this.stat.is_stop = false; },
  stronglyStop() { this.stat.is_strong_stop = true; },
  cancelStrongStop() { this.stat.is_strong_stop = false; },
  waitClick() { eventLayer = true; this.cancelStrongStop(); this.cancelWeakStop(); },
  setAuto(value) { this.stat.is_auto = value; if (!value) this.stat.is_wait_auto = false; },
  setSkip(value) { this.stat.is_skip = value; },
  readyAudio() {},
  getMessageInnerLayer() { return {find() {return {chars: shown.flatMap(part => part.chars)}}}; },
  layer: {showEventLayer() {eventLayer = true}, hideEventLayer() {eventLayer = false},
    layer_event: {css() {return eventLayer ? 'block' : 'none'}},
    showMessageLayers() {}, hideMessageLayers() {}},
};
context.TYRANO = {kag: k};
const handlers = Object.fromEntries(Object.entries(definitions).map(([name, value]) =>
  [name, Object.assign(Object.create(value), {kag: k})]));
const currentTag = () => k.ftag.array_tag[k.ftag.current_order_index];
const currentLine = () => {
  for (let i = k.ftag.current_order_index; i >= 0; i--) {
    const item = k.ftag.array_tag[i];
    if (item.name === 'label') return item.pm.label_name.replace('utterance_', '');
  }
  return null;
};
k.ftag = {array_tag: array, current_order_index: -1, master_tag: handlers,
  showNextImg() {}, hideNextImg() {},
  startTag(name, pm = {}) {
    const handler = handlers[name];
    if (!handler) throw Error('Unknown tag ' + name);
    if (name === 'stopse') stopCalls.push({...pm});
    k.trigger('tag-' + name, {target: pm, is_next_order: false});
    handler.start({...handler.pm, ...pm});
  },
  nextOrder() {
    if (k.stat.is_strong_stop || k.stat.is_adding_text) return;
    eventLayer = false;
    this.current_order_index++;
    if (this.current_order_index >= this.array_tag.length) return;
    const item = currentTag(); visited.push({index: this.current_order_index,
      name: item.name, line: currentLine()});
    k.trigger('nextorder', {scenario: k.stat.current_scenario, index: this.current_order_index});
    k.trigger('tag-' + item.name, {target: item.pm, in_scenario: true, is_macro: false});
    handlers[item.name].start({...handlers[item.name].pm, ...item.pm});
  },
};
k.key_mouse = {
  util: {canClick() {return eventLayer}, canShowMenu() {return eventLayer && !k.stat.is_wait},
    isMenuDisplayed() {return false}, isRemodalDisplayed() {return false},
    clearSkipAndAuto() { k.setSkip(false); k.setAuto(false); }},
  next() {
    if (!this.util.canClick()) return false;
    this.util.clearSkipAndAuto();
    if (k.stat.is_adding_text) k.stat.is_click_text = true;
    else if (!k.stat.is_click_text && !k.stat.is_stop) k.ftag.nextOrder();
    return true;
  },
};
const audioObjects = [];
function makeAudio() {
  const events = new Map();
  const audio = {active: true, stops: 0, unloaded: false,
    playing() {return this.active},
    stop() {this.active = false; this.stops++; return this},
    pause() {this.active = false; return this},
    play() {this.active = true; return this},
    unload() {this.unloaded = true; return null},
    on(name, callback) { if (!events.has(name)) events.set(name, []);
      events.get(name).push({callback, once: false}); return this; },
    once(name, callback) { if (!events.has(name)) events.set(name, []);
      events.get(name).push({callback, once: true}); return this; },
    off(name, callback) { if (!callback) events.set(name, []);
      else events.set(name, (events.get(name) || []).filter(x => x.callback !== callback));
      return this; },
    emitEnd() {
      this.active = false;
      // Howler queues a copy of every callback before removing once listeners.
      // A later off()/unload() cannot revoke these scheduled callbacks.
      for (const item of [...(events.get('end') || [])].reverse()) {
        schedule(() => item.callback.call(this), 0);
        if (item.once) this.off('end', item.callback);
      }
    },
  };
  audio.once('end', () => {
    k.tmp.is_se_play = false;
    if (k.tmp.is_se_play_wait && !Object.values(k.tmp.map_se).some(value => value.playing())) {
      k.tmp.is_se_play_wait = false;
      k.ftag.nextOrder();
    }
  });
  audioObjects.push(audio); return audio;
}
handlers.playse.start = () => {
  k.tmp.is_se_play = true; k.tmp.map_se['1'] = makeAudio(); k.ftag.nextOrder();
};
handlers.autostop = {start() { k.setAuto(false); }};
handlers.label = {start() { k.ftag.nextOrder(); }};
handlers.cm = {start() { shown = []; k.ftag.nextOrder(); }};
handlers.r = {start() { k.ftag.nextOrder(); }};
handlers.effect = {start(pm) {effects.push(pm.value); k.ftag.nextOrder(); }};
handlers.s = {start() { k.stronglyStop(); }};
// Use actual native text timer recursion and completion; stub only DOM rendering.
handlers.text.start = function(pm) {
  const chars = [...pm.val].map(value => ({value, visible: false}));
  const collection = {length: chars.length, chars, eq(index) {return chars[index]}};
  shown.push(collection); k.stat.current_message_str = pm.val;
  k.stat.is_adding_text = true; k.tmp.processed_click_interrupt = false; k.tmp.ch_speed = 20;
  k.waitClick('text'); this.addOneChar(0, collection, query(), query());
};
handlers.text.makeOneCharVisible = value => {value.visible = true};
handlers.text.makeAllCharsVisible = collection => collection.chars.forEach(x => {x.visible = true});
handlers.text.stopLipSyncWithText = () => {};
handlers.text.getMessageConfig = () => '0';
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
const playback = context.AutoDramaPlayback(k, script);
const displayedIds = () => playback.displayedLines().map(line => line.id);
const text = () => shown.map(part => part.chars.filter(x => x.visible).map(x => x.value).join(''));
const state = () => ({line: currentLine(), tag: currentTag()?.name, index: k.ftag.current_order_index,
  text: text(), adding: k.stat.is_adding_text, playing: k.tmp.is_se_play,
  wait: k.stat.is_wait, displayed: displayedIds(), stops: stopCalls.length,
  effects: [...effects], secondStarts: visited.filter(x => x.name === 'label' && x.line === 'two').length,
  thirdStarts: visited.filter(x => x.name === 'label' && x.line === 'three').length});
k.ftag.nextOrder();
const result = {};
if (request.case === 'reveal') {
  result.before = state(); playback.advance(); result.clicked = state();
  until(() => currentTag()?.name === 'wse' || currentTag()?.name === 'p');
  result.revealed = state();
  playback.advance(); result.secondClick = state();
  until(() => currentLine() === 'two'); result.next = state();
} else if (request.case === 'race' || request.case === 'advance' || request.case === 'wait') {
  until(() => currentTag()?.name === (request.voice === false ? 'p' : 'wse'));
  result.before = state();
  if (request.case === 'race') audioObjects[0].emitEnd();
  playback.advance();
  for (let i = 0; i < 10; i++) playback.advance();
  result.clicked = state();
  if (request.afterWait) {
    until(() => currentTag()?.name === 'wait'); result.duringWait = state();
    for (let i = 0; i < 10; i++) playback.advance();
    result.clickedDuringWait = state();
  }
  until(() => currentLine() === 'two'); result.next = state();
  until(() => currentTag()?.name === (request.secondVoice ? 'wse' : 'p'));
  result.nextFinished = state();
} else if (request.case === 'load') {
  until(() => currentTag()?.name === 'wse'); playback.advance();
  if (!request.loadMode || request.loadMode === 'reset') playback.reset();
  else if (request.loadMode === 'stat') k.stat = {...k.stat};
  else if (request.loadMode === 'tags') k.ftag.array_tag = [...k.ftag.array_tag];
  else if (request.loadMode === 'scenario') k.stat.current_scenario = 'restored.ks';
  k.trigger('load-before'); k.trigger('load-start');
  const restored = array.findIndex(item => item.name === 'label' && item.pm.label_name === 'utterance_three');
  k.ftag.current_order_index = restored;
  k.stat.is_adding_text = false; k.stat.is_click_text = false;
  k.tmp.is_se_play = false; k.tmp.is_se_play_wait = false;
  result.restored = state();
  while (timers.length) tick();
  result.afterTimers = state();
} else if (request.case === 'natural') {
  until(() => currentTag()?.name === 'wse'); result.before = state();
  audioObjects[0].emitEnd(); until(() => currentTag()?.name === 'p'); result.ended = state();
  playback.advance(); until(() => currentLine() === 'two'); result.next = state();
} else if (request.case === 'log') {
  result.partial = {canOpen: playback.canOpenLog(), ids: displayedIds()};
  until(() => currentTag()?.name === 'wse');
  result.full = {canOpen: playback.canOpenLog(), ids: displayedIds()};
  const before = k.ftag.current_order_index;
  playback.openLog(); audioObjects[0].emitEnd();
  while (timers.length) tick();
  result.open = {sameIndex: before === k.ftag.current_order_index, ...state()};
  playback.closeLog();
  result.closed = state();
} else if (request.case === 'logPartial') {
  const before = k.ftag.current_order_index;
  result.opened = playback.openLog();
  while (timers.length) tick();
  result.open = {sameIndex: before === k.ftag.current_order_index, ...state()};
  result.voicePaused = !audioObjects[0].playing();
  playback.closeLog();
  result.closed = state(); result.voiceResumed = audioObjects[0].playing();
}
result.stopCalls = stopCalls;
process.stdout.write(JSON.stringify(result));
"""


def run_playback(case: str, **options) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for native playback interaction tests")
    if not (ENGINE / "kag.tag.js").is_file():
        pytest.skip("Install the separately supplied Tyrano engine for native tag tests")
    completed = subprocess.run(
        [node, "-e", HARNESS, str(ENGINE), str(ROOT / "packages/tyrano_export/playback.js")],
        input=json.dumps({"case": case, **options}), text=True, encoding="utf-8",
        capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_first_click_reveals_text_without_stopping_voice_or_advancing():
    result = run_playback("reveal")
    assert result["before"]["adding"]
    assert result["clicked"]["line"] == "one"
    assert result["clicked"]["stops"] == 0
    assert result["clicked"]["text"] == ["今日も一緒に朝ご飯を食べよう。"]
    assert result["revealed"]["text"] == ["今日も一緒に朝ご飯を食べよう。"]
    assert result["revealed"]["line"] == "one"
    assert result["revealed"]["playing"]
    assert result["revealed"]["displayed"] == ["one"]
    assert result["next"]["line"] == "two"
    assert result["next"]["stops"] == 1
    assert result["next"]["adding"]


def test_reveal_latch_covers_speaker_name_and_all_body_lines():
    result = run_playback("reveal", name=True, multi=True)
    assert result["revealed"]["text"] == [
        "父親", "今日も一緒に朝ご飯を食べよう。", "まだ表示していない二行目。",
    ]
    assert result["revealed"]["line"] == "one"
    assert result["revealed"]["stops"] == 0
    assert result["next"]["adding"]


@pytest.mark.parametrize("case", ["advance", "race"])
def test_voice_interrupt_and_queued_end_advance_exactly_once_despite_rapid_clicks(case):
    result = run_playback(case)
    assert result["clicked"]["stops"] == 1
    assert result["stopCalls"] == [{"buf": "1", "stop": "true"}]
    assert result["next"]["line"] == "two"
    assert result["next"]["secondStarts"] == 1
    assert result["nextFinished"]["line"] == "two"
    assert result["nextFinished"]["thirdStarts"] == 0


def test_narration_page_advance_does_not_stop_audio_or_skip_next_text():
    result = run_playback("advance", voice=False)
    assert result["next"]["line"] == "two"
    assert result["next"]["adding"]
    assert result["nextFinished"]["thirdStarts"] == 0
    assert result["stopCalls"] == []


def test_queued_end_from_interrupted_voice_cannot_clear_following_voice_state():
    result = run_playback("race", secondVoice=True)
    assert result["nextFinished"]["line"] == "two"
    assert result["nextFinished"]["tag"] == "wse"
    assert result["nextFinished"]["playing"]
    assert result["nextFinished"]["thirdStarts"] == 0


def test_manual_voice_advance_executes_post_dialogue_wait_before_next_page():
    result = run_playback("wait", afterWait=True)
    assert result["duringWait"]["line"] == "one"
    assert result["duringWait"]["wait"]
    assert result["duringWait"]["secondStarts"] == 0
    assert result["clickedDuringWait"]["index"] == result["duringWait"]["index"]
    assert result["next"]["effects"] == ["after"]
    assert result["next"]["line"] == "two"
    assert result["nextFinished"]["thirdStarts"] == 0


@pytest.mark.parametrize("load_mode", ["reset", "stat", "tags", "scenario", "index"])
def test_reset_or_replaced_engine_state_invalidates_pending_click(load_mode):
    result = run_playback("load", loadMode=load_mode)
    assert result["restored"]["line"] == "three"
    assert result["afterTimers"]["index"] == result["restored"]["index"]
    assert result["afterTimers"]["line"] == "three"


def test_natural_voice_end_waits_for_click_and_manual_click_advances_once():
    result = run_playback("natural")
    assert result["ended"]["line"] == "one"
    assert result["ended"]["tag"] == "p"
    assert result["ended"]["displayed"] == ["one"]
    assert result["next"]["line"] == "two"
    assert result["next"]["secondStarts"] == 1


def test_log_contains_only_fully_displayed_text_and_does_not_advance_while_open():
    result = run_playback("log")
    assert result["partial"]["canOpen"]
    assert result["partial"]["ids"] == []
    assert result["full"]["canOpen"]
    assert result["full"]["ids"] == ["one"]
    assert result["open"]["sameIndex"]
    assert result["closed"]["line"] == "one"


def test_opening_log_during_typewriter_holds_progression_and_resumes_same_voice():
    result = run_playback("logPartial")
    assert result["opened"]
    assert result["open"]["sameIndex"]
    assert result["open"]["line"] == "one"
    assert result["voicePaused"]
    assert result["closed"]["line"] == "one"
    assert result["voiceResumed"]

