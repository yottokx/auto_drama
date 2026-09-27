"""Chapter-boundary, immutable-save and spoiler-free waiting UI regressions."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
class Element {
  constructor() {this.listeners = {}; this.children = []; this.value = '24'; this.hidden = true;}
  addEventListener(event, fn) {this.listeners[event] = fn;}
  querySelectorAll(selector) {return selector === 'button, select, input' ? this.children : [];}
  setAttribute(key, value) {this[key] = value;}
  close() {this.open = false; this.listeners.close?.();}
}
const elements = {}, handlers = {}, subscriptions = {}, requests = [], locations = [], storage = new Map();
const get = id => elements[id] ||= new Element();
get('ad-toolbar').children = ['ad-auto','ad-save','ad-load','ad-backlog','ad-font','ad-volume'].map(get);
const document = {getElementById:id => id === input.missingElement ? null : get(id),
  addEventListener(event, fn) {handlers[event] = fn;},
  querySelectorAll() {return [];}};
let tick, time = 0, nextResult = {status:'waiting',build_id:'first',next_build:null}, pending, saves = 0, loads = 0;
let hold = false, fail = false;
const story = {id:'script',characters:[],assets:[],utterances:[{id:'first_line',display_text:'READ'}]};
const tags = [{name:'label',pm:{label_name:'utterance_first_line'}}, {name:'text'}, {name:'p'},
  {name:'label',pm:{label_name:'auto_drama_chapter_end'}}, {name:'text'}, {name:'s'}];
const k = {config:{projectID:'script_hash'}, stat:{current_scenario:'first.ks',font:{},default_font:{}}, tmp:{},
  readyAudio() {}, on(event, fn) {subscriptions[event] = fn;},
  ftag:{array_tag:tags,current_order_index:2,nextOrder() {},
    startTag(name) {if (name === 'autostop') k.stat.is_auto = false;}},
  key_mouse:{next() {},util:{canShowMenu() {return true;}},
    qsave() {
      saves++;
      const value = {stat:k.stat,current_order_index:k.ftag.current_order_index - 1};
      storage.set(k.config.projectID + '_tyrano_quick_save', JSON.stringify(value));
      subscriptions['storage-quicksave']?.(); return true;
    },
    qload() {
      loads++; const data = JSON.parse(storage.get(k.config.projectID + '_tyrano_quick_save'));
      subscriptions['load-start']?.(); k.stat = data.stat;
      k.ftag.current_order_index = data.current_order_index + 1;
      subscriptions['load-complete']?.(); return true;
    }},
};
const context = {document, Date:{now:() => time}, setInterval(fn) {tick = fn;},
  $:{getStorage(key) {return storage.get(key);}},
  window:{TYRANO:{kag:k},localStorage:{getItem:key => storage.get(key)},
    location:{pathname:input.static ? '/static/index.html' : '/player/first/',assign:url => locations.push(url)}},
  fetch: async url => {
    requests.push(url);
    if (url === './script.json') {
      if (input.case === 'startup_pending') await new Promise(() => {});
      return {ok:input.case !== 'startup_script_error',json:async() => story};
    }
    if (url === './player-context.json' && input.case === 'startup_context_error') return {ok:false};
    if (url === './player-context.json') return {ok:true,json:async() => ({mode:'live',build_id:'first',
      project_id:'project',production_id:'series',storyline_id:'series',chapter_number:1,
      next_url:'/api/m3/builds/first/next'})};
    if (fail) throw Error('offline');
    if (hold) await new Promise(resolve => {pending = resolve;});
    return {ok:true,json:async() => nextResult};
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
handlers.DOMContentLoaded();
const flush = () => new Promise(resolve => setImmediate(resolve));
const click = id => get(id).listeners.click({stopPropagation(){}});
const end = () => {k.ftag.current_order_index = 5; k.stat.is_strong_stop = true; tick();};
const state = () => ({namespace:k.config.projectID, saves, loads, locations:[...locations], requests:[...requests],
  status:get('ad-next-status').textContent, hidden:get('ad-chapter-end').hidden,
  nextHidden:get('ad-next-button').hidden, saveDisabled:get('ad-save').disabled,
  autoDisabled:get('ad-auto').disabled, index:k.ftag.current_order_index,
  metadata:k.stat.auto_drama_position, notice:get('ad-status').textContent});
(async() => {
  if (input.case.startsWith('startup_')) {
    await flush();
    if (input.case === 'startup_engine_timeout') k.stat.current_scenario = '';
    time = 16000; tick?.();
    const error = {text:get('ad-start-button').textContent,disabled:get('ad-start-button').disabled};
    if (input.case === 'startup_engine_timeout') {k.stat.current_scenario = 'first.ks'; tick();}
    process.stdout.write(JSON.stringify({error,recovered:{
      text:get('ad-start-button').textContent,disabled:get('ad-start-button').disabled}, requests})); return;
  }
  await flush(); tick(); click('ad-start-button'); tick();
  const before = state();
  if (input.case === 'complete') nextResult = {status:'complete',build_id:'first',next_build:null};
  if (input.case === 'offline') fail = true;
  if (input.case === 'stale') {click('ad-save'); hold = true;}
  if (input.case === 'mismatch') {
    nextResult = {status:'ready',build_id:'other',next_build:{id:'second',chapter_number:2}};
  }
  end(); await flush(); tick();
  const waiting = state();
  if (input.case === 'ready') {
    nextResult = {status:'ready',build_id:'first',next_build:{id:'second',chapter_number:2,
      title:'SPOILER TITLE',plot:'SPOILER PLOT',player_url:'https://untrusted.invalid'}};
    time = 6000; tick(); await flush(); tick();
    const ready = state(); click('ad-next-button'); await flush();
    process.stdout.write(JSON.stringify({before,waiting,ready,after:state()})); return;
  }
  if (input.case === 'load') {
    k.ftag.current_order_index = 2; k.stat.is_strong_stop = false; click('ad-load'); tick();
  }
  if (input.case === 'tamper') {
    const key = k.config.projectID + '_tyrano_quick_save';
    const saved = JSON.parse(storage.get(key));
    if (input.field === 'event_id') saved.stat.auto_drama_position.event_id = 'first_line';
    else if (input.field === 'index') saved.current_order_index = 0;
    else saved.stat.auto_drama_position[input.field] = 'different';
    storage.set(key, JSON.stringify(saved)); click('ad-load');
  }
  if (input.case === 'stale') {
    // Restore an earlier valid save while a chapter-status response is pending.
    k.ftag.current_order_index = 2; k.stat.is_strong_stop = false; click('ad-save'); click('ad-load');
    nextResult = {status:'ready',build_id:'first',next_build:{id:'second',chapter_number:2}};
    pending(); await flush(); tick();
  }
  process.stdout.write(JSON.stringify({before,waiting,after:state()}));
})().catch(error => {console.error(error); process.exit(1);});
"""


def run(case, **options):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute chapter player controls")
    result = subprocess.run(
        [node, "-e", HARNESS, str(ROOT / "packages/tyrano_export/playback.js"),
         str(ROOT / "packages/tyrano_export/player.js")],
        input=json.dumps({"case": case, **options}), text=True, encoding="utf-8",
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_waiting_saves_current_build_and_ready_chapter_requires_explicit_transition():
    result = run("ready")
    assert result["before"]["requests"] == ["./script.json", "./player-context.json"]
    assert result["waiting"]["metadata"] == {
        "schema_version": 1, "project_id": "project", "production_id": "series",
        "storyline_id": "series",
        "build_id": "first", "event_id": "chapter_end",
    }
    assert result["waiting"]["namespace"] == "script_hash_series_first"
    assert result["waiting"]["saves"] == 1
    assert result["waiting"]["saveDisabled"] is False
    assert result["waiting"]["autoDisabled"] is True
    assert result["ready"]["nextHidden"] is False
    assert result["ready"]["locations"] == []
    assert "SPOILER" not in result["ready"]["status"]
    assert result["after"]["locations"] == ["/player/second/"]
    assert result["after"]["requests"].count("/api/m3/builds/first/next") == 3
    assert result["after"]["index"] == 5


def test_static_export_has_explicit_end_without_live_requests():
    result = run("static", static=True)["after"]
    assert result["requests"] == ["./script.json"]
    assert "書き出し" in result["status"]
    assert result["nextHidden"] is True and result["hidden"] is False
    assert result["saves"] == 1


def test_final_chapter_displays_complete_without_next_chapter_navigation():
    result = run("complete")["after"]
    assert "最後まで" in result["status"]
    assert result["nextHidden"] is True and result["locations"] == []


@pytest.mark.parametrize("case", ["offline", "mismatch"])
def test_unavailable_or_wrong_lineage_preserves_saved_position(case):
    result = run(case)["after"]
    assert result["metadata"]["event_id"] == "chapter_end"
    assert result["index"] == 5 and result["locations"] == []
    assert "再確認" in result["status"]


def test_waiting_save_reloads_the_same_build_at_the_chapter_boundary():
    result = run("load")["after"]
    assert result["loads"] == 1
    assert result["index"] == 5 and result["hidden"] is False
    assert result["metadata"]["build_id"] == "first"


@pytest.mark.parametrize("field", ["project_id", "storyline_id", "production_id", "build_id", "event_id", "index"])
def test_saves_with_different_identity_or_event_are_rejected(field):
    result = run("tamper", field=field)["after"]
    assert result["loads"] == 0
    assert "対応する保存データ" in result["notice"]


def test_old_next_chapter_response_cannot_reopen_waiting_ui_after_loading_earlier_save():
    result = run("stale")["after"]
    assert result["loads"] == 1 and result["index"] == 2
    assert result["hidden"] is True and result["nextHidden"] is True
    assert result["locations"] == []


@pytest.mark.parametrize("element", ["ad-chapter-end", "ad-resume-button"])
def test_mixed_launcher_versions_report_an_error_instead_of_loading_forever(element):
    result = run("startup_mixed", missingElement=element)
    assert "版が一致しません" in result["error"]["text"]
    assert result["error"]["disabled"] is True
    assert result["requests"] == []


@pytest.mark.parametrize("case, message", [
    ("startup_script_error", "作品を読み込めません"),
    ("startup_context_error", "公開版を確認できません"),
    ("startup_pending", "鑑賞の準備を完了できません"),
])
def test_startup_failures_remain_visible_after_the_loading_deadline(case, message):
    result = run(case)
    assert message in result["error"]["text"]
    assert result["error"]["disabled"] is True


def test_delayed_engine_reports_timeout_but_can_recover_without_reloading():
    result = run("startup_engine_timeout")
    assert "鑑賞の準備を完了できません" in result["error"]["text"]
    assert result["error"]["disabled"] is True
    assert result["recovered"] == {"text": "再生する", "disabled": False}
