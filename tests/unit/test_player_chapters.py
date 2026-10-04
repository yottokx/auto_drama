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
  constructor() {this.listeners = {}; this.children = []; this.value = '24'; this.hidden = true;
    this.style = {}; this.open = false;}
  addEventListener(event, fn) {this.listeners[event] = fn;}
  querySelectorAll(selector) {return selector === 'button, select, input' ? this.children : [];}
  setAttribute(key, value) {this[key] = value;}
  close() {this.open = false; this.listeners.close?.();}
}
const elements = {}, handlers = {}, subscriptions = {}, requests = [], locations = [], storage = new Map();
const session = new Map(), calls = [], timers = new Map();
let tick, time = 0, timerId = 0;
const schedule = (fn, delay) => {const id = ++timerId; timers.set(id, {fn,due:time + delay}); return id;};
const get = id => elements[id] ||= new Element();
get('ad-toolbar').children = ['ad-auto','ad-save','ad-load','ad-backlog','ad-font','ad-volume','ad-bgm-volume'].map(get);
get('ad-bgm-volume').value = '100';
const document = {getElementById:id => id === input.missingElement ? null : get(id),
  addEventListener(event, fn) {handlers[event] = fn;},
  querySelectorAll() {return [];}};
let nextResult = {status:'waiting',build_id:'first',next_build:null}, pending, saves = 0, loads = 0;
let hold = false, holdBody = false, fail = false, starts = 0, navTimes = [], pendingUnlock;
const story = {id:'script',characters:[],assets:[],utterances:[{id:'first_line',display_text:'READ'}]};
if(input.speech)story.utterances[0].audio_asset_id='voice';
const nextStory = {id:'nextscript',characters:[],assets:[{id:'track',kind:'music',filename:'track.mp3'},
  {id:'voice',kind:'audio',filename:'voice.wav'}, {id:'bg',kind:'background',filename:'bg.png'}],
  utterances:[{id:'nextline',audio_asset_id:'voice'}],
  music_cues:[{id:'nextmusic',utterance_id:'nextline',action:'play',asset_id:'track'}],
  scene_transitions:[{id:'nextstage',utterance_id:'nextline'}]};
const snapshot = {schema_version:1,cue_id:'loop',asset_id:'music',loop_start_seconds:2,
  loop_end_seconds:10,position_seconds:6,phase:'loop',track_volume:.35,user_volume:.6,paused:false};
let heldMusic = input.music ? snapshot : null;
const music = {
  setVolume(){}, snapshot(){return heldMusic;},
  unlock(){calls.push(['unlock']);return input.holdUnlock?new Promise(resolve=>pendingUnlock=resolve):
    Promise.resolve(!input.blockUnlock);},
  stopFaded({fadeOutMs,signal}) {
    calls.push(['fade',fadeOutMs,time]);
    return new Promise(resolve=>{let done=false;
      const finish=value=>{if(done)return;done=true;timers.delete(id);signal?.removeEventListener('abort',abort);
        if(value)heldMusic=null;resolve(value);};
      const abort=()=>finish(false),id=schedule(()=>finish(true),fadeOutMs);
      signal?.addEventListener('abort',abort,{once:true});if(signal?.aborted)abort();});
  },
  restore(value){heldMusic=value;calls.push(['restore',value]);return Promise.resolve(!!value);},
  resume(){calls.push(['resume']);return Promise.resolve(!!heldMusic);},
  pause(){calls.push(['pause']);},stop(){heldMusic=null;calls.push(['stop']);},
  dispose(){heldMusic=null;calls.push(['dispose']);},
};
const tags = [{name:'label',pm:{label_name:'utterance_first_line'}}, {name:'text'}, {name:'p'},
  {name:'label',pm:{label_name:'auto_drama_chapter_end'}}, {name:'text'}, {name:'s'}];
const k = {config:{projectID:'script_hash'}, stat:{current_scenario:'first.ks',font:{},default_font:{}}, tmp:{},
  readyAudio() {}, on(event, fn) {subscriptions[event] = fn;},
  ftag:{master_tag:{},array_tag:tags,current_order_index:2,nextOrder() {},
    startTag(name) {if (name === 'autostop') k.stat.is_auto = false;}},
  key_mouse:{next() {starts++;},util:{canShowMenu() {return true;}},
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
    ...(input.speech?{Howler:{ctx:{state:'suspended',resume(){
      if(input.voiceError)return Promise.reject(Error('voice autoplay denied'));
      if(input.voiceHold)return new Promise(resolve=>schedule(()=>{this.state='running';resolve();},500));
      if(!input.voiceBlocked)this.state='running';return Promise.resolve();}}}}:{}),
    AbortController,setTimeout:schedule,clearTimeout:id=>timers.delete(id),
    sessionStorage:{getItem:key=>session.get(key),setItem:(key,value)=>session.set(key,value),removeItem:key=>session.delete(key)},
    AutoDramaMusic:{create:()=>music},addEventListener:(event,fn)=>handlers[event]=fn,
    location:{pathname:input.static ? '/static/index.html' : '/player/'+(input.entry?'second':'first')+'/',
      assign:url => {if(input.assignFail)throw Error('navigation failed');locations.push(url);navTimes.push(time);}}},
  fetch: async url => {
    requests.push(url);
    if (url === './script.json') {
      if (input.case === 'startup_pending') await new Promise(() => {});
      return {ok:input.case !== 'startup_script_error',json:async() => story};
    }
    if (url === './player-context.json' && input.case === 'startup_context_error') return {ok:false};
    if (url === './player-context.json') return {ok:true,json:async() => ({mode:'live',build_id:input.entry?'second':'first',
      project_id:'project',production_id:input.entry?'production_second':'series',storyline_id:'series',chapter_number:input.entry?2:1,
      next_url:'/api/m3/builds/'+(input.entry?'second':'first')+'/next'})};
    if(url==='/player/second/script.json')return{ok:true,json:async()=>nextStory};
    if(url==='/player/second/data/others/auto_drama_stages.json')return{ok:true,json:async()=>({schema_version:1,
      scenes:[{id:'nextstage',background:{storage:'bg.png'},characters:[]}]})};
    if(url.startsWith('/player/second/data/'))return{ok:true,arrayBuffer:async()=>new ArrayBuffer(8)};
    if (fail) throw Error('offline');
    if (hold) await new Promise(resolve => {pending = resolve;});
    return {ok:true,json:async() => {if(holdBody)await new Promise(resolve=>pending=resolve);return nextResult;}};
  },
};
if(input.entry){session.set('auto_drama_next_chapter_entry_v1',JSON.stringify({schema_version:1,
 project_id:'project',storyline_id:'series',to_build:'second',chapter_number:2,created_at:0,...input.intent}));}
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
handlers.DOMContentLoaded();
const flush = () => new Promise(resolve => setImmediate(resolve));
const advance = async ms => {const target=time+ms;
  while(true){const due=[...timers].filter(([,value])=>value.due<=target).sort((a,b)=>a[1].due-b[1].due);
    if(!due.length)break;const[id,value]=due[0];time=value.due;timers.delete(id);value.fn();await flush();}
  time=target;await flush();};
const click = id => get(id).listeners.click({stopPropagation(){}});
const end = () => {k.ftag.current_order_index = 5; k.stat.is_strong_stop = true; tick();};
const state = () => ({namespace:k.config.projectID, saves, loads, locations:[...locations], requests:[...requests],
  status:get('ad-next-status').textContent, hidden:get('ad-chapter-end').hidden,
  nextHidden:get('ad-next-button').hidden, saveDisabled:get('ad-save').disabled,
  autoDisabled:get('ad-auto').disabled, index:k.ftag.current_order_index,
  metadata:k.stat.auto_drama_position, notice:get('ad-status').textContent,
  overlay:!get('ad-chapter-transition').hidden,busy:!!k.tmp.auto_drama_transition,
  nextDisabled:get('ad-next-button').disabled,music:heldMusic,calls:[...calls],starts,
  navTimes:[...navTimes],entry:session.get('auto_drama_next_chapter_entry_v1')??null,startHidden:get('ad-start').hidden});
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
  await flush(); tick(); await flush();
  if(input.entry){const initial=state();
    if(input.holdUnlock){await advance(300);pendingUnlock(true);await flush();}
    if(input.voiceHold){await advance(500);}
    if(input.case==='entry_manual'){click('ad-start-button');await flush();}
    process.stdout.write(JSON.stringify({initial,after:state()}));return;}
  if(input.case==='ordinary'){process.stdout.write(JSON.stringify({after:state()}));return;}
  click('ad-start-button'); tick();
  const before = state();
  if (input.case === 'complete') nextResult = {status:'complete',build_id:'first',next_build:null};
  if (input.case === 'offline') fail = true;
  if (input.case === 'stale') {click('ad-save'); hold = true;}
  if (input.case === 'mismatch') {
    nextResult = {status:'ready',build_id:'other',next_build:{id:'second',chapter_number:2}};
  }
  end(); await flush(); tick();
  const waiting = state();
  if (input.case.startsWith('next_') || input.case === 'ready') {
    nextResult = {status:'ready',build_id:'first',next_build:{id:'second',chapter_number:2,
      title:'SPOILER TITLE',plot:'SPOILER PLOT',player_url:'https://untrusted.invalid'}};
    time = 6000; tick(); await flush(); tick();
    const ready = state();
    if(input.case==='next_offline')fail=true;
    if(input.case==='next_waiting')nextResult={status:'waiting',build_id:'first',next_build:null};
    if(input.case==='next_complete')nextResult={status:'complete',build_id:'first',next_build:null};
    if(input.case==='next_mismatch')nextResult={status:'ready',build_id:'other',next_build:{id:'second',chapter_number:2}};
    if(['next_timeout','next_load','next_dispose','next_back'].includes(input.case))hold=true;
    if(input.case==='next_body_timeout')holdBody=true;
    click('ad-next-button');const immediate=state();click('ad-next-button');await flush();
    if(input.case==='next_load')k.key_mouse.qload();
    if(input.case==='next_dispose')handlers.pagehide({persisted:false});
    if(input.case==='next_back')handlers.pagehide({persisted:true});
    await advance(399);const fading=state();await advance(1);
    if(input.case==='next_back'){handlers.pageshow({persisted:true});await flush();}
    if(input.case==='next_timeout'||input.case==='next_body_timeout')await advance(7600);
    if(pending){pending();await flush();}
    if(input.case==='next_return'){handlers.pagehide({persisted:true});handlers.pageshow({persisted:true});await flush();}
    if(!input.case.startsWith('next_timeout')&&input.case!=='next_body_timeout')tick();
    process.stdout.write(JSON.stringify({before,waiting,ready,immediate,fading,after:state()})); return;
  }
  if (input.case === 'load') {
    k.ftag.current_order_index = 2; k.stat.is_strong_stop = false; click('ad-load'); await flush(); tick();
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
    assert result["immediate"]["overlay"] and result["immediate"]["busy"]
    assert result["immediate"]["nextDisabled"]
    assert result["immediate"]["locations"] == result["fading"]["locations"] == []
    assert result["after"]["navTimes"] == [6400]


def test_waiting_preloads_only_owned_first_scene_assets_without_starting_or_navigating():
    result = run("ready", music=True)
    requests = result["ready"]["requests"]
    assert "/player/second/script.json" in requests
    assert "/player/second/data/others/auto_drama_stages.json" in requests
    assert {"/player/second/data/bgm/track.mp3", "/player/second/data/sound/voice.wav",
            "/player/second/data/bgimage/bg.png"}.issubset(requests)
    assert not any("untrusted" in url for url in requests)
    assert result["ready"]["music"]["position_seconds"] == 6
    assert result["ready"]["locations"] == []


def test_next_click_saves_active_loop_then_fades_in_parallel_for_exactly_400ms():
    result = run("ready", music=True)
    assert result["waiting"]["music"]["phase"] == "loop"
    assert result["after"]["calls"].count(["fade", 400, 6000]) == 1
    assert result["immediate"]["music"]["position_seconds"] == 6
    assert result["fading"]["music"] is not None
    assert result["after"]["music"] is None
    marker = json.loads(result["after"]["entry"])
    assert marker == {"schema_version": 1, "project_id": "project", "storyline_id": "series",
                      "to_build": "second", "chapter_number": 2, "created_at": 6400}


@pytest.mark.parametrize("case", ["next_offline", "next_mismatch", "next_timeout", "next_body_timeout"])
def test_failed_or_timed_out_recheck_restores_original_loop_and_waiting_scene(case):
    result = run(case, music=True)["after"]
    assert result["locations"] == [] and not result["overlay"] and not result["busy"]
    assert result["music"]["position_seconds"] == 6 and result["music"]["user_volume"] == 0.6
    assert ["resume"] in result["calls"]
    assert "進めませんでした" in result["status"]
    assert result["entry"] is None


@pytest.mark.parametrize("case, message", [("next_waiting", "制作中"), ("next_complete", "最後まで")])
def test_changed_next_availability_rolls_back_without_abandoning_current_loop(case, message):
    result = run(case, music=True)["after"]
    assert result["locations"] == [] and result["nextHidden"] and not result["overlay"]
    assert result["music"]["phase"] == "loop"
    assert message in result["status"]


def test_navigation_api_failure_restores_audio_even_after_completed_fade():
    result = run("ready", music=True, assignFail=True)["after"]
    assert result["locations"] == [] and result["entry"] is None
    assert result["music"]["position_seconds"] == 6
    assert not result["overlay"] and not result["busy"]


@pytest.mark.parametrize("case", ["next_load", "next_dispose", "next_back"])
def test_cancelling_during_fade_prevents_old_fetch_from_navigating(case):
    result = run(case, music=True)["after"]
    assert result["locations"] == [] and not result["overlay"] and not result["busy"]
    assert result["entry"] is None
    if case == "next_back":
        assert result["music"]["phase"] == "loop" and ["resume"] in result["calls"]
    elif case == "next_dispose":
        assert ["dispose"] in result["calls"] and ["resume"] not in result["calls"]


def test_cached_back_after_success_restores_prefade_loop_position_and_reenables_next():
    result = run("next_return", music=True)["after"]
    assert result["locations"] == ["/player/second/"]
    assert result["music"]["position_seconds"] == 6 and result["music"]["phase"] == "loop"
    assert not result["overlay"] and not result["busy"] and not result["nextDisabled"]


def test_ordinary_open_stays_stopped_but_valid_next_intent_can_start_once_across_productions():
    assert run("ordinary")["after"]["starts"] == 0
    result = run("entry", entry=True)["after"]
    assert result["starts"] == 1 and result["entry"] is None
    assert "production_second" in result["namespace"]


@pytest.mark.parametrize("intent", [{"project_id": "other"}, {"storyline_id": "other"},
                                  {"chapter_number": 3}, {"created_at": -30001}, {"created_at": 1}])
def test_mismatched_or_expired_next_intent_is_consumed_without_autoplay(intent):
    result = run("entry", entry=True, intent=intent)["after"]
    assert result["starts"] == 0 and result["entry"] is None


def test_intent_for_another_build_is_not_consumed_by_an_unrelated_open():
    result = run("entry", entry=True, intent={"to_build": "other"})["after"]
    assert result["starts"] == 0 and result["entry"] is not None


@pytest.mark.parametrize("options", [{"blockUnlock": True}, {"holdUnlock": True},
                                    {"speech": True, "voiceError": True},
                                    {"speech": True, "voiceBlocked": True},
                                    {"speech": True, "voiceHold": True}])
def test_denied_or_delayed_audio_unlock_keeps_explicit_start_available(options):
    result = run("entry", entry=True, **options)["after"]
    assert result["starts"] == 0 and result["entry"] is None


def test_manual_start_after_autoplay_timeout_does_not_double_advance():
    result = run("entry_manual", entry=True, holdUnlock=True)["after"]
    assert result["starts"] == 1


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
