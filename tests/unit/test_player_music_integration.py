"""Owned game toolbar, cue tag, chapter, and save lifetimes without audio output."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

HARNESS = r'''
const fs=require('fs'),vm=require('vm');
const request=JSON.parse(fs.readFileSync(0,'utf8'));
class Element{
 constructor(){this.listeners={};this.children=[];this.value='100';this.hidden=true;this.open=false;}
 addEventListener(name,fn){this.listeners[name]=fn;}
 querySelectorAll(selector){return selector==='button, select, input'?this.children:[];}
 setAttribute(name,value){this[name]=value;}
 close(){this.open=false;this.listeners.close?.();}
 showModal(){this.open=true;}
 replaceChildren(){this.children=[];}
 append(item){this.children.push(item);}
}
const elements={},subscriptions={},storage=new Map(),calls=[];
const get=id=>elements[id]??=new Element();
get('ad-toolbar').children=['ad-auto','ad-save','ad-load','ad-backlog','ad-font','ad-volume','ad-bgm-volume'].map(get);
const cues=[{id:'music_first',utterance_id:'one',action:'play',asset_id:'music_a',volume:.35,
 loop_start_seconds:10,loop_end_seconds:60},
 {id:'music_continue',utterance_id:'two',action:'continue',volume:.35},
 {id:'music_none',utterance_id:'three',action:'stop',volume:.35}];
const story={id:'script',characters:[],assets:[],utterances:['one','two','three'].map(id=>({id,display_text:id})),music_cues:cues};
if(request.transition)story.scene_transitions=[{id:'boundary',utterance_id:'one'}];
const stages={schema_version:1,scenes:[{id:'boundary',utterance_id:'one'}]};
let tick,atEnd=false,pending=null,pendingStop=null;
let transitionPending=[],transitionGeneration=0,busyChanged=()=>{};
const transitionControl={
 warm(){return Promise.resolve(true);},
 run(cue,options){calls.push(['transition',cue,options]);const gen=++transitionGeneration;busyChanged(true);
  const promise=request.hold?new Promise(resolve=>transitionPending.push(resolve)):Promise.resolve(true);
  return promise.then(value=>{if(gen!==transitionGeneration)return false;busyChanged(false);return value;});},
 cancel(){transitionGeneration++;busyChanged(false);calls.push(['transition-cancel']);},
 dispose(){calls.push(['transition-dispose']);},skip(){calls.push(['transition-skip']);}
};
const savedMusic={schema_version:1,cue_id:'music_first',asset_id:'music_a',loop_start_seconds:10,
 loop_end_seconds:60,position_seconds:22,phase:'loop',track_volume:.35,user_volume:.42,paused:false};
let heldMusic=savedMusic,restorePending;
const music={
 cue(cue){calls.push(['cue',cue?.id]);return request.hold?new Promise(resolve=>pending=resolve):Promise.resolve(true);},
 unlock(){calls.push(['unlock']);return Promise.resolve(true);},
 setVolume(v){calls.push(['music-volume',v]);},snapshot(){return request.loadGate?heldMusic:savedMusic;},
 restore(value){calls.push(['restore',value]);return (request.holdRestore?new Promise(resolve=>restorePending=resolve):
  Promise.resolve(true)).then(()=>{heldMusic=value;return true;});},
 pause(){calls.push(['pause']);},resume(){calls.push(['resume']);return Promise.resolve(true);},
 stop(){heldMusic=null;calls.push(['stop']);pendingStop?.(false);},dispose(){calls.push(['dispose']);},
 stopFaded(options){calls.push(['stop-faded',options]);return request.holdStop?
  new Promise(resolve=>pendingStop=resolve):Promise.resolve(true);}
};
const playback={advance(){},displayedLines(){return[];},canOpenLog(){return true;},openLog(){return true;},
 closeLog(){calls.push(['log-close']);},reset(){calls.push(['reset']);},atChapterEnd(){return atEnd;},
 eventId(){return atEnd?'chapter_end':'two';}};
const tags=[{name:'label'},{name:'ad_music'},{name:'text'},{name:'p'},{name:'s'}];
const k={config:{projectID:'script_namespace',defaultSeVolume:100},
 stat:{current_scenario:'first.ks',font:{},default_font:{size:24}},tmp:{},
 readyAudio(){calls.push(['ready-audio']);},weaklyStop(){calls.push(['weak-stop']);},
 cancelWeakStop(){calls.push(['weak-resume']);},on(event,fn){subscriptions[event]=fn;},
 ftag:{master_tag:{return:{start(){calls.push(['native-return']);
  if(request.nativeEvent)subscriptions['load-complete']();k.stat.current_scenario='first.ks';}}},
  array_tag:tags,current_order_index:1,nextOrder(){calls.push(['next']);},
 startTag(name,pm){calls.push(['engine-volume',name,pm]);}},
 key_mouse:{next(){calls.push(['story-start']);},util:{canShowMenu(){return true;}},
 qsave(){const value={stat:structuredClone(k.stat),current_order_index:0};
  storage.set(k.config.projectID+'_tyrano_quick_save',JSON.stringify(value));return true;},
 qload(){subscriptions['load-start']();subscriptions['load-complete']();return true;}}
};
const handlers={};
const context={console,Promise,Map,Set,Date,setInterval(fn){tick=fn;},
 $:{getStorage(key){return storage.get(key);}},
 document:{getElementById:get,querySelectorAll(){return[];},createElement(){return new Element();},
 addEventListener(event,fn){handlers[event]=fn;}},
 fetch:async url=>({ok:true,json:async()=>url.includes('stages')?stages:story}),
 window:{TYRANO:{kag:k},localStorage:{getItem:key=>storage.get(key)},location:{pathname:'/static/index.html'},
  AutoDramaPlayback:()=>playback,AutoDramaMusic:{create(){calls.push(['create']);return music;}},
  AutoDramaTransitions:{create(_data,options){busyChanged=options.onBusyChange;return transitionControl;},kagAdapter(){return{};}},
  addEventListener(name,fn){handlers[name]=fn;}}
};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const click=id=>get(id).listeners.click({stopPropagation(){}});
(async()=>{
 handlers.DOMContentLoaded();await flush();tick();click('ad-start-button');await flush();
 if(request.case==='cue'){k.ftag.master_tag.ad_music.start({cue:'music_first'});await flush();}
 if(request.case==='transition'){k.ftag.master_tag.ad_transition.start({cue:'boundary'});await flush();}
 if(request.case==='transition-busy'){k.ftag.master_tag.ad_transition.start({cue:'boundary'});await flush();tick();
  calls.push(['disabled',get('ad-save').disabled,get('ad-load').disabled,get('ad-backlog').disabled]);
  click('ad-save');click('ad-load');click('ad-backlog');transitionPending[0](true);await flush();}
 if(request.case==='transition-stale'){k.ftag.master_tag.ad_transition.start({cue:'boundary'});await flush();
  subscriptions['load-start']();transitionPending[0](true);await flush();}
 if(request.case==='transition-bfcache'){k.ftag.master_tag.ad_transition.start({cue:'boundary'});await flush();
  handlers.pagehide({persisted:true});handlers.pageshow({persisted:true});await flush();
  transitionPending[0](true);await flush();transitionPending[1](true);await flush();}
 if(request.case==='chapter-fade'){k.ftag.master_tag.ad_music_stop.start({fade:'1000'});await flush();}
 if(request.case==='chapter-bfcache'){tags[1].name='ad_music_stop';
  k.ftag.master_tag.ad_music_stop.start({fade:'1000'});await flush();
  handlers.pagehide({persisted:true});await flush();handlers.pageshow({persisted:true});await flush();}
 if(request.case==='stale'){k.ftag.master_tag.ad_music.start({cue:'music_first'});
  subscriptions['load-start']();pending(true);await flush();}
 if(request.case==='save'){click('ad-save');}
 if(request.case==='load'){k.stat.auto_drama_music=savedMusic;k.stat.auto_drama_bgm_volume=.99;
  subscriptions['load-start']();subscriptions['load-complete']();await flush();}
 if(request.case==='legacy'){k.stat.auto_drama_position={event_id:'two'};
  subscriptions['load-start']();subscriptions['load-complete']();await flush();}
 if(request.case==='stop'){k.ftag.master_tag.ad_music.start({cue:'music_none'});await flush();}
 if(request.case==='chapter'||request.case==='chapter-log'||request.case==='chapter-return'){atEnd=true;tick();}
 if(request.case==='chapter-log'){click('ad-backlog');get('ad-log').close();await flush();}
 if(request.case==='chapter-return'){handlers.pagehide({persisted:true});handlers.pageshow({persisted:true});await flush();}
 if(request.case==='chapter-load'){atEnd=true;k.stat.auto_drama_music=savedMusic;
  k.stat.auto_drama_position={event_id:'chapter_end'};
  const saved=structuredClone(k.stat);subscriptions['load-start']();k.stat=saved;
  tick();click('ad-save');calls.push(['load-before-return',k.stat.auto_drama_music]);
  k.stat.current_scenario='make.ks';k.ftag.master_tag.return.start();await flush();tick();
  calls.push(['restoring-disabled',get('ad-save').disabled,get('ad-load').disabled,get('ad-backlog').disabled]);
  restorePending?.(true);await flush();tick();}
 if(request.case==='legacy-chapter'){atEnd=true;cues.pop();k.stat.auto_drama_position={event_id:'chapter_end'};
  subscriptions['load-start']();subscriptions['load-complete']();await flush();tick();}
 if(request.case==='volume'){get('ad-bgm-volume').listeners.input({target:{value:'20'}});}
 if(request.case==='voice'){get('ad-volume').listeners.input({target:{value:'25'}});}
 if(request.case==='log'){click('ad-backlog');get('ad-log').close();await flush();}
 if(request.case==='dispose'){handlers.pagehide();}
 if(request.case==='bfcache'){handlers.pagehide({persisted:true});handlers.pageshow({persisted:true});await flush();}
 if(request.case==='bfcache-log'){click('ad-backlog');handlers.pagehide({persisted:true});
  handlers.pageshow({persisted:true});await flush();}
 if(request.case==='bfcache-cue'){k.ftag.master_tag.ad_music.start({cue:'music_first'});
  handlers.pagehide({persisted:true});handlers.pageshow({persisted:true});pending(true);await flush();}
 process.stdout.write(JSON.stringify({calls,stat:k.stat,volume:get('ad-bgm-volume').value}));
})().catch(error=>{console.error(error);process.exit(1);});
'''


def run(case, **options):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for owned music player integration")
    result = subprocess.run(
        [node, "-e", HARNESS, str(ROOT / "packages/tyrano_export/player.js")],
        input=json.dumps({"case": case, **options}), encoding="utf-8", text=True,
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_music_cue_waits_for_adopted_playback_then_advances_once():
    calls = run("cue")["calls"]
    assert ["unlock"] in calls and ["cue", "music_first"] in calls
    assert calls[-3:] == [["cue", "music_first"], ["weak-resume"], ["next"]]


def test_scene_transition_owns_the_only_advance_and_does_not_run_immediate_music_tag():
    calls = run("transition", transition=True)["calls"]
    assert ["transition", "boundary", {"skip": False, "resume": False}] in calls
    assert calls.count(["next"]) == 1
    assert not any(item[0] == "cue" for item in calls)


def test_busy_transition_disables_and_guards_save_load_and_backlog():
    result = run("transition-busy", transition=True, hold=True)
    calls = result["calls"]
    assert ["disabled", True, True, True] in calls
    assert "auto_drama_position" not in result["stat"]
    assert not any(item[0] in {"restore", "pause", "reset"} for item in calls)
    assert calls.count(["next"]) == 1


def test_loaded_story_cannot_be_advanced_by_old_scene_transition_completion():
    calls = run("transition-stale", transition=True, hold=True)["calls"]
    assert ["transition-cancel"] in calls and ["reset"] in calls
    assert ["next"] not in calls and ["weak-resume"] not in calls


def test_cached_page_restarts_pending_transition_in_resume_mode_and_advances_once():
    calls = run("transition-bfcache", transition=True, hold=True)["calls"]
    assert ["transition", "boundary", {"skip": False, "resume": True}] in calls
    assert calls.count(["next"]) == 1
    assert ["pause"] in calls and ["resume"] in calls


def test_legacy_fade_tag_still_stops_music_before_the_old_final_story_stop():
    calls = run("chapter-fade", transition=True)["calls"]
    assert ["stop-faded", {"fadeOutMs": 1000}] in calls
    assert calls.count(["next"]) == 1


def test_cached_page_settles_chapter_stop_instead_of_preserving_the_old_track():
    calls = run("chapter-bfcache", transition=True, holdStop=True)["calls"]
    assert ["stop-faded", {"fadeOutMs": 1000}] in calls
    assert ["stop"] in calls and ["pause"] not in calls
    assert calls.index(["stop"]) < calls.index(["next"])
    assert calls.count(["next"]) == 1


def test_old_music_fetch_cannot_advance_new_loaded_story():
    calls = run("stale", hold=True)["calls"]
    assert ["stop"] in calls and ["reset"] in calls
    assert ["next"] not in calls and ["weak-resume"] not in calls


def test_music_position_phase_and_independent_gain_are_saved():
    stat = run("save")["stat"]
    saved = stat["auto_drama_music"]
    assert saved["position_seconds"] == 22 and saved["phase"] == "loop"
    assert saved["track_volume"] == 0.35 and saved["user_volume"] == 0.42
    assert stat["auto_drama_bgm_volume"] == 1


def test_explicit_load_restores_saved_track_and_snapshot_gain_before_resuming():
    result = run("load")
    assert result["volume"] == "42"
    restored = [item for item in result["calls"] if item[0] == "restore"]
    assert restored[0][1]["position_seconds"] == 22
    assert ["music-volume", 0.42] in result["calls"] and ["resume"] in result["calls"]


def test_legacy_save_restores_last_play_through_continue_scene():
    calls = run("legacy")["calls"]
    assert ["cue", "music_first"] in calls and ["resume"] in calls
    assert not any(item[0] == "restore" for item in calls)


def test_explicit_scene_stop_remains_supported_but_chapter_end_keeps_and_saves_loop():
    assert ["cue", "music_none"] in run("stop")["calls"]
    chapter = run("chapter")
    assert ["stop"] not in chapter["calls"]
    assert chapter["stat"]["auto_drama_position"]["event_id"] == "chapter_end"
    assert chapter["stat"]["auto_drama_music"]["phase"] == "loop"


@pytest.mark.parametrize("case", ["chapter-log", "chapter-return"])
def test_chapter_boundary_music_resumes_after_backlog_or_cached_back(case):
    calls = run(case)["calls"]
    assert ["pause"] in calls and ["resume"] in calls
    assert ["stop"] not in calls and ["dispose"] not in calls


@pytest.mark.parametrize("native_event", [True, False])
def test_native_strong_stop_load_does_not_overwrite_music_before_make_return(native_event):
    result = run("chapter-load", loadGate=True, holdRestore=True, nativeEvent=native_event)
    calls = result["calls"]
    original = next(value[1] for value in calls if value[0] == "load-before-return")
    assert original["position_seconds"] == 22 and original["phase"] == "loop"
    assert ["restoring-disabled", True, True, True] in calls
    assert sum(value[0] == "restore" for value in calls) == 1
    assert calls.index(["native-return"]) < next(index for index, value in enumerate(calls) if value[0] == "restore")
    assert ["resume"] in calls and result["stat"]["auto_drama_music"] == original
    assert result["stat"]["auto_drama_position"]["event_id"] == "chapter_end"


def test_legacy_chapter_save_without_track_snapshot_uses_last_play_through_continue():
    calls = run("legacy-chapter")["calls"]
    assert ["cue", "music_first"] in calls and ["resume"] in calls


def test_music_volume_does_not_touch_voice_and_voice_does_not_touch_music():
    music = run("volume")["calls"]
    assert ["music-volume", 0.2] in music
    assert not any(item[0] == "engine-volume" for item in music)
    voice = run("voice")["calls"]
    assert ["engine-volume", "seopt", {"volume": "25", "next": "false"}] in voice
    assert ["music-volume", 0.25] not in voice


def test_backlog_pauses_music_and_closing_restores_the_same_session():
    calls = run("log")["calls"]
    assert ["pause"] in calls and ["resume"] in calls
    assert ["cue", "music_first"] not in calls


def test_leaving_game_disposes_only_its_owned_music_layer():
    assert ["dispose"] in run("dispose")["calls"]


def test_cached_back_navigation_keeps_music_position_and_resumes_the_live_controller():
    calls = run("bfcache")["calls"]
    assert ["pause"] in calls and ["resume"] in calls
    assert ["dispose"] not in calls
    log_calls = run("bfcache-log")["calls"]
    assert ["resume"] not in log_calls and ["dispose"] not in log_calls


def test_cached_page_does_not_invalidate_its_pending_scene_cue():
    calls = run("bfcache-cue", hold=True)["calls"]
    assert ["dispose"] not in calls
    assert calls[-2:] == [["weak-resume"], ["next"]]
