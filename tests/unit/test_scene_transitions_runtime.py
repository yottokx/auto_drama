"""Cold preparation, visual/music barriers, and cancellation without audio output."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

HARNESS = r'''
const fs=require('fs'),vm=require('vm');
const request=JSON.parse(fs.readFileSync(0,'utf8'));
const calls=[],resolvers={};
const deferred=name=>new Promise(resolve=>resolvers[name]=resolve);
const flush=()=>new Promise(resolve=>setImmediate(resolve));
let frame='old',track='old',snapshot=null,cg=null;
const adapter={
 prepare(scene,signal){calls.push(['prepare-image',scene.id]);
  return request.cold?deferred('image'):Promise.resolve(request.fallback?{fallback:true}:true);},
 same(){return !!request.same;},
 cover(style,duration,signal){calls.push(['cover',style,duration,frame,track]);
  return request.barriers?deferred('cover'):Promise.resolve();},
 apply(scene){calls.push(['apply',scene.id]);frame=scene.id;cg=scene.event_cg||null;},
 reveal(duration){calls.push(['reveal',duration,frame,track]);
  return request.barriers?deferred('reveal'):Promise.resolve();},
 reset(){calls.push(['reset']);},
 finish(){calls.push(['finish-visual']);resolvers.cover?.();resolvers.reveal?.();}
};
const action=request.action||'play';
const music={
 getCue(id){return {id,action};},snapshot(){return snapshot;},
 prepareCue(id){calls.push(['prepare-audio',id,track]);
  if(request.failAudio)return Promise.resolve(false);
  return request.cold?deferred('audio'):Promise.resolve(true);},
 async transitionCue(id,options){
  calls.push(['music-out',options.fadeOutMs,track]);
  if(request.failSwitch)return false;
  if(request.barriers)await deferred('out');
  if(options.signal.aborted)return false;
  await options.beforeCommit({signal:options.signal});
  if(options.signal.aborted)return false;
  track=action==='stop'?null:id;snapshot=track?{cue_id:track}:null;
  calls.push(['music-commit',track]);
  calls.push(['music-in',options.fadeInMs,frame]);
  if(request.barriers)await deferred('in');
  return true;
 },
 finishTransition(){calls.push(['finish-audio']);resolvers.out?.();resolvers.in?.();}
 ,async stopFaded(){calls.push(['failed-music-fade']);track=null;return true;}
};
const context={console,Promise,Map,Set,AbortController};context.window=context;
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const scene={id:'new',visual:request.visual||'fade',duration_ms:500,music_fade_out_ms:1000,
 music_fade_in_ms:1000,music_cue_id:'next'};
if(request.cg)scene.event_cg={segment_id:'event',variant_id:'variant',storage:'cg.png'};
const controller=context.AutoDramaTransitions.create({scenes:[scene]},
 {music,adapter,report:message=>calls.push(['report',message]),onBusyChange:value=>calls.push(['busy',value])});
(async()=>{
 if(request.case==='resume')snapshot={cue_id:'next'};
 const result=controller.run('new',{skip:request.case==='skip',resume:request.case==='resume',visualOnly:request.visualOnly});
 await flush();
 if(request.cold){
  calls.push(['while-cold',frame,track,controller.busy()]);
  if(request.case==='cancel-cold')controller.cancel();
  resolvers.image(true);resolvers.audio(true);await flush();
 }
 if(request.barriers){
  if(request.case==='cancel-out')controller.cancel();
  if(request.case==='finish'){controller.skip();await flush();}
  else{resolvers.out?.();await flush();calls.push(['before-covered',frame,track]);
   resolvers.cover?.();await flush();}
  if(request.case==='cancel-reveal')controller.cancel();
  resolvers.reveal?.();resolvers.in?.();
 }
 let completed=await result;
 if(request.case==='retry-failed-switch'){
  frame='restored-old';track='restored-old';request.failSwitch=true;
  completed=await controller.run('new');
 }
 process.stdout.write(JSON.stringify({calls,completed,frame,track,cg,busy:controller.busy()}));
})().catch(error=>{console.error(error);process.exit(1);});
'''


def run(case="normal", **options):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for scene transition runtime checks")
    result = subprocess.run(
        [node, "-e", HARNESS, str(ROOT / "packages/tyrano_export/transitions.js")],
        input=json.dumps({"case": case, **options}), text=True, encoding="utf-8",
        capture_output=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout, "scene transition did not settle"
    return json.loads(result.stdout)


def test_cold_preparation_keeps_old_picture_and_music_until_both_are_ready():
    result = run(cold=True)
    assert ["while-cold", "old", "old", True] in result["calls"]
    assert result["completed"] and result["frame"] == "new" and result["track"] == "next"
    assert result["calls"].count(["busy", True]) == result["calls"].count(["busy", False]) == 1


def test_picture_exchange_waits_for_visual_cover_and_music_fade_out():
    result = run(barriers=True)
    calls = result["calls"]
    assert ["before-covered", "old", "old"] in calls
    assert calls.index(["apply", "new"]) < calls.index(["music-commit", "next"])
    assert ["reveal", 500, "new", "old"] in calls
    assert ["music-in", 1000, "new"] in calls
    assert result["completed"] and not result["busy"]


@pytest.mark.parametrize("case", ["cancel-cold", "cancel-out"])
def test_cancellation_cannot_apply_a_late_stage_or_replace_current_track(case):
    result = run(case, cold=case == "cancel-cold", barriers=case == "cancel-out")
    assert result["completed"] is False
    assert result["frame"] == result["track"] == "old"
    assert ["apply", "new"] not in result["calls"] and not result["busy"]


def test_cancellation_after_commit_does_not_restart_or_duplicate_new_music():
    result = run("cancel-reveal", barriers=True)
    assert not result["completed"] and result["frame"] == "new" and result["track"] == "next"
    assert result["calls"].count(["music-commit", "next"]) == 1


def test_continue_changes_the_picture_without_touching_music_position_or_gain():
    result = run(action="continue")
    assert result["completed"] and result["frame"] == "new" and result["track"] == "old"
    assert not any(row[0] in {"music-out", "music-commit", "music-in"} for row in result["calls"])


def test_unchanged_stage_is_never_hidden_or_rebuilt():
    result = run(same=True, action="continue")
    assert not any(row[0] in {"cover", "apply", "reveal"} for row in result["calls"])


def test_failed_music_preparation_fades_old_audio_then_advances_picture():
    result = run(failAudio=True)
    assert result["completed"] and result["frame"] == "new" and result["track"] is None
    assert any(row[0] == "report" for row in result["calls"])
    assert result["calls"].index(["failed-music-fade"]) < result["calls"].index(["apply", "new"])


def test_known_skip_uses_zero_visual_and_audio_duration():
    calls = run("skip")["calls"]
    assert ["cover", "fade", 0, "old", "old"] in calls
    assert ["music-out", 0, "old"] in calls and ["music-in", 0, "new"] in calls


def test_inflight_skip_finishes_visual_and_audio_without_new_commit():
    result = run("finish", barriers=True)
    assert result["completed"] and result["calls"].count(["music-commit", "next"]) == 1
    assert ["finish-visual"] in result["calls"] and ["finish-audio"] in result["calls"]


def test_bfcache_resume_does_not_restart_an_already_committed_music_cue():
    result = run("resume")
    assert result["completed"] and ["apply", "new"] in result["calls"]
    assert not any(row[0] == "music-commit" for row in result["calls"])


def test_failed_switch_after_preparation_fades_old_music_then_commits_stage():
    result = run(failSwitch=True)
    assert result["completed"] and result["frame"] == "new" and result["track"] is None
    assert ["failed-music-fade"] in result["calls"]


def test_restored_older_stage_can_reenter_previous_transition_after_music_failure():
    result = run("retry-failed-switch")
    assert result["completed"] and result["frame"] == "new" and result["track"] is None
    assert result["calls"].count(["apply", "new"]) == 2


def test_failed_cg_preparation_commits_normal_target_without_stale_cg():
    result = run(cg=True, fallback=True, action="continue")
    assert result["completed"] and result["frame"] == "new" and result["cg"] is None
    assert result["track"] == "old" and any(row[0] == "report" for row in result["calls"])


def test_restore_cg_picture_does_not_replay_its_music_cue():
    result = run(cg=True, visualOnly=True)
    assert result["completed"] and result["cg"]["variant_id"] == "variant"
    assert result["track"] == "old"
    assert not any(row[0] in {"prepare-audio", "music-out", "music-commit", "music-in"}
                   for row in result["calls"])
