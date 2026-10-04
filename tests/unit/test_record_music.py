"""Recorder mixing uses the real owned BGM layer with simulated audio nodes."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.record.record_player import AUDIO_START

ROOT = Path(__file__).resolve().parents[2]

HARNESS = r'''
const fs=require('fs'),vm=require('vm');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const contexts=[],recorders=[],writes=[],released=[];
class Node {
 constructor(ctx){this.ctx=ctx;this.inputs=new Set();this.outputs=new Set();}
 connect(target){if(target.ctx!==this.ctx)throw Error('cross-context audio node');
  target.inputs.add(this);this.outputs.add(target);return target;}
 disconnect(target){for(const node of [...this.outputs])if(!target||node===target){
  node.inputs.delete(this);this.outputs.delete(node);}}
 energy(){return [...this.inputs].reduce((sum,node)=>sum+node.energy(),0);}
}
class Gain extends Node {
 constructor(ctx){super(ctx);this.ramp=null;this.ramps=[];ctx.gains.push(this);const self=this;
  this.gain={value:1,setValueAtTime(value){this.value=value;self.ramp=null;},
   cancelScheduledValues(){self.ramp=null;},linearRampToValueAtTime(value,end){
    self.ramp={from:this.value,to:value,start:ctx.currentTime,end};self.ramps.push({...self.ramp});}};}
 energy(){const r=this.ramp,ratio=r?Math.max(0,Math.min(1,(this.ctx.currentTime-r.start)/(r.end-r.start))):0;
  return super.energy()*(r?r.from+(r.to-r.from)*ratio:this.gain.value);}
}
class Destination extends Node {
 constructor(ctx){super(ctx);this.track={stops:0,stop(){this.stops++;}};
  this.stream={getTracks:()=>[this.track],energy:()=>this.track.stops?0:this.energy()};}
}
class AudioContext {
 constructor(){this.currentTime=0;this.state='suspended';this.destination=new Node(this);
  this.destinations=[];this.sources=[];this.gains=[];contexts.push(this);}
 createGain(){return new Gain(this);}
 createMediaStreamDestination(){const node=new Destination(this);this.destinations.push(node);return node;}
 createMediaStreamSource(stream){if(input.case==='mix-failure')throw Error('mix creation failed');
  const node=new Node(this);node.energy=()=>stream.energy();return node;}
 createBufferSource(){const node=new Node(this);node.active=false;node.starts=[];
  node.energy=()=>node.active&&this.state==='running'?node.buffer.amplitude:0;
  node.start=(...args)=>{node.active=true;node.starts.push(args);};
  node.stop=()=>{node.active=false;};this.sources.push(node);return node;}
 async decodeAudioData(){return {duration:30,amplitude:.6};}
 async resume(){this.state='running';}
 async close(){this.state='closed';}
}
class MediaRecorder {
 constructor(stream,options){if(input.case==='start-failure')throw Error('recorder unavailable');
  this.stream=stream;this.options=options;this.state='inactive';this.stops=0;recorders.push(this);}
 start(ms){this.interval=ms;this.state='recording';}
 stop(){this.stops++;if(input.case==='stop-failure')throw Error('stop failed');this.state='inactive';
  this.ondataavailable({data:{size:2,arrayBuffer:async()=>new Uint8Array([1,2]).buffer}});
  this.ondataavailable({data:{size:1,arrayBuffer:async()=>new Uint8Array([3]).buffer}});
  queueMicrotask(()=>this.onstop());}
}
const hctx=new AudioContext(),master=new Gain(hctx),voice=new Node(hctx);
voice.energy=()=>.8;master.gain.value=.6;voice.connect(master);master.connect(hctx.destination);
const h={ctx:hctx,masterGain:master,usingWebAudio:true,volume(){}};
const context={console,Promise,AbortController,AudioContext,MediaRecorder,
 btoa:value=>Buffer.from(value,'binary').toString('base64'),
 fetch:async()=>({ok:true,arrayBuffer:async()=>new ArrayBuffer(1)})};
context.window=context;context.Howler=h;
context.recordAudioChunk=async value=>{await new Promise(resolve=>setImmediate(resolve));
 if(input.case==='write-failure')throw Error('write failed');writes.push(value);};
vm.createContext(context);
let music=null;
if(!['legacy','fade_legacy'].includes(input.case)){
 vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
 const script={utterances:[{id:'first'},{id:'second'}],assets:[
  {id:'theme',kind:'music',filename:'theme.mp3'},{id:'newtheme',kind:'music',filename:'newtheme.mp3'}],
  music_cues:input.case==='empty'?[]:[
   {id:'open',utterance_id:'first',action:'play',asset_id:'theme',volume:.4,loop_start_seconds:5,loop_end_seconds:15},
   {id:'switch',utterance_id:'second',action:'play',asset_id:'newtheme',volume:.3}]};
 music=context.AutoDramaMusic.create(script);music.setVolume(.5);
}
(async()=>{
 let error=null,before=null,afterSwitch=null,snapshot=null,sameFinish=null,fadeLevels=[],fadeSnapshot=null;
 try{
  await vm.runInContext('('+input.audioStart+')()',context);
  if(music&&input.case!=='empty'){
   await music.unlock();await music.cue({id:'open',utterance_id:'first',action:'play',asset_id:'theme',
    volume:.4,loop_start_seconds:5,loop_end_seconds:15});snapshot=music.snapshot();
  }
  before=recorders[0].stream.energy();
  if(input.case==='switch'){
   await music.cue({id:'switch',utterance_id:'second',action:'play',asset_id:'newtheme',volume:.3});
   afterSwitch=recorders[0].stream.energy();
  }
  if(input.case==='recorder-error')recorders[0].onerror({error:{message:'capture failed'}});
  if(input.case.startsWith('fade_')){
   context.fadeRecordingAudio(400);context.fadeRecordingAudio(400);
   for(const time of [0,.1,.2,.4]){hctx.currentTime=time;fadeLevels.push(recorders[0].stream.energy());}
   fadeSnapshot=music?.snapshot()??null;
  }
  const one=context.finishRecordingAudio(),two=context.finishRecordingAudio();sameFinish=one===two;
  await one;
 }catch(value){error=value.message;}
 const speakerLevels=contexts.map(ctx=>ctx.destination.energy());
 process.stdout.write(JSON.stringify({error,before,afterSwitch,snapshot,sameFinish,writes,speakerLevels,fadeLevels,fadeSnapshot,
  ramps:contexts.flatMap(ctx=>ctx.gains.flatMap(node=>node.ramps)),
  recordingContexts:contexts.length,autoUnlock:h.autoUnlock,autoSuspend:h.autoSuspend,
  destinationStops:contexts.flatMap(ctx=>ctx.destinations.map(node=>node.track.stops)),
  recorders:recorders.map(rec=>({stops:rec.stops,interval:rec.interval,mime:rec.options.mimeType})),
  sources:contexts.flatMap(ctx=>ctx.sources.map(node=>({starts:node.starts,loop:node.loop,
   a:node.loopStart,b:node.loopEnd}))) }));
})().catch(error=>{console.error(error);process.exit(1);});
'''


def run(case):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for owned recorder mixing tests")
    result = subprocess.run(
        [node, "-e", HARNESS, str(ROOT / "packages/tyrano_export/music.js")],
        input=json.dumps({"case": case, "audioStart": AUDIO_START}),
        capture_output=True, encoding="utf-8", text=True, check=False, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_recorder_mixes_voice_and_music_after_their_independent_gain_without_speakers():
    result = run("mix")
    assert result["error"] is None
    assert result["before"] == pytest.approx(0.8 * 0.6 + 0.6 * 0.4 * 0.5)
    assert result["recordingContexts"] == 2
    assert result["speakerLevels"] == [0, 0]
    assert result["snapshot"]["track_volume"] == 0.4
    assert result["snapshot"]["user_volume"] == 0.5
    assert result["snapshot"]["phase"] == "intro"
    assert result["sources"][0] == {"starts": [[0, 0]], "loop": True, "a": 5, "b": 15}
    assert result["sameFinish"] is True
    assert result["writes"] == ["AQI=", "Aw=="]
    assert result["recorders"] == [{"stops": 1, "interval": 1000, "mime": "audio/webm;codecs=opus"}]
    assert result["destinationStops"] == [1, 1]


def test_recording_music_switch_retains_user_gain_and_does_not_add_a_second_mix():
    result = run("switch")
    assert result["error"] is None
    assert result["afterSwitch"] == pytest.approx(0.8 * 0.6 + 0.6 * 0.3 * 0.5)
    assert result["recordingContexts"] == 2 and result["speakerLevels"] == [0, 0]


@pytest.mark.parametrize("case, level", [("fade_mix", 0.6), ("fade_legacy", 0.48)])
def test_recorder_final_fade_ramps_entire_mix_without_changing_reader_music(case, level):
    result = run(case)
    assert result["error"] is None
    assert result["fadeLevels"] == pytest.approx([level, level * .75, level * .5, 0])
    assert result["ramps"] == [{"from": 1, "to": 0, "start": 0, "end": .4}], "fade starts only once"
    if case == "fade_mix":
        assert result["fadeSnapshot"] == result["snapshot"], "recording never stops/rewinds reader BGM"
    assert result["writes"] == ["AQI=", "Aw=="]
    assert all(value == 0 for value in result["speakerLevels"])


@pytest.mark.parametrize("case", ["legacy", "empty"])
def test_old_publications_without_music_keep_the_original_voice_recording(case):
    result = run(case)
    assert result["error"] is None
    assert result["recordingContexts"] == 1
    assert result["before"] == pytest.approx(0.8 * 0.6)
    assert result["speakerLevels"] == [0] and result["writes"] == ["AQI=", "Aw=="]


@pytest.mark.parametrize("case, error", [
    ("start-failure", "recorder unavailable"), ("mix-failure", "mix creation failed"),
    ("stop-failure", "stop failed"), ("write-failure", "write failed"),
    ("recorder-error", "capture failed"),
])
def test_recording_failures_release_both_streams_and_keep_the_page_silent(case, error):
    result = run(case)
    assert result["error"] == error
    assert result["speakerLevels"] == [0, 0]
    assert result["destinationStops"] == [1, 1]
