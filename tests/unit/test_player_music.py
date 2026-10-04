"""Browser-owned BGM playback, loop phase, immutable saves and async races."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MUSIC = ROOT / "packages/tyrano_export/music.js"

HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const fetches = [], decodes = [], sources = [], contexts = [], pending = new Map();
const snapshots = {}, results = [], reports = [], heldFetch = [], heldDecode = [], captures = new Map();
const taps = [];
const signals = new Map(), timers = new Map(), heldCommits = new Map(), commits = [];
let wallMs = 0, timerId = 0;
const schedule = (fn, delay) => {
  const id = ++timerId; timers.set(id, {fn,due:wallMs + delay}); return id;
};
let holdFetch = false, holdDecode = false;
const media = new Map(input.script.assets.map((asset, index) =>
  ['./data/bgm/' + asset.filename, {index, duration:input.durations?.[asset.id] ?? 30}]));
const response = (url, ok = true) => ({ok, arrayBuffer:async() =>
  new Uint8Array([media.get(url)?.index ?? 0]).buffer});
class FakeAudioContext {
  constructor() {this.currentTime = 0; this.state = 'suspended'; this.sampleRate = input.contextSampleRate;
    this.destination = {kind:'music-output'};
    this.gains = []; contexts.push(this);}
  createGain() {
    const gain = {value:1, values:[], connects:[], outputs:[], disconnects:0,
      ramps:[], cancels:0,
      setValueAtTime(value, at) {this.value = value; this.values.push({value,at});},
      linearRampToValueAtTime(value, at) {
        ramp = {from:this.value,value,start:ctx.currentTime,end:at}; this.ramps.push({...ramp});
      },
      cancelScheduledValues() {fixed = this.value; ramp = null; this.cancels++;}};
    const ctx = this;
    let fixed = 1, ramp = null;
    Object.defineProperty(gain, 'value', {enumerable:true,
      get:() => ramp ? ramp.from + (ramp.value - ramp.from) *
        Math.max(0, Math.min(1, (ctx.currentTime - ramp.start) / (ramp.end - ramp.start))) : fixed,
      set:value => {fixed = value; ramp = null;}});
    const node = {kind:this.gains.length ? 'transition-gain' : 'scene-gain', gain, connect(target) {
        if (target.kind === 'music-capture' && input.tapConnectError) throw Error('tap connect failed');
        gain.connects.push(target.kind);
        if (!gain.outputs.includes(target)) gain.outputs.push(target);
      },
      disconnect(target) {
        if (target?.kind === 'music-output' && input.speakerDisconnectError) throw Error('speaker disconnect failed');
        gain.disconnects++;
        gain.outputs = target ? gain.outputs.filter(value => value !== target) : [];
      }};
    this.gains.push(node); return node;
  }
  createMediaStreamDestination() {
    if (input.tapCreateError) throw Error('tap creation failed');
    const track = {stops:0, stop() {this.stops++;}};
    const node = {kind:'music-capture', disconnects:0, track,
      stream:{getTracks:() => [track]}, disconnect() {this.disconnects++;}};
    taps.push(node); return node;
  }
  createBufferSource() {
    const record = {loop:false, loopStart:0, loopEnd:0, starts:[], stops:0, disconnects:0};
    const node = {get loop() {return record.loop}, set loop(value) {record.loop = value},
      get loopStart() {return record.loopStart}, set loopStart(value) {record.loopStart = value},
      get loopEnd() {return record.loopEnd}, set loopEnd(value) {record.loopEnd = value},
      set buffer(value) {record.duration = value.duration},
      connect(target) {record.ownGain = thisContext.gains.includes(target)},
      disconnect() {record.disconnects++}, stop() {record.stops++},
      start(...args) {record.starts.push(args); record.startedAt = thisContext.currentTime;
        if (input.startError) throw Error('start failed');}};
    const thisContext = this; record.node = node; sources.push(record); return node;
  }
  async resume() {if (input.resumeError) throw Error('autoplay denied');
    if (!input.resumeStaysSuspended) this.state = 'running';}
  async close() {this.state = 'closed';}
  async decodeAudioData(bytes) {
    const index = new Uint8Array(bytes)[0], asset = input.script.assets[index];
    decodes.push(asset.id);
    if (input.decodeError) throw Error('invalid MP3');
    const details = input.buffers?.[asset.id] || {};
    const duration = details.duration ?? (details.length !== undefined && details.sampleRate > 0 ?
      details.length / details.sampleRate : input.durations?.[asset.id] ?? 30);
    const buffer = {...details,duration};
    if (holdDecode) await new Promise(resolve => heldDecode.push(resolve));
    return buffer;
  }
}
const context = {console, AbortController, setTimeout:schedule, clearTimeout:id => timers.delete(id),
  AudioContext:input.noAudio ? undefined : FakeAudioContext,
  fetch:async (url, options = {}) => {
    const record = {url, aborted:false}; fetches.push(record);
    options.signal?.addEventListener('abort', () => {record.aborted = true;});
    if (holdFetch) return await new Promise((resolve, reject) => {
      heldFetch.push({resolve:ok => resolve(response(url, ok))});
      if (!input.ignoreAbort) options.signal?.addEventListener('abort', () => {
        const error = Error('aborted'); error.name = 'AbortError'; reject(error);
      });
    });
    if (input.fetchError) throw Error('offline');
    return response(url, !input.fetchStatusError);
  }};
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const adapterCalls = [], previews = new Map();
const adapter = async (url, {signal}) => {
  const record = {url,aborted:false}; adapterCalls.push(record);
  signal.addEventListener('abort', () => {record.aborted = true;}, {once:true});
  return {duration:media.get(url)?.duration ?? 30,sampleRate:48000};
};
const external = input.factoryExternal ? new FakeAudioContext() : null;
const player = context.AutoDramaMusic.create(input.script, {report:message => reports.push(message),
  ...(external ? {context:external} : {}), ...(input.factoryAdapter ? {loadAudio:adapter} : {})});
const flush = () => new Promise(resolve => setImmediate(resolve));
const cue = op => op.value ?? input.script.music_cues?.find(value => value.id === op.id);
const options = op => ({...op.options,
  ...(op.signal ? {signal:signals.get(op.signal).signal} : {}),
  ...(op.holdCommit || op.rejectCommit || op.observeCommit ? {beforeCommit:async ({signal}) => {
    commits.push({key:op.key,signal,snapshot:player.snapshot(),fade:contexts[0].gains[1].gain.value});
    if (op.rejectCommit) throw Error('visual commit failed');
    if (op.holdCommit) await new Promise(resolve => heldCommits.set(op.key, resolve));
  }} : {})});
(async() => {
  for (const op of input.ops) {
    let value;
    switch (op.op) {
      case 'cue': value = await player.cue(cue(op), op.options); break;
      case 'beginCue': pending.set(op.key, player.cue(cue(op), op.options)); break;
      case 'prepare': value = await player.prepareCue(op.id, options(op)); break;
      case 'beginPrepare': pending.set(op.key, player.prepareCue(op.id, options(op))); break;
      case 'transition': value = await player.transitionCue(op.id, options(op)); break;
      case 'beginTransition': pending.set(op.key, player.transitionCue(op.id, options(op))); break;
      case 'stopFaded': value = await player.stopFaded(options(op)); break;
      case 'beginStopFaded': pending.set(op.key, player.stopFaded(options(op))); break;
      case 'finishTransition': value = player.finishTransition(); break;
      case 'resolveCommit': heldCommits.get(op.key)(); break;
      case 'contextState': contexts[op.index ?? 0].state = op.value; break;
      case 'lookup': value = {cue:player.getCue(op.id),frozen:Object.isFrozen(player.getCue(op.id))}; break;
      case 'newSignal': {
        const controller = new AbortController(), listeners = new Set(), signal = controller.signal;
        const add = signal.addEventListener.bind(signal), remove = signal.removeEventListener.bind(signal);
        signal.addEventListener = (name,fn,opts) => {listeners.add(fn);add(name,fn,opts);};
        signal.removeEventListener = (name,fn,opts) => {listeners.delete(fn);remove(name,fn,opts);};
        signals.set(op.key,{controller,signal,listeners}); break;
      }
      case 'abortSignal': signals.get(op.key).controller.abort(); break;
      case 'createPreview': previews.set(op.key, context.AutoDramaMusic.create(input.script,
        {context:new FakeAudioContext(),loadAudio:adapter,registerOutput:false})); break;
      case 'previewCue': value = await previews.get(op.key).cue(cue(op), op.options); break;
      case 'previewDispose': previews.get(op.key).dispose(); break;
      case 'wait': value = await pending.get(op.key); break;
      case 'unlock': value = await player.unlock(); break;
      case 'resume': value = await player.resume(); break;
      case 'pause': player.pause(); break;
      case 'stop': player.stop(); break;
      case 'dispose': player.dispose(); break;
      case 'volume': value = player.setVolume(op.value); break;
      case 'advance': {
        contexts.forEach(ctx => {if (ctx.state === 'running') ctx.currentTime += op.seconds;});
        wallMs += op.seconds * 1000;
        for (;;) {
          const ready = [...timers.entries()].filter(([,timer]) => timer.due <= wallMs)
            .sort((a,b) => a[1].due - b[1].due)[0];
          if (!ready) break;
          timers.delete(ready[0]); ready[1].fn(); await flush();
        }
        break;
      }
      case 'snapshot': snapshots[op.key] = player.snapshot(); break;
      case 'mutateSnapshot': Object.assign(snapshots[op.key], op.patch); break;
      case 'mutateScript': Object.assign(input.script[op.collection][op.index], op.patch); break;
      case 'restore': {
        const saved = typeof op.value === 'string' ? snapshots[op.value] : op.value;
        value = await player.restore(op.patch ? {...saved,...op.patch} : saved); break;
      }
      case 'beginRestore': pending.set(op.key, player.restore(snapshots[op.value])); break;
      case 'holdFetch': holdFetch = op.value; break;
      case 'holdDecode': holdDecode = op.value; break;
      case 'resolveFetch': heldFetch[op.index].resolve(op.ok !== false); break;
      case 'resolveDecode': heldDecode[op.index](); break;
      case 'end': sources[op.index].node.onended?.(); break;
      case 'capture': {
        try {
          const captured = op.local ? player.captureOutput() : context.AutoDramaMusic.captureOutput();
          captures.set(op.key, captured); value = captured ? {active:true} : null;
        } catch (error) {value = {error:error.message};}
        break;
      }
      case 'sameCapture': value = captures.get(op.one) === captures.get(op.two); break;
      case 'releaseCapture': captures.get(op.key)?.release(op.options); break;
      case 'graph': snapshots[op.key] = contexts.map(ctx => ctx.gains.map(node => ({
        gain:node.gain.value, outputs:node.gain.outputs.map(target => target.kind)}))); break;
      default: throw Error('unknown harness operation');
    }
    await flush();
    if (value !== undefined) results.push({op:op.op,value});
  }
  process.stdout.write(JSON.stringify({snapshots,results,reports,fetches,decodes,adapterCalls,
    commits:commits.map(({signal,...record}) => ({...record,aborted:signal.aborted})),
    signalListeners:[...signals.values()].map(value => value.listeners.size),
    signalsAborted:[...signals.values()].map(value => value.signal.aborted),
    sources:sources.map(({node,...record}) => record),
    taps:taps.map(node => ({stops:node.track.stops,disconnects:node.disconnects})), pendingTimers:timers.size,
    contexts:contexts.map(ctx => ({state:ctx.state,currentTime:ctx.currentTime,
      gains:ctx.gains.map(node => ({...node.gain,outputs:node.gain.outputs.map(target => target.kind)}))}))}));
})().catch(error => {console.error(error); process.exitCode = 1;});
"""


@pytest.fixture
def script() -> dict:
    return {
        "utterances": [{"id": "first"}, {"id": "second"}],
        "assets": [
            {"id": "a", "kind": "music", "filename": "theme-a.mp3"},
            {"id": "b", "kind": "music", "filename": "theme-b.mp3"},
            {"id": "c", "kind": "music", "filename": "theme-c.mp3"},
            {"id": "d", "kind": "music", "filename": "theme-d.mp3"},
            {"id": "voice", "kind": "audio", "filename": "voice.mp3"},
        ],
        "music_cues": [
            {"id": "open", "utterance_id": "first", "action": "play", "asset_id": "a",
             "loop_start_seconds": 5, "loop_end_seconds": 15, "volume": 0.4},
            {"id": "once", "utterance_id": "second", "action": "play", "asset_id": "b", "volume": 0.75},
            {"id": "third", "utterance_id": "second", "action": "play", "asset_id": "c",
             "loop_start_seconds": 0, "loop_end_seconds": 10},
            {"id": "fourth", "utterance_id": "second", "action": "play", "asset_id": "d"},
            {"id": "keep", "utterance_id": "second", "action": "continue", "volume": 0.35},
            {"id": "silent", "utterance_id": "second", "action": "stop"},
        ],
    }


def run(script: dict, ops: list[dict], **options) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for browser music regressions")
    result = subprocess.run(
        [node, "-e", HARNESS, str(MUSIC)],
        input=json.dumps({"script": script, "ops": ops, **options}),
        capture_output=True, text=True, encoding="utf-8", check=True, timeout=15,
    )
    return json.loads(result.stdout)


def playing(*ops: dict) -> list[dict]:
    return [{"op": "unlock"}, {"op": "cue", "id": "open"}, *ops]


def test_native_intro_then_loop_uses_audio_render_thread(script):
    result = run(script, playing(
        {"op": "advance", "seconds": 4}, {"op": "snapshot", "key": "intro"},
        {"op": "advance", "seconds": 11}, {"op": "snapshot", "key": "boundary"},
        {"op": "advance", "seconds": 23}, {"op": "snapshot", "key": "loop"},
    ))
    assert result["sources"][0]["starts"] == [[0, 0]]
    assert result["sources"][0]["loop"] is True
    assert result["sources"][0]["loopStart"] == 5
    assert result["sources"][0]["loopEnd"] == 15
    assert result["snapshots"]["intro"]["position_seconds"] == 4
    assert result["snapshots"]["intro"]["phase"] == "intro"
    assert result["snapshots"]["boundary"]["position_seconds"] == 5
    assert result["snapshots"]["boundary"]["phase"] == "loop"
    assert result["snapshots"]["loop"]["position_seconds"] == 8


def test_pause_resume_preserves_loop_phase_and_cached_pcm(script):
    result = run(script, playing(
        {"op": "advance", "seconds": 17}, {"op": "pause"},
        {"op": "advance", "seconds": 100}, {"op": "snapshot", "key": "paused"},
        {"op": "resume"}, {"op": "advance", "seconds": 9},
        {"op": "snapshot", "key": "resumed"},
    ))
    assert result["snapshots"]["paused"]["position_seconds"] == 7
    assert result["snapshots"]["paused"]["paused"] is True
    assert result["sources"][1]["starts"] == [[0, 7]]
    assert result["sources"][0]["stops"] == 1
    assert result["snapshots"]["resumed"]["position_seconds"] == 6
    assert result["snapshots"]["resumed"]["phase"] == "loop"
    assert result["decodes"] == ["a"]


@pytest.mark.parametrize("elapsed,phase,position", [(2, "intro", 2), (17, "loop", 7)])
def test_save_restore_resumes_correct_phase(script, elapsed, phase, position):
    result = run(script, playing(
        {"op": "volume", "value": 0.5}, {"op": "advance", "seconds": elapsed},
        {"op": "snapshot", "key": "saved"}, {"op": "cue", "id": "once"},
        {"op": "restore", "value": "saved"}, {"op": "snapshot", "key": "restored"},
    ))
    assert result["sources"][2]["starts"] == [[0, position]]
    assert result["snapshots"]["restored"] == result["snapshots"]["saved"]
    assert result["snapshots"]["restored"]["phase"] == phase
    assert result["decodes"] == ["a", "b"]
    assert result["contexts"][0]["gains"][0]["value"] == pytest.approx(0.2)


def test_restore_paused_waits_until_resume(script):
    result = run(script, playing(
        {"op": "advance", "seconds": 3}, {"op": "pause"},
        {"op": "snapshot", "key": "saved"}, {"op": "stop"},
        {"op": "restore", "value": "saved"}, {"op": "snapshot", "key": "restored"},
        {"op": "advance", "seconds": 20}, {"op": "resume"},
        {"op": "snapshot", "key": "resumed"},
    ))
    assert len(result["sources"]) == 2
    assert result["sources"][1]["startedAt"] == 23
    assert result["sources"][1]["starts"] == [[0, 3]]
    assert result["snapshots"]["restored"]["paused"] is True
    assert result["snapshots"]["resumed"]["position_seconds"] == 3


def test_track_user_gain_independent_continue_does_not_reset(script):
    result = run(script, playing(
        {"op": "volume", "value": 0.25}, {"op": "advance", "seconds": 18},
        {"op": "cue", "id": "keep"}, {"op": "snapshot", "key": "kept"},
        {"op": "volume", "value": 0}, {"op": "snapshot", "key": "muted"},
    ))
    assert len(result["sources"]) == 1
    assert result["sources"][0]["ownGain"] is True
    assert result["snapshots"]["kept"]["cue_id"] == "open"
    assert result["snapshots"]["kept"]["position_seconds"] == 8
    assert result["snapshots"]["kept"]["track_volume"] == 0.4
    assert result["contexts"][0]["gains"][0]["values"][-2]["value"] == pytest.approx(0.1)
    assert result["contexts"][0]["gains"][0]["value"] == 0
    assert result["snapshots"]["muted"]["phase"] == "loop"


@pytest.mark.parametrize("value", [-0.01, 1.01, "0.5", None, True])
def test_user_volume_rejects_invalid_values_without_changing_track(script, value):
    result = run(script, playing({"op": "volume", "value": value}, {"op": "snapshot", "key": "after"}))
    assert result["results"][-1]["value"] is False
    assert result["snapshots"]["after"]["user_volume"] == 1


def test_one_shot_finishes_without_full_track_loop_or_replaying_on_restore(script):
    result = run(script, [
        {"op": "unlock"}, {"op": "cue", "id": "once"},
        {"op": "advance", "seconds": 31}, {"op": "end", "index": 0},
        {"op": "snapshot", "key": "finished"}, {"op": "restore", "value": "finished"},
        {"op": "resume"}, {"op": "snapshot", "key": "restored"},
    ])
    assert len(result["sources"]) == 1
    assert result["sources"][0]["loop"] is False
    assert result["snapshots"]["finished"]["phase"] == "ended"
    assert result["snapshots"]["restored"]["position_seconds"] == 30
    assert result["snapshots"]["restored"]["phase"] == "ended"


def test_one_shot_elapsed_snapshot_handles_delayed_ended_callback(script):
    result = run(script, [{"op": "unlock"}, {"op": "cue", "id": "once"},
                          {"op": "advance", "seconds": 35}, {"op": "snapshot", "key": "ended"}])
    assert result["snapshots"]["ended"]["phase"] == "ended"
    assert result["snapshots"]["ended"]["position_seconds"] == 30


@pytest.mark.parametrize("op", [{"op": "stop"}, {"op": "cue", "id": "silent"},
                                {"op": "restore", "value": None}])
def test_stop_removes_audio_and_position(script, op):
    result = run(script, playing(op, {"op": "snapshot", "key": "stopped"}, {"op": "resume"}))
    assert result["snapshots"]["stopped"] is None
    assert len(result["sources"]) == 1
    assert result["sources"][0]["stops"] == 1
    assert result["contexts"][0]["gains"][0]["value"] == 0


def test_dispose_closes_only_owned_context_and_is_idempotent(script):
    result = run(script, playing({"op": "dispose"}, {"op": "dispose"}, {"op": "stop"},
                                {"op": "resume"}, {"op": "cue", "id": "open"},
                                {"op": "snapshot", "key": "disposed"}))
    assert result["contexts"][0]["state"] == "closed"
    assert len(result["contexts"]) == 1
    assert result["snapshots"]["disposed"] is None
    assert result["sources"][0]["stops"] == 1


def test_old_script_without_cues_never_creates_context_or_fetches(script):
    del script["music_cues"]
    result = run(script, [{"op": "unlock"}, {"op": "cue", "value": {"action": "play"}},
                          {"op": "resume"}, {"op": "snapshot", "key": "legacy"}])
    assert result["contexts"] == result["fetches"] == result["sources"] == []
    assert result["snapshots"]["legacy"] is None
    assert result["reports"] == []


def test_fetch_switch_aborts_own_request_and_late_response_cannot_start_old_track(script):
    result = run(script, [
        {"op": "unlock"}, {"op": "holdFetch", "value": True},
        {"op": "beginCue", "id": "open", "key": "old"},
        {"op": "holdFetch", "value": False}, {"op": "cue", "id": "once"},
        {"op": "resolveFetch", "index": 0}, {"op": "wait", "key": "old"},
        {"op": "snapshot", "key": "current"},
    ], ignoreAbort=True)
    assert result["fetches"][0]["aborted"] is True
    assert result["decodes"] == ["b"]
    assert len(result["sources"]) == 1
    assert result["snapshots"]["current"]["asset_id"] == "b"
    assert result["results"][-1]["value"] is False
    assert result["reports"] == []


def test_stop_during_decode_ignores_late_non_abortable_result(script):
    result = run(script, [
        {"op": "unlock"}, {"op": "holdDecode", "value": True},
        {"op": "beginCue", "id": "open", "key": "old"}, {"op": "stop"},
        {"op": "resolveDecode", "index": 0}, {"op": "wait", "key": "old"},
        {"op": "snapshot", "key": "stopped"},
    ])
    assert result["sources"] == []
    assert result["snapshots"]["stopped"] is None
    assert result["results"][-1]["value"] is False
    assert result["reports"] == []


def test_same_asset_retry_never_reuses_cancelled_fetch_promise(script):
    result = run(script, [
        {"op": "unlock"}, {"op": "holdFetch", "value": True},
        {"op": "beginCue", "id": "open", "key": "old"}, {"op": "stop"},
        {"op": "holdFetch", "value": False}, {"op": "cue", "id": "open"},
        {"op": "wait", "key": "old"}, {"op": "snapshot", "key": "new"},
    ])
    assert len(result["fetches"]) == 2
    assert result["fetches"][0]["aborted"] is True
    assert result["decodes"] == ["a"]
    assert len(result["sources"]) == 1
    assert result["snapshots"]["new"]["asset_id"] == "a"


def test_pause_while_loading_prevents_autoplay_until_resume(script):
    result = run(script, [
        {"op": "unlock"}, {"op": "holdFetch", "value": True},
        {"op": "beginCue", "id": "open", "key": "pending"}, {"op": "pause"},
        {"op": "resolveFetch", "index": 0}, {"op": "wait", "key": "pending"},
        {"op": "snapshot", "key": "paused"}, {"op": "advance", "seconds": 20},
        {"op": "resume"}, {"op": "snapshot", "key": "started"},
    ])
    assert result["snapshots"]["paused"]["paused"] is True
    assert result["snapshots"]["paused"]["position_seconds"] == 0
    assert len(result["sources"]) == 1
    assert result["sources"][0]["startedAt"] == 20


def test_restore_inflight_invalidated_by_new_cue(script):
    result = run(script, playing(
        {"op": "advance", "seconds": 18}, {"op": "snapshot", "key": "saved"},
        # Evict A before restoring it, so restore must load asynchronously.
        {"op": "cue", "id": "once"}, {"op": "cue", "id": "third"},
        {"op": "holdFetch", "value": True},
        {"op": "beginRestore", "value": "saved", "key": "old"},
        {"op": "holdFetch", "value": False}, {"op": "cue", "id": "once"},
        {"op": "resolveFetch", "index": 0}, {"op": "wait", "key": "old"},
        {"op": "snapshot", "key": "current"},
    ), ignoreAbort=True)
    assert result["snapshots"]["current"]["asset_id"] == "b"
    assert result["decodes"] == ["a", "b", "c"]
    assert len(result["sources"]) == 4


def test_decoded_pcm_cache_is_two_entry_lru(script):
    result = run(script, playing(
        {"op": "cue", "id": "once"}, {"op": "cue", "id": "open"},
        {"op": "cue", "id": "third"}, {"op": "cue", "id": "open"},
        {"op": "cue", "id": "once"},
    ))
    assert result["decodes"] == ["a", "b", "c", "b"]


@pytest.mark.parametrize("field,value", [
    ("asset_id", "b"), ("loop_start_seconds", 6), ("loop_end_seconds", 16),
    ("volume", 0.5), ("utterance_id", "second"), ("id", "unknown"),
])
def test_mutated_or_unknown_cue_cannot_select_audio(script, field, value):
    supplied = {**script["music_cues"][0], field: value}
    result = run(script, [{"op": "cue", "value": supplied}])
    assert result["results"][0]["value"] is False
    assert result["contexts"] == result["fetches"] == []


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("cue_id", "once"), ("asset_id", "b"),
    ("track_volume", 0.7), ("user_volume", 2), ("user_volume", "0.5"),
    ("paused", "false"), ("position_seconds", -1), ("phase", "ended"),
    ("loop_start_seconds", 0), ("loop_end_seconds", 30),
])
def test_tampered_save_rejected_without_interrupting_current_track(script, field, value):
    result = run(script, playing(
        {"op": "advance", "seconds": 2}, {"op": "snapshot", "key": "saved"},
        {"op": "restore", "value": "saved", "patch": {field: value}},
        {"op": "snapshot", "key": "after"},
    ))
    assert result["results"][-1]["value"] is False
    assert result["snapshots"]["after"] == result["snapshots"]["saved"]
    assert len(result["sources"]) == 1


@pytest.mark.parametrize("phase,position", [("intro", 15), ("loop", 4), ("loop", 15)])
def test_saved_phase_bounds_are_strict(script, phase, position):
    result = run(script, playing(
        {"op": "snapshot", "key": "saved"},
        {"op": "restore", "value": "saved", "patch": {"phase": phase, "position_seconds": position}},
    ))
    assert result["results"][-1]["value"] is False


@pytest.mark.parametrize("phase,position", [("once", 31), ("ended", 29)])
def test_saved_one_shot_position_rechecked_against_decoded_duration(script, phase, position):
    result = run(script, [{"op": "unlock"}, {"op": "cue", "id": "once"},
                          {"op": "snapshot", "key": "saved"},
                          {"op": "restore", "value": "saved",
                           "patch": {"phase": phase, "position_seconds": position}},
                          {"op": "snapshot", "key": "invalid"}])
    assert result["results"][-1]["value"] is False
    assert result["snapshots"]["invalid"] is None
    assert len(result["sources"]) == 1


@pytest.mark.parametrize("name", ["../theme.mp3", "x/theme.mp3", "x\\theme.mp3", "https://x/t.mp3",
                                  "theme.mp3?x", "theme.wav", "CON.mp3", "a\n.mp3"])
def test_manifest_music_filename_cannot_escape_owned_folder(script, name):
    script["assets"][0]["filename"] = name
    result = run(script, [{"op": "cue", "id": "open"}])
    assert result["fetches"] == result["contexts"] == []


@pytest.mark.parametrize("patch", [
    {"asset_id": "voice"}, {"asset_id": "missing"}, {"loop_start_seconds": -1},
    {"loop_end_seconds": 5}, {"loop_end_seconds": None}, {"volume": 2},
    {"utterance_id": "unpublished"}, {"action": "javascript"},
])
def test_invalid_manifest_cues_remain_inert(script, patch):
    script["music_cues"][0].update(patch)
    result = run(script, [{"op": "cue", "id": "open"}])
    assert result["fetches"] == result["contexts"] == []


def test_loop_end_rechecked_after_mp3_decode(script):
    result = run(script, [{"op": "cue", "id": "open"}, {"op": "snapshot", "key": "bad"}],
                 durations={"a": 14.9})
    assert result["decodes"] == ["a"]
    assert result["sources"] == []
    assert result["snapshots"]["bad"] is None
    assert len(result["reports"]) == 1


@pytest.mark.parametrize("option", ["fetchError", "fetchStatusError", "decodeError", "noAudio", "startError"])
def test_load_errors_are_reported_without_hanging_or_starting_stale_audio(script, option):
    result = run(script, [{"op": "cue", "id": "open"}, {"op": "snapshot", "key": "failed"}],
                 **{option: True})
    assert result["results"][0]["value"] is False
    assert result["snapshots"]["failed"] is None
    assert len(result["reports"]) == 1


def test_browser_suspension_does_not_falsely_advance_snapshot_position(script):
    result = run(script, [{"op": "cue", "id": "open"},
                          {"op": "advance", "seconds": 100}, {"op": "snapshot", "key": "blocked"},
                          {"op": "unlock"}, {"op": "advance", "seconds": 4},
                          {"op": "snapshot", "key": "unlocked"}])
    assert result["snapshots"]["blocked"]["position_seconds"] == 0
    assert result["snapshots"]["unlocked"]["position_seconds"] == 4


def test_autoplay_resume_rejection_is_reported(script):
    result = run(script, [{"op": "unlock"}], resumeError=True)
    assert result["results"][0]["value"] is False
    assert len(result["reports"]) == 1


def test_resolved_resume_does_not_claim_autoplay_when_audio_context_remains_suspended(script):
    result = run(script, [{"op": "unlock"}], resumeStaysSuspended=True)
    assert result["results"][0]["value"] is False
    assert result["sources"] == []


def test_duplicate_asset_and_cue_ids_cannot_ambiguously_choose_music(script):
    script["assets"].append({"id": "a", "kind": "music", "filename": "other.mp3"})
    script["music_cues"].append({**script["music_cues"][1], "asset_id": "c"})
    result = run(script, [{"op": "cue", "id": "open"}, {"op": "cue", "id": "once"}])
    assert result["contexts"] == result["fetches"] == []


def test_cue_map_is_pinned_to_factory_input_values(script):
    canonical = script["music_cues"][0].copy()
    result = run(script, [
        {"op": "mutateScript", "collection": "music_cues", "index": 0, "patch": {"volume": 0.9}},
        {"op": "mutateScript", "collection": "assets", "index": 0,
         "patch": {"filename": "https://untrusted.invalid/theme.mp3"}},
        {"op": "cue", "value": canonical}, {"op": "snapshot", "key": "canonical"},
    ])
    assert result["snapshots"]["canonical"]["track_volume"] == 0.4
    assert result["fetches"][0]["url"] == "./data/bgm/theme-a.mp3"


def test_rapid_switches_bound_native_decoding_and_skip_obsolete_queued_cues(script):
    ids = ["open", "once", "third", "fourth", "open", "once"]
    result = run(script, [
        {"op": "unlock"}, {"op": "holdDecode", "value": True},
        *[{"op": "beginCue", "id": cue_id, "key": str(index)} for index, cue_id in enumerate(ids)],
        {"op": "resolveDecode", "index": 0}, {"op": "resolveDecode", "index": 1},
        *[{"op": "wait", "key": str(index)} for index in range(len(ids))],
        {"op": "snapshot", "key": "last"},
    ])
    assert result["decodes"] == ["a", "b"]
    assert len(result["sources"]) == 1
    assert result["snapshots"]["last"]["asset_id"] == "b"
    assert [item["value"] for item in result["results"] if item["op"] == "wait"] == [False] * 5 + [True]


@pytest.mark.parametrize("initial,later,success", [(4, 31, True), (31, 4, False)])
def test_pending_restore_uses_immutable_validated_save_copy(script, initial, later, success):
    result = run(script, [
        {"op": "unlock"}, {"op": "cue", "id": "once"}, {"op": "advance", "seconds": 4},
        {"op": "snapshot", "key": "saved"},
        {"op": "cue", "id": "open"}, {"op": "cue", "id": "third"},
        {"op": "mutateSnapshot", "key": "saved", "patch": {"position_seconds": initial}},
        {"op": "holdFetch", "value": True},
        {"op": "beginRestore", "value": "saved", "key": "restoring"},
        {"op": "mutateSnapshot", "key": "saved", "patch": {"position_seconds": later}},
        {"op": "resolveFetch", "index": 0}, {"op": "wait", "key": "restoring"},
        {"op": "snapshot", "key": "after"},
    ])
    assert result["results"][-1]["value"] is success
    if success:
        assert result["sources"][-1]["starts"] == [[0, 4]]
        assert result["snapshots"]["after"]["position_seconds"] == 4
    else:
        assert len(result["sources"]) == 3
        assert result["snapshots"]["after"] is None


def test_capture_before_first_cue_routes_future_tracks_after_independent_gain(script):
    result = run(script, [
        {"op": "capture", "key": "recording"}, {"op": "graph", "key": "prepared"},
        {"op": "volume", "value": 0.5}, {"op": "unlock"}, {"op": "cue", "id": "open"},
        {"op": "graph", "key": "playing"}, {"op": "snapshot", "key": "state"},
        {"op": "cue", "id": "once"}, {"op": "graph", "key": "switched"},
    ])
    assert len(result["contexts"]) == len(result["taps"]) == 1
    assert result["snapshots"]["prepared"][0][1]["outputs"] == ["music-capture"]
    assert result["snapshots"]["playing"][0][0] == {"gain": 0.2, "outputs": ["transition-gain"]}
    assert result["snapshots"]["switched"][0][0] == {"gain": 0.375, "outputs": ["transition-gain"]}
    assert result["snapshots"]["playing"][0][1] == {"gain": 1, "outputs": ["music-capture"]}
    assert result["snapshots"]["state"]["track_volume"] == 0.4
    assert result["snapshots"]["state"]["user_volume"] == 0.5
    assert all(source["ownGain"] for source in result["sources"])


def test_capture_initializes_suspended_context_without_starting_audio(script):
    result = run(script, [{"op": "capture", "key": "prepared"}])
    assert len(result["contexts"]) == len(result["taps"]) == 1
    assert result["contexts"][0]["state"] == "suspended"
    assert result["sources"] == result["fetches"] == []


@pytest.mark.parametrize("legacy", [True, False])
def test_capture_without_canonical_play_cues_creates_no_context(script, legacy):
    if legacy:
        del script["music_cues"]
    else:
        script["music_cues"] = [script["music_cues"][-1]]
    result = run(script, [{"op": "capture", "key": "empty"}])
    assert result["results"][0]["value"] is None
    assert result["contexts"] == result["taps"] == []


def test_duplicate_capture_and_release_are_idempotent_and_keep_speakers_silent(script):
    result = run(script, playing(
        {"op": "capture", "key": "one"}, {"op": "capture", "key": "two", "local": True},
        {"op": "sameCapture", "one": "one", "two": "two"},
        {"op": "releaseCapture", "key": "one"},
        {"op": "releaseCapture", "key": "two", "options": {"restoreSpeakers": True}},
        {"op": "cue", "id": "once"}, {"op": "graph", "key": "muxing"},
    ))
    assert [item["value"] for item in result["results"] if item["op"] == "sameCapture"] == [True]
    assert len(result["taps"]) == 1
    assert result["taps"][0]["stops"] == 1
    assert result["snapshots"]["muxing"][0][1]["outputs"] == []
    assert result["snapshots"]["muxing"][0][0]["gain"] == 0.75


def test_explicit_release_can_restore_speakers_once_and_capture_can_be_recreated(script):
    result = run(script, playing(
        {"op": "capture", "key": "one"},
        {"op": "releaseCapture", "key": "one", "options": {"restoreSpeakers": True}},
        {"op": "releaseCapture", "key": "one", "options": {"restoreSpeakers": True}},
        {"op": "graph", "key": "restored"},
        {"op": "capture", "key": "two"}, {"op": "graph", "key": "recaptured"},
        {"op": "releaseCapture", "key": "two", "options": {"restoreSpeakers": False}},
    ))
    assert result["snapshots"]["restored"][0][1]["outputs"] == ["music-output"]
    assert result["snapshots"]["recaptured"][0][1]["outputs"] == ["music-capture"]
    assert len(result["contexts"]) == 1
    assert len(result["taps"]) == 2
    assert [tap["stops"] for tap in result["taps"]] == [1, 1]


def test_dispose_releases_capture_without_reconnecting_even_on_late_explicit_release(script):
    result = run(script, playing(
        {"op": "capture", "key": "active"}, {"op": "dispose"},
        {"op": "releaseCapture", "key": "active", "options": {"restoreSpeakers": True}},
        {"op": "capture", "key": "disposed"}, {"op": "graph", "key": "closed"},
    ))
    assert result["results"][-1]["value"] is None
    assert result["contexts"][0]["state"] == "closed"
    assert result["snapshots"]["closed"][0][0]["outputs"] == []
    assert result["taps"][0]["stops"] == 1


@pytest.mark.parametrize("option", ["tapCreateError", "tapConnectError", "speakerDisconnectError"])
def test_capture_initialization_failure_cleans_partial_graph_and_leaves_speakers_silent(script, option):
    result = run(script, [
        {"op": "capture", "key": "failed"}, {"op": "unlock"}, {"op": "cue", "id": "open"},
        {"op": "graph", "key": "quiet"},
    ], **{option: True})
    assert result["results"][0]["value"] == {"error": "BGM recording output unavailable"}
    assert result["snapshots"]["quiet"][0][1]["outputs"] == []
    assert all(tap["stops"] == 1 for tap in result["taps"])
    assert len(result["contexts"]) == 1


def test_published_44100_to_48000_rounding_keeps_canonical_bounds_and_runtime_loop_phase(script):
    canonical_b = 69.47410430839003
    sample_rate, length = 48000, 3334757
    effective_b = length / sample_rate
    half_frame = 0.5 / sample_rate
    script["music_cues"][0]["loop_end_seconds"] = canonical_b
    result = run(script, playing(
        {"op": "advance", "seconds": effective_b - half_frame},
        {"op": "snapshot", "key": "intro"},
        {"op": "advance", "seconds": 2 * half_frame},
        {"op": "snapshot", "key": "crossed"}, {"op": "pause"},
        {"op": "snapshot", "key": "paused"}, {"op": "advance", "seconds": 100},
        {"op": "resume"}, {"op": "snapshot", "key": "resumed"},
        {"op": "stop"}, {"op": "restore", "value": "paused"},
        {"op": "resume"}, {"op": "snapshot", "key": "restored"},
    ), buffers={"a": {"sampleRate": sample_rate, "length": length}})
    assert result["reports"] == []
    assert result["snapshots"]["intro"]["phase"] == "intro"
    assert result["snapshots"]["crossed"]["phase"] == "loop"
    assert result["snapshots"]["crossed"]["position_seconds"] == pytest.approx(5 + half_frame, abs=1e-10)
    for key in ("intro", "crossed", "paused", "resumed", "restored"):
        assert result["snapshots"][key]["loop_start_seconds"] == 5
        assert result["snapshots"][key]["loop_end_seconds"] == canonical_b
    for key in ("paused", "resumed", "restored"):
        assert result["snapshots"][key]["position_seconds"] == pytest.approx(5 + half_frame, abs=1e-10)
    assert len(result["sources"]) == 3
    assert all(source["loopEnd"] == effective_b for source in result["sources"])
    assert result["sources"][1]["starts"][0][1] == pytest.approx(5 + half_frame, abs=1e-10)
    assert result["sources"][2]["starts"][0][1] == pytest.approx(5 + half_frame, abs=1e-10)
    assert result["decodes"] == ["a"]


@pytest.mark.parametrize("phase", ["intro", "loop"])
@pytest.mark.parametrize("remainder_frames", [0, 0.5])
def test_restore_position_in_resampled_tail_wraps_to_effective_interval(script, phase, remainder_frames):
    sample_rate = 48000
    canonical_b = 15
    effective_b = canonical_b - 0.75 / sample_rate
    position = effective_b + remainder_frames / sample_rate
    result = run(script, playing(
        {"op": "snapshot", "key": "saved"},
        {"op": "restore", "value": "saved", "patch": {
            "phase": phase, "position_seconds": position, "paused": True,
        }},
        {"op": "snapshot", "key": "normalized"}, {"op": "resume"},
        {"op": "snapshot", "key": "resumed"},
    ), buffers={"a": {"sampleRate": sample_rate, "duration": effective_b}})
    assert result["reports"] == []
    for key in ("normalized", "resumed"):
        assert result["snapshots"][key]["phase"] == "loop"
        assert result["snapshots"][key]["position_seconds"] == pytest.approx(5 + remainder_frames / sample_rate)
        assert result["snapshots"][key]["loop_end_seconds"] == canonical_b
    assert result["sources"][-1]["loopEnd"] == effective_b
    assert result["sources"][-1]["starts"][0][1] == pytest.approx(5 + remainder_frames / sample_rate)


@pytest.mark.parametrize("sample_rate", [8000, 44100, 48000, 96000, 192000])
@pytest.mark.parametrize("frames,accepted", [(1, True), (2, True), (2.01, False), (480, False)])
def test_decoded_end_tolerance_is_at_most_two_samples(script, sample_rate, frames, accepted):
    result = run(script, [{"op": "cue", "id": "open"}, {"op": "snapshot", "key": "state"}],
                 buffers={"a": {"sampleRate": sample_rate, "duration": 15 - frames / sample_rate}})
    assert result["results"][0]["value"] is accepted
    if accepted:
        assert result["sources"][0]["loopEnd"] == 15 - frames / sample_rate
        assert result["snapshots"]["state"]["loop_end_seconds"] == 15
    else:
        assert result["sources"] == []
        assert result["snapshots"]["state"] is None


@pytest.mark.parametrize("buffer_rate", [None, 0, 1, -48000, "48000", True])
@pytest.mark.parametrize("frames,accepted", [(1.5, True), (2.5, False)])
def test_invalid_buffer_rate_uses_finite_context_rate_fallback(script, buffer_rate, frames, accepted):
    context_rate = 96000
    result = run(script, [{"op": "cue", "id": "open"}], contextSampleRate=context_rate,
                 buffers={"a": {"sampleRate": buffer_rate, "duration": 15 - frames / context_rate}})
    assert result["results"][0]["value"] is accepted


@pytest.mark.parametrize("frames,accepted", [(1.5, True), (2.5, False)])
def test_missing_native_rates_uses_narrow_48000_fallback(script, frames, accepted):
    result = run(script, [{"op": "cue", "id": "open"}], contextSampleRate=0,
                 buffers={"a": {"duration": 15 - frames / 48000}})
    assert result["results"][0]["value"] is accepted


@pytest.mark.parametrize("offset_frames", [0, 0.5])
def test_rounding_tolerance_rejects_loop_collapsed_by_effective_end(script, offset_frames):
    effective_b = 15 - 1 / 48000
    script["music_cues"][0]["loop_start_seconds"] = effective_b + offset_frames / 48000
    result = run(script, [{"op": "cue", "id": "open"}],
                 buffers={"a": {"sampleRate": 48000, "duration": effective_b}})
    assert result["results"][0]["value"] is False
    assert result["sources"] == []


@pytest.mark.parametrize("phase", ["once", "ended"])
def test_one_shot_save_from_other_sample_rate_normalizes_rounded_away_tail(script, phase):
    source_end = 1323001 / 44100
    decoded_end = 1440001 / 48000
    position = source_end if phase == "ended" else (source_end + decoded_end) / 2
    result = run(script, [
        {"op": "unlock"}, {"op": "cue", "id": "once"}, {"op": "snapshot", "key": "saved"},
        {"op": "restore", "value": "saved", "patch": {"phase": phase, "position_seconds": position}},
        {"op": "snapshot", "key": "restored"}, {"op": "resume"},
    ], buffers={"b": {"sampleRate": 48000, "length": 1440001}})
    assert result["reports"] == []
    assert result["snapshots"]["restored"]["phase"] == "ended"
    assert result["snapshots"]["restored"]["position_seconds"] == decoded_end
    assert result["snapshots"]["restored"]["loop_end_seconds"] is None
    assert len(result["sources"]) == 1


def test_material_one_shot_ended_position_difference_still_rejected(script):
    result = run(script, [
        {"op": "cue", "id": "once"}, {"op": "snapshot", "key": "saved"},
        {"op": "restore", "value": "saved", "patch": {"phase": "ended", "position_seconds": 30 + 3 / 48000}},
    ], buffers={"b": {"sampleRate": 48000, "length": 1440000}})
    assert result["results"][-1]["value"] is False
    assert len(result["sources"]) == 1


def test_prepare_is_pure_and_commit_uses_cached_decoded_buffer(script):
    result = run(script, playing(
        {"op": "advance", "seconds": 17}, {"op": "snapshot", "key": "before"},
        {"op": "prepare", "id": "once"}, {"op": "snapshot", "key": "prepared"},
        {"op": "transition", "id": "once", "options": {"fadeOutMs": 0, "fadeInMs": 0}},
        {"op": "snapshot", "key": "committed"},
    ))
    assert result["snapshots"]["prepared"] == result["snapshots"]["before"]
    assert result["snapshots"]["committed"]["cue_id"] == "once"
    assert result["decodes"] == ["a", "b"]
    assert len(result["fetches"]) == len(result["sources"]) == 2
    assert result["sources"][0]["stops"] == 1
    assert result["sources"][1]["startedAt"] == 17
    assert result["pendingTimers"] == 0


def test_transition_fades_sequentially_and_visual_barrier_precedes_source_commit(script):
    result = run(script, playing(
        {"op": "capture", "key": "recording"}, {"op": "volume", "value": 0.5},
        {"op": "beginTransition", "id": "once", "key": "change", "observeCommit": True},
        {"op": "advance", "seconds": 0.5}, {"op": "graph", "key": "out"},
        {"op": "snapshot", "key": "old"}, {"op": "volume", "value": 0.25},
        {"op": "advance", "seconds": 0.5}, {"op": "graph", "key": "commit"},
        {"op": "snapshot", "key": "new"},
        {"op": "advance", "seconds": 0.5}, {"op": "graph", "key": "in"},
        {"op": "advance", "seconds": 0.5}, {"op": "wait", "key": "change"},
    ))
    assert result["snapshots"]["old"]["cue_id"] == "open"
    assert result["snapshots"]["new"]["cue_id"] == "once"
    assert result["snapshots"]["out"][0] == [
        {"gain": 0.2, "outputs": ["transition-gain"]}, {"gain": 0.5, "outputs": ["music-capture"]},
    ]
    assert result["snapshots"]["commit"][0][0]["gain"] == 0.75 * 0.25
    assert result["snapshots"]["commit"][0][1]["gain"] == 0
    assert result["snapshots"]["in"][0][1]["gain"] == 0.5
    assert result["commits"][0]["snapshot"]["cue_id"] == "open"
    assert result["commits"][0]["fade"] == 0
    assert result["sources"][1]["startedAt"] == 1
    assert result["results"][-1]["value"] is True
    assert result["contexts"][0]["gains"][1]["value"] == 1
    assert result["pendingTimers"] == 0


@pytest.mark.parametrize("kind", ["beginPrepare", "beginTransition"])
def test_preparation_timeout_preserves_audible_source_and_discards_late_result(script, kind):
    result = run(script, playing(
        {"op": "holdFetch", "value": True}, {"op": kind, "id": "once", "key": "pending"},
        {"op": "advance", "seconds": 10.1}, {"op": "wait", "key": "pending"},
        {"op": "snapshot", "key": "kept"}, {"op": "graph", "key": "full"},
        {"op": "holdFetch", "value": False}, {"op": "resolveFetch", "index": 0},
        {"op": "prepare", "id": "once"},
    ), ignoreAbort=True)
    assert result["snapshots"]["kept"]["cue_id"] == "open"
    assert result["snapshots"]["kept"]["position_seconds"] == pytest.approx(10.1)
    assert result["snapshots"]["full"][0][1]["gain"] == 1
    assert len(result["sources"]) == 1 and result["sources"][0]["stops"] == 0
    assert result["fetches"][1]["aborted"] is True
    assert result["decodes"] == ["a", "b"]
    assert len(result["fetches"]) == 3
    assert result["pendingTimers"] == 0


def test_decode_timeout_cannot_cache_late_pcm_or_poison_retry(script):
    result = run(script, playing(
        {"op": "holdDecode", "value": True}, {"op": "beginPrepare", "id": "once", "key": "old"},
        {"op": "advance", "seconds": 11}, {"op": "wait", "key": "old"},
        {"op": "resolveDecode", "index": 0}, {"op": "holdDecode", "value": False},
        {"op": "prepare", "id": "once"}, {"op": "snapshot", "key": "kept"},
    ))
    assert result["decodes"] == ["a", "b", "b"]
    assert len(result["sources"]) == 1 and result["sources"][0]["stops"] == 0
    assert result["snapshots"]["kept"]["cue_id"] == "open"
    assert result["pendingTimers"] == 0


def test_current_pcm_is_pinned_while_only_latest_next_preparation_is_retained(script):
    result = run(script, playing(
        {"op": "prepare", "id": "once"}, {"op": "prepare", "id": "third"},
        {"op": "prepare", "id": "open"}, {"op": "prepare", "id": "once"},
        {"op": "snapshot", "key": "kept"},
    ))
    assert result["decodes"] == ["a", "b", "c", "b"]
    assert result["snapshots"]["kept"]["cue_id"] == "open"
    assert len(result["sources"]) == 1


def test_commit_keeps_captured_pcm_when_another_preparation_evicts_it(script):
    result = run(script, playing(
        {"op": "beginTransition", "id": "once", "key": "change"},
        {"op": "prepare", "id": "third"},
        {"op": "advance", "seconds": 1}, {"op": "advance", "seconds": 1},
        {"op": "wait", "key": "change"}, {"op": "snapshot", "key": "committed"},
    ))
    assert result["decodes"] == ["a", "b", "c"]
    assert len(result["fetches"]) == 3
    assert result["snapshots"]["committed"]["cue_id"] == "once"
    assert len(result["sources"]) == 2


def test_latest_transition_wins_and_stale_timer_cannot_change_new_gain(script):
    result = run(script, playing(
        {"op": "beginTransition", "id": "once", "key": "old"},
        {"op": "advance", "seconds": 0.4},
        {"op": "beginTransition", "id": "third", "key": "new"},
        {"op": "wait", "key": "old"}, {"op": "advance", "seconds": 0.5},
        {"op": "graph", "key": "new_out"}, {"op": "advance", "seconds": 0.5},
        {"op": "advance", "seconds": 1}, {"op": "wait", "key": "new"},
        {"op": "snapshot", "key": "final"},
    ))
    assert result["snapshots"]["new_out"][0][1]["gain"] == pytest.approx(0.5)
    assert result["snapshots"]["final"]["cue_id"] == "third"
    assert len(result["sources"]) == 2
    assert [item["value"] for item in result["results"] if item["op"] == "wait"] == [False, True]
    assert result["pendingTimers"] == 0


@pytest.mark.parametrize("steps,expected", [([0.4], "open"), ([1, 0.4], "once")])
def test_pause_cancels_fade_and_preserves_whichever_track_has_committed(script, steps, expected):
    result = run(script, playing(
        {"op": "beginTransition", "id": "once", "key": "change"},
        *[{"op": "advance", "seconds": seconds} for seconds in steps],
        {"op": "pause"}, {"op": "wait", "key": "change"}, {"op": "snapshot", "key": "paused"},
        {"op": "advance", "seconds": 10}, {"op": "resume"}, {"op": "snapshot", "key": "resumed"},
    ))
    assert result["snapshots"]["paused"]["cue_id"] == expected
    assert result["snapshots"]["paused"]["position_seconds"] == pytest.approx(0.4)
    assert result["snapshots"]["paused"]["paused"] is True
    assert result["snapshots"]["resumed"]["position_seconds"] == pytest.approx(0.4)
    assert result["sources"][-1]["starts"][0][1] == pytest.approx(0.4)
    assert result["contexts"][0]["gains"][1]["value"] == 1
    assert result["pendingTimers"] == 0


def test_abort_while_trusted_visual_barrier_is_pending_prevents_late_commit(script):
    result = run(script, playing(
        {"op": "newSignal", "key": "external"},
        {"op": "beginTransition", "id": "once", "key": "change", "holdCommit": True, "signal": "external"},
        {"op": "advance", "seconds": 1}, {"op": "abortSignal", "key": "external"},
        {"op": "wait", "key": "change"}, {"op": "resolveCommit", "key": "change"},
        {"op": "snapshot", "key": "kept"},
    ))
    assert result["snapshots"]["kept"]["cue_id"] == "open"
    assert len(result["sources"]) == 1
    assert result["commits"][0]["aborted"] is True
    assert result["contexts"][0]["gains"][1]["value"] == 1
    assert result["signalListeners"] == [0]
    assert result["pendingTimers"] == 0


def test_visual_callback_failure_restores_old_gain_and_track(script):
    result = run(script, playing(
        {"op": "beginTransition", "id": "once", "key": "change", "rejectCommit": True},
        {"op": "advance", "seconds": 1}, {"op": "wait", "key": "change"},
        {"op": "snapshot", "key": "kept"},
    ))
    assert result["results"][-1]["value"] is False
    assert result["snapshots"]["kept"]["cue_id"] == "open"
    assert len(result["sources"]) == 1
    assert result["contexts"][0]["gains"][1]["value"] == 1


@pytest.mark.parametrize("operation", [{"op": "stop"}, {"op": "dispose"}, {"op": "cue", "id": "third"}])
def test_immediate_operations_cancel_pending_fade_without_late_source_changes(script, operation):
    result = run(script, playing(
        {"op": "beginTransition", "id": "once", "key": "change"},
        {"op": "advance", "seconds": 0.5}, operation,
        {"op": "wait", "key": "change"}, {"op": "advance", "seconds": 5},
        {"op": "snapshot", "key": "after"},
    ))
    assert result["results"][-1]["value"] is False
    assert len(result["sources"]) == (2 if operation["op"] == "cue" else 1)
    assert (result["snapshots"]["after"] or {}).get("cue_id") == ("third" if operation["op"] == "cue" else None)
    assert result["pendingTimers"] == 0


def test_faded_chapter_stop_cancels_preparation_and_leaves_no_track(script):
    result = run(script, playing(
        {"op": "holdFetch", "value": True}, {"op": "beginPrepare", "id": "once", "key": "prepare"},
        {"op": "beginStopFaded", "key": "end"}, {"op": "wait", "key": "prepare"},
        {"op": "advance", "seconds": 0.5}, {"op": "graph", "key": "out"},
        {"op": "advance", "seconds": 0.5}, {"op": "wait", "key": "end"},
        {"op": "snapshot", "key": "ended"}, {"op": "resolveFetch", "index": 0},
    ), ignoreAbort=True)
    assert result["snapshots"]["out"][0][1]["gain"] == 0.5
    assert result["snapshots"]["ended"] is None
    assert result["fetches"][1]["aborted"] is True
    assert result["decodes"] == ["a"]
    assert result["sources"][0]["stops"] == 1
    assert result["pendingTimers"] == 0


def test_old_chapter_stop_cannot_stop_new_immediate_cue(script):
    result = run(script, playing(
        {"op": "beginStopFaded", "key": "old"}, {"op": "advance", "seconds": 0.4},
        {"op": "cue", "id": "third"}, {"op": "wait", "key": "old"},
        {"op": "advance", "seconds": 2}, {"op": "snapshot", "key": "new"},
    ))
    assert result["results"][-1]["value"] is False
    assert result["snapshots"]["new"]["cue_id"] == "third"
    assert result["sources"][1]["stops"] == 0
    assert result["contexts"][0]["gains"][1]["value"] == 1


@pytest.mark.parametrize("steps", [[0.3], [1, 0.3]])
def test_dynamic_skip_finishes_remaining_fades_without_restarting_committed_intro(script, steps):
    result = run(script, playing(
        {"op": "beginTransition", "id": "once", "key": "change"},
        *[{"op": "advance", "seconds": seconds} for seconds in steps],
        {"op": "finishTransition"}, {"op": "wait", "key": "change"},
        {"op": "snapshot", "key": "final"},
    ))
    assert result["results"][-1]["value"] is True
    assert len(result["sources"]) == 2
    assert result["snapshots"]["final"]["position_seconds"] == pytest.approx(0 if len(steps) == 1 else 0.3)
    assert result["sources"][1]["starts"] == [[0, 0]]
    assert result["contexts"][0]["gains"][1]["value"] == 1
    assert result["pendingTimers"] == 0


def test_continue_preserves_track_phase_and_volume_without_visual_callback(script):
    result = run(script, playing(
        {"op": "advance", "seconds": 18}, {"op": "volume", "value": 0.5},
        {"op": "snapshot", "key": "before"},
        {"op": "transition", "id": "keep", "observeCommit": True},
        {"op": "snapshot", "key": "after"},
    ))
    assert result["snapshots"]["after"] == result["snapshots"]["before"]
    assert result["commits"] == []
    assert len(result["sources"]) == 1
    assert result["contexts"][0]["gains"][0]["value"] == 0.2


def test_continue_does_not_abort_pending_immediate_intro_load(script):
    result = run(script, [
        {"op": "unlock"}, {"op": "holdFetch", "value": True},
        {"op": "beginCue", "id": "open", "key": "opening"},
        {"op": "transition", "id": "keep"}, {"op": "resolveFetch", "index": 0},
        {"op": "wait", "key": "opening"}, {"op": "snapshot", "key": "playing"},
    ])
    assert result["results"][-1]["value"] is True
    assert result["fetches"][0]["aborted"] is False
    assert len(result["sources"]) == 1
    assert result["snapshots"]["playing"]["cue_id"] == "open"


def test_continue_preserves_independent_next_preparation(script):
    result = run(script, playing(
        {"op": "holdFetch", "value": True},
        {"op": "beginPrepare", "id": "once", "key": "next"},
        {"op": "transition", "id": "keep"}, {"op": "resolveFetch", "index": 0},
        {"op": "wait", "key": "next"}, {"op": "snapshot", "key": "kept"},
    ))
    assert result["results"][-1]["value"] is True
    assert result["fetches"][1]["aborted"] is False
    assert len(result["sources"]) == 1
    assert result["snapshots"]["kept"]["cue_id"] == "open"


@pytest.mark.parametrize("stage", ["prepare", "fade_out", "fade_in"])
def test_external_abort_cleans_listeners_and_preserves_committed_track(script, stage):
    before = [{"op": "holdFetch", "value": True}] if stage == "prepare" else []
    during = [{"op": "advance", "seconds": 1}] if stage == "fade_in" else []
    result = run(script, playing(
        {"op": "newSignal", "key": "external"}, *before,
        {"op": "beginTransition", "id": "once", "key": "change", "signal": "external"},
        *during, {"op": "abortSignal", "key": "external"}, {"op": "wait", "key": "change"},
        {"op": "snapshot", "key": "kept"},
    ))
    assert result["results"][-1]["value"] is False
    assert result["snapshots"]["kept"]["cue_id"] == ("once" if stage == "fade_in" else "open")
    assert result["contexts"][0]["gains"][1]["value"] == 1
    assert result["signalListeners"] == [0]
    assert result["pendingTimers"] == 0


def test_internal_stop_cancels_own_scope_without_aborting_caller_signal(script):
    result = run(script, playing(
        {"op": "newSignal", "key": "external"}, {"op": "holdFetch", "value": True},
        {"op": "beginPrepare", "id": "once", "key": "pending", "signal": "external"},
        {"op": "stop"}, {"op": "wait", "key": "pending"},
    ))
    assert result["signalsAborted"] == [False]
    assert result["signalListeners"] == [0]
    assert result["fetches"][1]["aborted"] is True
    assert result["pendingTimers"] == 0


def test_first_play_skips_inaudible_fade_out_but_waits_for_visual_barrier(script):
    result = run(script, [
        {"op": "unlock"}, {"op": "beginTransition", "id": "open", "key": "first", "holdCommit": True},
        {"op": "snapshot", "key": "before"}, {"op": "resolveCommit", "key": "first"},
        {"op": "advance", "seconds": 1}, {"op": "wait", "key": "first"},
    ])
    assert result["snapshots"]["before"] is None
    assert result["commits"][0]["snapshot"] is None
    assert result["sources"][0]["startedAt"] == 0
    assert result["results"][-1]["value"] is True


def test_suspended_context_settles_fades_without_waiting_for_frozen_audio_clock(script):
    result = run(script, playing(
        {"op": "contextState", "value": "suspended"},
        {"op": "transition", "id": "once"}, {"op": "snapshot", "key": "new"},
    ))
    assert result["snapshots"]["new"]["cue_id"] == "once"
    assert result["contexts"][0]["currentTime"] == 0
    assert result["contexts"][0]["gains"][1]["value"] == 1
    assert result["pendingTimers"] == 0


def test_context_suspension_during_fade_settles_future_phases_coherently(script):
    result = run(script, playing(
        {"op": "beginTransition", "id": "once", "key": "change"},
        {"op": "advance", "seconds": 0.4}, {"op": "contextState", "value": "suspended"},
        {"op": "advance", "seconds": 1}, {"op": "wait", "key": "change"},
        {"op": "snapshot", "key": "new"},
    ))
    assert result["snapshots"]["new"]["cue_id"] == "once"
    assert result["snapshots"]["new"]["position_seconds"] == 0
    assert result["contexts"][0]["currentTime"] == 0.4
    assert result["contexts"][0]["gains"][1]["value"] == 1


def test_preview_uses_external_context_adapter_and_near_boundary_offset_without_owning_context(script):
    result = run(script, [
        {"op": "cue", "id": "open", "options": {"position": 12}}, {"op": "unlock"},
        {"op": "advance", "seconds": 4}, {"op": "snapshot", "key": "loop"},
        {"op": "capture", "key": "global"}, {"op": "dispose"},
    ], factoryExternal=True, factoryAdapter=True, noAudio=True)
    assert result["sources"][0]["starts"] == [[0, 12]]
    assert result["snapshots"]["loop"]["phase"] == "loop"
    assert result["snapshots"]["loop"]["position_seconds"] == 6
    assert result["contexts"][0]["state"] == "running"
    assert result["fetches"] == result["decodes"] == result["taps"] == []
    assert len(result["adapterCalls"]) == 1


def test_preview_factory_cannot_replace_registered_game_recording_output(script):
    result = run(script, playing(
        {"op": "createPreview", "key": "preview"},
        {"op": "previewCue", "key": "preview", "id": "once", "options": {"position": 20}},
        {"op": "capture", "key": "game"}, {"op": "graph", "key": "registered"},
        {"op": "previewDispose", "key": "preview"},
    ))
    assert result["snapshots"]["registered"][0][1]["outputs"] == ["music-capture"]
    assert result["snapshots"]["registered"][1][1]["outputs"] == ["music-output"]
    assert result["contexts"][1]["state"] == "suspended"
    assert result["sources"][0]["stops"] == 0


def test_getcue_returns_frozen_canonical_copy_and_unknown_id_is_null(script):
    result = run(script, [{"op": "lookup", "id": "open"}, {"op": "lookup", "id": "missing"}])
    assert result["results"][0]["value"]["frozen"] is True
    assert result["results"][0]["value"]["cue"]["asset_id"] == "a"
    assert result["results"][1]["value"]["cue"] is None


@pytest.mark.parametrize("operation,options", [
    ("prepare", {"timeoutMs": 10001}), ("prepare", {"timeoutMs": 0}),
    ("transition", {"timeoutMs": 10001}), ("transition", {"timeoutMs": 0}),
    ("transition", {"fadeOutMs": -1}), ("transition", {"fadeInMs": "1000"}),
])
def test_bounded_preparation_and_fade_options_cannot_schedule_unbounded_work(script, operation, options):
    result = run(script, playing({"op": operation, "id": "once", "options": options},
                                {"op": "snapshot", "key": "kept"}))
    assert result["results"][-1]["value"] is False
    assert result["snapshots"]["kept"]["cue_id"] == "open"
    assert len(result["sources"]) == 1
    assert result["pendingTimers"] == 0
