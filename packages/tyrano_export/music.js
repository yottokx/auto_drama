"use strict";
// Package-owned playback code. Story data can select manifest assets and numeric
// loop points only; it never supplies URLs, engine tags, or executable code.
(() => {
  let owned = null;
  const identifier = value => typeof value === "string" && /^[a-z][a-z0-9_-]{0,63}$/.test(value);
  const number = value => typeof value === "number" && Number.isFinite(value);
  const volume = value => number(value) && value >= 0 && value <= 1;
  const filename = value => typeof value === "string" && value.length <= 128 &&
    /^[A-Za-z0-9][A-Za-z0-9_.-]*\.mp3$/i.test(value) &&
    !/^(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9]|LPT[1-9])\./i.test(value);
  function create(script, {report = () => {}, context: externalContext = null, loadAudio = null,
    registerOutput = !externalContext && !loadAudio} = {}) {
    const notify = message => { try { report(message); } catch (_) {} };
    const assets = new Map(), cues = new Map(), decoded = new Map();
    const duplicateAssets = new Set(), duplicateCues = new Set();
    const list = value => Array.isArray(value) ? value : [];
    const utterances = new Set(list(script?.utterances).map(value => value?.id).filter(identifier));
    for (const asset of list(script?.assets)) {
      if (asset?.kind !== "music" || !identifier(asset.id) || !filename(asset.filename)) continue;
      if (assets.has(asset.id) || duplicateAssets.has(asset.id)) {
        assets.delete(asset.id); duplicateAssets.add(asset.id);
      } else assets.set(asset.id, "./data/bgm/" + asset.filename);
    }
    const normalize = value => {
      if (!value || !identifier(value.id) || !utterances.has(value.utterance_id) ||
          !["play", "stop", "continue"].includes(value.action)) return null;
      const result = {id: value.id, utterance_id: value.utterance_id, action: value.action,
        asset_id: value.asset_id ?? null, loop_start_seconds: value.loop_start_seconds ?? null,
        loop_end_seconds: value.loop_end_seconds ?? null, volume: value.volume ?? 0.35};
      if (!volume(result.volume)) return null;
      if (result.action !== "play") {
        return result.asset_id === null && result.loop_start_seconds === null &&
          result.loop_end_seconds === null ? result : null;
      }
      if (!assets.has(result.asset_id)) return null;
      const a = result.loop_start_seconds, b = result.loop_end_seconds;
      if (!(a === null && b === null) &&
          !(number(a) && number(b) && a >= 0 && b > a)) return null;
      return result;
    };
    for (const value of list(script?.music_cues)) {
      const cue = normalize(value);
      if (!cue) continue;
      if (cues.has(cue.id) || duplicateCues.has(cue.id)) {
        cues.delete(cue.id); duplicateCues.add(cue.id);
      } else cues.set(cue.id, Object.freeze(cue));
    }
    const activeMusic = [...cues.values()].some(value => value.action === "play");
    let context = null, gain = null, fadeGain = null, source = null, current = null, request = null;
    let generation = 0, userVolume = 1, paused = false, disposed = false;
    let decodeTail = Promise.resolve(), capture = null, speakersConnected = false;
    let preparation = null, transition = null;
    const ensureContext = () => {
      if (disposed || !activeMusic) return null;
      if (!context) {
        const Context = window.AudioContext || window.webkitAudioContext;
        if (!Context && !externalContext) throw Error("Web Audio unavailable");
        const created = externalContext || new Context();
        let createdGain = null, createdFadeGain = null;
        try {
          createdGain = created.createGain(); createdFadeGain = created.createGain();
          createdGain.connect(createdFadeGain); createdFadeGain.connect(created.destination);
          context = created; gain = createdGain; fadeGain = createdFadeGain;
          speakersConnected = true;
        } catch (error) {
          try { createdGain?.disconnect(); } catch (_) {}
          try { createdFadeGain?.disconnect(); } catch (_) {}
          if (!externalContext) { try { created.close()?.catch(() => {}); } catch (_) {} }
          throw error;
        }
      }
      return context;
    };
    const applyVolume = () => {
      if (!gain) return;
      const value = (current?.cue.volume ?? 0) * userVolume;
      gain.gain.setValueAtTime(value, context.currentTime);
    };
    const setFade = value => {
      if (!fadeGain) return;
      fadeGain.gain.cancelScheduledValues?.(context.currentTime);
      fadeGain.gain.setValueAtTime(value, context.currentTime);
    };
    const captureOutput = () => {
      if (disposed || !activeMusic) return null;
      if (capture) return capture;
      let tap = null;
      try {
        // The recorder prepares its mix before the first play cue, so this may
        // create a suspended context. A user gesture still owns actual unlock.
        const ctx = ensureContext();
        tap = ctx.createMediaStreamDestination();
        if (!tap.stream || typeof tap.stream.getTracks !== "function") throw Error("Missing stream");
        fadeGain.connect(tap);
        // Prepare the capture graph before removing the owned speaker output.
        if (speakersConnected) { fadeGain.disconnect(ctx.destination); speakersConnected = false; }
        let released = false;
        const handle = Object.freeze({stream: tap.stream, release(options = {}) {
          if (released) return;
          released = true;
          if (capture === handle) capture = null;
          try { fadeGain.disconnect(tap); } catch (_) { try { fadeGain.disconnect(); } catch (_) {} }
          try { tap.disconnect(); } catch (_) {}
          for (const track of tap.stream.getTracks()) { try { track.stop(); } catch (_) {} }
          // Recording cleanup defaults to silence while FFmpeg muxes. Disposal
          // never restores speakers, even if a caller requests it afterwards.
          if (options?.restoreSpeakers === true && !disposed && !speakersConnected) {
            try { fadeGain.connect(context.destination); speakersConnected = true; }
            catch (_) { notify("BGMのスピーカー出力を再開できませんでした。"); }
          }
        }});
        capture = handle;
        return handle;
      } catch (_) {
        // A failed recording initialization must not leak audio to speakers or
        // leave a partial live MediaStream graph behind.
        try { fadeGain?.disconnect(); } catch (_) {}
        speakersConnected = false;
        try { tap?.disconnect(); } catch (_) {}
        for (const track of tap?.stream?.getTracks?.() || []) { try { track.stop(); } catch (_) {} }
        throw Error("BGM recording output unavailable");
      }
    };
    const detach = () => {
      const old = source; source = null;
      if (!old) return;
      old.onended = null;
      try { old.stop(); } catch (_) {}
      try { old.disconnect(); } catch (_) {}
    };
    const cancel = () => {
      generation++;
      request?.abort(); request = null;
      detach();
    };
    const duration = (value, fallback, minimum = 0) => {
      const result = value ?? fallback;
      if (!number(result) || result < minimum || result > 10000) throw Error("Invalid music timing");
      return result;
    };
    const scope = (signal, timeoutMs = null) => {
      const controller = new window.AbortController();
      let timer = null, timedOut = false;
      const forward = () => controller.abort();
      let rejectAbort;
      const aborted = new Promise((_, reject) => { rejectAbort = reject; });
      aborted.catch(() => {});
      const reject = () => rejectAbort(Error("Music operation canceled"));
      controller.signal.addEventListener("abort", reject, {once: true});
      if (signal?.aborted) controller.abort();
      else signal?.addEventListener("abort", forward, {once: true});
      if (timeoutMs !== null && !controller.signal.aborted) {
        timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
      }
      return {controller, get timedOut() { return timedOut; },
        race: promise => Promise.race([promise, aborted]),
        cleanup() {
          if (timer !== null) window.clearTimeout(timer);
          signal?.removeEventListener("abort", forward);
          controller.signal.removeEventListener("abort", reject);
        },
      };
    };
    const cancelPreparation = () => {
      const old = preparation; preparation = null;
      old?.controller.abort();
    };
    const cancelTransition = () => {
      const old = transition; transition = null;
      old?.scope.controller.abort();
      if (old) setFade(1);
    };
    const queued = (operation, valid) => {
      const next = decodeTail.then(() => valid() ? operation() : null);
      decodeTail = next.then(() => {}, () => {});
      return next;
    };
    const readBuffer = async (cue, controller, valid) => {
      const ctx = ensureContext(), url = assets.get(cue.asset_id);
      if (!valid()) throw Error("Music operation canceled");
      let buffer;
      if (typeof loadAudio === "function") {
        buffer = await queued(() => loadAudio(url, {signal: controller.signal}), valid);
      } else {
        const response = await window.fetch(url, {signal: controller.signal});
        if (!valid()) throw Error("Music operation canceled");
        if (!response.ok) throw Error("Music fetch failed");
        const bytes = await response.arrayBuffer();
        if (!valid()) throw Error("Music operation canceled");
        buffer = await queued(() => ctx.decodeAudioData(bytes), valid);
      }
      if (!valid()) throw Error("Music operation canceled");
      return buffer;
    };
    const bufferInfo = (cue, buffer) => {
      if (!number(buffer?.duration) || buffer.duration <= 0) throw Error("Invalid decoded music");
      const validRate = value => number(value) && value >= 8000;
      const rate = validRate(buffer.sampleRate) ? buffer.sampleRate :
        (validRate(context?.sampleRate) ? context.sampleRate : 48000);
      const epsilon = Number.EPSILON * Math.max(1, buffer.duration, cue.loop_end_seconds ?? 0) * 4;
      const tolerance = 2 / rate + epsilon;
      // Resampling can shorten a valid adopted PCM endpoint by a fraction of
      // a frame. Runtime B uses the decode; cue/save metadata stays immutable.
      const b = cue.loop_end_seconds === null ? null : Math.min(cue.loop_end_seconds, buffer.duration);
      if (b !== null && (cue.loop_end_seconds - buffer.duration > tolerance || cue.loop_start_seconds >= b)) {
        throw Error("Music loop outside decoded buffer");
      }
      return {effective_loop_end_seconds: b, tolerance};
    };
    const remember = (assetId, buffer) => {
      decoded.delete(assetId); decoded.set(assetId, buffer);
      while (decoded.size > 2) {
        const victim = [...decoded.keys()].find(id => id !== current?.cue.asset_id && id !== assetId);
        if (victim === undefined) break;
        decoded.delete(victim);
      }
    };
    const prepare = async (cue, {signal, timeoutMs = 10000} = {}) => {
      if (disposed || signal?.aborted) return null;
      let pending;
      try {
        const limit = duration(timeoutMs, 10000, 1);
        cancelPreparation();
        const cached = decoded.get(cue.asset_id);
        if (cached) {
          const info = bufferInfo(cue, cached); remember(cue.asset_id, cached);
          return {cue, buffer: cached, ...info};
        }
        pending = scope(signal, limit); preparation = pending;
        const valid = () => !disposed && preparation === pending && !pending.controller.signal.aborted;
        const buffer = await pending.race(readBuffer(cue, pending.controller, valid));
        if (!valid()) return null;
        const info = bufferInfo(cue, buffer);
        remember(cue.asset_id, buffer);
        return {cue, buffer, ...info};
      } catch (_) {
        if (!disposed && (pending?.timedOut || !pending?.controller.signal.aborted)) {
          notify("次のBGMを準備できませんでした。");
        }
        return null;
      } finally {
        pending?.cleanup();
        if (preparation === pending) preparation = null;
      }
    };
    const location = track => {
      const elapsed = track.started_at === null ? 0 : Math.max(0, context.currentTime - track.started_at);
      const a = track.cue.loop_start_seconds, b = track.effective_loop_end_seconds;
      const position = track.position_seconds + elapsed;
      if (a !== null) {
        if (track.phase === "intro" && position < b) return {position_seconds: position, phase: "intro"};
        const distance = track.phase === "intro" ? position - b : position - a;
        return {position_seconds: a + distance % (b - a), phase: "loop"};
      }
      const end = track.buffer?.duration;
      if (end !== undefined && position >= end) return {position_seconds: end, phase: "ended"};
      return {position_seconds: position, phase: track.phase};
    };
    const trackFrom = (cue, buffer, saved = null, position = 0) => {
      const info = bufferInfo(cue, buffer);
      const track = {cue, buffer, position_seconds: saved?.position_seconds ?? position,
        effective_loop_end_seconds: info.effective_loop_end_seconds,
        phase: saved?.phase ?? (cue.loop_start_seconds === null ? "once" : "intro"), started_at: null};
      if (cue.loop_start_seconds !== null) Object.assign(track, location(track));
      else {
        if ((track.phase === "ended" && Math.abs(track.position_seconds - buffer.duration) > info.tolerance) ||
            (track.phase === "once" && track.position_seconds - buffer.duration > info.tolerance)) {
          throw Error("Saved music position outside decoded buffer");
        }
        if (track.phase === "ended" || track.position_seconds >= buffer.duration) {
          track.position_seconds = buffer.duration; track.phase = "ended";
        }
      }
      return track;
    };
    const freeze = () => {
      if (!current) return;
      Object.assign(current, location(current)); current.started_at = null;
      detach();
    };
    const start = () => {
      if (!current?.buffer || source || paused || disposed || current.phase === "ended") return;
      const track = current, token = generation, node = context.createBufferSource();
      node.buffer = track.buffer;
      node.loop = track.cue.loop_start_seconds !== null;
      if (node.loop) {
        node.loopStart = track.cue.loop_start_seconds;
        node.loopEnd = track.effective_loop_end_seconds;
      }
      node.connect(gain); source = node; applyVolume();
      node.onended = () => {
        if (token !== generation || current !== track || source !== node || node.loop) return;
        track.position_seconds = track.buffer.duration; track.phase = "ended"; track.started_at = null;
        source = null; node.onended = null;
        try { node.disconnect(); } catch (_) {}
      };
      track.started_at = context.currentTime;
      // No duration argument: play the intro once, then let the audio render
      // thread wrap B -> A. JavaScript timers never schedule loop boundaries.
      try { node.start(0, track.position_seconds); }
      catch (error) { track.started_at = null; detach(); throw error; }
    };
    const load = async (cue, saved = null, position = 0) => {
      cancel();
      const token = generation;
      if (saved) { userVolume = saved.user_volume; paused = saved.paused; }
      const track = {cue, buffer: null, position_seconds: saved?.position_seconds ?? position,
        effective_loop_end_seconds: cue.loop_end_seconds,
        phase: saved?.phase ?? (cue.loop_start_seconds === null ? "once" : "intro"), started_at: null};
      current = track;
      try {
        ensureContext();
        let buffer = decoded.get(cue.asset_id);
        if (!buffer) {
          const controller = new window.AbortController(); request = controller;
          buffer = await readBuffer(cue, controller, () => token === generation &&
            !disposed && !controller.signal.aborted);
          if (token !== generation || disposed) return false;
          if (request === controller) request = null;
        }
        if (token !== generation || disposed) return false;
        Object.assign(track, trackFrom(cue, buffer, saved, position));
        remember(cue.asset_id, buffer);
        applyVolume(); start(); return true;
      } catch (_) {
        if (token !== generation || disposed) return false;
        request?.abort(); request = null; detach(); current = null; applyVolume();
        notify("BGMを再生できませんでした。音楽ファイルとループ位置を確認してください。");
        return false;
      }
    };
    const fadeTo = async (target, milliseconds, task) => {
      const valid = () => !disposed && transition === task && !task.scope.controller.signal.aborted;
      if (!valid()) return false;
      // Frozen audio clocks cannot complete ramps. Settle them immediately;
      // a paused controller still keeps its source stopped, and a suspended
      // external context still requires its ordinary gesture-owned unlock.
      if (!fadeGain || task.fast || milliseconds === 0 || paused || context.state !== "running" ||
          typeof fadeGain.gain.linearRampToValueAtTime !== "function") {
        setFade(target); return true;
      }
      const param = fadeGain.gain, from = param.value, time = context.currentTime;
      param.cancelScheduledValues?.(time);
      param.setValueAtTime(from, time);
      param.linearRampToValueAtTime(target, time + milliseconds / 1000);
      return await new Promise(resolve => {
        let settled = false, timer;
        const signal = task.scope.controller.signal;
        const finish = () => {
          if (settled) return;
          settled = true; window.clearTimeout(timer); signal.removeEventListener("abort", finish);
          if (task.fade?.finish === finish) task.fade = null;
          if (valid()) setFade(target);
          resolve(valid());
        };
        task.fade = {finish};
        timer = window.setTimeout(finish, milliseconds);
        signal.addEventListener("abort", finish, {once: true});
        if (signal.aborted) finish();
      });
    };
    const runTransition = async (cue, {fadeOutMs = 1000, fadeInMs = 1000, signal,
      timeoutMs = 10000, beforeCommit} = {}) => {
      if (disposed || signal?.aborted) return false;
      let task;
      try {
        const out = duration(fadeOutMs, 1000), into = duration(fadeInMs, 1000);
        duration(timeoutMs, 10000, 1);
        if (beforeCommit !== undefined && typeof beforeCommit !== "function") throw Error("Invalid commit callback");
        cancelTransition();
        // Continue preserves an immediate load or independent preparation too;
        // it only supersedes an obsolete scene transition.
        if (cue?.action === "continue") return true;
        cancelPreparation();
        // A new scene/chapter command supersedes a pending immediate load.
        // Already audible sources stay connected until their fade-out finishes.
        if (request) {
          generation++; request.abort(); request = null;
          if (!current?.buffer) { current = null; applyVolume(); }
        }
        task = {scope: scope(signal), fast: false, fade: null}; transition = task;
        const valid = () => !disposed && transition === task && !task.scope.controller.signal.aborted;
        const prepared = cue?.action === "play" ?
          await prepare(cue, {signal: task.scope.controller.signal, timeoutMs}) : null;
        if (!valid() || (cue?.action === "play" && !prepared)) return false;
        const audibleOut = current?.buffer && source && location(current).phase !== "ended" ? out : 0;
        if (!await fadeTo(0, audibleOut, task) || !valid()) return false;
        if (beforeCommit) await task.scope.race(Promise.resolve().then(() =>
          valid() ? beforeCommit({signal: task.scope.controller.signal}) : undefined));
        if (!valid()) return false;
        // Commit the captured validated PCM synchronously. Another preparation
        // may alter the tiny cache but cannot force a late refetch here.
        const next = prepared ? trackFrom(cue, prepared.buffer) : null;
        cancel(); current = next;
        if (next) remember(cue.asset_id, prepared.buffer);
        applyVolume(); start();
        if (!await fadeTo(1, prepared ? into : 0, task) || !valid()) return false;
        return true;
      } catch (_) {
        if (!disposed && task && transition === task && !task.scope.controller.signal.aborted) {
          notify("BGMの切替を完了できませんでした。");
        }
        return false;
      } finally {
        task?.scope.cleanup();
        if (transition === task) { transition = null; setFade(1); }
      }
    };
    const stop = () => {
      if (disposed) return;
      cancelTransition(); cancelPreparation(); cancel(); current = null; applyVolume(); setFade(1);
    };
    const unlock = async () => {
      if (disposed) return false;
      if (!activeMusic) return true;
      try {
        const ctx = ensureContext();
        if (ctx.state !== "running") await ctx.resume();
        return !disposed && ctx.state === "running";
      } catch (_) {
        if (!disposed) notify("BGMの再生を開始できませんでした。再開ボタンを押してください。");
        return false;
      }
    };
    const controller = Object.freeze({
      getCue: id => cues.get(id) ?? null,
      async prepareCue(id, options = {}) {
        const cue = cues.get(id);
        if (disposed || !cue || options.signal?.aborted) return false;
        if (cue.action !== "play") return true;
        return !!await prepare(cue, options);
      },
      async transitionCue(id, options = {}) {
        const cue = cues.get(id);
        if (!cue) return false;
        return runTransition(cue, options);
      },
      stopFaded: options => runTransition(null, {...options, fadeInMs: 0}),
      finishTransition() {
        if (!transition || transition.scope.controller.signal.aborted) return false;
        transition.fast = true; transition.fade?.finish(); return true;
      },
      async cue(value, {position = 0} = {}) {
        if (disposed || !activeMusic) return false;
        const supplied = normalize(value), canonical = supplied && cues.get(supplied.id);
        if (!canonical || JSON.stringify(supplied) !== JSON.stringify(canonical)) {
          notify("BGMの指定がこの章の音楽データと一致しません。"); return false;
        }
        if (!number(position) || position < 0 ||
            (canonical.loop_end_seconds !== null && position >= canonical.loop_end_seconds)) return false;
        if (canonical.action === "stop") { stop(); return true; }
        // A serialized default volume on "continue" must not reset the current
        // track's authored volume or restart its intro/loop phase.
        if (canonical.action === "continue") { cancelTransition(); return true; }
        cancelTransition(); cancelPreparation(); setFade(1);
        return load(canonical, null, position);
      },
      unlock,
      setVolume(value) {
        if (disposed || !volume(value)) return false;
        userVolume = value; applyVolume(); return true;
      },
      snapshot() {
        if (!current || disposed) return null;
        return {schema_version: 1, asset_id: current.cue.asset_id, cue_id: current.cue.id,
          loop_start_seconds: current.cue.loop_start_seconds, loop_end_seconds: current.cue.loop_end_seconds,
          ...location(current), track_volume: current.cue.volume, user_volume: userVolume, paused};
      },
      async restore(saved) {
        if (disposed) return false;
        if (saved === null || saved === undefined) { stop(); return true; }
        const cue = cues.get(saved.cue_id);
        const loop = cue?.loop_start_seconds !== null;
        if (saved.schema_version !== 1 || !cue || cue.action !== "play" ||
            saved.asset_id !== cue.asset_id || saved.loop_start_seconds !== cue.loop_start_seconds ||
            saved.loop_end_seconds !== cue.loop_end_seconds || saved.track_volume !== cue.volume ||
            !volume(saved.user_volume) || typeof saved.paused !== "boolean" ||
            !number(saved.position_seconds) || saved.position_seconds < 0 ||
            (loop ? !((saved.phase === "intro" && saved.position_seconds < cue.loop_end_seconds) ||
              (saved.phase === "loop" && saved.position_seconds >= cue.loop_start_seconds &&
                saved.position_seconds < cue.loop_end_seconds)) : !["once", "ended"].includes(saved.phase))) {
          notify("保存されたBGMの位置がこの章の音楽データと一致しません。"); return false;
        }
        // The save object belongs to the game engine and may change during an
        // async fetch. Keep validation and playback tied to this exact copy.
        cancelTransition(); cancelPreparation(); setFade(1);
        return load(cue, Object.freeze({...saved}));
      },
      pause() {
        if (!disposed) { cancelTransition(); cancelPreparation(); paused = true; setFade(1); freeze(); }
      },
      async resume() {
        if (disposed) return false;
        paused = false;
        const token = generation;
        if (!await unlock()) return false;
        if (token !== generation || paused || disposed) return false;
        try { start(); return true; }
        catch (_) { stop(); notify("BGMを再開できませんでした。"); return false; }
      },
      stop,
      captureOutput,
      dispose() {
        if (disposed) return;
        stop(); disposed = true; decoded.clear();
        capture?.release();
        if (owned === controller) owned = null;
        try { gain?.disconnect(); } catch (_) {}
        try { fadeGain?.disconnect(); } catch (_) {}
        if (!externalContext) { try { context?.close()?.catch(() => {}); } catch (_) {} }
      },
    });
    if (registerOutput) owned = controller;
    return controller;
  }
  window.AutoDramaMusic = Object.freeze({create, captureOutput: () => owned?.captureOutput() ?? null});
})();
