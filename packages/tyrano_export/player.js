"use strict";
// Only compiler-owned code runs here. All story data enters DOM text sinks.
(() => {
  const byId = id => document.getElementById(id);
  const kag = () => window.TYRANO && window.TYRANO.kag;
  let script = null, started = false, bound = false, playback = null, music = null;
  let musicEpoch = 0;
  let stages = null, transitions = null, transitionEpoch = 0, pendingTransition = null;
  const livePath = window.location?.pathname?.match(/^\/player\/([a-zA-Z0-9_-]+)\/(?:index.html)?$/);
  let context = livePath ? null : {mode: "static"};
  let identity = null, ended = false, nextBuild = null, pollAt = 0, polling = false, epoch = 0;
  let chapterNavigation = null, nextPreparation = null, returnMusic, pageSuspended = false;
  let storyLoading = false, loadRestoreStarted = false;
  let entryAttempted = false, entryGeneration = 0;
  const entryKey = "auto_drama_next_chapter_entry_v1";
  const status = text => { byId("ad-status").textContent = text; };
  const pauseReplay = () => byId("ad-log").querySelectorAll("audio").forEach(a => a.pause());
  const stopAuto = () => {
    const k = kag();
    // Native keyboard roles are guarded while text/audio is running. The owned
    // toolbar can always stop auto without moving the current scenario index.
    if (k && k.stat.is_auto) k.ftag.startTag("autostop", {next: "false"});
  };
  const controls = () => byId("ad-toolbar").querySelectorAll("button, select, input");
  const saveIdentity = eventId => ({schema_version: 1, ...identity, event_id: eventId});
  function readSave() {
    try {
      const k = kag();
      if (!window.localStorage?.getItem(k.config.projectID + "_tyrano_quick_save")) return null;
      const value = JSON.parse($.getStorage(k.config.projectID + "_tyrano_quick_save", "webstorage"));
      const saved = value?.stat?.auto_drama_position;
      if (!identity || !saved || saved.schema_version !== 1 ||
          Object.keys(identity).some(key => saved[key] !== identity[key]) ||
          value.stat.current_scenario !== "first.ks" ||
          !Number.isInteger(value.current_order_index) ||
          value.current_order_index < -1 || value.current_order_index >= kag().ftag.array_tag.length - 1 ||
          saved.event_id !== playback.eventId(value.current_order_index + 1)) return null;
      return value;
    } catch (_) { return null; }
  }
  function savePosition() {
    const k = kag();
    if (storyLoading || k.tmp.auto_drama_transition) return false;
    k.stat.auto_drama_position = saveIdentity(playback.atChapterEnd() ? "chapter_end" : playback.eventId());
    if (music) {
      k.stat.auto_drama_music = music.snapshot();
      k.stat.auto_drama_bgm_volume = Number(byId("ad-bgm-volume").value) / 100;
    }
    return k.key_mouse.qsave();
  }
  function musicForEvent(eventId) {
    if (!eventId || eventId === "chapter_start") return null;
    const cues = new Map((script.music_cues || []).map(cue => [cue.utterance_id, cue]));
    let current = null;
    for (const line of script.utterances) {
      const cue = cues.get(line.id);
      if (cue?.action === "play") current = cue;
      else if (cue?.action === "stop") current = null;
      if (line.id === eventId) return current;
    }
    return eventId === "chapter_end" ? current : null;
  }
  function restoreMusic() {
    if (!music) return Promise.resolve();
    const k = kag(), stat = k.stat, token = musicEpoch, saved = stat.auto_drama_music;
    let volume = saved?.user_volume ?? k.stat.auto_drama_bgm_volume ?? 1;
    if (!Number.isFinite(volume) || volume < 0 || volume > 1) volume = 1;
    byId("ad-bgm-volume").value = String(volume * 100);
    music.setVolume(volume);
    if (Object.prototype.hasOwnProperty.call(k.stat, "auto_drama_music")) {
      return music.restore(saved).then(restored => {
        if (restored && token === musicEpoch && stat === k.stat && started &&
            !pageSuspended && !byId("ad-log").open) return music.resume();
      }).catch(() => { if (token === musicEpoch && stat === k.stat) status("BGMを復元できませんでした。"); });
    } else {
      const cue = musicForEvent(k.stat.auto_drama_position?.event_id);
      if (cue) return music.cue(cue).then(restored => {
        if (restored && token === musicEpoch && stat === k.stat && started &&
            !pageSuspended && !byId("ad-log").open) return music.resume();
      }).catch(() => { if (token === musicEpoch && stat === k.stat) status("BGMを再開できませんでした。"); });
    }
  }
  function bindMusic(k) {
    if (!window.AutoDramaMusic?.create) return;
    music = window.AutoDramaMusic.create(script, {report: message => status(message)});
    music.setVolume(Number(byId("ad-bgm-volume").value) / 100);
    if (!(script.music_cues || []).length) return;
    const cues = new Map(script.music_cues.map(cue => [cue.id, cue]));
    // Compiler-owned tags select adopted cue IDs. Author text cannot provide
    // URLs, engine parameters, or executable JavaScript.
    k.ftag.master_tag.ad_music = {kag: k, pm: {cue: ""}, vital: ["cue"], start(pm) {
      const stat = k.stat, tags = k.ftag.array_tag, index = k.ftag.current_order_index;
      const token = ++musicEpoch;
      k.weaklyStop();
      Promise.resolve(music.cue(cues.get(pm.cue))).catch(() => {
        status("BGMを再生できませんでした。本文の再生を続けます。");
      }).finally(() => {
        if (token !== musicEpoch || stat !== k.stat || tags !== k.ftag.array_tag ||
            index !== k.ftag.current_order_index) return;
        k.cancelWeakStop();
        k.ftag.nextOrder();
      });
    }};
    k.ftag.master_tag.ad_music_stop = {kag: k, pm: {fade: "0"}, start(pm = {}) {
      const fade = Number(pm.fade || 0);
      if (!fade || !music.stopFaded || !(script.scene_transitions || []).length) {
        musicEpoch++; music.stop(); k.ftag.nextOrder(); return;
      }
      const stat = k.stat, tags = k.ftag.array_tag, index = k.ftag.current_order_index;
      const token = ++musicEpoch;
      k.tmp.auto_drama_transition = true; k.weaklyStop();
      music.stopFaded({fadeOutMs: k.stat.is_skip ? 0 : fade}).catch(() => {}).finally(() => {
        if (token !== musicEpoch || stat !== k.stat || tags !== k.ftag.array_tag ||
            index !== k.ftag.current_order_index) return;
        k.tmp.auto_drama_transition = false; k.cancelWeakStop(); k.ftag.nextOrder();
      });
    }};
  }
  function cancelTransitions({remember = false} = {}) {
    transitionEpoch++;
    transitions?.cancel();
    if (kag()?.tmp) kag().tmp.auto_drama_transition = false;
    if (!remember) pendingTransition = null;
  }
  function bindTransitions(k) {
    if (!(script.scene_transitions || []).length) return;
    transitions = window.AutoDramaTransitions.create(stages, {
      music, adapter: window.AutoDramaTransitions.kagAdapter(k), report: status,
      onBusyChange: value => { k.tmp.auto_drama_transition = value; },
    });
    const valid = state => state && state.stat === k.stat && state.tags === k.ftag.array_tag &&
      state.index === k.ftag.current_order_index;
    k.ftag.master_tag.ad_transition = {kag: k, pm: {cue: ""}, vital: ["cue"], start(pm, resume = false) {
      const state = {pm, stat: k.stat, tags: k.ftag.array_tag, index: k.ftag.current_order_index};
      const token = ++transitionEpoch;
      pendingTransition = state;
      k.weaklyStop();
      transitions.run(pm.cue, {skip: !!k.stat.is_skip, resume}).then(completed => {
        if (!completed || token !== transitionEpoch || !valid(state)) return;
        pendingTransition = null;
        k.cancelWeakStop(); k.ftag.nextOrder();
      }).catch(() => {
        if (token !== transitionEpoch || !valid(state)) return;
        pendingTransition = null;
        status("場面転換を完了できませんでした。本文の再生を続けます。");
        k.cancelWeakStop(); k.ftag.nextOrder();
      });
    }};
    transitions.warm(stages.scenes[0]?.id).catch(() => {});
  }
  function clearEnd() {
    epoch++;
    cancelChapterNavigation();
    nextPreparation?.controller.abort(); nextPreparation = null;
    ended = false; nextBuild = null; polling = false; pollAt = 0;
    byId("ad-chapter-end").hidden = true;
    byId("ad-next-button").hidden = true;
  }
  function boundedRead(url, {signal, cache = "no-store", bytes = false, timeoutMs = 8000} = {}) {
    return new Promise((resolve, reject) => {
      const controller = new window.AbortController();
      let done = false, timer;
      const finish = (error, value) => {
        if (done) return;
        done = true; window.clearTimeout(timer); signal?.removeEventListener("abort", abort);
        if (error) { controller.abort(); reject(error); } else resolve(value);
      };
      const abort = () => finish(Error("chapter request cancelled"));
      signal?.addEventListener("abort", abort, {once: true});
      if (signal?.aborted) { abort(); return; }
      timer = window.setTimeout(() => finish(Error("chapter request timed out")), timeoutMs);
      Promise.resolve().then(() => fetch(url, {cache, signal: controller.signal})).then(response => {
        if (!response.ok) throw Error("next chapter unavailable");
        return bytes ? response.arrayBuffer() : response.json();
      }).then(value => finish(null, value), error => finish(error));
    });
  }
  function validatedNext(result) {
    if (result.build_id !== identity.build_id || !["ready", "waiting", "complete"].includes(result.status)) {
      throw Error("invalid chapter lineage");
    }
    if (result.status === "ready") {
      const next = result.next_build;
      if (!next || !/^[a-zA-Z0-9_-]+$/.test(next.id) || next.id === identity.build_id ||
          next.chapter_number !== context.chapter_number + 1) throw Error("invalid next chapter");
    }
    return result;
  }
  function prepareNext(id) {
    if (nextPreparation?.id === id) return;
    nextPreparation?.controller.abort();
    const controller = new window.AbortController(), prefix = "/player/" + encodeURIComponent(id) + "/";
    nextPreparation = {id, controller};
    (async () => {
      const next = await boundedRead(prefix + "script.json", {signal: controller.signal, cache: "force-cache"});
      if (!Array.isArray(next.utterances) || !Array.isArray(next.assets)) return;
      const first = next.utterances[0], selected = new Set();
      if (first?.audio_asset_id) selected.add(first.audio_asset_id);
      const cue = (next.music_cues || []).find(value => value.utterance_id === first?.id && value.action === "play");
      if (cue) selected.add(cue.asset_id);
      if ((next.scene_transitions || []).length) {
        const targets = await boundedRead(prefix + "data/others/auto_drama_stages.json",
          {signal: controller.signal, cache: "force-cache"});
        const stage = targets.scenes?.[0];
        if (stage?.background) {
          const asset = next.assets.find(value => value.kind === "background" && value.filename === stage.background.storage);
          if (asset) selected.add(asset.id);
        }
        for (const character of (stage?.characters || []).slice(0, 3)) {
          const asset = next.assets.find(value => value.kind === "character" && value.filename === character.storage);
          if (asset) selected.add(asset.id);
        }
      } else {
        for (const direction of (next.directions || []).filter(value => value.utterance_id === first?.id)) {
          if (direction.kind === "background") selected.add(direction.asset_id);
          if (direction.kind === "enter") {
            const character = next.characters?.find(value => value.id === direction.character_id);
            if (character?.image_asset_id) selected.add(character.image_asset_id);
          }
        }
      }
      const folders = {background: "bgimage", character: "fgimage", audio: "sound", music: "bgm"};
      const assets = next.assets.filter(value => selected.has(value.id)).slice(0, 8);
      await Promise.all(assets.map(value => {
        if (!folders[value.kind] || !/^[A-Za-z0-9][A-Za-z0-9_.-]*\.[A-Za-z0-9]+$/.test(value.filename)) return;
        return boundedRead(prefix + "data/" + folders[value.kind] + "/" + value.filename,
          {signal: controller.signal, cache: "force-cache", bytes: true}).catch(() => {});
      }));
    })().catch(() => {}); // Best-effort warming never changes the current chapter.
  }
  function showNext(result) {
    nextBuild = null; byId("ad-next-button").hidden = true;
    if (result.status === "ready") {
      nextBuild = result.next_build.id;
      byId("ad-next-status").textContent = "次の章を鑑賞できます。準備ができたら進んでください。";
      byId("ad-next-button").hidden = false; pollAt = Infinity;
      prepareNext(nextBuild);
    } else if (result.status === "complete") {
      byId("ad-next-status").textContent = "物語はここまでです。最後までご鑑賞いただきありがとうございました。";
      pollAt = Infinity;
    } else {
      byId("ad-next-status").textContent = "次の章を制作中です。この位置を保ったままお待ちください。";
      pollAt = Date.now() + 5000;
    }
  }
  async function checkNext() {
    if (!ended || context?.mode !== "live" || polling || chapterNavigation || pageSuspended) return;
    const requestEpoch = epoch;
    polling = true;
    byId("ad-next-button").disabled = true;
    try {
      const result = validatedNext(await boundedRead(context.next_url));
      if (requestEpoch !== epoch || !ended) return;
      showNext(result);
    } catch (_) {
      if (requestEpoch !== epoch || !ended) return;
      nextBuild = null;
      byId("ad-next-button").hidden = true;
      byId("ad-next-status").textContent = "次の章の公開状況を確認できません。この位置を保って再確認します。";
      pollAt = Date.now() + 5000;
    } finally {
      if (requestEpoch === epoch) {
        polling = false;
        byId("ad-next-button").disabled = !nextBuild;
      }
    }
  }
  function cancelChapterNavigation({keepReturn = false} = {}) {
    const pending = chapterNavigation; chapterNavigation = null;
    pending?.controller.abort(); pending?.animation?.cancel();
    if (pending?.timer) window.clearTimeout(pending.timer);
    const overlay = byId("ad-chapter-transition");
    if (overlay) { overlay.hidden = true; if (overlay.style) overlay.style.opacity = "0"; }
    if (pending && kag()?.tmp) kag().tmp.auto_drama_transition = false;
    if (!keepReturn) returnMusic = undefined;
  }
  function restoreBoundaryMusic(snapshot) {
    if (!music) return Promise.resolve();
    const token = ++musicEpoch, stat = kag().stat;
    return music.restore(snapshot).then(restored => {
      if (restored && token === musicEpoch && stat === kag().stat && started &&
          !pageSuspended && !byId("ad-log").open && !chapterNavigation) return music.resume();
    }).catch(() => status("BGMを復元できませんでした。"));
  }
  async function enterNextChapter() {
    const k = kag();
    if (!nextBuild || !ended || context?.mode !== "live" || polling || chapterNavigation ||
        storyLoading || k.tmp.auto_drama_transition || byId("ad-log").open || pageSuspended) return;
    savePosition(); // Native [s] position and the live loop, before fading/busy.
    const state = {controller: new window.AbortController(), epoch, stat: k.stat,
      tags: k.ftag.array_tag, index: k.ftag.current_order_index, snapshot: music?.snapshot() ?? null};
    chapterNavigation = state; returnMusic = state.snapshot;
    const valid = () => chapterNavigation === state && state.epoch === epoch && ended &&
      state.stat === k.stat && state.tags === k.ftag.array_tag && state.index === k.ftag.current_order_index;
    k.tmp.auto_drama_transition = true;
    byId("ad-next-button").disabled = true;
    controls().forEach(control => { control.disabled = true; });
    const overlay = byId("ad-chapter-transition");
    overlay.hidden = false; overlay.style.opacity = "0";
    let visual;
    if (overlay.animate) {
      state.animation = overlay.animate([{opacity: 0}, {opacity: 1}], {duration: 400, fill: "forwards"});
      visual = state.animation.finished;
    } else visual = new Promise(resolve => { state.timer = window.setTimeout(resolve, 400); });
    visual.catch(() => {});
    try {
      const [result] = await Promise.all([
        boundedRead(context.next_url, {signal: state.controller.signal}).then(validatedNext),
        visual, music?.stopFaded({fadeOutMs: 400, signal: state.controller.signal}) ?? Promise.resolve(true),
      ]);
      if (!valid()) return;
      showNext(result);
      if (result.status !== "ready") throw Error("next chapter not ready");
      overlay.style.opacity = "1";
      try {
        window.sessionStorage?.setItem(entryKey, JSON.stringify({schema_version: 1,
          project_id: identity.project_id, storyline_id: identity.storyline_id,
          to_build: result.next_build.id, chapter_number: result.next_build.chapter_number,
          created_at: Date.now()}));
      } catch (_) {}
      window.location.assign("/player/" + encodeURIComponent(result.next_build.id) + "/");
    } catch (_) {
      if (!valid()) return;
      const snapshot = state.snapshot;
      try { window.sessionStorage?.removeItem(entryKey); } catch (_) {}
      cancelChapterNavigation();
      if (nextBuild) {
        byId("ad-next-status").textContent = "次の章へ進めませんでした。この位置で待機し、もう一度確認できます。";
        pollAt = Date.now() + 5000;
      }
      byId("ad-next-button").disabled = !nextBuild;
      await restoreBoundaryMusic(snapshot);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const start = byId("ad-start-button"), log = byId("ad-log");
    const toolbar = byId("ad-toolbar");
    let startupError = false;
    const startupFailure = message => {
      startupError = true;
      start.textContent = message;
      start.disabled = true;
    };
    if (![toolbar, log, byId("ad-chapter-end"), byId("ad-resume-button"),
      byId("ad-next-button"), byId("ad-next-status")].every(Boolean)) {
      startupFailure("鑑賞画面のファイルの版が一致しません。制御サーバーを再起動し、制作画面から章を開き直してください。");
      return;
    }
    const loadingSince = Date.now();
    function beginChapter() {
      const k = kag();
      if (started || !k || !script) return;
      k.readyAudio();
      music?.unlock().catch(() => status("BGMの再生を準備できませんでした。"));
      if (window.Howler && Howler.ctx) Howler.ctx.resume().catch(() => {});
      started = true;
      byId("ad-start").hidden = true;
      k.key_mouse.next();
    }
    async function tryChapterEntry() {
      if (entryAttempted || context?.mode !== "live" || !identity) return;
      entryAttempted = true;
      let intent;
      try {
        intent = JSON.parse(window.sessionStorage?.getItem(entryKey) || "null");
        if (intent?.to_build !== identity.build_id) return;
        window.sessionStorage.removeItem(entryKey); // Single-use, including rejected intents.
      } catch (_) { return; }
      const age = Date.now() - intent.created_at;
      if (intent.schema_version !== 1 || intent.project_id !== identity.project_id ||
          intent.storyline_id !== identity.storyline_id || intent.chapter_number !== context.chapter_number ||
          !Number.isFinite(age) || age < 0 || age > 30000) return;
      const token = ++entryGeneration;
      let timer;
      try {
        kag().readyAudio();
        const speech = script.utterances.some(line => line.audio_asset_id);
        const voice = !speech ? Promise.resolve(true) : window.Howler?.ctx ?
          Promise.resolve(window.Howler.ctx.resume()).then(() => window.Howler.ctx.state === "running") : Promise.resolve(false);
        const unlocked = Promise.all([music?.unlock() ?? true, voice]).then(values => values.every(Boolean));
        const allowed = await Promise.race([unlocked,
          new Promise(resolve => { timer = window.setTimeout(() => resolve(false), 300); })]);
        if (allowed && token === entryGeneration && !started && !pageSuspended && !startupError) beginChapter();
      } catch (_) {} finally { if (timer) window.clearTimeout(timer); }
    }
    // Tyrano stops auto on document mousedown/touchstart, before a button's
    // click runs. Isolate the complete interaction so clicking an active auto
    // toggle cannot stop it on press and then unintentionally restart on click.
    // Do not preventDefault: focus, range dragging and native audio still work.
    [toolbar, log, byId("ad-chapter-end")].forEach(surface => {
      ["pointerdown", "mousedown", "mouseup", "touchstart", "touchend",
        "keydown", "keyup", "click"].forEach(type => {
        surface.addEventListener(type, e => e.stopPropagation());
      });
    });
    const isControl = target => target?.closest?.(
      "#ad-toolbar, #ad-log, #ad-start, #ad-chapter-end, button, input, select, textarea, audio, a");
    const inMessageWindow = e => {
      if (isControl(e.target)) return false;
      const point = e.changedTouches?.[0] || e;
      return [...document.querySelectorAll(".message_outer")].some(element => {
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 && point.clientX >= rect.left &&
          point.clientX <= rect.right && point.clientY >= rect.top && point.clientY <= rect.bottom;
      });
    };
    // The native event layer disappears during [wse], so hit-test the visible
    // message rectangle and handle input before native handlers can advance.
    ["pointerdown", "mousedown", "mouseup", "touchstart", "touchend"].forEach(type => {
      document.addEventListener(type, e => {
        if (started && inMessageWindow(e)) e.stopPropagation();
      }, true);
    });
    document.addEventListener("click", e => {
      if (!started || e.button !== 0 || !inMessageWindow(e)) return;
      e.preventDefault(); e.stopPropagation();
      if (!log.open && !storyLoading) playback?.advance();
    }, true);
    document.addEventListener("keydown", e => {
      if (!started) { e.stopPropagation(); return; }
      if (e.key !== "Enter" || isControl(e.target) || log.open) return;
      e.preventDefault(); e.stopPropagation();
      if (!e.repeat && !storyLoading) playback?.advance();
    }, true);
    start.addEventListener("click", e => {
      e.stopPropagation();
      beginChapter();
    });
    byId("ad-resume-button").addEventListener("click", e => {
      e.stopPropagation();
      const k = kag();
      if (started || !readSave()) return;
      k.readyAudio();
      music?.unlock().catch(() => status("BGMの再生を準備できませんでした。"));
      if (window.Howler && Howler.ctx) Howler.ctx.resume().catch(() => {});
      started = true;
      byId("ad-start").hidden = true;
      k.key_mouse.qload();
    });
    byId("ad-next-button").addEventListener("click", () => {
      enterNextChapter();
    });
    byId("ad-auto").addEventListener("click", () => {
      const k = kag();
      if (k.stat.is_auto) stopAuto();
      // The native role advances a click-wait correctly. During active text or
      // voice, only arm the flag; the next [p] will start the automatic wait.
      else if (k.tmp.auto_drama_transition || k.stat.is_adding_text || k.tmp.is_se_play || k.stat.is_wait) k.setAuto(true);
      else if (!k.key_mouse.auto()) k.setAuto(true);
      byId("ad-auto").setAttribute("aria-pressed", String(!!k.stat.is_auto));
    });
    byId("ad-save").addEventListener("click", () => {
      stopAuto();
      if (!savePosition()) status("台詞の表示と音声が終わってから保存してください。");
    });
    byId("ad-load").addEventListener("click", () => {
      const k = kag();
      if (storyLoading || k.tmp.auto_drama_transition) return;
      if (!readSave()) {
        status("この版の章に対応する保存データはありません。"); return;
      }
      stopAuto();
      if (!k.key_mouse.qload()) status("台詞の表示と音声が終わってから読み込んでください。");
    });
    byId("ad-font").addEventListener("change", e => {
      const k = kag(), size = e.target.value;
      k.stat.font.size = size;
      k.stat.default_font.size = size;
      document.querySelectorAll(".message_inner span").forEach(span => {
        span.style.fontSize = size + "px";
      });
    });
    byId("ad-volume").addEventListener("input", e => {
      // next:false is essential: a setting must not advance the scenario.
      kag().ftag.startTag("seopt", {volume: e.target.value, next: "false"});
      log.querySelectorAll("audio").forEach(a => { a.volume = Number(e.target.value) / 100; });
    });
    byId("ad-bgm-volume")?.addEventListener("input", e => {
      music?.setVolume(Number(e.target.value) / 100);
    });
    byId("ad-backlog").addEventListener("click", () => {
      if (storyLoading || kag().tmp.auto_drama_transition) return;
      if (!playback?.openLog()) return;
      music?.pause();
      const items = byId("ad-log-items");
      items.replaceChildren();
      const lines = playback.displayedLines();
      if (!lines.length) items.textContent = "表示が完了した本文はまだありません。";
      lines.forEach(line => {
        const article = document.createElement("article");
        const speaker = script.characters.find(c => c.id === line.speaker_id);
        if (speaker) {
          const name = document.createElement("strong");
          name.textContent = speaker.name; article.append(name);
        }
        const text = document.createElement("p");
        text.textContent = line.display_text; article.append(text);
        const asset = script.assets.find(a => a.id === line.audio_asset_id);
        if (asset) {
          const audio = document.createElement("audio");
          audio.controls = true; audio.preload = "none";
          audio.src = "./data/sound/" + encodeURIComponent(asset.filename);
          audio.volume = Number(byId("ad-volume").value) / 100;
          audio.addEventListener("play", () => {
            log.querySelectorAll("audio").forEach(a => { if (a !== audio) a.pause(); });
          });
          article.append(audio);
        }
        items.append(article);
      });
      log.showModal();
      items.lastElementChild?.scrollIntoView({block: "nearest"});
    });
    byId("ad-log-close").addEventListener("click", () => log.close());
    log.addEventListener("close", () => {
      pauseReplay(); playback?.closeLog();
      if (started && !pageSuspended && !chapterNavigation) music?.resume().catch(() => {});
    });
    window.addEventListener?.("pagehide", event => {
      entryGeneration++;
      const hadNavigation = !!chapterNavigation;
      if (hadNavigation) cancelChapterNavigation({keepReturn: !!event?.persisted});
      if (event?.persisted) {
        // A cached page returns with the same script/controller. Keep its track
        // and pending cue alive; discarding either would break Back navigation.
        pageSuspended = true;
        if (pendingTransition) cancelTransitions({remember: true});
        // A chapter already exiting to silence must never preserve its old
        // track for Back navigation. Settle this stop when the page is hidden;
        // ordinary scene transitions retain their resumable track instead.
        const k = kag();
        if (k?.ftag?.array_tag[k.ftag.current_order_index]?.name === "ad_music_stop") music?.stop();
        else music?.pause();
      } else {
        epoch++; musicEpoch++; cancelChapterNavigation(); nextPreparation?.controller.abort();
        cancelTransitions(); transitions?.dispose(); music?.dispose();
      }
    });
    window.addEventListener?.("pageshow", event => {
      if (!event?.persisted || !pageSuspended) return;
      pageSuspended = false;
      if (started && !log.open) {
        if (returnMusic !== undefined) {
          const savedMusic = returnMusic; returnMusic = undefined;
          restoreBoundaryMusic(savedMusic);
        } else music?.resume().catch(() => {});
        const saved = pendingTransition, k = kag();
        if (saved && saved.stat === k.stat && saved.tags === k.ftag.array_tag &&
            saved.index === k.ftag.current_order_index) {
          k.ftag.master_tag.ad_transition.start(saved.pm, true);
        }
      }
      byId("ad-next-button").disabled = !nextBuild;
    });
    fetch("./script.json").then(response => {
      if (!response.ok) throw Error("script unavailable");
      return response.json();
    }).then(async value => {
      script = value;
      if (!(value.scene_transitions || []).length) return;
      const response = await fetch("./data/others/auto_drama_stages.json");
      if (!response.ok) throw Error("stage targets unavailable");
      const result = await response.json();
      if (result.schema_version !== 1 || !Array.isArray(result.scenes) ||
          result.scenes.length !== value.scene_transitions.length ||
          result.scenes.some(scene => !value.scene_transitions.some(cue => cue.id === scene.id))) {
        throw Error("stage targets do not match adopted script");
      }
      stages = result;
    }).catch(() => {
      startupFailure("作品を読み込めません。再読み込みしてください。");
    });
    if (livePath) fetch("./player-context.json", {cache: "no-store"}).then(response => {
      if (!response.ok) throw Error("player identity unavailable");
      return response.json();
    }).then(value => {
      if (value.mode !== "live" || value.build_id !== livePath[1] ||
          typeof value.production_id !== "string" || typeof value.project_id !== "string" ||
          typeof value.storyline_id !== "string" ||
          !Number.isInteger(value.chapter_number) || value.chapter_number < 1 ||
          value.next_url !== "/api/m3/builds/" + livePath[1] + "/next") throw Error("invalid player identity");
      context = value;
    }).catch(() => {
      startupFailure("公開版を確認できません。制御サーバーを再起動して再読み込みしてください。");
    });
    setInterval(() => {
      const k = kag();
      const ready = !!(context && script && (!(script.scene_transitions || []).length || stages) &&
        k && k.key_mouse && k.stat && k.stat.current_scenario &&
        k.ftag && k.ftag.array_tag);
      const stable = ready && !storyLoading && !k.tmp.auto_drama_transition && !k.stat.is_adding_text && !k.tmp.is_se_play &&
        !k.stat.is_wait && k.key_mouse.util.canShowMenu();
      if (!started && !stable && !startupError && Date.now() - loadingSince >= 15000) {
        startupFailure("鑑賞の準備を完了できません。通信状態を確認して再読み込みしてください。");
      }
      if (ready && !bound) {
        if ((script.music_cues || []).length && !window.AutoDramaMusic?.create) {
          startupFailure("BGM再生ファイルの版が一致しません。サーバーを再起動して公開版を開き直してください。");
          return;
        }
        if ((script.scene_transitions || []).length && !window.AutoDramaTransitions?.create) {
          startupFailure("場面転換ファイルの版が一致しません。サーバーを再起動して公開版を開き直してください。");
          return;
        }
        bound = true;
        identity = context.mode === "live" ? {
          project_id: context.project_id, storyline_id: context.storyline_id,
          production_id: context.production_id, build_id: context.build_id,
        } : {project_id: script.id, storyline_id: k.config.projectID,
          production_id: k.config.projectID, build_id: k.config.projectID};
        if (context.mode === "live") k.config.projectID += "_" + context.production_id + "_" + context.build_id;
        playback = window.AutoDramaPlayback(k, script);
        bindMusic(k);
        bindTransitions(k);
        k.on("storage-quicksave", () => status(readSave() ?
          "このブラウザーに保存しました。" : "保存できませんでした。ブラウザーの保存設定を確認してください。"));
        k.on("load-start", () => {
          storyLoading = true;
          loadRestoreStarted = false;
          musicEpoch++;
          cancelTransitions();
          music?.stop();
          clearEnd();
          playback.reset();
          if (log.open) log.close();
          pauseReplay();
        });
        const completeStoryLoad = () => {
          if (!storyLoading || loadRestoreStarted) return;
          loadRestoreStarted = true;
          const token = musicEpoch, stat = k.stat;
          // Native loads expose their saved strong-stop flag before make.ks
          // returns. Hold boundary autosave until its saved track is restored.
          Promise.resolve(restoreMusic()).finally(() => {
            if (token === musicEpoch && stat === k.stat) storyLoading = false;
          });
          status("保存した位置から再開しました。");
          byId("ad-font").value = String(k.stat.default_font.size);
          byId("ad-volume").value = String(k.config.defaultSeVolume);
        };
        k.on("load-complete", completeStoryLoad);
        // Native [s] saves have no pending page-clear flag, so this engine's
        // make.ks return omits load-complete. Adapt that same trusted return
        // without changing engine files or restoring twice for ordinary saves.
        const returnTag = k.ftag.master_tag?.return;
        if (returnTag?.start) {
          const originalReturn = returnTag.start;
          returnTag.start = function (...args) {
            const loadingMake = storyLoading && k.stat.current_scenario === "make.ks";
            const result = originalReturn.apply(this, args);
            if (loadingMake) completeStoryLoad();
            return result;
          };
        }
      }
      if (!started) {
        start.disabled = !stable;
        if (stable) start.textContent = "再生する";
        const resume = byId("ad-resume-button");
        resume.hidden = !ready || !readSave();
        resume.disabled = !stable;
        if (stable && bound) tryChapterEntry();
      }
      controls().forEach(control => { control.disabled = !ready || !started || storyLoading || !!chapterNavigation; });
      ["ad-save", "ad-load"].forEach(id => {
        byId(id).disabled = !stable || !started;
      });
      byId("ad-backlog").disabled = !started || !ready || storyLoading || k.tmp.auto_drama_transition || !playback?.canOpenLog();
      if (ready && k.tmp.auto_drama_transition && k.stat.is_skip) {
        transitions?.skip(); music?.finishTransition?.();
      }
      if (ready && started && stable && playback.atChapterEnd() && !log.open) {
        if (!ended) {
          ended = true;
          musicEpoch++;
          stopAuto();
          savePosition();
          byId("ad-chapter-end").hidden = false;
          byId("ad-next-status").textContent = context.mode === "live" ?
            "次の章の公開状況を確認しています。" :
            "この書き出しに含まれる章はここまでです。続きを鑑賞するには制作画面から公開済みの章を開いてください。";
        }
        if (context.mode === "live" && Date.now() >= pollAt) checkNext();
      }
      if (ended) byId("ad-auto").disabled = true;
      if (ready) byId("ad-auto").setAttribute("aria-pressed", String(!!k.stat.is_auto));
    }, 200);
  });
})();
