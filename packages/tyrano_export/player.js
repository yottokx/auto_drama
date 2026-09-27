"use strict";
// Only compiler-owned code runs here. All story data enters DOM text sinks.
(() => {
  const byId = id => document.getElementById(id);
  const kag = () => window.TYRANO && window.TYRANO.kag;
  let script = null, started = false, bound = false, playback = null;
  const livePath = window.location?.pathname?.match(/^\/player\/([a-zA-Z0-9_-]+)\/(?:index.html)?$/);
  let context = livePath ? null : {mode: "static"};
  let identity = null, ended = false, nextBuild = null, pollAt = 0, polling = false, epoch = 0;
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
    k.stat.auto_drama_position = saveIdentity(playback.atChapterEnd() ? "chapter_end" : playback.eventId());
    return k.key_mouse.qsave();
  }
  function clearEnd() {
    epoch++;
    ended = false; nextBuild = null; polling = false; pollAt = 0;
    byId("ad-chapter-end").hidden = true;
    byId("ad-next-button").hidden = true;
  }
  async function checkNext(navigate = false) {
    if (!ended || context?.mode !== "live" || polling) return;
    const requestEpoch = epoch;
    polling = true;
    byId("ad-next-button").disabled = true;
    try {
      const response = await fetch(context.next_url, {cache: "no-store"});
      if (!response.ok) throw Error("next chapter unavailable");
      const result = await response.json();
      if (requestEpoch !== epoch || !ended) return;
      if (result.build_id !== identity.build_id || !["ready", "waiting", "complete"].includes(result.status)) {
        throw Error("invalid chapter lineage");
      }
      nextBuild = null;
      byId("ad-next-button").hidden = true;
      if (result.status === "ready") {
        const next = result.next_build;
        if (!next || !/^[a-zA-Z0-9_-]+$/.test(next.id) || next.id === identity.build_id ||
            next.chapter_number !== context.chapter_number + 1) throw Error("invalid next chapter");
        nextBuild = next.id;
        byId("ad-next-status").textContent = "次の章を鑑賞できます。準備ができたら進んでください。";
        byId("ad-next-button").hidden = false;
        pollAt = Infinity;
        if (navigate) {
          // Revalidate the lineage at the explicit chapter boundary, then keep
          // that exact build URL for the whole next chapter.
          savePosition();
          window.location.assign("/player/" + encodeURIComponent(nextBuild) + "/");
        }
      } else if (result.status === "complete") {
        byId("ad-next-status").textContent = "物語はここまでです。最後までご鑑賞いただきありがとうございました。";
        pollAt = Infinity;
      } else {
        byId("ad-next-status").textContent = "次の章を制作中です。この位置を保ったままお待ちください。";
        pollAt = Date.now() + 5000;
      }
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
      if (!log.open) playback?.advance();
    }, true);
    document.addEventListener("keydown", e => {
      if (!started) { e.stopPropagation(); return; }
      if (e.key !== "Enter" || isControl(e.target) || log.open) return;
      e.preventDefault(); e.stopPropagation();
      if (!e.repeat) playback?.advance();
    }, true);
    start.addEventListener("click", e => {
      e.stopPropagation();
      const k = kag();
      if (!k || !script) return;
      k.readyAudio();
      if (window.Howler && Howler.ctx) Howler.ctx.resume().catch(() => {});
      started = true;
      byId("ad-start").hidden = true;
      k.key_mouse.next();
    });
    byId("ad-resume-button").addEventListener("click", e => {
      e.stopPropagation();
      const k = kag();
      if (!readSave()) return;
      k.readyAudio();
      if (window.Howler && Howler.ctx) Howler.ctx.resume().catch(() => {});
      started = true;
      byId("ad-start").hidden = true;
      k.key_mouse.qload();
    });
    byId("ad-next-button").addEventListener("click", () => {
      if (nextBuild) checkNext(true);
    });
    byId("ad-auto").addEventListener("click", () => {
      const k = kag();
      if (k.stat.is_auto) stopAuto();
      // The native role advances a click-wait correctly. During active text or
      // voice, only arm the flag; the next [p] will start the automatic wait.
      else if (k.stat.is_adding_text || k.tmp.is_se_play || k.stat.is_wait) k.setAuto(true);
      else if (!k.key_mouse.auto()) k.setAuto(true);
      byId("ad-auto").setAttribute("aria-pressed", String(!!k.stat.is_auto));
    });
    byId("ad-save").addEventListener("click", () => {
      stopAuto();
      if (!savePosition()) status("台詞の表示と音声が終わってから保存してください。");
    });
    byId("ad-load").addEventListener("click", () => {
      const k = kag();
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
    byId("ad-backlog").addEventListener("click", () => {
      if (!playback?.openLog()) return;
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
    log.addEventListener("close", () => { pauseReplay(); playback?.closeLog(); });
    fetch("./script.json").then(response => {
      if (!response.ok) throw Error("script unavailable");
      return response.json();
    }).then(value => { script = value; }).catch(() => {
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
      const ready = !!(context && script && k && k.key_mouse && k.stat && k.stat.current_scenario &&
        k.ftag && k.ftag.array_tag);
      const stable = ready && !k.stat.is_adding_text && !k.tmp.is_se_play &&
        !k.stat.is_wait && k.key_mouse.util.canShowMenu();
      if (!started && !stable && !startupError && Date.now() - loadingSince >= 15000) {
        startupFailure("鑑賞の準備を完了できません。通信状態を確認して再読み込みしてください。");
      }
      if (ready && !bound) {
        bound = true;
        identity = context.mode === "live" ? {
          project_id: context.project_id, storyline_id: context.storyline_id,
          production_id: context.production_id, build_id: context.build_id,
        } : {project_id: script.id, storyline_id: k.config.projectID,
          production_id: k.config.projectID, build_id: k.config.projectID};
        if (context.mode === "live") k.config.projectID += "_" + context.production_id + "_" + context.build_id;
        playback = window.AutoDramaPlayback(k, script);
        k.on("storage-quicksave", () => status(readSave() ?
          "このブラウザーに保存しました。" : "保存できませんでした。ブラウザーの保存設定を確認してください。"));
        k.on("load-start", () => {
          clearEnd();
          playback.reset();
          if (log.open) log.close();
          pauseReplay();
        });
        k.on("load-complete", () => {
          status("保存した位置から再開しました。");
          byId("ad-font").value = String(k.stat.default_font.size);
          byId("ad-volume").value = String(k.config.defaultSeVolume);
        });
      }
      if (!started) {
        start.disabled = !stable;
        if (stable) start.textContent = "再生する";
        const resume = byId("ad-resume-button");
        resume.hidden = !ready || !readSave();
        resume.disabled = !stable;
      }
      controls().forEach(control => { control.disabled = !ready || !started; });
      ["ad-save", "ad-load"].forEach(id => {
        byId(id).disabled = !stable || !started;
      });
      byId("ad-backlog").disabled = !started || !ready || !playback?.canOpenLog();
      if (ready && started && stable && playback.atChapterEnd() && !log.open) {
        if (!ended) {
          ended = true;
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
