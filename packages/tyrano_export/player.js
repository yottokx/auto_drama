"use strict";
// The viewing screen on the installed Tyrano engine. The engine keeps the
// stage (backgrounds, portraits, masks) and the scenario order; this file owns the
// message window, input, voice, backlog and settings. Story text reaches the DOM
// only through text sinks, looked up in script.json by utterance ID.
(() => {
  const TAGS = window.tyrano.plugin.kag.tag;
  const livePath = window.location.pathname.match(/^\/player\/([a-zA-Z0-9_-]+)\/(?:index.html)?$/);
  const SETTINGS_KEY = "adn_settings_v1", ENTRY_KEY = "auto_drama_next_chapter_entry_v1";
  const $ = id => document.getElementById(id);

  let k = null, script = null, states = null, stages = null, context = livePath ? null : {mode: "static"};
  let music = null, transitions = null, stageAdapter = null, bound = false, failed = false;
  let lines = [], lineIndex = new Map(), characters = new Map(), assets = new Map(), cues = new Map(), scenes = new Map();
  // gen invalidates every pending callback when a jump replaces the current position.
  let gen = 0, started = false, gate = false, seeking = false, busy = false, ended = false, fontsReady = false;
  // gate → line → (transition) → say → after → wait → … → end
  let tagState = "idle";
  const cur = {index: -1, pages: [], page: 0, typed: 0, phase: "idle", advance: false};
  let auto = false, skipping = false, logOpen = false, logAtEnd = true, logOpenedAt = 0, pendingComplete = false;
  let voice = null, voiceOn = false, warmVoice = null, logVoice = null;
  let typeTimer, autoTimer, toastTimer, pauseFinish = null;
  let nextBuild = null, polling = false, pollAt = 0, entering = false;

  const cfg = {size: 24, face: "gothic", speed: 62, wait: 14, voice: 100, bgm: 100, alpha: 80};
  try { Object.assign(cfg, JSON.parse(window.localStorage.getItem(SETTINGS_KEY) || "{}")); } catch (_) {}
  const storeSettings = () => { try { window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(cfg)); } catch (_) {} };

  document.body.insertAdjacentHTML("beforeend", `
<div id="adn-shield"><div id="adn">
  <section id="adn-win" aria-live="polite">
    <div id="adn-col">
      <div id="adn-namerow"><span id="adn-name"></span><span id="adn-voice" aria-hidden="true"><i></i><i></i><i></i></span><span id="adn-rule"></span><span id="adn-pages"></span></div>
      <div id="adn-body"></div>
      <div id="adn-measure" aria-hidden="true"></div>
    </div>
    <div id="adn-foot">
      <nav id="adn-menu" aria-label="鑑賞操作">
        <button id="adn-m-auto">オート</button><button id="adn-m-save">セーブ</button><button id="adn-m-load">ロード</button>
        <button id="adn-m-log" title="ホイール上 / ↑">ログ</button><button id="adn-m-config">設定</button>
        <button id="adn-m-hide" title="右クリック">非表示</button>
      </nav>
      <button id="adn-mode" aria-label="解除"><span id="adn-mode-label">AUTO</span></button>
    </div>
    <div id="adn-gauge"></div>
  </section>
  <div id="adn-toast" role="status"></div>
  <section id="adn-log" aria-label="バックログ">
    <div class="sheet-head"><h2>ログ</h2><button id="adn-log-close">閉じる</button></div>
    <div id="adn-log-list"></div>
  </section>
  <aside id="adn-config" aria-label="設定">
    <div class="sheet-head"><h2>設定</h2><button id="adn-config-close">閉じる</button></div>
    <div class="row"><span>文字の大きさ</span>
      <div class="seg" data-key="size"><button data-v="21">小</button><button data-v="24">標準</button><button data-v="28">大</button></div></div>
    <div class="row"><span>書体</span>
      <div class="seg" data-key="face"><button data-v="gothic">ゴシック</button><button data-v="mincho">明朝</button></div></div>
    <label class="row"><span>文字の速さ <output></output></span><input type="range" data-key="speed" min="0" max="100"></label>
    <label class="row"><span>オートの待ち時間 <output></output></span><input type="range" data-key="wait" min="3" max="40"></label>
    <label class="row"><span>音声の音量 <output></output></span><input type="range" data-key="voice" min="0" max="100"></label>
    <label class="row"><span>BGMの音量 <output></output></span><input type="range" data-key="bgm" min="0" max="100"></label>
    <label class="row"><span>ウィンドウの濃さ <output></output></span><input type="range" data-key="alpha" min="30" max="100"></label>
  </aside>
  <section id="adn-end" aria-label="章の終わり" hidden>
    <p id="adn-end-note" role="status"></p>
    <div class="choices"><button id="adn-next" hidden>次の章へ</button></div>
  </section>
  <section id="adn-start" aria-label="再生開始">
    <p id="adn-chapter"></p><h1 id="adn-title"></h1>
    <p id="adn-start-note" role="status">読み込んでいます</p>
    <div class="choices"><button id="adn-begin" disabled>はじめから</button><button id="adn-resume" hidden>つづきから</button></div>
  </section>
  <div id="adn-cover"></div>
</div></div>`);
  const shield = $("adn-shield"), ui = $("adn"), body = $("adn-body"), measure = $("adn-measure");
  const root = document.documentElement;

  function toast(message) {
    const el = $("adn-toast"); el.textContent = message; el.classList.add("on");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove("on"), 2200);
  }
  function fade(element, to, duration) {
    const from = Number(element.style.opacity || 0);
    element.style.opacity = String(to);
    if (!duration || !element.animate || from === to) return Promise.resolve();
    return element.animate([{opacity: from}, {opacity: to}], {duration, easing: "linear"}).finished.catch(() => {});
  }
  // Follow the engine's own stage rectangle so the window lines up with the portraits at any size.
  function syncStage() {
    const rect = $("tyrano_base")?.getBoundingClientRect();
    let scale = Math.min(window.innerWidth / 960, window.innerHeight / 640);
    let left = (window.innerWidth - 960 * scale) / 2, top = (window.innerHeight - 640 * scale) / 2;
    if (rect && rect.width > 0 && rect.height > 0) { scale = rect.width / 960; left = rect.left; top = rect.top; }
    ui.style.transform = `translate(${left}px, ${top}px) scale(${scale})`;
  }

  /* ── 設定 ── */
  function applySettings() {
    root.style.setProperty("--fs", cfg.size + "px");
    root.style.setProperty("--body-font", `var(--${cfg.face === "mincho" ? "mincho" : "gothic"})`);
    root.style.setProperty("--win-a", String(cfg.alpha / 100));
    music?.setVolume(cfg.bgm / 100);
    for (const playing of [voice, logVoice]) playing?.volume(cfg.voice / 100);
  }
  function showRange(input) {
    const value = cfg[input.dataset.key], out = input.closest(".row").querySelector("output");
    input.value = String(value);
    out.textContent = input.dataset.key === "speed" ? (value >= 98 ? "一括" : value) :
      input.dataset.key === "wait" ? (value / 10).toFixed(1) + " 秒" :
      input.dataset.key === "alpha" ? value + "%" : value;
  }
  function showSegments() {
    document.querySelectorAll("#adn-config .seg button").forEach(button => {
      button.setAttribute("aria-pressed", String(String(cfg[button.parentElement.dataset.key]) === button.dataset.v));
    });
  }
  document.querySelectorAll("#adn-config input[type=range]").forEach(input => {
    showRange(input);
    input.addEventListener("input", () => {
      cfg[input.dataset.key] = Number(input.value); showRange(input); applySettings(); storeSettings();
    });
  });
  document.querySelectorAll("#adn-config .seg").forEach(seg => seg.addEventListener("click", event => {
    const button = event.target.closest("button"); if (!button) return;
    cfg[seg.dataset.key] = seg.dataset.key === "size" ? Number(button.dataset.v) : button.dataset.v;
    showSegments(); applySettings(); storeSettings(); relayout();
  }));
  showSegments(); applySettings();

  /* ── 改ページ: 句点で区切り、表示行数に収まるだけ詰める。1文で溢れる場合のみ読点、最後は文字単位 ── */
  function fits(text) {
    measure.textContent = text + "　"; // 待機マークの分を確保
    return measure.scrollHeight <= parseFloat(getComputedStyle(body).height) + 1;
  }
  function units(text) {
    const out = [];
    for (const sentence of text.match(/[^。！？\n]*(?:[。！？]+[」』）]*|\n)|[^。！？\n]+$/g) || [text]) {
      if (fits(sentence)) { out.push(sentence); continue; }
      for (const clause of sentence.match(/[^、]*、|[^、]+$/g) || [sentence]) {
        if (fits(clause)) { out.push(clause); continue; }
        let chunk = "";
        for (const ch of clause) { if (chunk && !fits(chunk + ch)) { out.push(chunk); chunk = ""; } chunk += ch; }
        if (chunk) out.push(chunk);
      }
    }
    return out;
  }
  function paginate(text) {
    const pages = [];
    let current = "";
    for (const unit of units(text)) {
      if (current && !fits(current + unit)) { pages.push(current.replace(/\n+$/, "")); current = ""; }
      current += unit;
    }
    if (current) pages.push(current);
    return pages.length ? pages : [text];
  }

  /* ── 音声 ── */
  const soundUrl = id => {
    const asset = assets.get(id);
    return asset ? "./data/sound/" + encodeURIComponent(asset.filename) : null;
  };
  // Voices play through the engine's bundled Howler: the recorder captures Howler's
  // master gain, and BGM keeps its own independent gain in music.js.
  function sound(url, done) {
    let finished = false;
    const finish = () => { if (finished) return; finished = true; howl.unload(); done(); };
    const howl = new window.Howl({src: [url], volume: cfg.voice / 100,
      onend: finish, onloaderror: finish, onplayerror: finish});
    howl.play();
    return {stop() { if (finished) return; finished = true; howl.stop(); howl.unload(); },
      volume: value => howl.volume(value), duration: () => howl.duration(),
      position: () => Number(howl.seek()) || 0};
  }
  function stopVoice() {
    const playing = voice; voice = null; voiceOn = false;
    playing?.stop();
    $("adn-voice").classList.remove("on");
  }
  function playVoice(line) {
    stopVoice();
    const url = soundUrl(line.audio_asset_id);
    if (!url) return;
    const playing = sound(url, () => {
      if (voice !== playing) return;
      voice = null; voiceOn = false; $("adn-voice").classList.remove("on");
      // The last page has waited for the voice; anything earlier keeps its own pace.
      if (tagState === "say" && lastPage() && cur.typed >= cur.pages[cur.page].length) completeSay(false);
    });
    voice = playing; voiceOn = true; $("adn-voice").classList.add("on");
  }
  // Decode the next line ahead of time. Howler shares decoded audio by URL, so the
  // line's own Howl starts without a fetch; the previous warm copy is then released.
  function warmNextVoice(index) {
    const next = lines[index + 1], url = next && soundUrl(next.audio_asset_id);
    const previous = warmVoice;
    warmVoice = url ? new window.Howl({src: [url], preload: true}) : null;
    previous?.unload();
  }

  /* ── 1行の表示 ── */
  const lastPage = () => cur.page >= cur.pages.length - 1;
  const next = () => k.ftag.nextOrder();
  function clearText() {
    clearTimeout(typeTimer); clearAuto();
    $("adn-name").textContent = ""; $("adn-pages").textContent = ""; body.textContent = "";
  }
  function paint() {
    const text = cur.pages[cur.page];
    body.firstChild.textContent = text.slice(0, cur.typed);
    body.lastChild.textContent = text.slice(cur.typed);
  }
  function showMarker() {
    let mark = body.querySelector(".mark");
    if (!mark) { mark = document.createElement("span"); body.firstChild.after(mark); }
    mark.className = "mark " + (lastPage() ? "end" : "more");
  }
  function showPage(page, {complete = false} = {}) {
    clearTimeout(typeTimer); clearAuto();
    cur.page = page; cur.typed = complete ? cur.pages[page].length : 0; cur.phase = "typing";
    $("adn-pages").textContent = cur.pages.length > 1 ? `${page + 1} / ${cur.pages.length}` : "";
    body.innerHTML = '<span></span><span class="rest"></span>';
    paint();
    if (!complete) tick();
  }
  function tick() {
    const text = cur.pages[cur.page], interval = Math.round(90 - cfg.speed * .86);
    if (interval <= 5) cur.typed = text.length; else cur.typed++;
    paint();
    if (cur.typed < text.length) { typeTimer = setTimeout(tick, interval); return; }
    // 続きのページがある間は音声を待たない。最後のページだけ音声の終わりを待つ。
    if (!lastPage()) toWait();
    else { cur.phase = "speaking"; if (!voiceOn) completeSay(false); }
  }
  // 待機状態: 全文を出して待機マークを表示する。音声は止めない。
  function toWait() {
    clearTimeout(typeTimer);
    cur.typed = cur.pages[cur.page].length; paint();
    cur.phase = "wait"; showMarker();
    if (!lastPage()) { if (auto) scheduleAuto(pageDelay()); }
    else if (!voiceOn) completeSay(false);
  }
  // [ad_say] を終えて、行のあとの演出と [ad_wait] へ進む。
  function completeSay(advance) {
    if (tagState !== "say") return;
    if (logOpen && !advance) { pendingComplete = true; return; }
    clearTimeout(typeTimer); clearAuto();
    tagState = "after"; cur.advance = advance;
    if (advance) { stopVoice(); body.querySelector(".mark")?.remove(); }
    next();
  }
  function proceedWait() {
    if (tagState !== "wait") return;
    tagState = "idle"; clearAuto(); body.querySelector(".mark")?.remove();
    next();
  }
  function advance() {
    if (!started || seeking || busy) return;
    if (tagState === "say") {
      if (cur.phase !== "wait") toWait(); // 待機マークが出るまでは待機状態へ移るだけ
      else if (!lastPage()) showPage(cur.page + 1);
      else completeSay(true);
    } else if (tagState === "after") {
      if (cur.phase === "wait") cur.advance = true;
    } else if (tagState === "wait") proceedWait();
  }
  function relayout() {
    if (!["say", "after", "wait"].includes(tagState) || cur.index < 0) return;
    const waiting = cur.phase === "wait" || tagState !== "say";
    cur.pages = paginate(lines[cur.index].display_text);
    showPage(tagState === "say" ? Math.min(cur.page, cur.pages.length - 1) : cur.pages.length - 1, {complete: true});
    if (waiting) { cur.phase = "wait"; showMarker(); } else if (lastPage()) cur.phase = "speaking"; else toWait();
  }

  /* ── オート / スキップ ── */
  function clearAuto() {
    clearTimeout(autoTimer);
    const gauge = $("adn-gauge"); gauge.style.transition = "none"; gauge.style.width = "0";
  }
  function pageDelay() {
    // 音声の長さを文字数で按分して、読み上げ位置に合わせてめくる
    const duration = voiceOn ? voice.duration() : 0;
    if (duration > 0) {
      const done = cur.pages.slice(0, cur.page + 1).join("").length / cur.pages.join("").length;
      return Math.max(300, (duration * done - voice.position()) * 1000);
    }
    return cfg.wait * 60 + readingAllowance();
  }
  // 音声のない行は、表示中に読み進められるので、読み終えるための上乗せは短くとどめる。
  const readingAllowance = () => Math.min(900, cur.pages[cur.page].length * 12);
  const finalDelay = () => cfg.wait * 100 + (lines[cur.index]?.audio_asset_id ? 0 : readingAllowance());
  function scheduleAuto(delay) {
    clearAuto();
    const gauge = $("adn-gauge"); void gauge.offsetWidth;
    gauge.style.transition = `width ${delay}ms linear`; gauge.style.width = "100%";
    autoTimer = setTimeout(() => {
      if (tagState === "say" && cur.phase === "wait" && !lastPage()) showPage(cur.page + 1);
      else if (tagState === "wait") proceedWait();
    }, delay);
  }
  function setAuto(on) {
    if (on && (!started || ended || seeking)) return;
    auto = on; ui.classList.toggle("auto", on);
    if (on) $("adn-mode-label").textContent = "AUTO";
    clearAuto();
    if (!on) return;
    if (tagState === "wait") scheduleAuto(finalDelay());
    else if (tagState === "say" && cur.phase === "wait" && !lastPage()) scheduleAuto(pageDelay());
  }
  function setSkip(on) {
    if (on === skipping) return;
    if (on && (!started || ended || seeking || overlayOpen() || ui.classList.contains("bare"))) return;
    if (on) setAuto(false);
    skipping = on; k.stat.is_skip = on; ui.classList.toggle("skip", on);
    if (on) $("adn-mode-label").textContent = "SKIP";
    if (!on) return;
    if (pauseFinish) pauseFinish();
    else if (busy) { transitions?.skip(); music?.finishTransition?.(); }
    else if (tagState === "say") completeSay(true);
    else if (tagState === "wait") proceedWait();
  }

  /* ── ティラノ側から呼ばれるタグ ── */
  const define = (name, pm, start) => {
    TAGS[name] = {pm, vital: Object.keys(pm), start(values) { k = this.kag; start(values); }};
  };
  define("ad_gate", {}, () => { tagState = "gate"; gate = true; });
  define("ad_line", {id: ""}, pm => {
    tagState = "line"; cur.index = lineIndex.get(pm.id); cur.phase = "idle"; cur.advance = false;
    clearText(); stopVoice(); remember("auto", cur.index, true);
    next();
  });
  define("ad_transition", {cue: ""}, pm => {
    if (!transitions) { next(); return; }
    const token = gen, scene = scenes.get(pm.cue);
    // 暗転やクロスフェードの間はウィンドウも引く。カットでは残す。
    if (scene && scene.visual !== "cut" && scene.duration_ms > 0 && !k.stat.is_skip) ui.classList.add("away");
    busy = true;
    transitions.run(pm.cue, {skip: !!k.stat.is_skip}).catch(() => {
      toast("場面転換を完了できませんでした。本文の再生を続けます。");
    }).then(() => { if (token !== gen) return; busy = false; next(); });
  });
  define("ad_music", {cue: ""}, pm => {
    const token = gen;
    Promise.resolve(music?.cue(cues.get(pm.cue))).catch(() => {
      toast("BGMを再生できませんでした。本文の再生を続けます。");
    }).then(() => { if (token === gen) next(); });
  });
  define("ad_pause", {time: "0"}, pm => {
    if (k.stat.is_skip) { next(); return; }
    const token = gen;
    const finish = () => { clearTimeout(timer); pauseFinish = null; if (token === gen) next(); };
    const timer = setTimeout(finish, Number(pm.time) || 0);
    pauseFinish = finish;
  });
  define("ad_say", {id: ""}, pm => {
    const index = lineIndex.get(pm.id), line = lines[index], token = gen;
    tagState = "say"; cur.index = index; cur.advance = false; pendingComplete = false;
    ui.classList.remove("away");
    $("adn-name").textContent = characters.get(line.speaker_id)?.name || "";
    cur.pages = paginate(line.display_text);
    if (k.stat.is_skip) {
      showPage(cur.pages.length - 1, {complete: true});
      setTimeout(() => {
        if (token !== gen || tagState !== "say") return;
        if (k.stat.is_skip) completeSay(true); else toWait();
      }, 40);
      return;
    }
    playVoice(line); warmNextVoice(index);
    showPage(0);
  });
  define("ad_wait", {}, () => {
    tagState = "wait";
    if (cur.advance || k.stat.is_skip) { proceedWait(); return; }
    cur.phase = "wait"; showMarker();
    if (auto) scheduleAuto(finalDelay());
  });
  define("ad_end", {}, () => {
    tagState = "end"; ended = true;
    setAuto(false); setSkip(false); clearText(); stopVoice();
    remember("auto", lines.length - 1, true);
    $("adn-end").hidden = false; $("adn-next").hidden = true; nextBuild = null; pollAt = 0;
    $("adn-end-note").textContent = context.mode === "live" ? "次の章の公開状況を確認しています。" :
      "この書き出しに含まれる章はここまでです。";
    next();
  });

  /* ── セーブ / ロード / 行へ戻る ── */
  const saveKey = slot => `adn_${slot}_v1:${context?.build_id || script?.id}`;
  function recall(slot) {
    try { return lineIndex.get(JSON.parse(window.localStorage.getItem(saveKey(slot)) || "null")?.id); }
    catch (_) { return undefined; }
  }
  // 「つづきから」は最も進んだ位置を指す。戻って読み返しても巻き戻さない。
  function remember(slot, index, furthest = false) {
    if (index === undefined || index < 0 || (furthest && (recall(slot) ?? -1) >= index)) return false;
    try { window.localStorage.setItem(saveKey(slot), JSON.stringify({id: lines[index].id})); return true; }
    catch (_) { return false; }
  }
  const atRest = () => started && !seeking && !busy && ["say", "wait", "end"].includes(tagState);
  // Read-only view for the recorder and browser checks; nothing here can drive playback.
  window.AutoDramaPlayer = Object.freeze({state: () => ({
    started, ended, seeking, busy, auto, skipping, tag: tagState, phase: cur.phase,
    line: lines[cur.index]?.id ?? null, index: cur.index, total: lines.length,
    page: cur.page, pages: cur.pages.length, voice: voiceOn, log: logOpen,
  })});
  async function jumpTo(index) {
    if (seeking || !lines[index]) return;
    seeking = true; gen++;
    setAuto(false); setSkip(false); stopVoice(); clearText();
    pendingComplete = false; pauseFinish = null; busy = false; ended = false; nextBuild = null;
    entering = false; $("adn-next").disabled = false;
    transitions?.cancel();
    await fade($("adn-cover"), 1, 160);
    $("adn-end").hidden = true; ui.classList.remove("away");
    const state = states.entries[lines[index].id];
    try { await stageAdapter.prepare(state); } catch (_) {}
    stageAdapter.apply(state); stageAdapter.remember(state);
    if (music) {
      const cue = state.music_cue_id ? cues.get(state.music_cue_id) : null;
      // 同じ曲の区間内へ戻るときは鳴らしたままにする。
      if (!cue) music.stop();
      else if (music.snapshot()?.cue_id !== cue.id) await Promise.resolve(music.cue(cue)).catch(() => {});
    }
    tagState = "idle"; seeking = false;
    k.ftag.nextOrderWithLabel("utterance_" + lines[index].id);
    fade($("adn-cover"), 0, 220);
  }

  /* ── バックログ ── */
  const overlayOpen = () => logOpen || $("adn-config").classList.contains("on");
  function stopLogVoice() {
    const playing = logVoice; logVoice = null; playing?.stop();
    $("adn-log-list").querySelectorAll("button[data-on]").forEach(b => { delete b.dataset.on; b.textContent = "音声を聴く"; });
  }
  function openLog() {
    if (overlayOpen() || !atRest() || cur.index < 0) return;
    setAuto(false);
    if (tagState === "say") { // 閉じたら待機状態から再開する
      stopVoice(); clearTimeout(typeTimer);
      cur.typed = cur.pages[cur.page].length; paint(); cur.phase = "wait"; showMarker();
      if (lastPage()) pendingComplete = true;
    }
    const list = $("adn-log-list"); list.textContent = "";
    lines.slice(0, cur.index + 1).forEach((line, index) => {
      const entry = document.createElement("article");
      entry.className = "entry" + (index === cur.index && !ended ? " now" : "");
      const who = document.createElement("div"); who.className = "who";
      who.textContent = characters.get(line.speaker_id)?.name || "";
      const text = document.createElement("p"); text.textContent = line.display_text;
      const acts = document.createElement("div"); acts.className = "acts";
      const url = soundUrl(line.audio_asset_id);
      if (url) {
        const play = document.createElement("button"); play.textContent = "音声を聴く";
        play.addEventListener("click", () => {
          const playing = play.dataset.on; stopLogVoice();
          if (playing) return;
          play.dataset.on = "1"; play.textContent = "止める";
          const replay = sound(url, () => { if (logVoice === replay) stopLogVoice(); });
          logVoice = replay;
        });
        acts.append(play);
      }
      if (index !== cur.index || ended) {
        const back = document.createElement("button"); back.textContent = "ここへ戻る";
        back.addEventListener("click", () => { closeLog({resume: false}); jumpTo(index); });
        acts.append(back);
      }
      entry.append(who, text, acts); list.append(entry);
    });
    logOpen = true; logAtEnd = true; logOpenedAt = performance.now();
    $("adn-log").classList.add("on");
    list.scrollTop = list.scrollHeight;
  }
  function closeLog({resume = true} = {}) {
    if (!logOpen) return;
    stopLogVoice(); logOpen = false; $("adn-log").classList.remove("on");
    const complete = pendingComplete; pendingComplete = false;
    if (resume && complete) completeSay(false);
  }
  function openConfig() { if (!overlayOpen() && started) { setAuto(false); $("adn-config").classList.add("on"); } }
  function closeConfig() {
    const panel = $("adn-config");
    if (panel.contains(document.activeElement)) document.activeElement.blur(); // e.g. a slider just dragged
    panel.classList.remove("on");
  }

  /* ── 章の終わりと次の章 ── */
  function showNext(result) {
    nextBuild = null; $("adn-next").hidden = true;
    if (result.status === "ready") {
      nextBuild = result.next_build.id; pollAt = Infinity;
      $("adn-end-note").textContent = "次の章を鑑賞できます。"; $("adn-next").hidden = false;
    } else if (result.status === "complete") {
      pollAt = Infinity; $("adn-end-note").textContent = "物語はここまでです。最後までご鑑賞いただきありがとうございました。";
    } else {
      pollAt = Date.now() + 5000; $("adn-end-note").textContent = "次の章を制作中です。この位置を保ったままお待ちください。";
    }
  }
  async function readNext() {
    const response = await fetch(context.next_url, {cache: "no-store"});
    if (!response.ok) throw Error("next chapter unavailable");
    const result = await response.json();
    if (result.build_id !== context.build_id || !["ready", "waiting", "complete"].includes(result.status) ||
        (result.status === "ready" && (!/^[a-zA-Z0-9_-]+$/.test(result.next_build?.id || "") ||
          result.next_build.id === context.build_id ||
          result.next_build.chapter_number !== context.chapter_number + 1))) throw Error("invalid chapter lineage");
    return result;
  }
  async function checkNext() {
    if (!ended || context.mode !== "live" || polling || entering) return;
    polling = true; const token = gen;
    try {
      const result = await readNext();
      if (token === gen && ended) showNext(result);
    } catch (_) {
      if (token === gen && ended) {
        nextBuild = null; $("adn-next").hidden = true; pollAt = Date.now() + 5000;
        $("adn-end-note").textContent = "次の章の公開状況を確認できません。この位置を保って再確認します。";
      }
    } finally { polling = false; }
  }
  async function enterNext() {
    if (!nextBuild || !ended || entering) return;
    entering = true; $("adn-next").disabled = true;
    const token = gen, snapshot = music?.snapshot() ?? null;
    let leaving = false;
    try {
      // Confirm the successor again while the picture and the music fade together.
      const [result] = await Promise.all([readNext(), fade($("adn-cover"), 1, 400),
        Promise.resolve(music?.stopFaded({fadeOutMs: 400})).catch(() => {})]);
      if (token !== gen || !ended) return;
      showNext(result);
      if (result.status !== "ready") throw Error("next chapter not ready");
      try {
        window.sessionStorage.setItem(ENTRY_KEY, JSON.stringify({schema_version: 1, project_id: context.project_id,
          storyline_id: context.storyline_id, to_build: nextBuild, chapter_number: context.chapter_number + 1,
          created_at: Date.now()}));
      } catch (_) {}
      leaving = true; // Keep the button locked until the next page replaces this one.
      window.location.assign("/player/" + encodeURIComponent(nextBuild) + "/");
    } catch (_) {
      if (token !== gen || !ended) return;
      // Stay on this chapter's last picture with its music, and allow another try.
      try { window.sessionStorage.removeItem(ENTRY_KEY); } catch (_) {}
      if (nextBuild) {
        pollAt = Date.now() + 5000;
        $("adn-end-note").textContent = "次の章へ進めませんでした。この位置で待機し、もう一度確認できます。";
      }
      if (music) await music.restore(snapshot).then(restored => restored && music.resume()).catch(() => {});
      await fade($("adn-cover"), 0, 200);
    } finally {
      if (token === gen && !leaving) { entering = false; $("adn-next").disabled = false; }
    }
  }

  /* ── 開始 ── */
  function begin(resumeIndex) {
    if (started || !bound || !gate) return;
    started = true;
    music?.unlock().catch(() => toast("BGMの再生を準備できませんでした。"));
    window.Howler?.ctx?.resume?.().catch(() => {});
    if (resumeIndex === undefined) { $("adn-start").hidden = true; tagState = "idle"; next(); return; }
    $("adn-cover").style.opacity = "1"; $("adn-start").hidden = true;
    jumpTo(resumeIndex);
  }
  // 前の章から「次の章へ」で来たときは、音を出せる状態なら開始画面を挟まない。
  async function tryChapterEntry() {
    let intent;
    try {
      intent = JSON.parse(window.sessionStorage.getItem(ENTRY_KEY) || "null");
      if (intent?.to_build !== context.build_id) return;
      window.sessionStorage.removeItem(ENTRY_KEY);
    } catch (_) { return; }
    const age = Date.now() - intent.created_at;
    if (intent.schema_version !== 1 || intent.project_id !== context.project_id ||
        intent.storyline_id !== context.storyline_id || intent.chapter_number !== context.chapter_number ||
        !(age >= 0 && age <= 30000)) return;
    // Both the BGM context and Howler's voice context must already be allowed to sound.
    const speech = lines.some(line => line.audio_asset_id), howler = window.Howler;
    howler?.volume(); // Howler creates its AudioContext lazily through this public accessor.
    const voices = !speech ? true : howler?.ctx ?
      Promise.resolve(howler.ctx.resume()).then(() => howler.ctx.state === "running") : false;
    const unlocked = Promise.all([music?.unlock() ?? true, voices]).then(values => values.every(Boolean));
    const allowed = await Promise.race([unlocked.catch(() => false),
      new Promise(resolve => setTimeout(() => resolve(false), 300))]);
    if (allowed && !started) begin();
  }

  /* ── 入力 ── */
  $("adn-m-auto").addEventListener("click", () => setAuto(true));
  $("adn-mode").addEventListener("click", () => setAuto(false));
  $("adn-m-save").addEventListener("click", () => {
    if (!["say", "after", "wait"].includes(tagState) || seeking) return;
    toast(remember("save", cur.index) ? "セーブしました" : "セーブできませんでした。ブラウザーの保存設定を確認してください");
  });
  $("adn-m-load").addEventListener("click", () => {
    const index = recall("save");
    if (index === undefined) { toast("セーブした場面がありません"); return; }
    if (atRest()) jumpTo(index).then(() => toast("ロードしました"));
  });
  $("adn-m-log").addEventListener("click", openLog);
  $("adn-m-config").addEventListener("click", openConfig);
  // Without this the same click reaches the stage handler, which shows the UI again at once.
  $("adn-m-hide").addEventListener("click", event => {
    event.stopPropagation(); if (event.detail) event.currentTarget.blur();
    setAuto(false); ui.classList.add("bare");
  });
  $("adn-log-close").addEventListener("click", () => closeLog());
  $("adn-config-close").addEventListener("click", closeConfig);
  $("adn-next").addEventListener("click", enterNext);
  $("adn-begin").addEventListener("click", () => begin());
  $("adn-resume").addEventListener("click", () => begin(recall("auto")));

  function stageClick(target) {
    if (!started) return;
    if (ui.classList.contains("bare")) { ui.classList.remove("bare"); return; } // 戻すだけで送らない
    if (target?.closest?.("#adn-menu, #adn-mode, #adn-log, #adn-config, #adn-end .choices")) return;
    if ($("adn-config").classList.contains("on")) { closeConfig(); return; }
    if (logOpen) return;
    if (auto) { setAuto(false); return; }
    advance();
  }
  shield.addEventListener("click", event => {
    // A button pressed with the pointer must not keep focus, or Enter and Space would
    // press it again instead of advancing. Keyboard activation (detail 0) keeps focus.
    if (event.detail) event.target.closest?.("button")?.blur();
    stageClick(event.target);
  });
  shield.addEventListener("contextmenu", event => {
    event.preventDefault();
    if (logOpen) closeLog();
    else if ($("adn-config").classList.contains("on")) closeConfig();
    else if (started && !ended) { setAuto(false); ui.classList.toggle("bare"); }
  });
  shield.addEventListener("wheel", event => {
    if ($("adn-config").classList.contains("on")) return;
    if (!logOpen) { event.preventDefault(); if (event.deltaY < 0 && !ui.classList.contains("bare")) openLog(); return; }
    const list = $("adn-log-list");
    const atEnd = list.scrollTop + list.clientHeight >= list.scrollHeight - 1;
    // 最下部でさらに下へ回すと閉じる
    if (event.deltaY > 0 && atEnd && logAtEnd && performance.now() - logOpenedAt > 350) closeLog();
    logAtEnd = atEnd;
  }, {passive: false});
  // Capture phase: the engine must not see keys either.
  document.addEventListener("keydown", event => {
    event.stopPropagation();
    if (event.key === "Control") { setSkip(true); return; }
    if (event.repeat) return;
    if (event.key === "Escape") {
      if (logOpen) closeLog();
      else if ($("adn-config").classList.contains("on")) closeConfig();
      else if (ui.classList.contains("bare")) ui.classList.remove("bare");
      else setAuto(false);
    } else if ((event.key === "ArrowUp" || event.key === "PageUp") && !ui.classList.contains("bare")) {
      openLog();
    } else if ((event.key === "Enter" || event.key === " ") && !event.target.closest?.("button, input")) {
      event.preventDefault(); stageClick(null);
    }
  }, true);
  document.addEventListener("keyup", event => {
    event.stopPropagation();
    if (event.key === "Control") setSkip(false);
  }, true);
  window.addEventListener("blur", () => { if (skipping) setSkip(false); });
  window.addEventListener("resize", syncStage);
  window.addEventListener("pagehide", () => { stopVoice(); stopLogVoice(); music?.pause(); });
  window.addEventListener("pageshow", event => { if (event.persisted) window.location.reload(); });

  /* ── 読み込み ── */
  const readJson = (url, options) => fetch(url, options).then(response => {
    if (!response.ok) throw Error(url + " unavailable");
    return response.json();
  });
  const fail = message => { failed = true; $("adn-start-note").textContent = message; };
  const needsStages = value => !!((value.scene_transitions || []).length || (value.event_cg_segments || []).length);
  // script.json is the published script and never changes. The stage data is regenerated
  // by the running server, so a copy cached from an earlier screen must not be reused.
  const fresh = {cache: "no-store"};
  Promise.all([readJson("./script.json"),
    readJson("./data/others/auto_drama_states.json", fresh)]).then(async ([value, entries]) => {
    if (needsStages(value)) stages = await readJson("./data/others/auto_drama_stages.json", fresh);
    lines = value.utterances; lineIndex = new Map(lines.map((line, index) => [line.id, index]));
    characters = new Map(value.characters.map(item => [item.id, item]));
    assets = new Map(value.assets.map(item => [item.id, item]));
    cues = new Map((value.music_cues || []).map(item => [item.id, item]));
    scenes = new Map((stages?.scenes || []).map(item => [item.id, item]));
    states = entries; script = value;
    $("adn-title").textContent = value.title;
  }).catch(() => fail("作品を読み込めません。再読み込みしてください。"));
  if (livePath) readJson("./player-context.json", {cache: "no-store"}).then(value => {
    if (value.mode !== "live" || value.build_id !== livePath[1]) throw Error("invalid player identity");
    context = value;
    $("adn-chapter").textContent = `第 ${value.chapter_number} 章`;
  }).catch(() => fail("公開版を確認できません。再読み込みしてください。"));
  Promise.race([document.fonts?.ready, new Promise(resolve => setTimeout(resolve, 2500))])
    .then(() => { fontsReady = true; });

  const loadingSince = Date.now();
  let entryTried = false;
  setInterval(() => {
    syncStage();
    const engine = window.TYRANO && window.TYRANO.kag;
    if (!bound && !failed && script && states && context && engine?.ftag?.array_tag?.length && engine.layer) {
      k = engine; bound = true;
      music = window.AutoDramaMusic.create(script, {report: toast});
      music.setVolume(cfg.bgm / 100);
      stageAdapter = window.AutoDramaTransitions.kagAdapter(k);
      if (stages) {
        transitions = window.AutoDramaTransitions.create(stages, {
          music, adapter: window.AutoDramaTransitions.kagAdapter(k), report: toast});
        transitions.warm(stages.scenes[0]?.id).catch(() => {});
      }
    }
    if (!started && !failed) {
      const ready = bound && gate && fontsReady;
      $("adn-begin").disabled = !ready;
      if (ready) {
        $("adn-start-note").textContent = "音声が流れます。";
        const saved = recall("auto");
        $("adn-resume").hidden = !(saved > 0);
        if (!entryTried && context.mode === "live") { entryTried = true; tryChapterEntry(); }
      } else if (Date.now() - loadingSince >= 15000) {
        fail("鑑賞の準備を完了できません。通信状態を確認して再読み込みしてください。");
      }
    }
    if (ended && context.mode === "live" && Date.now() >= pollAt) checkNext();
  }, 200);
  syncStage();
})();
