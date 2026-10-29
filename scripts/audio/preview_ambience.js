"use strict";
// Ambience preview layer for an unpacked chapter (scripts/audio/preview_ambience_tyrano.py).
// It only watches the player's current line through the read-only AutoDramaPlayer.state()
// and plays the span's loop beside the BGM; the published player is not changed.
(() => {
  const FADE = 1.5, POLL = 250;
  const MUFFLE_HZ = 500, MUFFLE_GAIN = 0.7;
  const MODES = {off: "オフ", as_is: "そのまま", position: "位置で処理"};
  const POSITIONS = {same_space: "同じ空間", beyond_wall: "壁の向こう", inside_device: "機械の中"};
  let mode = "position", level = 1, data = null, order = null, context = null, master = null;
  let playing = null, pending = "", buffers = new Map(), wanted = null;
  try { mode = window.localStorage.getItem("ambience_preview_mode") || mode; } catch (_) {}
  try { level = Number(window.localStorage.getItem("ambience_preview_level") ?? 1); } catch (_) {}

  const userBgm = () => {
    try { return (JSON.parse(window.localStorage.getItem("adn_settings_v1") || "{}").bgm ?? 100) / 100; }
    catch (_) { return 1; }
  };
  const ensure = () => {
    if (!context) {
      context = new (window.AudioContext || window.webkitAudioContext)();
      master = context.createGain(); master.connect(context.destination);
    }
    if (context.state !== "running") context.resume().catch(() => {});
    master.gain.setTargetAtTime(level * userBgm(), context.currentTime, 0.05);
    return context;
  };
  const load = async id => {
    if (!buffers.has(id)) {
      buffers.set(id, fetch("./ambience/" + data.assets[id].file).then(r => r.arrayBuffer())
        .then(bytes => ensure().decodeAudioData(bytes)));
    }
    return buffers.get(id);
  };
  // How the active span sounds in the current mode: null for silence.
  const treatment = span => {
    if (!span || mode === "off") return null;
    if (mode === "as_is") return {span, muffled: false, gain: span.gain};
    if (span.position === "same_space") return {span, muffled: false, gain: span.gain};
    if (span.position === "beyond_wall") return {span, muffled: true, gain: span.gain * MUFFLE_GAIN};
    return null;
  };
  const key = value => value ? `${value.span.id}:${value.muffled}` : "";
  const fadeOut = voice => {
    if (!voice) return;
    const now = context.currentTime;
    voice.fade.gain.cancelScheduledValues(now);
    voice.fade.gain.setValueAtTime(voice.fade.gain.value, now);
    voice.fade.gain.linearRampToValueAtTime(0, now + FADE);
    try { voice.source.stop(now + FADE + 0.05); } catch (_) {}
  };
  const start = async next => {
    pending = key(next);
    let buffer;
    try { buffer = await load(next.span.asset); } finally { if (pending === key(next)) pending = ""; }
    if (key(wanted) !== key(next)) return;
    const ctx = ensure(), source = ctx.createBufferSource(), filter = ctx.createBiquadFilter();
    const fade = ctx.createGain(), gain = ctx.createGain();
    source.buffer = buffer; source.loop = true;
    filter.type = "lowpass"; filter.frequency.value = next.muffled ? MUFFLE_HZ : 20000; filter.Q.value = 0.7;
    gain.gain.value = next.gain;
    source.connect(filter); filter.connect(gain); gain.connect(fade); fade.connect(master);
    fade.gain.setValueAtTime(0, ctx.currentTime);
    fade.gain.linearRampToValueAtTime(1, ctx.currentTime + FADE);
    // Start somewhere inside the loop, as a scene that was already going on.
    source.start(0, Math.random() * buffer.duration);
    fadeOut(playing);
    playing = {key: key(next), source, fade};
  };
  const update = () => {
    if (!data || !order) return;
    const state = window.AutoDramaPlayer?.state?.();
    const index = state?.started && !state.ended ? order.get(state.line) : undefined;
    const span = index === undefined ? null :
      data.spans.find(value => index >= order.get(value.from) &&
        (value.until === null || index < order.get(value.until)));
    const next = treatment(span);
    wanted = next;
    show(span, next, state);
    if (context) master.gain.setTargetAtTime(level * userBgm(), context.currentTime, 0.05);
    if (key(next) === (playing?.key ?? "") || (next && key(next) === pending)) return;
    if (!next) { if (context) fadeOut(playing); playing = null; return; }
    start(next).catch(() => { label.textContent = "環境音を読み込めませんでした。"; });
  };

  const panel = document.createElement("div");
  panel.id = "ambience-preview";
  panel.innerHTML = `<style>
    #ambience-preview{position:fixed;top:8px;right:8px;z-index:2147483000;background:rgba(20,20,24,.86);
      color:#eee;font:12px/1.5 sans-serif;padding:8px 10px;border-radius:6px;max-width:300px}
    #ambience-preview button{font:inherit;margin:0 4px 4px 0;padding:2px 8px;border:1px solid #888;
      background:#333;color:#eee;border-radius:4px;cursor:pointer}
    #ambience-preview button[aria-pressed=true]{background:#2d6cdf;border-color:#2d6cdf}
    #ambience-preview input{width:120px;vertical-align:middle}
  </style><div><b>環境音</b></div><div class="modes"></div>
  <div>音量 <input type="range" min="0" max="200" step="5"> <output></output></div><div class="label"></div>`;
  const label = panel.querySelector(".label"), range = panel.querySelector("input"), out = panel.querySelector("output");
  for (const [value, text] of Object.entries(MODES)) {
    const button = document.createElement("button");
    button.textContent = text; button.dataset.mode = value;
    button.addEventListener("click", () => {
      mode = value; try { window.localStorage.setItem("ambience_preview_mode", mode); } catch (_) {}
      button.blur(); ensure(); update();
    });
    panel.querySelector(".modes").append(button);
  }
  range.value = String(Math.round(level * 100)); out.textContent = range.value + "%";
  range.addEventListener("input", () => {
    level = Number(range.value) / 100; out.textContent = range.value + "%";
    try { window.localStorage.setItem("ambience_preview_level", String(level)); } catch (_) {}
    update();
  });
  for (const type of ["click", "pointerdown", "mousedown", "wheel", "contextmenu"]) {
    panel.addEventListener(type, event => event.stopPropagation());
  }
  const show = (span, next, state) => {
    panel.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    if (!state?.started) { label.textContent = "本編を始めると鳴ります。"; return; }
    if (!span) { label.textContent = "この行に環境音はありません。"; return; }
    const how = next ? (next.muffled ? "こもらせて鳴らす" : "そのまま鳴らす") : "鳴らさない";
    label.textContent = `${span.sound}（${span.from}〜）→ ${data.assets[span.asset].label}` +
      `／判定: ${POSITIONS[span.position]}／${how}`;
  };
  // Audio may start only after a gesture; the player's own start button is one.
  window.addEventListener("pointerdown", () => { if (context || mode !== "off") ensure(); }, true);
  window.addEventListener("keydown", () => { if (context || mode !== "off") ensure(); }, true);
  window.addEventListener("DOMContentLoaded", () => document.body.append(panel));
  Promise.all([fetch("./ambience.json").then(r => r.json()), fetch("./script.json").then(r => r.json())])
    .then(([ambience, script]) => {
      data = ambience;
      order = new Map(script.utterances.map((line, index) => [line.id, index]));
      window.setInterval(update, POLL);
    })
    .catch(() => { label.textContent = "環境音の設定を読めませんでした。"; });
})();
