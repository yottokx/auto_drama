"use strict";
// Compiler-owned stage targets only. No story text is interpreted as code/HTML.
(() => {
  const root = typeof window === "undefined" ? globalThis : window;
  const aborted = signal => { if (signal?.aborted) throw Error("transition cancelled"); };
  function create(data, {music = null, adapter, report = () => {}, onBusyChange = () => {}} = {}) {
    const scenes = new Map((data?.scenes || []).map(scene => [scene.id, scene]));
    const sequence = [...scenes.keys()], prepared = new Map();
    let operation = null, generation = 0, disposed = false, currentId = null;
    const busy = () => !!operation;
    function prepareImages(scene) {
      if (!prepared.has(scene.id)) {
        const controller = new AbortController();
        const promise = Promise.resolve().then(() => adapter.prepare(scene, controller.signal));
        prepared.set(scene.id, {controller, promise});
        promise.catch(() => {});
        while (prepared.size > 2) {
          const victim = [...prepared.keys()].find(id => id !== currentId && id !== scene.id);
          if (!victim) break;
          prepared.get(victim).controller.abort(); prepared.delete(victim);
        }
      }
      return prepared.get(scene.id).promise;
    }
    function warm(id) {
      const scene = scenes.get(id);
      if (!scene || disposed) return Promise.resolve(false);
      return Promise.all([
        prepareImages(scene).catch(() => false),
        scene.music_cue_id && music?.prepareCue ? music.prepareCue(scene.music_cue_id).catch(() => false) : true,
      ]).then(values => values.every(Boolean));
    }
    function warmNext(id) {
      const next = sequence[sequence.indexOf(id) + 1];
      if (next) warm(next).catch(() => {});
    }
    function cancel() {
      generation++;
      const previous = operation; operation = null;
      previous?.controller.abort();
      adapter.reset();
      for (const value of prepared.values()) value.controller.abort();
      prepared.clear();
      if (previous) onBusyChange(false);
    }
    function skip() {
      if (!operation) return false;
      operation.fast = true;
      adapter.finish();
      music?.finishTransition?.();
      return true;
    }
    async function run(id, {skip: fast = false, resume = false} = {}) {
      const scene = scenes.get(id);
      if (!scene || disposed) return false;
      if (operation) cancel();
      const state = {controller: new AbortController(), token: ++generation, fast};
      operation = state; onBusyChange(true);
      const signal = state.controller.signal;
      const valid = () => !disposed && operation === state && state.token === generation && !signal.aborted;
      let reveal = Promise.resolve(), committed = false;
      try {
        // Both preparations are pure: the old screen and source remain active.
        const [imagesReady, audioReady] = await Promise.all([
          prepareImages(scene).then(value => value !== false).catch(() => false),
          scene.music_cue_id && music?.prepareCue ?
            music.prepareCue(scene.music_cue_id, {signal}).catch(() => false) : true,
        ]);
        if (!valid()) return false;
        if (!imagesReady) report("場面の画像を準備できませんでした。本文の再生を続けます。");
        const changed = imagesReady && !adapter.same(scene);
        const duration = () => state.fast ? 0 : scene.duration_ms;
        const covered = changed ? adapter.cover(scene.visual, duration(), signal) : Promise.resolve();
        covered.catch(() => {});
        const commit = async ({signal: audioSignal} = {}) => {
          await covered; aborted(signal);
          aborted(audioSignal);
          if (!valid()) throw Error("transition cancelled");
          if (changed) adapter.apply(scene);
          currentId = scene.id;
          committed = true;
          // Start the visual reveal and the new source's fade-in together.
          reveal = changed ? adapter.reveal(duration(), signal) : Promise.resolve();
          reveal.catch(() => {});
        };
        const cue = scene.music_cue_id && music?.getCue?.(scene.music_cue_id);
        const snapshot = resume ? music?.snapshot?.() : null;
        const alreadyCommitted = resume && cue && (cue.action === "stop" ? !snapshot :
          cue.action === "play" && snapshot?.cue_id === cue.id);
        const silence = async () => {
          if (music?.stopFaded) {
            await music.stopFaded({fadeOutMs: state.fast ? 0 : scene.music_fade_out_ms, signal})
              .catch(() => { if (valid()) music.stop(); });
          } else if (valid()) music?.stop?.();
        };
        if (audioReady && scene.music_cue_id && music?.transitionCue &&
            cue?.action !== "continue" && !alreadyCommitted) {
          const success = await music.transitionCue(scene.music_cue_id, {
            fadeOutMs: state.fast ? 0 : scene.music_fade_out_ms,
            fadeInMs: state.fast ? 0 : scene.music_fade_in_ms,
            signal, beforeCommit: commit,
          });
          if (!valid()) return false;
          // Missing/unsupported audio must not strand the scene behind a mask.
          if (!success) {
            report("BGMを切り替えられませんでした。音楽を止めて本文の再生を続けます。");
            await silence();
            if (!valid()) return false;
            if (!committed) await commit();
          }
        } else {
          if (!audioReady) {
            report("BGMを準備できませんでした。音楽を止めて本文の再生を続けます。");
            await silence();
            if (!valid()) return false;
          }
          await commit();
        }
        await reveal;
        if (!valid()) return false;
        adapter.reset();
        warmNext(id);
        return true;
      } catch (_) {
        if (!valid()) return false;
        report("場面転換を完了できませんでした。本文の再生を続けます。");
        return true;
      } finally {
        if (operation === state) {
          adapter.reset(); operation = null; onBusyChange(false);
        }
      }
    }
    return {run, warm, warmNext, cancel, skip, busy,
      dispose() { disposed = true; cancel(); }};
  }

  function kagAdapter(k, {timeoutMs = 10000} = {}) {
    let overlay = null;
    const animations = new Set();
    const url = (kind, storage) => "./data/" + kind + "/" + encodeURIComponent(storage);
    function image(src, signal) {
      return new Promise((resolve, reject) => {
        const value = new Image();
        let timer;
        const finish = error => {
          clearTimeout(timer); value.onload = value.onerror = null;
          signal?.removeEventListener("abort", cancel);
          if (error) { value.src = ""; reject(error); } else resolve(value);
        };
        const cancel = () => finish(Error("image preparation cancelled"));
        value.onload = () => finish(); value.onerror = () => finish(Error("image unavailable"));
        signal?.addEventListener("abort", cancel, {once: true});
        if (signal?.aborted) { cancel(); return; }
        timer = setTimeout(() => finish(Error("image preparation timed out")), timeoutMs);
        value.src = src;
      });
    }
    function animate(element, from, to, duration, signal) {
      aborted(signal);
      element.style.opacity = String(from);
      if (!duration || !element.animate) { element.style.opacity = String(to); return Promise.resolve(); }
      const animation = element.animate([{opacity: from}, {opacity: to}],
        {duration, easing: "linear", fill: "forwards"});
      animations.add(animation);
      const cancel = () => animation.cancel();
      signal.addEventListener("abort", cancel, {once: true});
      return animation.finished.then(() => { aborted(signal); element.style.opacity = String(to); })
        .finally(() => { signal.removeEventListener("abort", cancel); animations.delete(animation); });
    }
    function reset() {
      for (const animation of animations) animation.cancel();
      animations.clear(); overlay?.remove(); overlay = null;
    }
    function same(scene) {
      const base = k.layer.getLayer("base", "fore").get(0);
      const source = base?.style.backgroundImage?.match(/url\(["']?(.*?)["']?\)/)?.[1] || "";
      if (scene.background && source.split("/").pop() !== scene.background.storage) return false;
      const visible = [...k.layer.getLayer("0", "fore").get(0).querySelectorAll(".tyrano_chara")];
      if (visible.length !== scene.characters.length) return false;
      return scene.characters.every(value => {
        const element = visible.find(item => item.classList.contains("ad_" + value.id));
        return element && ["left", "top", "width", "height"].every(key =>
          parseFloat(element.style[key]) === value[key]);
      });
    }
    function apply(scene) {
      const base = k.layer.getLayer("base", "fore");
      if (scene.background) {
        base.css({"background-image": "url(" + url("bgimage", scene.background.storage) + ")",
          "background-size": scene.background.position ? "cover" : "100% 100%",
          "background-position": scene.background.position || "center", display: "block", opacity: 1});
      }
      const layer = k.layer.getLayer("0", "fore"), desired = new Set(scene.characters.map(value => "ad_" + value.id));
      layer.find(".tyrano_chara").each(function () {
        const name = [...this.classList].find(value => desired.has(value));
        if (!name) this.remove();
      });
      for (const [name, value] of Object.entries(k.stat.charas || {})) {
        value.is_show = desired.has(name) ? "true" : "false";
      }
      for (const value of scene.characters) {
        const name = "ad_" + value.id;
        let element = layer.get(0).querySelector("." + name);
        if (!element) {
          element = document.createElement("div");
          element.classList.add("tyrano_chara", name);
          const img = document.createElement("img");
          img.classList.add("chara_img"); img.src = url("fgimage", value.storage);
          element.append(img); layer.get(0).append(element);
        }
        Object.assign(element.style, {position: "absolute", display: "block", opacity: "1",
          left: value.left + "px", top: value.top + "px", width: value.width + "px",
          height: value.height + "px", zIndex: String(value.z_index)});
        const img = element.querySelector(".chara_img");
        Object.assign(img.style, {width: value.width + "px", height: value.height + "px"});
        const character = k.stat.charas?.[name];
        if (character) Object.assign(character, {is_show: "true", layer: "0", width: value.width, height: value.height});
      }
      if (scene.characters.length) layer.show();
    }
    function cover(visual, duration, signal) {
      reset();
      if (!["fade", "dissolve"].includes(visual) || !duration) return Promise.resolve();
      const game = document.getElementById("root_layer_game");
      overlay = document.createElement("div");
      overlay.className = "ad-stage-transition";
      if (visual === "fade") overlay.style.background = "#000";
      else {
        const copy = game.cloneNode(true);
        copy.removeAttribute("id");
        copy.querySelectorAll("[id]").forEach(element => element.removeAttribute("id"));
        overlay.append(copy);
      }
      document.getElementById("tyrano_base").append(overlay);
      return visual === "fade" ? animate(overlay, 0, 1, duration, signal) : Promise.resolve();
    }
    return {same, apply, cover, reset,
      prepare(scene, signal) {
        return Promise.all([
          ...(scene.background ? [image(url("bgimage", scene.background.storage), signal)] : []),
          ...scene.characters.map(value => image(url("fgimage", value.storage), signal)),
        ]).then(() => true);
      },
      reveal(duration, signal) { return overlay ? animate(overlay, 1, 0, duration, signal) : Promise.resolve(); },
      finish() { for (const animation of animations) animation.finish(); },
    };
  }
  root.AutoDramaTransitions = {create, kagAdapter};
})();
