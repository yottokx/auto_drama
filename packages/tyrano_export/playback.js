"use strict";
// Adapt the installed engine through its existing tag handlers; never alter it.
window.AutoDramaPlayback = function (k, script) {
  let source = null, ranges = [], generation = 0;
  let reveal = null, advancing = null, log = null;
  const snapshot = () => ({stat: k.stat, tags: k.ftag.array_tag,
    scenario: k.stat.current_scenario, generation});
  const matches = state => state && state.generation === generation && state.stat === k.stat &&
    state.tags === k.ftag.array_tag && state.scenario === k.stat.current_scenario;
  const index = () => k.ftag.current_order_index;
  const tag = () => k.ftag.array_tag[index()];
  const stopModes = () => {
    if (k.stat.is_auto) k.ftag.startTag("autostop", {next: "false"});
    if (k.stat.is_skip) k.setSkip(false);
    k.stat.is_wait_auto = false;
    k.tmp.is_vo_play_wait = false;
  };
  function lineRanges() {
    if (source === k.ftag.array_tag) return ranges;
    source = k.ftag.array_tag;
    ranges = [];
    let current = null;
    source.forEach((item, i) => {
      if (item.name === "label") {
        const name = item.pm.label_name;
        current = name.startsWith("utterance_") ?
          {id: name.slice("utterance_".length), start: i, lastText: -1} : null;
      } else if (current && item.name === "text") current.lastText = i;
      else if (current && item.name === "p") {
        if (current.lastText >= 0) ranges.push({...current, end: i});
        current = null; // The chapter-end text belongs to no utterance.
      }
    });
    return ranges;
  }
  const currentLine = () => lineRanges().find(line => line.start <= index() && index() <= line.end);
  const complete = line => index() > line.lastText ||
    (index() === line.lastText && !k.stat.is_adding_text);
  function displayedLines() {
    const seen = new Set(lineRanges().filter(complete).map(line => line.id));
    return script.utterances.filter(line => seen.has(line.id));
  }
  function atChapterEnd() {
    // Native saves resume one tag before their saved stop. Both the old M3
    // compiler and the labeled M4 compiler end at the final [s].
    const end = k.ftag.array_tag.length - 1;
    return end >= 0 && k.ftag.array_tag[end]?.name === "s" &&
      index() >= end - 1 && !!k.stat.is_strong_stop;
  }
  function eventId(order = index()) {
    const end = k.ftag.array_tag.length - 1;
    if (order >= end && k.ftag.array_tag[end]?.name === "s") return "chapter_end";
    const line = lineRanges().find(item => item.start <= order && order <= item.end);
    return line ? line.id : "chapter_start";
  }

  // Block only the engine's next step while a modal is open. Text already on the
  // current page can finish, but a callback cannot enter another text/tag/page.
  const nextOrder = k.ftag.nextOrder;
  k.ftag.nextOrder = function (...args) {
    if (matches(log)) { log.pending = true; return; }
    return nextOrder.apply(this, args);
  };

  // Let the single native typewriter timer consume the reveal request. Calling
  // finishAddingChars directly on click would leave that timer running twice.
  const text = k.ftag.master_tag?.text;
  if (text?.addOneChar) {
    const addOneChar = text.addOneChar;
    text.addOneChar = function (charIndex, chars, message, inner) {
      if (matches(reveal) && currentLine()?.id === reveal.id) {
        this.makeAllCharsVisible(chars);
        if (k.tmp.popopo?.key) k.tmp.popopo.player.stop();
        this.finishAddingChars();
        return;
      }
      return addOneChar.call(this, charIndex, chars, message, inner);
    };
  }
  const page = k.ftag.master_tag?.p;
  if (page?.start) {
    const startPage = page.start;
    page.start = function (...args) {
      const skipThisWait = matches(advancing) && index() === advancing.end;
      if (matches(reveal) && currentLine()?.id === reveal.id) reveal = null;
      if (skipThisWait) advancing = null;
      const result = startPage.apply(this, args);
      if (skipThisWait) {
        k.ftag.hideNextImg();
        k.ftag.nextOrder();
      }
      return result;
    };
  }

  function advance() {
    if (log || advancing) return false;
    const line = currentLine();
    if (!line) return false;
    stopModes();
    if (!complete(line)) {
      // During scene entry/audio loading there is no displayed text to finish.
      if (!k.stat.is_adding_text) return false;
      reveal = {...snapshot(), id: line.id};
      k.stat.is_click_text = true;
      if (text && k.getMessageInnerLayer) {
        text.makeAllCharsVisible(k.getMessageInnerLayer().find(".char"));
      }
      return true;
    }
    // Effects after the text keep their own timer. Do not force nextOrder while
    // an animation or explicit pause is in flight.
    if (!["wse", "p"].includes(tag()?.name)) return false;
    reveal = null;
    const request = {...snapshot(), id: line.id, end: line.end, from: index()};
    advancing = request;
    k.tmp.is_se_play_wait = false;
    k.tmp.is_vo_play_wait = false;
    if (k.tmp.map_se?.["1"] || k.tmp.is_se_play) {
      k.ftag.startTag("stopse", {buf: "1", stop: "true"});
    }
    // Howler may already have queued the old end callback. Let it drain before
    // a new voice starts; flags from that old voice must not affect the new one.
    setTimeout(() => {
      if (advancing !== request) return;
      if (!matches(request) || index() !== request.from) { advancing = null; return; }
      if (tag()?.name === "p") advancing = null;
      k.stat.is_click_text = false;
      k.ftag.hideNextImg();
      k.ftag.nextOrder();
    }, 0);
    return true;
  }

  function canOpenLog() {
    return !advancing && !log && (!!k.key_mouse.util.canShowMenu() || tag()?.name === "wse");
  }
  function openLog() {
    if (!canOpenLog()) return false;
    stopModes();
    const voice = k.tmp.map_se?.["1"];
    log = {...snapshot(), pending: false, voice,
      wasPlaying: !!voice?.playing()};
    if (log.wasPlaying) voice.pause();
    return true;
  }
  function closeLog() {
    const saved = log;
    log = null;
    if (!matches(saved)) return;
    if (saved.wasPlaying && k.tmp.is_se_play && k.tmp.map_se?.["1"] === saved.voice) {
      saved.voice.play();
    }
    if (saved.pending) k.ftag.nextOrder();
  }
  function reset() {
    generation++;
    reveal = null;
    advancing = null;
    log = null;
  }
  return {advance, displayedLines, canOpenLog, openLog, closeLog, reset, atChapterEnd, eventId};
};
