"use strict";
// How a voice sounds when its speaker is not there: through a phone, a radio, a speaker or a
// recording, straight into the head, or remembered. Built from Web Audio nodes when the voice
// plays, so the published audio stays the speaker's own. Each treatment is trimmed to sound as
// loud as the untreated voice: narrow bands sit where the ear is most sensitive and would
// otherwise pierce.
(() => {
  // Measured on real lines with K-weighted, gated loudness (treated against untreated).
  const TRIM = {phone: 0.59, voicemail: 0.605, radio: 0.472, speaker: 0.788, telepathy: 1.073, recollection: 0.952};
  const impulses = new Map();

  function impulse(ctx, seconds) {
    const key = seconds + "@" + ctx.sampleRate;
    if (impulses.has(key)) return impulses.get(key);
    const length = Math.round(ctx.sampleRate * seconds), buffer = ctx.createBuffer(2, length, ctx.sampleRate);
    for (let side = 0; side < 2; side++) {
      const data = buffer.getChannelData(side);
      for (let i = 0; i < length; i++) data[i] = (Math.random() * 2 - 1) * Math.exp(-6.9 * i / length);
    }
    impulses.set(key, buffer);
    return buffer;
  }
  function noise(ctx, level) {
    const buffer = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate), data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    const source = ctx.createBufferSource(), output = gain(ctx, level);
    source.buffer = buffer; source.loop = true;
    source.connect(output); source.start();
    return {source, output};
  }
  function filter(ctx, type, frequency, extra = {}) {
    const node = ctx.createBiquadFilter();
    node.type = type; node.frequency.value = frequency;
    if (extra.gain !== undefined) node.gain.value = extra.gain;
    if (extra.q !== undefined) node.Q.value = extra.q;
    return node;
  }
  function shaper(ctx, curve) {
    const node = ctx.createWaveShaper(), points = new Float32Array(1024);
    for (let i = 0; i < points.length; i++) points[i] = curve(i / (points.length - 1) * 2 - 1);
    node.curve = points; node.oversample = "2x";
    return node;
  }
  function gain(ctx, value) {
    const node = ctx.createGain();
    node.gain.value = value;
    return node;
  }
  function series(nodes) {
    for (let i = 1; i < nodes.length; i++) nodes[i - 1].connect(nodes[i]);
    return {input: nodes[0], output: nodes[nodes.length - 1]};
  }
  const band = (ctx, low, high) => [0, 1, 2].map(() => filter(ctx, "highpass", low))
    .concat([0, 1, 2].map(() => filter(ctx, "lowpass", high)));
  // A gentle dip where the ear is most sensitive keeps the narrow bands from piercing.
  const soften = ctx => filter(ctx, "peaking", 2600, {gain: -4, q: 0.9});
  function compressor(ctx, threshold, ratio) {
    const node = ctx.createDynamicsCompressor();
    node.threshold.value = threshold; node.ratio.value = ratio;
    return node;
  }

  // {input, output, sources}: sources are players to stop when the voice ends.
  function untrimmed(ctx, kind) {
    const sources = [];
    if (kind === "phone" || kind === "voicemail") {
      const chain = series([shaper(ctx, x => Math.tanh(1.2 * x) / Math.tanh(1.2)), compressor(ctx, -24, 3),
                            ...band(ctx, 300, kind === "voicemail" ? 3000 : 3400), soften(ctx)]);
      if (kind === "phone") return {...chain, sources};
      // A recording: a little hiss and coarser samples.
      const hiss = noise(ctx, 0.006), shaped = series(band(ctx, 400, 3000));
      hiss.output.connect(shaped.input); shaped.output.connect(chain.output); sources.push(hiss.source);
      const crush = shaper(ctx, x => Math.round(x * 256) / 256);
      chain.output.connect(crush);
      return {input: chain.input, output: crush, sources};
    }
    if (kind === "radio") {
      const chain = series([gain(ctx, 1.5), shaper(ctx, x => Math.atan(2.5 * x) / Math.atan(2.5)),
                            compressor(ctx, -26, 4), ...band(ctx, 500, 2600), soften(ctx)]);
      const hiss = noise(ctx, 0.008), shaped = series(band(ctx, 600, 2500));
      hiss.output.connect(shaped.input); shaped.output.connect(chain.output); sources.push(hiss.source);
      return {...chain, sources, squelch: true};
    }
    if (kind === "speaker" || kind === "telepathy") {
      const input = gain(ctx, 1), output = gain(ctx, 1), room = ctx.createConvolver();
      const telepathy = kind === "telepathy";
      room.buffer = impulse(ctx, telepathy ? 2.4 : 0.35);
      const voice = telepathy ? [filter(ctx, "highpass", 120)]
        : [filter(ctx, "highpass", 180), filter(ctx, "lowpass", 6000), filter(ctx, "peaking", 1800, {gain: 4, q: 1})];
      const dry = series([...voice, gain(ctx, telepathy ? 0.8 : 1)]), wet = gain(ctx, telepathy ? 0.6 : 0.35);
      input.connect(dry.input); dry.output.connect(output);
      dry.output.connect(room); room.connect(wet); wet.connect(output);
      if (telepathy) {
        // A slow chorus: the voice wavers slightly, as if heard from inside.
        const delay = ctx.createDelay(0.1), lfo = ctx.createOscillator(), depth = gain(ctx, 0.003), mix = gain(ctx, 0.4);
        delay.delayTime.value = 0.02; lfo.frequency.value = 0.3;
        lfo.connect(depth); depth.connect(delay.delayTime); lfo.start(); sources.push(lfo);
        dry.output.connect(delay); delay.connect(mix); mix.connect(output);
      }
      return {input, output, sources};
    }
    if (kind === "recollection") {
      // A remembered voice: echoes that fade, as words come back in memory.
      const input = gain(ctx, 1), output = gain(ctx, 1);
      input.connect(output);
      for (const [seconds, level] of [[0.18, 0.35], [0.36, 0.18]]) {
        const delay = ctx.createDelay(1), tap = gain(ctx, level);
        delay.delayTime.value = seconds;
        input.connect(delay); delay.connect(tap); tap.connect(output);
      }
      return {input, output, sources};
    }
    const pass = gain(ctx, 1);
    return {input: pass, output: pass, sources};
  }
  function treatment(ctx, kind, trimmed = true) {
    const chain = untrimmed(ctx, kind), level = gain(ctx, trimmed ? TRIM[kind] ?? 1 : 1);
    chain.output.connect(level);
    return {...chain, output: level};
  }
  // A radio's key click: a short burst of band-limited noise.
  function squelch(ctx, destination, length) {
    const burst = noise(ctx, 0.04), shaped = series(band(ctx, 500, 2600));
    burst.output.connect(shaped.input); shaped.output.connect(destination);
    burst.source.stop(ctx.currentTime + length);
  }

  // Send a Howler sound through a treatment once its Web Audio node exists.
  function routeSounds(howl, kind) {
    const Howler = window.Howler, ctx = Howler && Howler.ctx;
    if (!ctx || !Howler.usingWebAudio) return;
    for (const sound of howl._sounds || []) {
      if (!sound._node || sound._adnTreated) continue;
      sound._adnTreated = true;
      const chain = treatment(ctx, kind);
      sound._node.disconnect();
      sound._node.connect(chain.input);
      chain.output.connect(Howler.masterGain);
      if (chain.squelch) squelch(ctx, Howler.masterGain, 0.09);
      const finish = () => {
        for (const source of chain.sources) { try { source.stop(); } catch (_) { /* already stopped */ } }
        if (chain.squelch) squelch(ctx, Howler.masterGain, 0.14);
      };
      howl.once("end", finish); howl.once("stop", finish);
    }
  }
  function route(howl, kind) {
    if (!kind || kind === "message") return;
    routeSounds(howl, kind);
    howl.once("play", () => routeSounds(howl, kind));
  }

  window.AutoDramaVoiceEffects = Object.freeze({route, treatment});
})();
