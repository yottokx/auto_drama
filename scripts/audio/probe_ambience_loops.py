"""Natural ambience loops, one method per kind of sound, with the kind of each sound given.

Experiment 1 of the ambience library: does each method make a loop that sounds natural when it is
told what kind of sound it has? Whether an LLM can tell the kind from a description is a separate
experiment. The methods:

  texture  rain, wind, fire, static   a long region whose seams avoid salient moments, whose inside
                                      has few standout events, and whose ends sound alike; joined by
                                      an equal-power crossfade of several seconds
  waves    sea                        a region from one trough of the swell to another, joined by a
                                      crossfade centred on the trough
  ticks    clock                      an even number of steady ticks, cut in the quiet gap before a
                                      tick so the interval across the seam stays the same
  event    distant train              one pass cut out of a take and played at scattered intervals
                                      over a quiet bed, which is itself a texture loop

`generate` runs Stable Audio 3 Medium (run it with the stable_audio3 runtime's Python while the
worker is stopped); `loop` cuts the loops, renders previews and writes index.html.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SFX = "TrackType: SFX. "
HOP = 0.05
TICK_HOP = 0.002
TEXTURE_FADE = 3.0
MIN_LOOP = 30.0
MIN_TICK_LOOP = 10.0
LEVEL_WINDOW = 4.0
SEAM_PREVIEW = 8.0
SCENE_SECONDS = 180.0
SCENE_FADE = 3.0
LOOP_RMS, BED_RMS, EVENT_RMS = 0.03, 0.01, 0.04
TAKES, LOOP_SECONDS, EVENT_SECONDS = 3, 90.0, 25.0

# Prompts follow what the earlier trials taught: texture and density, no surfaces for things to hit,
# nothing named that should be absent.
ITEMS = [
    {"id": "rain_light", "label": "しとしと降る雨", "kind": "texture",
     "prompt": "Soft light rain falling evenly, the gentle hiss of countless tiny raindrops blending into one "
               "smooth continuous sound, a quiet rainy day, calm ambience."},
    {"id": "rain_medium", "label": "本降りの雨", "kind": "texture",
     "prompt": "Steady moderate rain, a dense even wash of countless raindrops blending into one continuous "
               "rushing sound, a rainy evening, calm ambience."},
    {"id": "rain_heavy", "label": "激しい雨", "kind": "texture",
     "prompt": "Heavy downpour, a thick roaring wash of dense rain falling hard and evenly, continuous rain "
               "ambience."},
    {"id": "wind_light", "label": "穏やかな夜風", "kind": "texture",
     "prompt": "A soft night breeze, gentle airy wind slowly rising and falling, quiet open air, continuous "
               "ambience."},
    {"id": "wind_strong", "label": "強い風", "kind": "texture",
     "prompt": "Strong cold wind howling across open high ground, powerful gusts rising and falling, continuous "
               "wind ambience."},
    {"id": "fire_hearth", "label": "暖炉・松明の火", "kind": "texture",
     "prompt": "A small fire burning steadily, soft flickering flames with gentle occasional crackles, warm and "
               "quiet, continuous ambience."},
    {"id": "radio_static", "label": "無線のノイズ", "kind": "texture",
     "prompt": "Radio static from a shortwave receiver, a continuous hiss of noise with faint crackles and slowly "
               "drifting interference tones."},
    {"id": "waves_calm", "label": "穏やかな波", "kind": "waves",
     "prompt": "Gentle ocean waves on a sandy beach, each wave slowly rolling in, breaking softly and drawing "
               "back, calm seaside ambience, continuous."},
    {"id": "waves_rough", "label": "荒れた海", "kind": "waves",
     "prompt": "Rough sea, large waves surging and breaking heavily, deep roaring water, continuous ambience."},
    {"id": "clock_ticking", "label": "時計の秒針", "kind": "ticks",
     "prompt": "A wall clock ticking steadily in a quiet room, clear regular ticks, close perspective, "
               "continuous."},
    {"id": "evening_air", "label": "静かな夕方の空気（電車の土台）", "kind": "texture", "bed": True,
     "prompt": "Quiet evening air outdoors in a residential neighborhood, a very faint distant hum of the town, "
               "calm and still, continuous ambience."},
    {"id": "train_pass", "label": "遠くを走る電車", "kind": "event", "over": "evening_air",
     "prompt": "A single train passing once in the distance, a faint rumble slowly rising as it approaches, the "
               "rhythmic clatter of wheels, then fading away into quiet evening air."},
]
KIND_LABELS = {"texture": "一様な質感", "waves": "周期のある音", "ticks": "規則的に刻む音", "event": "時々起きる出来事"}


# ---- signal helpers (numpy only, so the unit tests run in the main environment) ----

def mono(audio):
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def frame_db(signal, rate, hop=HOP):
    size = max(1, round(rate * hop))
    count = len(signal) // size
    frames = signal[:count * size].reshape(count, size).astype(np.float64)
    return 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)


def moving_mean(values, width):
    width = max(1, int(width))
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def moving_median(values, width):
    width = max(1, int(width)) | 1
    padded = np.pad(values, width // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)


def find_peaks(values, *, distance, height=None, prominence=None, reach=None):
    """Local maxima, strongest first within `distance`; prominence is measured within `reach` frames."""
    inner = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])) + 1
    if height is not None:
        inner = inner[values[inner] >= height]
    if prominence is not None:
        reach = reach or len(values)
        kept = []
        for index in inner:
            left = values[max(0, index - reach):index]
            right = values[index + 1:index + 1 + reach]
            base = max(left.min() if len(left) else values[index], right.min() if len(right) else values[index])
            if values[index] - base >= prominence:
                kept.append(index)
        inner = np.asarray(kept, dtype=int)
    chosen = []
    for index in sorted(inner, key=lambda i: -values[i]):
        if all(abs(index - other) >= distance for other in chosen):
            chosen.append(index)
    return np.asarray(sorted(chosen), dtype=int)


def usable_range(db, hop=HOP):
    """Seconds between the generator's fade-in and fade-out, judged by the level of the middle."""
    count = len(db)
    level = float(np.median(db[count // 5:count - count // 5]))
    steady = moving_mean(db, LEVEL_WINDOW / hop) >= level - 4
    first = int(np.argmax(steady))
    last = count - int(np.argmax(steady[::-1]))
    guard = round(1.0 / hop)
    return (first + guard) * hop, (last - guard) * hop


def salience(db, hop=HOP):
    """How far each frame stands above its surroundings, in dB."""
    return db - moving_median(db, 3.0 / hop)


def band_profile(signal, rate, start, end, bands=8):
    segment = signal[int(start * rate):int(end * rate)].astype(np.float64)
    power = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))**2
    freq = np.fft.rfftfreq(len(segment), 1 / rate)
    edges = np.geomspace(50, min(16000, rate / 2), bands + 1)
    return np.array([10 * np.log10(power[(freq >= lo) & (freq < hi)].sum() + 1e-12)
                     for lo, hi in itertools.pairwise(edges)])


def profile_distance(first, second):
    """Mean band difference after removing the overall level difference, which the score counts apart."""
    difference = first - second
    return float(np.mean(np.abs(difference - difference.mean())))


def window_max(values, start, end, hop=HOP):
    lo, hi = max(0, int(start / hop)), min(len(values), int(np.ceil(end / hop)))
    return float(values[lo:hi].max()) if hi > lo else 0.0


def window_mean(values, start, end, hop=HOP):
    lo, hi = max(0, int(start / hop)), min(len(values), int(np.ceil(end / hop)))
    return float(values[lo:hi].mean()) if hi > lo else 0.0


# ---- joining ----

def equal_power_loop(audio, start, end, fade):
    """One period; its tail blends into the audio just before `start`, so it plays on into `start`.

    Sine/cosine weights keep the power of uncorrelated sound such as noise; the music loops' weights,
    which sum to one, dip by about 3 dB in the middle of the fade for such sound."""
    if not 0 < fade <= start or end - start <= 2 * fade or end > len(audio):
        raise ValueError("loop region or crossfade does not fit the audio")
    result = audio[start:end].astype(np.float32).copy()
    t = (np.arange(fade) + 0.5) / fade
    shape = (fade,) + (1,) * (audio.ndim - 1)
    result[-fade:] = (result[-fade:] * np.cos(np.pi / 2 * t).reshape(shape)
                      + audio[start - fade:start] * np.sin(np.pi / 2 * t).reshape(shape))
    return result


def equal_gain_loop(audio, start, end, fade):
    """The music loops' raised-cosine join, kept only to measure its dip on the same region."""
    result = audio[start:end].astype(np.float32).copy()
    weights = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
    shape = (fade,) + (1,) * (audio.ndim - 1)
    result[-fade:] = result[-fade:] * (1 - weights).reshape(shape) + audio[start - fade:start] * weights.reshape(shape)
    return result


# ---- region choice per kind ----

def texture_region(audio, rate, *, fade=TEXTURE_FADE, min_loop=MIN_LOOP, step=0.5):
    signal = mono(audio)
    db = frame_db(signal, rate)
    standout = salience(db)
    lo, hi = usable_range(db)
    strong = max(9.0, float(np.percentile(standout, 99)) + 3)
    events = find_peaks(standout, distance=round(1 / HOP), height=strong) * HOP
    points = np.arange(lo + fade, hi + 1e-9, step)
    if len(points) < 2 or points[-1] - points[0] < min_loop:
        return None
    seam = np.array([window_max(standout, p - fade, p + 0.5) for p in points])
    level = np.array([window_mean(db, p - fade, p) for p in points])
    profiles = [band_profile(signal, rate, p - fade, p) for p in points]
    # Swells and bursts last longer than a standout; a seam inside one, or a region full of them, is
    # what makes a loop recognisable, so seams keep to the usual level and the region to a steady one.
    second = moving_mean(db, 1.0 / HOP)
    usual = float(np.median(second[int(lo / HOP):int(hi / HOP)]))
    ordinary = np.array([window_max(np.abs(second - usual), p - fade, p + 0.5) for p in points])
    sums, squares = np.concatenate(([0.0], np.cumsum(second))), np.concatenate(([0.0], np.cumsum(second**2)))
    best = None
    for i, start in enumerate(points):
        for j in np.flatnonzero(points >= start + min_loop):
            inside = int(np.count_nonzero((events > start) & (events < points[j] - fade)))
            a, b = int(start / HOP), int(points[j] / HOP)
            mean = (sums[b] - sums[a]) / (b - a)
            spread = float(np.sqrt(max((squares[b] - squares[a]) / (b - a) - mean**2, 0.0)))
            parts = {"seam_db": float(seam[i] + seam[j]), "level_db": abs(float(level[i] - level[j])),
                     "unusual_seam_db": 0.5 * float(ordinary[i] + ordinary[j]), "unsteady_db": spread,
                     "timbre_db": profile_distance(profiles[i], profiles[j]),
                     "shortening": 0.1 * float((hi - lo) - (points[j] - start)), "standouts": 2.0 * inside}
            score = sum(parts.values())
            if best is None or score < best["score"]:
                best = {"start": float(start), "end": float(points[j]), "fade": fade, "score": score,
                        "parts": parts, "standouts_inside": inside}
    best.update({"usable": [lo, hi], "standout_threshold_db": strong,
                 "standout_times": [round(float(t), 2) for t in events if lo <= t <= hi]})
    return best


def wave_region(audio, rate, *, min_loop=MIN_LOOP):
    signal = mono(audio)
    db = frame_db(signal, rate)
    swell = moving_mean(db, 0.5 / HOP)
    lo, hi = usable_range(db)
    troughs = find_peaks(-swell, distance=round(2 / HOP), prominence=3.0, reach=round(6 / HOP)) * HOP
    troughs = troughs[(troughs >= lo) & (troughs <= hi)]
    if len(troughs) < 3:
        return {"failed": "うねりの谷が3つ未満", "troughs": troughs.tolist(), "usable": [lo, hi]}
    cycle = float(np.median(np.diff(troughs)))
    fade = float(np.clip(0.25 * cycle, 0.5, 2.0))
    cuts = troughs + fade / 2
    best = None
    for i, start in enumerate(cuts):
        if start - fade < lo:
            continue
        for j in np.flatnonzero((cuts >= start + min_loop) & (cuts <= hi)):
            parts = {"level_db": abs(window_mean(db, start - fade, start) - window_mean(db, cuts[j] - fade, cuts[j])),
                     "timbre_db": profile_distance(band_profile(signal, rate, start - fade, start),
                                                   band_profile(signal, rate, cuts[j] - fade, cuts[j])),
                     "shortening": 0.1 * float((hi - lo) - (cuts[j] - start))}
            score = sum(parts.values())
            if best is None or score < best["score"]:
                best = {"start": float(start), "end": float(cuts[j]), "fade": fade, "score": score, "parts": parts,
                        "cycles": int(j - i)}
    if best is None:
        return {"failed": f"谷から谷まで{MIN_LOOP:.0f}秒以上取れない", "troughs": troughs.tolist(), "usable": [lo, hi]}
    best.update({"usable": [lo, hi], "cycle_seconds": cycle, "troughs": [round(float(t), 2) for t in troughs]})
    return best


def tick_strength(signal, rate, hop=TICK_HOP):
    """How far the first difference, which favours clicks, stands above its floor, in dB per frame."""
    db = frame_db(np.diff(signal), rate, hop)
    return db - moving_median(db, 1.0 / hop)


def tick_period(strength, hop=TICK_HOP, shortest=0.25, longest=2.0):
    """The shortest lag whose autocorrelation is near the best, so tick-tock pairs count as two ticks."""
    envelope = np.clip(strength, 0, None)
    envelope = envelope - envelope.mean()
    size = 1 << int(np.ceil(np.log2(2 * len(envelope))))
    spectrum = np.fft.rfft(envelope, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:len(envelope)]
    lo, hi = round(shortest / hop), round(longest / hop)
    lags = find_peaks(correlation[lo:hi], distance=1) + lo
    if not len(lags):
        return None
    best = correlation[lags].max()
    return float(lags[np.flatnonzero(correlation[lags] >= 0.8 * best)[0]] * hop)


def tick_times(signal, rate, *, hop=TICK_HOP, height=15.0):
    """Ticks followed one period at a time from the strongest one, taking the strongest click near each
    expected time, so a second click inside a tick or an echo is not taken for a tick. A tick that is
    not there leaves NaN, which breaks any steady run."""
    strength = tick_strength(signal, rate, hop)
    period = tick_period(strength, hop)
    if period is None:
        return np.array([])
    anchor = int(np.argmax(strength))
    found = {anchor: True}
    for direction in (1, -1):
        last, step = anchor, period / hop
        while True:
            expected = last + direction * step
            lo, hi = int(expected - 0.2 * step), int(expected + 0.2 * step) + 1
            if lo < 0 or hi > len(strength):
                break
            index = lo + int(np.argmax(strength[lo:hi]))
            if strength[index] >= height:
                if found.get(last):
                    step = 0.9 * step + 0.1 * abs(index - last)
                found[index] = True
                last = index
            else:
                found[round(expected)] = False
                last = round(expected)
    return np.array([index * hop if real else np.nan for index, real in sorted(found.items())])


def tick_region(audio, rate, *, min_loop=MIN_TICK_LOOP, tolerance=0.08):
    signal = mono(audio)
    lo, hi = usable_range(frame_db(signal, rate))
    ticks = tick_times(signal, rate)
    ticks = ticks[np.isnan(ticks) | ((ticks >= lo) & (ticks <= hi))]
    if np.count_nonzero(~np.isnan(ticks)) < 4:
        return {"failed": "刻みが検出できない", "ticks": int(np.count_nonzero(~np.isnan(ticks))), "usable": [lo, hi]}
    intervals = np.diff(ticks)
    period = float(np.nanmedian(intervals))
    with np.errstate(invalid="ignore"):
        steady = np.abs(intervals / period - 1) <= tolerance
    run_start, run_length, current = 0, 0, 0
    for index, ok in enumerate(steady):
        current = current + 1 if ok else 0
        if current > run_length:
            run_start, run_length = index - current + 1, current
    half = len(intervals) // 2
    drift = float(np.nanmedian(intervals[half:]) / np.nanmedian(intervals[:half]) - 1) if half else 0.0
    count = run_length - run_length % 2
    base = {"period": period, "ticks": int(np.count_nonzero(~np.isnan(ticks))), "steady_run": run_length,
            "tempo_drift": drift, "usable": [lo, hi]}
    if count * period < min_loop:
        return {**base, "failed": f"一定の刻みが{min_loop:.0f}秒続かない"}
    first, last = ticks[run_start], ticks[run_start + count]
    fade = min(0.08, 0.2 * period)
    start, end = first - 0.35 * period, last - 0.35 * period
    run = intervals[run_start:run_start + count]
    return {**base, "start": float(start), "end": float(end), "fade": fade, "count": count,
            "jitter": float(np.std(run) / period), "interval_across_seam": float(run[-1] / period)}


def event_cut(audio, rate, *, fade_in=1.0, fade_out=2.0):
    """The one pass in a take: the span above (peak - 30 dB) around the loudest moment."""
    db = moving_mean(frame_db(mono(audio), rate), 0.25 / HOP)
    peak = int(np.argmax(db))
    above = db >= db[peak] - 30
    left = peak - int(np.argmax(~above[peak::-1])) + 1 if not above[:peak + 1].all() else 0
    right = peak + int(np.argmax(~above[peak:])) if not above[peak:].all() else len(db)
    loud = (db >= db[peak] - 12).astype(int)
    rises = np.flatnonzero(np.diff(np.concatenate(([0], loud))) == 1) * HOP
    passes = 1 + int(np.count_nonzero(np.diff(rises) > 3.0)) if len(rises) else 0
    start, end = round(left * HOP * rate), min(len(audio), round(right * HOP * rate))
    clip = audio[start:end].astype(np.float32).copy()
    inn, out = min(len(clip) // 4, round(fade_in * rate)), min(len(clip) // 4, round(fade_out * rate))
    clip[:inn] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, inn)))[:, None]
    clip[len(clip) - out:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, out)))[:, None]
    flags = []
    if left == 0:
        flags.append("頭から鳴っている")
    if right >= len(db):
        flags.append("最後まで鳴り止まない")
    if passes > 1:
        flags.append(f"山が{passes}つある")
    return clip, {"start": start / rate, "end": end / rate, "passes": passes, "flags": flags}


def event_scene(bed, events, rate, rng, *, seconds=SCENE_SECONDS, gap=(35.0, 75.0), first=(10.0, 30.0)):
    """The bed from a random offset, with passes at scattered times and levels; faded in and out."""
    total = round(seconds * rate)
    offset = int(rng.integers(len(bed)))
    scene = np.resize(np.roll(bed, -offset, axis=0), (total, bed.shape[1])).astype(np.float32)
    placed, at = [], float(rng.uniform(*first))
    while events:
        choice = int(rng.integers(len(events)))
        clip = events[choice] * 10 ** (float(rng.uniform(-3, 2)) / 20)
        begin = round(at * rate)
        if begin + len(clip) > total:
            break
        scene[begin:begin + len(clip)] += clip
        placed.append({"at": round(at, 1), "take": choice + 1})
        at += len(clip) / rate + float(rng.uniform(*gap))
    edge = round(SCENE_FADE * rate)
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, edge)))[:, None]
    scene[:edge] *= ramp
    scene[-edge:] *= ramp[::-1]
    return scene, placed


# ---- checks on the finished loop ----

def spectral_flux(signal, rate, hop=HOP):
    size = round(rate * hop)
    count = len(signal) // size
    frames = signal[:count * size].reshape(count, size) * np.hanning(size)
    magnitude = np.log1p(np.abs(np.fft.rfft(frames, axis=1)) * 100)
    return np.concatenate(([0.0], np.maximum(np.diff(magnitude, axis=0), 0).mean(axis=1)))


def join_levels(loop, rate, fade_seconds):
    """Played twice through: the level step from before the fade to after the seam, and how far the
    fade itself falls below the line between them."""
    twice = mono(np.concatenate((loop, loop)))
    seam = len(loop) / rate
    db = moving_mean(frame_db(twice, rate), 0.25 / HOP)
    before = window_mean(db, seam - fade_seconds - 1.0, seam - fade_seconds)
    after = window_mean(db, seam, seam + 1.0)
    return after - before, window_mean(db, seam - fade_seconds, seam) - (before + after) / 2


def seam_check(loop, rate, fade_seconds, *, joined_other=None):
    """The level across the join, and how the spectral change at the seam ranks among every same-sized
    window of the loop (1.0: nowhere else changes as much)."""
    step, dip = join_levels(loop, rate, fade_seconds)
    twice = mono(np.concatenate((loop, loop)))
    seam = len(loop) / rate
    flux = spectral_flux(twice, rate)
    width = 7
    rolling = np.lib.stride_tricks.sliding_window_view(flux[:len(flux) // 2], width).max(axis=1)
    result = {"step_db": step, "dip_db": dip,
              "flux_percentile": float(np.mean(rolling < window_max(flux, seam - 0.15, seam + 0.2)))}
    if joined_other is not None:
        result["equal_gain_dip_db"] = join_levels(joined_other, rate, fade_seconds)[1]
    return result


# ---- generation ----

def normalized(audio, target_rms, peak=0.99):
    rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    return (audio * min(target_rms / max(rms, 1e-9), peak / max(float(np.abs(audio).max()), 1e-9))).astype(np.float32)


def generate(output, items):
    import soundfile

    from services.worker.generation.music import stable_audio

    sources = output / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    base = {"model": "medium", "model_path": str(ROOT / "services/worker/runtimes/stable_audio3/models/medium"),
            "prompt": "x", "duration": LOOP_SECONDS, "steps": 8, "seed": 0, "device": "cuda", "dtype": "float32"}
    loaded = stable_audio.load_pipeline(base)
    for number, item in enumerate(ITEMS):
        if item["id"] not in items:
            continue
        for take in range(1, TAKES + 1):
            path = sources / f"{item['id']}_{take}.flac"
            if path.exists():
                continue
            seed = 4000 + 10 * number + take
            duration = EVENT_SECONDS if item["kind"] == "event" else LOOP_SECONDS
            audio, rate, _ = stable_audio.infer_pcm(loaded, stable_audio.validate_request(
                {**base, "prompt": SFX + item["prompt"], "duration": duration, "seed": seed}))
            soundfile.write(path, normalized(audio, LOOP_RMS), rate, subtype="PCM_24")
            print(path.name, f"seed={seed}", flush=True)


# ---- loops, previews and the page ----

def save_mp3(audio, rate, path):
    from services.worker.generation.music import stable_audio

    path.unlink(missing_ok=True)
    stable_audio._save_audio(np.asarray(audio, dtype=np.float32).T[None], rate, path, np)


def seam_preview(loop, rate):
    span = min(round(SEAM_PREVIEW * rate), len(loop) // 2)
    return np.concatenate((loop[-span:], loop[:span])), span / rate


def loop_take(item, audio, rate, folder, name):
    kind = item["kind"]
    info = {"kind_used": kind}
    if kind == "waves":
        info = wave_region(audio, rate)
        if "failed" in info:
            info = {**texture_region(audio, rate), "fallback": info["failed"], "kind_used": "texture"}
    elif kind == "ticks":
        info = tick_region(audio, rate)
        if "failed" in info:
            return info
    else:
        info = texture_region(audio, rate)
        if info is None:
            return {"failed": "使える区間が短すぎる"}
    info.setdefault("kind_used", kind)
    start, end, fade = round(info["start"] * rate), round(info["end"] * rate), round(info["fade"] * rate)
    loop = equal_power_loop(audio, start, end, fade)
    if kind == "ticks":
        # A fade this short spans no level window; the interval across the seam is the one before the cut tick.
        info["check"] = {key: value for key, value in seam_check(loop, rate, info["fade"]).items() if key != "dip_db"}
    else:
        info["check"] = seam_check(loop, rate, info["fade"], joined_other=equal_gain_loop(audio, start, end, fade))
    clip, at = seam_preview(loop, rate)
    save_mp3(np.concatenate((loop, loop)), rate, folder / f"{name}_twice.mp3")
    save_mp3(clip, rate, folder / f"{name}_seam.mp3")
    info.update({"loop_seconds": len(loop) / rate, "seam_at": at, "twice": f"loops/{name}_twice.mp3",
                 "seam": f"loops/{name}_seam.mp3"})
    return info, loop


def build(output):
    import soundfile

    folder = output / "loops"
    folder.mkdir(parents=True, exist_ok=True)
    results, beds = {}, {}
    for item in ITEMS:
        if item["kind"] == "event":
            continue
        rows = []
        for take in range(1, TAKES + 1):
            name = f"{item['id']}_{take}"
            audio, rate = soundfile.read(output / "sources" / f"{name}.flac", dtype="float32", always_2d=True)
            outcome = loop_take(item, audio, rate, folder, name)
            info = outcome[0] if isinstance(outcome, tuple) else outcome
            if isinstance(outcome, tuple) and item.get("bed"):
                beds.setdefault(item["id"], []).append(outcome[1])
            rows.append({"take": take, "source": f"sources_mp3/{name}.mp3", **info})
            print(name, json.dumps({k: v for k, v in info.items() if k in ("start", "end", "failed", "fallback", "check")},
                                   ensure_ascii=False, default=float), flush=True)
        results[item["id"]] = rows
    rng = np.random.default_rng(20261029)
    for item in ITEMS:
        if item["kind"] != "event":
            continue
        clips, rows = [], []
        for take in range(1, TAKES + 1):
            name = f"{item['id']}_{take}"
            audio, rate = soundfile.read(output / "sources" / f"{name}.flac", dtype="float32", always_2d=True)
            clip, info = event_cut(audio, rate)
            clips.append(clip * (EVENT_RMS / max(window_peak_rms(clip, rate), 1e-9)))
            save_mp3(clips[-1], rate, folder / f"{name}_cut.mp3")
            rows.append({"take": take, "source": f"sources_mp3/{name}.mp3", "cut": f"loops/{name}_cut.mp3", **info})
        bed = beds[item["over"]][0]
        bed = bed * (BED_RMS / float(np.sqrt(np.mean(bed.astype(np.float64)**2))))
        scenes = []
        for number in range(1, 3):
            scene, placed = event_scene(bed, clips, rate, rng)
            save_mp3(scene, rate, folder / f"{item['id']}_scene_{number}.mp3")
            scenes.append({"file": f"loops/{item['id']}_scene_{number}.mp3", "placed": placed})
        results[item["id"]] = {"takes": rows, "scenes": scenes, "bed": f"{item['over']} の1テイク目のループ"}
    mp3s = output / "sources_mp3"
    mp3s.mkdir(exist_ok=True)
    for path in sorted((output / "sources").glob("*.flac")):
        audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
        save_mp3(audio, rate, mp3s / f"{path.stem}.mp3")
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float),
                                         encoding="utf-8")
    (output / "index.html").write_text(page(results), encoding="utf-8")


def window_peak_rms(clip, rate):
    db = moving_mean(frame_db(mono(clip), rate), 1.0 / HOP)
    return float(10 ** (db.max() / 20))


def player(path):
    return f'<audio controls preload="none" src="{html.escape(path)}"></audio>'


def loop_cell(row):
    if "failed" in row:
        return f"<td colspan=3>ループにできない: {html.escape(row['failed'])}</td>"
    check = row["check"]
    notes = [f"{row['start']:.1f}〜{row['end']:.1f}秒（{row['loop_seconds']:.0f}秒）", f"つなぎ {row['fade']:.2f}秒"]
    if "fallback" in row:
        notes.append(f"質感方式で代用（{html.escape(row['fallback'])}）")
    if "cycles" in row:
        notes.append(f"うねり{row['cycles']}回分")
    if "count" in row:
        notes.append(f"{row['count']}刻み、刻みのばらつき {row['jitter']:.1%}、テイク内の速さの変化 {row['tempo_drift']:+.1%}")
    if row.get("standouts_inside"):
        notes.append(f"区間内の目立つ音 {row['standouts_inside']}回")
    notes.append(f"つなぎ目の前後の音量差 {check['step_db']:+.1f}dB")
    if "dip_db" in check:
        dip = f"つなぎの間の落ち込み {check['dip_db']:+.1f}dB"
        if "equal_gain_dip_db" in check:
            dip += f"（BGM用のつなぎ方なら {check['equal_gain_dip_db']:+.1f}dB）"
        notes.append(dip)
    if "interval_across_seam" in row:
        notes.append(f"つなぎ目をまたぐ刻み間隔 {row['interval_across_seam']:.2f}周期")
    return (f"<td>{player(row['twice'])}<br><small>2周（つなぎ目は{row['loop_seconds']:.0f}秒目）</small></td>"
            f"<td>{player(row['seam'])}<br><small>つなぎ目は{row['seam_at']:.0f}秒目</small></td>"
            f"<td><small>{'<br>'.join(notes)}</small></td>")


def page(results):
    parts = ["<!doctype html><meta charset=utf-8><title>環境音ループ実験</title>",
             ("<style>body{font-family:sans-serif;margin:16px;max-width:1200px}table{border-collapse:collapse}"
             "td,th{border:1px solid #ccc;padding:6px;vertical-align:top}audio{width:260px}</style>"),
             "<h1>環境音のループ（実験1：種類を指定した場合）</h1>",
             ("<p>「2周」はループを2回続けて再生したもの、「つなぎ目」はつなぎ目の前後8秒ずつです。"
             "元の音源は生成したままのもの（90秒）です。</p>")]
    for item in ITEMS:
        rows = results[item["id"]]
        parts.append(f"<h2>{html.escape(item['label'])} <small>（{KIND_LABELS[item['kind']]}）</small></h2>")
        if item["kind"] == "event":
            parts.append("<table><tr><th>テイク</th><th>元の音源</th><th>切り出した1回分</th><th>メモ</th></tr>")
            for row in rows["takes"]:
                flags = "、".join(row["flags"]) or "—"
                parts.append(f"<tr><td>{row['take']}</td><td>{player(row['source'])}</td><td>{player(row['cut'])}</td>"
                             f"<td><small>{row['start']:.1f}〜{row['end']:.1f}秒を切り出し<br>{html.escape(flags)}"
                             "</small></td></tr>")
            parts.append("</table><p>場面での鳴り方（3分、土台は" + html.escape(rows["bed"]) + "）</p><table>")
            for scene in rows["scenes"]:
                times = "、".join(f"{p['at']:.0f}秒（テイク{p['take']}）" for p in scene["placed"])
                parts.append(f"<tr><td>{player(scene['file'])}</td><td><small>通過: {times}</small></td></tr>")
            parts.append("</table>")
            continue
        parts.append("<table><tr><th>テイク</th><th>元の音源</th><th>2周</th><th>つなぎ目</th><th>メモ</th></tr>")
        for row in rows:
            parts.append(f"<tr><td>{row['take']}</td><td>{player(row['source'])}</td>{loop_cell(row)}</tr>")
        parts.append("</table>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("generate", "loop"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ambience-loops-20261029")
    parser.add_argument("--only", nargs="*", default=[item["id"] for item in ITEMS])
    args = parser.parse_args()
    if args.step == "generate":
        generate(args.output_dir, set(args.only))
    else:
        build(args.output_dir)


if __name__ == "__main__":
    main()
