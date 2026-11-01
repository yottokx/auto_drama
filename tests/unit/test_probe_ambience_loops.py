"""Ambience loops on synthetic sound: joins that keep the level, seams away from standout moments,
cuts on troughs and between ticks, and one pass cut out of a take."""
import pytest

np = pytest.importorskip("numpy")

import itertools

from scripts.audio import probe_ambience_loops as probe

RATE = 8000


def noise(seconds, seed=0):
    return np.random.default_rng(seed).normal(0, 0.1, (round(seconds * RATE), 2)).astype(np.float32)


def test_equal_power_join_keeps_the_level_of_noise_where_the_music_join_dips():
    audio = noise(20)
    start, end, fade = 2 * RATE, 18 * RATE, 2 * RATE
    joined = probe.equal_power_loop(audio, start, end, fade)
    check = probe.seam_check(joined, RATE, 2.0, joined_other=probe.equal_gain_loop(audio, start, end, fade))
    assert len(joined) == end - start
    assert abs(check["dip_db"]) < 0.7
    assert check["equal_gain_dip_db"] < -1.0


def test_join_plays_on_into_the_start_of_the_region():
    audio = noise(10)
    start, end, fade = 2 * RATE, 8 * RATE, RATE
    joined = probe.equal_power_loop(audio, start, end, fade)
    assert np.allclose(joined[-1], audio[start - 1], atol=0.01)
    assert np.array_equal(joined[:RATE], audio[start:start + RATE])


def test_usable_range_skips_the_generated_fade_in():
    audio = noise(60)
    audio[:5 * RATE] *= np.linspace(0, 1, 5 * RATE)[:, None] ** 3
    lo, hi = probe.usable_range(probe.frame_db(probe.mono(audio), RATE))
    assert lo > 3.5
    assert hi > 55


def test_texture_seams_keep_out_of_a_short_burst():
    audio = noise(80)
    audio[round(20 * RATE):round(23 * RATE)] *= 10
    region = probe.texture_region(audio, RATE)
    for seam in (region["start"], region["end"]):
        assert not 19.5 <= seam <= 26.5


def test_texture_seams_keep_out_of_a_swell():
    audio = noise(80)
    audio[round(30 * RATE):round(40 * RATE)] *= 4
    region = probe.texture_region(audio, RATE)
    for seam in (region["start"], region["end"]):
        assert not 29 <= seam <= 44


def test_texture_seams_avoid_standout_moments():
    audio = noise(70)
    bursts = (4.5, 35.0, 65.5)
    for at in bursts:
        audio[round(at * RATE):round((at + 0.15) * RATE)] *= 12
    region = probe.texture_region(audio, RATE)
    for seam in (region["start"], region["end"]):
        assert all(not seam - region["fade"] - 0.2 <= at <= seam + 0.6 for at in bursts)
    assert region["end"] - region["start"] >= probe.MIN_LOOP
    assert all(min(abs(at - t) for t in region["standout_times"]) < 0.2 for at in bursts)


def test_waves_are_cut_on_troughs_and_hold_whole_swells():
    seconds, period = 90, 8.0
    t = np.arange(round(seconds * RATE)) / RATE
    swell = 0.1 + 0.9 * (0.5 - 0.5 * np.cos(2 * np.pi * t / period))
    audio = noise(seconds) * swell[:, None].astype(np.float32)
    region = probe.wave_region(audio, RATE)
    trough = region["start"] - region["fade"] / 2
    assert min(trough % period, period - trough % period) < 0.3
    length = region["end"] - region["start"]
    assert abs(length / period - round(length / period)) < 0.05
    assert region["cycle_seconds"] == pytest.approx(period, abs=0.3)


def clock(seconds, period=1.0, speedup_after=None, second_click=False):
    audio = noise(seconds, 1) * 0.02
    at, index = 0.5, 0
    while at < seconds - 0.4:
        sample = round(at * RATE)
        audio[sample:sample + 40] += (0.8 if index % 2 == 0 else 0.5) * np.hanning(40)[:, None]
        if second_click and index % 3 == 0:
            later = sample + round(0.18 * RATE)
            audio[later:later + 40] += 0.4 * np.hanning(40)[:, None]
        step = period * (0.85 if speedup_after is not None and at > speedup_after else 1.0)
        at, index = at + step, index + 1
    return audio


def test_ticks_loop_on_an_even_count_and_keep_the_interval_across_the_seam():
    audio = clock(40)
    region = probe.tick_region(audio, RATE)
    assert region["count"] % 2 == 0
    assert region["end"] - region["start"] == pytest.approx(region["count"] * 1.0, abs=0.02)
    loop = probe.equal_power_loop(audio, round(region["start"] * RATE), round(region["end"] * RATE),
                                  round(region["fade"] * RATE))
    ticks = probe.tick_times(probe.mono(np.concatenate((loop, loop))), RATE)
    seam = len(loop) / RATE
    across = [b - a for a, b in itertools.pairwise(ticks) if a < seam <= b]
    assert across[0] == pytest.approx(1.0, abs=0.02) == region["interval_across_seam"]


def test_a_second_click_inside_some_ticks_is_not_taken_for_a_tick():
    region = probe.tick_region(clock(40, second_click=True), RATE)
    assert region["period"] == pytest.approx(1.0, abs=0.02)
    assert region["end"] - region["start"] > 30


def test_ticks_that_speed_up_are_reported_and_looped_only_on_the_steady_part():
    audio = clock(60, speedup_after=30)
    region = probe.tick_region(audio, RATE)
    assert region["tempo_drift"] < -0.1
    assert region["end"] - region["start"] < 32


def test_event_cut_takes_the_one_pass_and_flags_a_second():
    audio = noise(25) * 0.001
    t = np.arange(len(audio)) / RATE
    single = audio + (0.3 * np.exp(-((t - 10) / 2.5) ** 2))[:, None] * noise(25, 2)
    clip, info = probe.event_cut(single, RATE)
    assert 2 < info["start"] < 8 and 12 < info["end"] < 18
    assert info["passes"] == 1 and not info["flags"]
    assert abs(clip[0]).max() < 1e-3
    double = single + (0.3 * np.exp(-((t - 20) / 1.5) ** 2))[:, None] * noise(25, 3)
    assert probe.event_cut(double, RATE)[1]["passes"] == 2


def test_event_scene_scatters_passes_inside_the_scene_and_fades_its_edges():
    bed = noise(10) * 0.1
    events = [noise(4, 5), noise(6, 6)]
    scene, placed = probe.event_scene(bed, events, RATE, np.random.default_rng(1), seconds=180)
    assert len(scene) == 180 * RATE
    assert 1 <= len(placed) <= 4
    assert all(b["at"] - a["at"] >= 35 for a, b in itertools.pairwise(placed))
    assert abs(scene[0]).max() < 1e-3 and abs(scene[-1]).max() < 1e-3
