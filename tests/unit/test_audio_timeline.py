"""Tk-thread timeline checks without audio devices or playback subprocesses."""

import gc
import tkinter
from types import SimpleNamespace

import pytest

from scripts.audio.timeline import AudioTimeline


@pytest.fixture
def timeline():
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    seeks = []
    widget = AudioTimeline(root, seeks.append)
    widget.pack(fill="x")
    root.update_idletasks()
    widget._on_resize(SimpleNamespace(width=920))
    yield widget, seeks
    root.destroy()
    # Release Tk fonts and bindings on the same thread that created the widgets.
    widget.__dict__.clear()
    root.__dict__.clear()
    gc.collect()


def pointer(x):
    return SimpleNamespace(x=x, y=51)


def test_timeline_shows_intro_repeat_repaired_area_and_file_duration(timeline):
    widget, seeks = timeline
    widget.set_audio(120, loop_start_seconds=20, loop_end_seconds=110,
                     edit_start_seconds=106, edit_end_seconds=114)
    canvas = widget.canvas
    left, right = widget._plot_bounds()
    assert canvas.coords("intro_region") == pytest.approx([left, 43, widget._x_for_seconds(20), 59])
    assert canvas.coords("loop_region") == pytest.approx([widget._x_for_seconds(20), 43, widget._x_for_seconds(110), 59])
    assert canvas.coords("edit_region") == pytest.approx([widget._x_for_seconds(106), 51, widget._x_for_seconds(114), 59])
    assert "A 戻り先" in canvas.itemcget("loop_start_label", "text")
    assert "00:20.00" in canvas.itemcget("loop_start_label", "text")
    assert "B 終端" in canvas.itemcget("loop_end_label", "text")
    assert canvas.itemcget("duration_label", "text") == "02:00.00"
    assert canvas.coords("duration_label")[0] == right
    assert seeks == []


def test_click_drag_and_release_seek_with_clamping_and_no_duplicate_release(timeline):
    widget, seeks = timeline
    widget.set_audio(120, 20, 120)
    left, right = widget._plot_bounds()
    widget._on_press(pointer(left - 50))
    widget._on_drag(pointer(widget._x_for_seconds(30)))
    widget._on_release(pointer(right + 50))
    assert seeks == pytest.approx([0, 30, 120])
    assert widget.canvas.itemcget("position_label", "text") == "再生 02:00.00"
    widget._on_press(pointer(widget._x_for_seconds(60)))
    widget._on_release(pointer(widget._x_for_seconds(60)))
    assert seeks == pytest.approx([0, 30, 120, 60])


def test_drag_without_pointer_press_does_not_seek(timeline):
    widget, seeks = timeline
    widget.set_audio(120)
    widget._on_drag(pointer(widget._x_for_seconds(30)))
    widget._on_release(pointer(widget._x_for_seconds(30)))
    assert seeks == []


def test_programmatic_playhead_tracks_b_to_a_jump_without_callback(timeline):
    widget, seeks = timeline
    widget.set_audio(120, 20, 110)
    widget.set_position(110)
    assert widget.canvas.coords("playhead_line")[0] == pytest.approx(widget._x_for_seconds(110))
    assert widget.canvas.itemcget("position_label", "text") == "再生 01:50.00"
    widget.set_position(20)
    assert widget.canvas.coords("playhead_line")[0] == pytest.approx(widget._x_for_seconds(20))
    assert widget.canvas.itemcget("position_label", "text") == "再生 00:20.00"
    assert seeks == []


@pytest.mark.parametrize("seconds, label", [(-100, "00:00.00"), (1200, "02:00.00"), (12.345, "00:12.34")])
def test_playhead_clamps_and_displays_hundredths_without_feedback(timeline, seconds, label):
    widget, seeks = timeline
    widget.set_audio(120)
    widget.set_position(seconds)
    assert widget.canvas.itemcget("position_label", "text") == "再生 " + label
    assert seeks == []


def test_resize_repositions_markers_and_preserves_seconds_mapping(timeline):
    widget, seeks = timeline
    widget.set_audio(120, 20, 110, 106, 114)
    widget.set_position(35)
    before = widget.canvas.coords("loop_start_marker")[0]
    widget._on_resize(SimpleNamespace(width=460))
    assert widget.canvas.coords("loop_start_marker")[0] != before
    assert widget.canvas.coords("loop_start_marker")[0] == pytest.approx(widget._x_for_seconds(20))
    assert widget.canvas.coords("loop_end_marker")[0] == pytest.approx(widget._x_for_seconds(110))
    assert widget.canvas.coords("playhead_line")[0] == pytest.approx(widget._x_for_seconds(35))
    widget._on_press(pointer(widget._x_for_seconds(30)))
    assert seeks == pytest.approx([30])


@pytest.mark.parametrize("setting", ["widget", "seeking"])
def test_disabled_widget_or_seam_preview_blocks_pointer_but_keeps_playhead(timeline, setting):
    widget, seeks = timeline
    widget.set_audio(120, 20, 110)
    if setting == "widget":
        widget.set_enabled(False)
    else:
        widget.set_seeking_enabled(False)
    widget._on_press(pointer(widget._x_for_seconds(30)))
    widget._on_drag(pointer(widget._x_for_seconds(40)))
    widget._on_release(pointer(widget._x_for_seconds(50)))
    widget.set_position(65)
    assert widget.canvas.itemcget("position_label", "text") == "再生 01:05.00"
    assert widget.canvas.find_withtag("loop_region")
    assert seeks == []
    widget.set_enabled(True)
    widget.set_seeking_enabled(True)
    widget._on_press(pointer(widget._x_for_seconds(30)))
    assert seeks == pytest.approx([30])


def test_disabling_seeking_cancels_existing_drag(timeline):
    widget, seeks = timeline
    widget.set_audio(120)
    widget._on_press(pointer(widget._x_for_seconds(10)))
    widget.set_seeking_enabled(False)
    widget.set_seeking_enabled(True)
    widget._on_drag(pointer(widget._x_for_seconds(20)))
    widget._on_release(pointer(widget._x_for_seconds(20)))
    assert seeks == pytest.approx([10])


def test_starting_normal_playback_during_pointer_press_keeps_drag_active(timeline):
    widget, seeks = timeline
    widget.set_audio(120)
    # Normal playback startup refreshes the enabled state inside on_seek.
    widget.on_seek = lambda seconds: (seeks.append(seconds), widget.set_seeking_enabled(True))
    widget._on_press(pointer(widget._x_for_seconds(10)))
    widget._on_drag(pointer(widget._x_for_seconds(20)))
    widget._on_release(pointer(widget._x_for_seconds(30)))
    assert seeks == pytest.approx([10, 20, 30])


@pytest.mark.parametrize("start,end", [(None, 100), (100, None), (110, 20), (20, 20), (float("nan"), 100)])
def test_incomplete_or_invalid_loop_markers_are_not_displayed(timeline, start, end):
    widget, _ = timeline
    widget.set_audio(120, start, end)
    assert not widget.canvas.find_withtag("loop_region")
    assert not widget.canvas.find_withtag("loop_start_label")


def test_out_of_file_ranges_are_clamped_and_labels_fit_history_width(timeline):
    widget, _ = timeline
    widget.set_audio(120, -20, 200, -10, 150)
    left, right = widget._plot_bounds()
    assert widget.canvas.coords("loop_region") == pytest.approx([left, 43, right, 59])
    assert widget.canvas.coords("edit_region") == pytest.approx([left, 51, right, 59])
    for tag in ("loop_start_label", "loop_end_label", "duration_label", "zero_label", "position_label"):
        x1, _, x2, _ = widget.canvas.bbox(tag)
        assert 0 <= x1 < x2 <= 920


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), None])
def test_invalid_duration_resets_and_blocks_seeking(timeline, duration):
    widget, seeks = timeline
    widget.set_audio(120, 20, 110, 106, 114)
    widget.set_audio(duration)
    widget._on_press(pointer(500))
    assert widget.canvas.find_withtag("empty_label")
    assert not widget.canvas.find_withtag("loop_region")
    assert not widget.canvas.find_withtag("edit_region")
    assert not widget.canvas.find_withtag("playhead_line")
    assert seeks == []


def test_reset_clears_current_audio_and_new_audio_resets_position(timeline):
    widget, seeks = timeline
    widget.set_audio(120, 20, 110)
    widget.set_position(70)
    widget.reset()
    assert widget.canvas.itemcget("position_label", "text") == "再生 --:--.--"
    widget.set_audio(30)
    assert widget.canvas.itemcget("position_label", "text") == "再生 00:00.00"
    assert not widget.canvas.find_withtag("loop_region")
    assert seeks == []
