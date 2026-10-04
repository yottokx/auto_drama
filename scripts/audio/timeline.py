"""Small Tk timeline for one-time openings and explicitly marked repeat regions."""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont
from tkinter import ttk


def _finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _time_label(seconds: float) -> str:
    hundredths = max(0, round(seconds * 100))
    minutes, remainder = divmod(hundredths, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


class AudioTimeline(ttk.Frame):
    """Call only from the Tk thread; callbacks occur only on pointer input."""

    def __init__(self, parent, on_seek: Callable[[float], None], **kwargs):
        super().__init__(parent, **kwargs)
        self.on_seek = on_seek
        self._duration = 0.0
        self._position = 0.0
        self._loop_start = None
        self._loop_end = None
        self._edit_start = None
        self._edit_end = None
        self._enabled = True
        self._seeking_enabled = True
        self._dragging = False
        self._last_pointer_position = None
        self._canvas_width = 640.0
        self._font = tkfont.Font(root=self, family="Yu Gothic UI", size=9)
        self.canvas = tk.Canvas(self, height=96, width=640, background="#fafafa",
                                highlightthickness=0, borderwidth=0, takefocus=False)
        self.canvas.pack(fill="x", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self._redraw()

    def set_audio(self, duration_seconds, loop_start_seconds=None, loop_end_seconds=None,
                  edit_start_seconds=None, edit_end_seconds=None):
        duration = _finite_number(duration_seconds)
        if duration is None or duration <= 0:
            self.reset()
            return
        self._duration = duration
        self._position = 0.0
        self._dragging = False
        self._last_pointer_position = None
        self._loop_start, self._loop_end = self._bounded_range(loop_start_seconds, loop_end_seconds)
        self._edit_start, self._edit_end = self._bounded_range(edit_start_seconds, edit_end_seconds)
        self._redraw()

    def set_position(self, seconds):
        position = _finite_number(seconds)
        self._position = min(self._duration, max(0.0, position if position is not None else 0.0))
        self._draw_position()

    def set_enabled(self, enabled: bool):
        self._enabled = bool(enabled)
        self._dragging = False
        self._redraw()

    def set_seeking_enabled(self, enabled: bool):
        self._seeking_enabled = bool(enabled)
        if not self._seeking_enabled:
            self._dragging = False
        self._update_cursor()

    def reset(self):
        self._duration = self._position = 0.0
        self._loop_start = self._loop_end = self._edit_start = self._edit_end = None
        self._dragging = False
        self._last_pointer_position = None
        self._redraw()

    def _bounded_range(self, start, end):
        start, end = _finite_number(start), _finite_number(end)
        if start is None or end is None:
            return None, None
        start, end = max(0.0, min(self._duration, start)), max(0.0, min(self._duration, end))
        return (start, end) if start < end else (None, None)

    def _plot_bounds(self):
        margin = min(18.0, self._canvas_width / 5)
        return margin, max(margin + 1.0, self._canvas_width - margin)

    def _x_for_seconds(self, seconds):
        left, right = self._plot_bounds()
        return left + (right - left) * min(self._duration, max(0.0, seconds)) / max(self._duration, 1e-12)

    def _seconds_for_x(self, x):
        left, right = self._plot_bounds()
        return self._duration * min(1.0, max(0.0, (x - left) / (right - left)))

    def _can_seek(self):
        return self._enabled and self._seeking_enabled and self._duration > 0

    def _update_cursor(self):
        self.canvas.configure(cursor="hand2" if self._can_seek() else "arrow")

    def _on_resize(self, event):
        self._canvas_width = max(1.0, float(event.width))
        self._redraw()

    def _seek_pointer(self, event):
        if not self._can_seek():
            return
        seconds = self._seconds_for_x(float(event.x))
        self.set_position(seconds)
        if seconds != self._last_pointer_position:
            self._last_pointer_position = seconds
            self.on_seek(seconds)

    def _on_press(self, event):
        self._dragging = self._can_seek()
        self._last_pointer_position = None
        if self._dragging:
            self._seek_pointer(event)

    def _on_drag(self, event):
        if self._dragging:
            self._seek_pointer(event)

    def _on_release(self, event):
        if self._dragging:
            self._seek_pointer(event)
        self._dragging = False

    def _label_x(self, x, text):
        half = self._font.measure(text) / 2 + 3
        if 2 * half >= self._canvas_width:
            return self._canvas_width / 2
        return min(self._canvas_width - half, max(half, x))

    def _redraw(self):
        self.canvas.delete("all")
        self._update_cursor()
        left, right = self._plot_bounds()
        text_color = "#263442" if self._enabled else "#77818b"
        if not self._duration:
            self.canvas.create_text(self._canvas_width / 2, 51, text="音声を選ぶと再生位置とA〜Bを表示します。",
                                    fill=text_color, font=self._font, tags="empty_label")
            self._draw_position()
            return
        self.canvas.create_rectangle(left, 43, right, 59, fill="#e3e7ec", outline="",
                                     tags="audio_region")
        if self._loop_start is not None:
            start_x, end_x = self._x_for_seconds(self._loop_start), self._x_for_seconds(self._loop_end)
            self.canvas.create_rectangle(left, 43, start_x, 59, fill="#c4cbd4", outline="",
                                         tags="intro_region")
            self.canvas.create_rectangle(start_x, 43, end_x, 59,
                                         fill="#a8d1fa" if self._enabled else "#ccd7e3", outline="",
                                         tags="loop_region")
            for x, seconds, letter, description, y, tag in (
                (start_x, self._loop_start, "A", "戻り先", 28, "loop_start"),
                (end_x, self._loop_end, "B", "終端", 72, "loop_end"),
            ):
                self.canvas.create_line(x, 36, x, 65, width=2,
                                        fill="#1d65a9" if self._enabled else "#8a97a5", tags=tag + "_marker")
                label = f"{letter} {description}  {_time_label(seconds)}"
                self.canvas.create_text(self._label_x(x, label), y, text=label,
                                        fill=text_color, font=self._font, tags=tag + "_label")
        if self._edit_start is not None:
            self.canvas.create_rectangle(self._x_for_seconds(self._edit_start), 51,
                                         self._x_for_seconds(self._edit_end), 59,
                                         fill="#ed9e48" if self._enabled else "#d7c1a8", outline="",
                                         tags="edit_region")
            self.canvas.create_text(right, 10, text="橙: AI修復範囲   青: 繰り返し区間", anchor="e",
                                    fill=text_color, font=self._font, tags="legend_label")
        elif self._loop_start is not None:
            self.canvas.create_text(right, 10, text="灰: 冒頭を一度再生   青: 繰り返し", anchor="e",
                                    fill=text_color, font=self._font, tags="legend_label")
        self.canvas.create_text(left, 88, text="00:00.00", anchor="w",
                                fill=text_color, font=self._font, tags="zero_label")
        self.canvas.create_text(right, 88, text=_time_label(self._duration), anchor="e",
                                fill=text_color, font=self._font, tags="duration_label")
        self.canvas.create_text(self._canvas_width / 2, 88, text="クリック・ドラッグで再生位置を移動",
                                fill=text_color, font=self._font, tags="seek_hint")
        self._draw_position()

    def _draw_position(self):
        self.canvas.delete("position_label", "playhead_line", "playhead_head")
        label = "再生 " + (_time_label(self._position) if self._duration else "--:--.--")
        self.canvas.create_text(12, 10, text=label, anchor="w", font=self._font,
                                fill="#263442" if self._enabled else "#77818b", tags="position_label")
        if self._duration:
            x = self._x_for_seconds(self._position)
            color = "#b32436" if self._enabled else "#8a929b"
            self.canvas.create_line(x, 37, x, 65, fill=color, width=2, tags="playhead_line")
            self.canvas.create_polygon(x - 4, 35, x + 4, 35, x, 40, fill=color, outline="",
                                       tags="playhead_head")
