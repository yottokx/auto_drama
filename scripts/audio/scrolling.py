"""Local vertical scrolling for notebook pages, leaving playback controls fixed."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """Build the page inside ``content``; add this wrapper to the notebook."""

    def __init__(self, parent, *, padding=0, viewport_height=320, **kwargs):
        super().__init__(parent, **kwargs)
        self._viewport_width = 640
        self._viewport_height = max(1, int(viewport_height))
        self._content_height = self._viewport_height
        self._layout_after_id = None
        background = ttk.Style(self).lookup("TFrame", "background") or "#f0f0f0"
        self.canvas = tk.Canvas(self, width=self._viewport_width, height=self._viewport_height,
                                highlightthickness=0, borderwidth=0, background=background,
                                takefocus=False)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.content = ttk.Frame(self.canvas, padding=padding)
        self._window_id = self.canvas.create_window(0, 0, window=self.content, anchor="nw")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.content.bind("<Configure>", self._on_content_configure)
        # These local bindings do not intercept wheel handling in text inputs,
        # combo boxes, other pages, or the fixed history/playback panel.
        for surface in (self.canvas, self.content):
            surface.bind("<MouseWheel>", self._on_mousewheel)
            surface.bind("<Button-4>", self._on_mousewheel)
            surface.bind("<Button-5>", self._on_mousewheel)
        self._queue_layout()

    def _queue_layout(self):
        if self._layout_after_id is None:
            self._layout_after_id = self.after_idle(self._update_layout)

    def _on_canvas_configure(self, event):
        self._viewport_width = max(1, int(event.width))
        self._viewport_height = max(1, int(event.height))
        self._queue_layout()

    def _on_content_configure(self, event):
        self._queue_layout()

    def _update_layout(self):
        self._layout_after_id = None
        self._content_height = max(self.content.winfo_reqheight(), self._viewport_height)
        self.canvas.itemconfigure(self._window_id, width=self._viewport_width,
                                  height=self._content_height)
        self.canvas.configure(scrollregion=(0, 0, self._viewport_width, self._content_height))

    def _on_mousewheel(self, event):
        if self._content_height <= self._viewport_height:
            return None
        button = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        if button in (4, 5):
            units = -1 if button == 4 else 1
        elif delta:
            units = -int(delta / 120) or (-1 if delta > 0 else 1)
        else:
            return None
        self.canvas.yview_scroll(units, "units")
        return "break"

    def destroy(self):
        if self._layout_after_id is not None:
            self.after_cancel(self._layout_after_id)
            self._layout_after_id = None
        super().destroy()
