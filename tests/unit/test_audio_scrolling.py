"""Notebook scrolling geometry without global pointer interception."""

import gc
import tkinter
from tkinter import ttk
from types import SimpleNamespace

import pytest

from scripts.audio.scrolling import ScrollableFrame


@pytest.fixture
def page():
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    root.attributes("-alpha", 0)
    root.geometry("640x300")
    widget = ScrollableFrame(root, padding=10, viewport_height=240)
    widget.pack(fill="both", expand=True)
    root.deiconify()
    root.update()
    yield widget
    root.destroy()
    widget.__dict__.clear()
    root.__dict__.clear()
    gc.collect()


def settle(page, *, width=600, height=240):
    root = page.winfo_toplevel()
    root.geometry(f"{width + page.scrollbar.winfo_reqwidth()}x{height}")
    for _ in range(3):
        root.update()


def tall_contents(page):
    page.content.columnconfigure(0, weight=1)
    for row in range(24):
        ttk.Label(page.content, text=f"設定 {row + 1}").grid(row=row, column=0, sticky="w", pady=4)
    settle(page)


def content_window_height(page):
    return int(float(page.canvas.itemcget(page._window_id, "height")))


def test_long_page_keeps_natural_height_and_last_setting_is_reachable(page):
    tall_contents(page)
    natural = page.content.winfo_reqheight()
    assert natural > 240
    assert content_window_height(page) == natural
    assert int(float(page.canvas.itemcget(page._window_id, "width"))) == 600
    assert tuple(map(float, page.canvas.cget("scrollregion").split())) == (0, 0, 600, natural)
    page.canvas.yview_moveto(1)
    first, last = page.canvas.yview()
    assert first > 0 and last == pytest.approx(1)
    last_setting = page.content.winfo_children()[-1]
    assert last_setting.winfo_y() + last_setting.winfo_height() <= page.canvas.canvasy(page.canvas.winfo_height())


def test_short_page_stretches_to_viewport_for_internal_row_weights(page):
    page.content.rowconfigure(0, weight=1)
    child = ttk.Frame(page.content)
    child.grid(row=0, column=0, sticky="nsew")
    settle(page, width=600, height=400)
    assert page.content.winfo_reqheight() < 400
    assert content_window_height(page) == 400
    assert page.content.winfo_height() == 400
    assert child.winfo_height() == 380  # Natural content grows inside the 10px page padding.
    assert int(float(page.canvas.itemcget(page._window_id, "width"))) == 600
    assert page._on_mousewheel(SimpleNamespace(delta=-120)) is None


def test_resizing_changes_content_width_without_cropping_natural_height(page):
    tall_contents(page)
    natural = page.content.winfo_reqheight()
    settle(page, width=360, height=200)
    assert int(float(page.canvas.itemcget(page._window_id, "width"))) == 360
    assert content_window_height(page) == natural
    settle(page, width=720, height=natural + 100)
    assert int(float(page.canvas.itemcget(page._window_id, "width"))) == 720
    assert content_window_height(page) == natural + 100


@pytest.mark.parametrize("event", [SimpleNamespace(delta=-120), SimpleNamespace(delta=-15), SimpleNamespace(num=5, delta=0)])
def test_local_wheel_moves_long_page_down(page, event):
    tall_contents(page)
    before = page.canvas.yview()[0]
    assert page._on_mousewheel(event) == "break"
    assert page.canvas.yview()[0] > before


def test_reverse_wheel_moves_back_up_and_zero_delta_is_ignored(page):
    tall_contents(page)
    page.canvas.yview_moveto(0.5)
    before = page.canvas.yview()[0]
    assert page._on_mousewheel(SimpleNamespace(delta=120)) == "break"
    assert page.canvas.yview()[0] < before
    assert page._on_mousewheel(SimpleNamespace(delta=0)) is None


def test_shrinking_page_clamps_scroll_position_back_to_top(page):
    tall_contents(page)
    page.canvas.yview_moveto(1)
    for child in page.content.winfo_children():
        child.destroy()
    ttk.Label(page.content, text="短い設定ページ").grid(row=0, column=0)
    settle(page)
    assert content_window_height(page) == 240
    assert page.canvas.yview()[0] == pytest.approx(0)


def test_scrollbar_controls_canvas_and_bindings_are_local(page):
    root = page.winfo_toplevel()
    before = root.bind_all("<MouseWheel>")
    another = ScrollableFrame(root)
    assert root.bind_all("<MouseWheel>") == before
    assert page.canvas.bind("<MouseWheel>")
    assert page.content.bind("<MouseWheel>")
    assert str(page.scrollbar.cget("orient")) == "vertical"
    text = tkinter.Text(page.content, height=2)
    assert text.bind("<MouseWheel>") == ""
    assert text.bindtags()[1:] == ("Text", str(root), "all")
    another.destroy()


def test_destroy_cancels_pending_layout_callback(page):
    page.update_idletasks()
    page._queue_layout()
    pending = page._layout_after_id
    assert pending is not None
    root = page.winfo_toplevel()
    page.destroy()
    assert pending not in root.tk.call("after", "info")
