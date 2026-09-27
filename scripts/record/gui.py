"""Small Tkinter launcher for recording a published project."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import StringVar, Tk, filedialog, messagebox, ttk
from urllib.parse import quote, urljoin, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
RECORDER = Path(__file__).with_name("record_player.py")
API = "http://127.0.0.1:8000"
PHASES = {
    "starting": "起動中", "loading": "章を読み込み中", "recording": "録画中",
    "encoding": "映像を保存中", "muxing_chapter": "章の音声を合成中",
    "checking_next": "次の章を確認中", "joining": "動画を結合中",
}


def duration(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def progress_text(state: dict, fallback_elapsed: float = 0, now: float | None = None) -> tuple[str, str]:
    elapsed = state.get("elapsed_seconds", fallback_elapsed)
    # The encoder can run for a while without a status update. Only the elapsed
    # time keeps increasing; video duration and line position stay measured values.
    if state.get("status") == "running" and "updated_at_epoch" in state:
        elapsed += max(0, (time.time() if now is None else now) - state["updated_at_epoch"])
    timing = f"録画時間 {duration(state.get('recorded_seconds', 0))}　実行時間 {duration(elapsed)}"
    chapter = state.get("current_chapter")
    total = state.get("utterance_total")
    position = "台詞・地の文の位置を取得中…"
    if chapter:
        count = f"（{state.get('utterance_current', 0)}/{total}）" if total is not None else "（取得中）"
        position = f"録画{chapter}章目　台詞・地の文 {count}"
    return timing, position


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.load(response)


def display_name(project: dict) -> str:
    return f"{project.get('title') or '無題の作品'}  [{project['id'][:8]}]"


def make_output(parent: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S")
    return parent / f"recording-{stamp}-{uuid.uuid4().hex[:6]}"


class RecordingApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("作品録画")
        self.root.geometry("660x340")
        self.root.minsize(600, 320)
        self.projects: list[dict] = []
        self.events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.process: subprocess.Popen | None = None
        self.output: Path | None = None
        self.loading = False
        self.run_started: float | None = None
        self.last_progress: dict = {}
        self.server = StringVar(value=API)
        self.output_parent = StringVar(value=str(ROOT / "outputs"))
        self.status = StringVar(value="サーバーから作品一覧を取得します。")
        self.timing = StringVar(value="録画時間 00:00:00　実行時間 00:00:00")
        self.position = StringVar(value="台詞・地の文の位置は録画開始後に表示します。")

        frame = ttk.Frame(root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="サーバー").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.server).grid(row=0, column=1, sticky="ew", padx=8)
        self.refresh_button = ttk.Button(frame, text="更新", command=self.refresh)
        self.refresh_button.grid(row=0, column=2)

        ttk.Label(frame, text="作品").grid(row=1, column=0, sticky="w", pady=6)
        self.project_box = ttk.Combobox(frame, state="readonly")
        self.project_box.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0))

        ttk.Label(frame, text="保存先").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.output_parent).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="参照", command=self.browse).grid(row=2, column=2)
        ttk.Label(frame, text="選んだフォルダー内に録画ごとの新しいフォルダーを作成します。")\
            .grid(row=3, column=1, columnspan=2, sticky="w", padx=8)

        self.start_button = ttk.Button(frame, text="録画開始", command=self.start)
        self.start_button.grid(row=4, column=1, sticky="w", padx=8, pady=(16, 9))
        self.open_button = ttk.Button(frame, text="保存先を開く", command=self.open_output,
                                      state="disabled")
        self.open_button.grid(row=4, column=2, pady=(16, 9))
        ttk.Label(frame, textvariable=self.timing)\
            .grid(row=5, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(frame, textvariable=self.position)\
            .grid(row=6, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(frame, textvariable=self.status, wraplength=570, justify="left")\
            .grid(row=7, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        root.after(100, self.drain_events)
        root.after(100, self.refresh)

    def api_base(self) -> str:
        value = self.server.get().strip().rstrip("/")
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query:
            raise ValueError("サーバーには http://127.0.0.1:8000 のようなURLを指定してください。")
        return value

    def browse(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.output_parent.get() or str(ROOT))
        if chosen:
            self.output_parent.set(chosen)

    def refresh(self) -> None:
        if self.loading:
            return
        try:
            base = self.api_base()
        except ValueError as exc:
            messagebox.showerror("サーバーURL", str(exc))
            return
        self.loading = True
        self.refresh_button.configure(state="disabled")
        self.status.set("作品一覧を取得中…")

        def work():
            try:
                projects = fetch_json(base + "/api/projects")["projects"]
                if not isinstance(projects, list):
                    raise TypeError("作品一覧の形式が不正です。")
                self.events.put(("projects", projects))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.events.put(("refresh_error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def start(self) -> None:
        index = self.project_box.current()
        if index < 0 or index >= len(self.projects):
            messagebox.showinfo("作品", "作品を選択してください。")
            return
        try:
            base = self.api_base()
            parent = Path(self.output_parent.get().strip()).expanduser().resolve()
            if not parent.is_dir():
                raise ValueError("保存先フォルダーが見つかりません。")
            if not RECORDER.is_file():
                raise ValueError("録画スクリプトが見つかりません。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("録画開始", str(exc))
            return
        project = self.projects[index]
        self.run_started = time.monotonic()
        self.last_progress = {}
        self.timing.set("録画時間 00:00:00　実行時間 00:00:00")
        self.position.set("台詞・地の文の位置を取得中…")
        self.start_button.configure(state="disabled")
        self.status.set("公開済みの章を確認中…")

        def work():
            try:
                project_id = quote(project["id"], safe="")
                production = fetch_json(base + "/api/m3/projects/" + project_id).get("production") or {}
                player = production.get("player_url")
                if not player:
                    raise ValueError("この作品には録画できる公開章がありません。")
                output = make_output(parent)
                command = [str(ROOT / ".venv/Scripts/python.exe"), "-u", "-X", "utf8",
                           str(RECORDER), "--url", urljoin(base + "/", player),
                           "--output-dir", str(output)]
                if not Path(command[0]).is_file():
                    raise ValueError("プロジェクトのPython環境が見つかりません。")
                with (parent / (output.name + ".stdout.log")).open("wb") as stdout, \
                        (parent / (output.name + ".stderr.log")).open("wb") as stderr:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                               creationflags=subprocess.CREATE_NO_WINDOW)
                self.events.put(("started", (process, output)))
            except (OSError, ValueError, KeyError) as exc:
                self.events.put(("start_error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def drain_events(self) -> None:
        while not self.events.empty():
            event, data = self.events.get_nowait()
            if event == "projects":
                self.projects = data
                self.project_box["values"] = [display_name(project) for project in data]
                self.project_box.set("")
                self.loading = False
                self.refresh_button.configure(state="normal")
                self.status.set(f"{len(data)}件の作品を取得しました。")
            elif event == "refresh_error":
                self.loading = False
                self.refresh_button.configure(state="normal")
                self.status.set("作品一覧を取得できませんでした。")
                messagebox.showerror("作品一覧", str(data))
            elif event == "started":
                self.process, self.output = data
                self.open_button.configure(state="normal")
                self.status.set(f"録画中… 保存先: {self.output}")
            elif event == "start_error":
                self.start_button.configure(state="normal")
                self.status.set("録画を開始できませんでした。")
                messagebox.showerror("録画開始", str(data))
        if self.process is not None:
            self.update_progress()
        self.root.after(500, self.drain_events)

    def update_progress(self) -> None:
        state_path = self.output / "status.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # The recorder may be writing the file at exactly this moment.
            state = self.last_progress
        self.last_progress = state
        elapsed = time.monotonic() - self.run_started if self.run_started is not None else 0
        timing, position = progress_text(state, elapsed)
        self.timing.set(timing)
        self.position.set(position)
        status = state.get("status")
        if status == "completed":
            self.status.set(f"録画完了: {state['output']}")
        elif status == "failed":
            self.status.set(f"録画失敗: {state.get('error', 'ログを確認してください。')}")
        elif self.process.poll() is not None:
            self.status.set(f"録画プロセスが終了しました。ログ: {self.output}.stderr.log")
        else:
            phase = PHASES.get(state.get("phase"), "起動中")
            self.status.set(f"{phase}… 保存先: {self.output}")
            return
        self.process = None
        self.start_button.configure(state="normal")

    def open_output(self) -> None:
        if self.output:
            os.startfile(self.output if self.output.exists() else self.output.parent)


def main() -> None:
    root = Tk()
    RecordingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
