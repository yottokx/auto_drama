"""Standalone, subprocess-isolated BGM generation and comparison workbench."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, StringVar, Text, Tk, filedialog, messagebox, ttk
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.audio.catalog import CoordinatorCatalog
from scripts.audio.engine import MODEL_SPECS, MP3_BITRATES, validate_request
from scripts.audio.hub_auth import hub_environment
from scripts.audio.json_io import write_json
from scripts.audio.loop_playback import loop_playback_plan
from scripts.audio.process_tree import WindowsChildJob
from scripts.audio.prompts import (
    MOOD_PRESETS,
    STYLE_PRESETS,
    generate_prompt,
    normalize_ace_metadata,
)
from scripts.audio.scrolling import ScrollableFrame
from scripts.audio.timeline import AudioTimeline

RUNTIME = ROOT / "services/worker/runtimes/stable_audio3"
SETTINGS = ROOT / "private/audio/gui-settings.json"
PLAYBACK_ROOT = ROOT / "private/audio/playback"
QUALITY_PRESETS = {
    "温かな会話 / Acoustic Pop": (
        "Genre: Acoustic Pop. Instruments: Piano, Acoustic Guitar, Bass, Brushes. "
        "Warm, relaxed instrumental music at 80 BPM in a major key. A clear, memorable piano "
        "melody over consonant chords, gentle fingerpicked guitar and a steady soft rhythm. "
        "Balanced, clean studio production with natural acoustic tones. A consistent, flowing "
        "arrangement with subtle variation, suitable for quiet conversation. "
        "TrackType: Music, VocalType: Instrumental"
    ),
    "穏やかな情感 / Piano Soundtrack": (
        "Genre: Soundtrack, Modern Classical. Instruments: Piano, Strings. "
        "Tender, reflective instrumental music at 72 BPM. A lyrical piano melody supported by "
        "warm, sustained strings and consonant harmony. Gentle dynamics, rounded piano tone "
        "and clear, spacious studio production. An intimate, reassuring atmosphere with a "
        "coherent recurring theme and smooth phrasing throughout. "
        "TrackType: Music, VocalType: Instrumental"
    ),
    "軽快な日常 / Lounge Jazz": (
        "Genre: Lounge Jazz. Instruments: Piano, Upright Bass, Brushes, Vibraphone. "
        "Lighthearted, easygoing instrumental music at 96 BPM. A tuneful piano motif, "
        "warm jazz chords and a relaxed, steady groove. Sparse vibraphone accents, soft brushed "
        "drums and clear acoustic production. A playful, friendly atmosphere with balanced "
        "dynamics and a consistent small ensemble arrangement. "
        "TrackType: Music, VocalType: Instrumental"
    ),
    "激しい緊迫 / Industrial Metal": (
        "Genre: Aggressive Industrial Metal. Instruments: Distorted palm-muted electric guitars, "
        "overdriven bass, pounding double-kick drums, sharp snare, abrasive synthesizers. "
        "Instrumental music at 160 BPM in D minor. Relentless syncopated riffs, hard staccato "
        "attacks, heavy low-end and fierce driving energy throughout. Dense dry production, "
        "crushing mechanical groove and tense dissonant accents. "
        "TrackType: Music, VocalType: Instrumental"
    ),
    "冷たい圧迫 / Dissonant Industrial": (
        "Genre: Dark Dissonant Industrial Score. Instruments: Metallic synthesizer stabs, "
        "distorted low bass, harsh industrial percussion and cold electronic drones. "
        "Instrumental music at 65 BPM in C minor. Heavy mechanical impacts, grating minor "
        "seconds and tritones, static unresolved harmony and oppressive robotic menace. "
        "Sharp attacks and a hollow sterile reverberation sustain cold tension throughout. "
        "TrackType: Music, VocalType: Instrumental"
    ),
    "高速コミカル / Electro Funk": (
        "Genre: Energetic Comic Electro Funk. Instruments: Clipped funk guitar, popping slap "
        "bass, bright brassy synthesizer stabs, punchy drum machine and sharp handclaps. "
        "Instrumental music at 132 BPM in C major. Bouncy sixteenth-note syncopation, cheeky "
        "short melodic hooks, crisp stop-start accents and lively playful call-and-response. "
        "Tight modern production and buoyant danceable energy throughout. "
        "TrackType: Music, VocalType: Instrumental"
    ),
}
QUALITY_METADATA = {
    "激しい緊迫 / Industrial Metal": {"bpm": 160, "keyscale": "D minor", "timesignature": "4"},
    "冷たい圧迫 / Dissonant Industrial": {"bpm": 65, "keyscale": "C minor", "timesignature": "4"},
    "高速コミカル / Electro Funk": {"bpm": 132, "keyscale": "C major", "timesignature": "4"},
}
LOOP_METHODS = {
    "whole_crossfade": "曲全体をクロスフェード",
    "smart_region": "ループ区間を自動探索",
    "ai_bridge": "AIで接続部分を補修",
    "smart_ai_bridge": "区間を自動探索＋インペインティング",
}
DEFAULT_PROMPT = next(iter(QUALITY_PRESETS.values()))


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def new_run(parent: Path, kind: str = "bgm") -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return parent / f"{kind}-{stamp}-{uuid.uuid4().hex[:8]}"


class AudioTestApp:
    def __init__(self, root: Tk, *, auto_refresh: bool = True):
        self.root = root
        root.title("BGM比較テスト • Stable Audio 3 / ACE-Step 1.5")
        root.geometry("1120x900")
        root.minsize(980, 780)
        self.events = queue.SimpleQueue()
        self.projects = []
        self.chapters = []
        self.scenes = []
        self.catalog_revision = 0
        self.prompt_revision = 0
        self._active_scene_key = None
        self._prompt_source_kind = "quality_control"
        self._llm_base_prompt = ""
        self._prompt_snapshot = DEFAULT_PROMPT
        self.pending_llm = None
        self.llm_run_revision = None
        self.process = None
        self._process_job = None
        self.run_dir = None
        self.run_kind = None
        self.started_at = 0.0
        self.stop_at = None
        self.close_pending = False
        self.results = {}
        self._log_stream = None
        self.playback_process = None
        self._playback_job = None
        self._playback_session = None
        self._playback_status_path = None
        self._playback_control_path = None
        self._playback_key = None
        self._playback_mode = None
        self._playback_sequence = 0
        self._timeline_key = None
        self.loop_source_metadata = {}
        self._loop_source_path = None
        saved = read_json(SETTINGS)
        self.server = StringVar(value=saved.get("server", "http://127.0.0.1:8000"))
        self.python = StringVar(value=saved.get("python", str(RUNTIME / ".venv/Scripts/python.exe")))
        self.output = StringVar(value=saved.get("output", str(ROOT / "outputs/audio-tests")))
        self.model = StringVar(value="medium")
        self.model_choice = StringVar()
        self.paths = {key: StringVar(value=saved.get(f"path_{key}", str(RUNTIME / "models" / key)))
                      for key in MODEL_SPECS}
        self.duration = StringVar(value="30")
        self.steps = StringVar(value="8")
        self.seed = StringVar(value="-1")
        self.quality_preset = StringVar(value=next(iter(QUALITY_PRESETS)))
        self.device = StringVar(value="auto")
        self.dtype = StringVar(value="auto")
        self.offload = BooleanVar(value=False)
        self.output_format = StringVar(value=saved.get("output_format", "mp3"))
        self.mp3_bitrate = StringVar(value=str(saved.get("mp3_bitrate", 192)))
        self.keep_wav = BooleanVar(value=saved.get("keep_wav", False))
        self.loop_source = StringVar(value="")
        self.loop_method = StringVar(value=LOOP_METHODS["smart_region"])
        self.loop_duration = StringVar(value="60")
        self.loop_fade = StringVar(value="0.25")
        self.loop_candidates = StringVar(value="3")
        self.loop_bridge = StringVar(value="4")
        self.loop_context = StringVar(value="15")
        self.loop_hint = StringVar(value="結果一覧から元の曲を選び、「入力にする」を押してください。")
        self.loop_points = StringVar(value="曲を選ぶと再生範囲とループの戻り先を表示します。")
        self.playback_label = StringVar(value="音源を選択してください。")
        self.style = StringVar(value="auto")
        self.mood = StringVar(value="auto")
        self.auto_scene_prompt = BooleanVar(value=saved.get("auto_scene_prompt", False))
        self.prompt_source = StringVar(value="品質確認用")
        self.scene_interpretation = StringVar(value="")
        self.tempo = StringVar(value="0")
        self.ace_bpm = StringVar(value="0")
        self.ace_keyscale = StringVar(value="")
        self.ace_timesignature = StringVar(value="")
        self.ace_planner_enabled = BooleanVar(value=saved.get("ace_planner_enabled", True))
        self.ace_planner_path = StringVar(value=saved.get(
            "ace_planner_path", str(RUNTIME / "models" / "ace15_planner_1_7b"),
        ))
        self.ace_continuous = BooleanVar(value=saved.get("ace_continuous", True))
        self.llm_url = StringVar(value=saved.get("llm_url", "http://127.0.0.1:8080/v1"))
        self.llm_model = StringVar(value=saved.get("llm_model", "local-model"))
        self.llm_backend = StringVar(value=saved.get("llm_backend", "llama_cpp"))
        if self.llm_backend.get() not in {"llama_cpp", "ollama", "openai"}:
            self.llm_backend.set("llama_cpp")
        self.llm_local_model = StringVar(value=saved.get("llm_local_model", ""))
        self.llm_hint = StringVar()
        self.loop = BooleanVar(value=False)
        self.rating = StringVar(value="未評価")
        self.note = StringVar(value="")
        self.status = StringVar(value="作品を選ぶか、プロンプトを直接入力して生成できます。")
        self.catalog_status = StringVar(value="作品サーバーの接続先を確認してください。")
        self.story_context_hint = StringVar(value="")
        self.model_hint = StringVar()
        self.generation_hint = StringVar()
        self.metrics = StringVar(value="生成結果を選択すると、時間・ピーク・RMSを表示します。")
        self._build()
        self._model_changed()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(150, self.poll)
        if auto_refresh:
            root.after(200, self.refresh)

    def _build(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.rowconfigure(3, weight=0)
        ttk.Label(outer, text="BGM生成・比較 / Stable Audio 3 ・ ACE-Step 1.5", font=("Yu Gothic UI", 17, "bold"))\
            .grid(row=0, column=0, sticky="w", pady=(0, 8))
        tabs = ttk.Notebook(outer)
        self.tabs = tabs
        tabs.grid(row=1, column=0, sticky="nsew")
        main_page = ScrollableFrame(tabs, padding=10)
        settings_page = ScrollableFrame(tabs, padding=14)
        llm_page = ScrollableFrame(tabs, padding=14)
        loop_page = ScrollableFrame(tabs, padding=14)
        main, settings = main_page.content, settings_page.content
        llm_settings, loop_settings = llm_page.content, loop_page.content
        tabs.add(main_page, text="シーンと生成")
        tabs.add(loop_page, text="ループ実験")
        self.loop_tab = loop_page
        tabs.add(llm_page, text="LLM設定")
        tabs.add(settings_page, text="環境・モデル準備")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)
        source = ttk.LabelFrame(main, text="作品のシーン", padding=10)
        source.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        source.columnconfigure(1, weight=1)
        self.project_box = self._combo(source, "作品", 0, self.project_changed)
        self.chapter_box = self._combo(source, "章", 1, self.chapter_changed)
        self.scene_box = self._combo(source, "シーン", 2, self.scene_changed)
        ttk.Label(source, textvariable=self.catalog_status, wraplength=460)\
            .grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        self.preview = Text(source, height=4, wrap="word", state="disabled", font=("Yu Gothic UI", 10))
        self.preview.grid(row=4, column=0, columnspan=2, sticky="ew")
        ttk.Label(source, textvariable=self.story_context_hint, wraplength=460)\
            .grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        controls = ttk.LabelFrame(main, text="音声生成", padding=10)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.columnconfigure(1, weight=1)
        self.model_box = self._combo(controls, "モデル", 0, self._model_selected,
                                     variable=self.model_choice,
                                     values=[spec["label"] for spec in MODEL_SPECS.values()])
        ttk.Label(controls, textvariable=self.model_hint, wraplength=450)\
            .grid(row=1, column=0, columnspan=4, sticky="w", pady=4)
        self._entry(controls, "秒数", self.duration, 2)
        self._entry(controls, "ステップ", self.steps, 3)
        self._entry(controls, "Seed（-1でランダム）", self.seed, 4)
        devices = ttk.Frame(controls)
        devices.grid(row=5, column=0, columnspan=4, sticky="ew", pady=4)
        ttk.Label(devices, text="実行先").pack(side="left")
        ttk.Combobox(devices, textvariable=self.device, values=["auto", "cuda", "cpu"],
                     state="readonly", width=7).pack(side="left", padx=6)
        ttk.Label(devices, text="精度").pack(side="left")
        ttk.Combobox(devices, textvariable=self.dtype,
                     values=["auto", "float32", "float16", "bfloat16"],
                     state="readonly", width=10).pack(side="left", padx=6)
        ttk.Checkbutton(controls, text="CPUオフロード（CUDAのVRAMを節約）", variable=self.offload)\
            .grid(row=6, column=0, columnspan=4, sticky="w")
        ttk.Label(controls, textvariable=self.generation_hint,
                  wraplength=450).grid(row=7, column=0, columnspan=4, sticky="w", pady=4)
        prompt_frame = ttk.LabelFrame(main, text="音楽プロンプト（英語・編集可）", padding=10)
        prompt_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(1, weight=1)
        options = ttk.Frame(prompt_frame)
        options.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(options, text="スタイル").pack(side="left")
        self.style_box = ttk.Combobox(options, state="readonly", width=16,
                                     values=list(STYLE_PRESETS.values()))
        self.style_box.current(0)
        self.style_box.pack(side="left", padx=5)
        ttk.Label(options, text="感情").pack(side="left")
        self.mood_box = ttk.Combobox(options, state="readonly", width=13,
                                    values=list(MOOD_PRESETS.values()))
        self.mood_box.current(0)
        self.mood_box.pack(side="left", padx=5)
        ttk.Label(options, text="希望BPM（0=自動）").pack(side="left")
        ttk.Entry(options, textvariable=self.tempo, width=5).pack(side="left", padx=5)
        ttk.Button(options, text="シーンから作成", command=self.draft).pack(side="left", padx=5)
        self.ai_button = ttk.Button(options, text="LLMで作成", command=self.draft_ai)
        self.ai_button.pack(side="left", padx=5)
        self.prompt = Text(prompt_frame, height=5, wrap="word", font=("Yu Gothic UI", 11), undo=True)
        self.prompt.grid(row=1, column=0, sticky="nsew")
        self.prompt.insert("1.0", DEFAULT_PROMPT)
        self.prompt.edit_modified(False)
        self.prompt.bind("<<Modified>>", self._prompt_edited)
        interpretation = ttk.Frame(prompt_frame)
        interpretation.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        interpretation.columnconfigure(1, weight=1)
        ttk.Label(interpretation, textvariable=self.prompt_source).grid(row=0, column=0, sticky="nw", padx=(0, 8))
        ttk.Label(interpretation, textvariable=self.scene_interpretation, wraplength=840)\
            .grid(row=0, column=1, sticky="w")
        quality = ttk.Frame(prompt_frame)
        quality.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(quality, text="品質確認用").pack(side="left")
        ttk.Combobox(quality, textvariable=self.quality_preset, values=list(QUALITY_PRESETS),
                     state="readonly", width=36).pack(side="left", padx=5)
        ttk.Button(quality, text="プロンプトをセット", command=self.draft_quality).pack(side="left", padx=5)
        ttk.Label(quality, text="シーン解釈を使わず比較 / Seed -1で別の曲").pack(side="left", padx=5)
        ttk.Checkbutton(prompt_frame, text="シーン選択時に簡易プロンプトを入れる", variable=self.auto_scene_prompt)\
            .grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.ace_controls = ttk.Frame(prompt_frame)
        self.ace_controls.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(self.ace_controls, text="生成BPM").pack(side="left")
        ttk.Entry(self.ace_controls, textvariable=self.ace_bpm, width=5).pack(side="left", padx=5)
        ttk.Label(self.ace_controls, text="調性").pack(side="left")
        keys = [""] + [f"{note} {mode}" for mode in ("major", "minor")
                       for note in ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")]
        ttk.Combobox(self.ace_controls, textvariable=self.ace_keyscale, values=keys,
                     state="readonly", width=12).pack(side="left", padx=5)
        ttk.Label(self.ace_controls, text="拍子").pack(side="left")
        ttk.Combobox(self.ace_controls, textvariable=self.ace_timesignature,
                     values=["", "2", "3", "4", "6"], state="readonly", width=4).pack(side="left", padx=5)
        ttk.Label(self.ace_controls, text="インスト固定 / 希望BPM 0なら場面ごとに自動選択")\
            .pack(side="left", padx=6)
        self.ace_planner_controls = ttk.Frame(prompt_frame)
        self.ace_planner_controls.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(self.ace_planner_controls, text="専用plannerを使う（1.7B）",
                        variable=self.ace_planner_enabled, command=self._model_changed).pack(side="left")
        ttk.Label(self.ace_planner_controls, text="ON/OFFで比較 / 同じプロンプト・BPM・Seedを使用")\
            .pack(side="left", padx=10)
        ttk.Checkbutton(self.ace_planner_controls, text="長い無音を抑える", variable=self.ace_continuous)\
            .pack(side="left", padx=8)
        self._settings(settings)
        self._llm_settings(llm_settings)
        self._loop_settings(loop_settings)
        actions = ttk.Frame(outer)
        actions.grid(row=2, column=0, sticky="ew", pady=8)
        self.generate_button = ttk.Button(actions, text="BGMを生成", command=self.generate)
        self.generate_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(actions, text="処理を停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 10))
        self.progress = ttk.Progressbar(actions, length=150, mode="indeterminate")
        self.progress.pack(side="left", padx=6)
        ttk.Label(actions, textvariable=self.status, wraplength=620).pack(side="left", padx=6)
        history = ttk.LabelFrame(outer, text="今回の生成結果 / モデル・Seedを変えて比較", padding=8)
        history.grid(row=3, column=0, sticky="nsew")
        history.columnconfigure(0, weight=1)
        history.rowconfigure(0, weight=1)
        self.history = ttk.Treeview(history, columns=("model", "seconds", "seed", "time", "scene"),
                                    show="headings", height=4, selectmode="browse")
        for key, label, width in [("model", "モデル", 140), ("seconds", "秒数", 60),
                                  ("seed", "Seed", 100), ("time", "生成時間", 100),
                                  ("scene", "シーン / 実行名", 470)]:
            self.history.heading(key, text=label)
            self.history.column(key, width=width, stretch=key == "scene")
        self.history.grid(row=0, column=0, sticky="nsew")
        self.history.bind("<<TreeviewSelect>>", self.result_selected)
        playback_info = ttk.Frame(history)
        playback_info.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        ttk.Label(playback_info, textvariable=self.playback_label, wraplength=940, justify="left").pack(anchor="w")
        ttk.Label(playback_info, textvariable=self.loop_points, wraplength=940, justify="left").pack(anchor="w")
        self.timeline = AudioTimeline(history, on_seek=self._seek_playback)
        self.timeline.grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Label(history, textvariable=self.metrics, wraplength=940, justify="left").grid(
            row=3, column=0, sticky="w", pady=5)
        play = ttk.Frame(history)
        play.grid(row=4, column=0, sticky="ew")
        ttk.Button(play, text="▶ 冒頭から試聴", command=self.play).pack(side="left", padx=(0, 6))
        ttk.Button(play, text="■ 停止", command=self.stop_playback).pack(side="left", padx=6)
        ttk.Button(play, text="継ぎ目だけ試聴", command=lambda: self.play(seam_preview=True)).pack(side="left", padx=6)
        ttk.Checkbutton(play, text="繰り返し試聴", variable=self.loop,
                        command=self._loop_mode_changed).pack(side="left", padx=6)
        ttk.Button(play, text="保存先を開く", command=self.open_output).pack(side="left", padx=6)
        ttk.Button(play, text="過去の結果を読み込む", command=self.load_history).pack(side="left", padx=6)
        ttk.Button(play, text="ループ入力にする", command=self.choose_loop_source).pack(side="left", padx=6)
        review = ttk.Frame(history)
        review.grid(row=5, column=0, sticky="ew", pady=(7, 0))
        review.columnconfigure(3, weight=1)
        ttk.Label(review, text="BGM適性").grid(row=0, column=0)
        ttk.Combobox(review, textvariable=self.rating, state="readonly", width=9,
                     values=["未評価", "1", "2", "3", "4", "5"]).grid(row=0, column=1, padx=6)
        ttk.Label(review, text="メモ").grid(row=0, column=2)
        ttk.Entry(review, textvariable=self.note).grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Button(review, text="評価を保存", command=self.save_review).grid(row=0, column=4)

    def _settings(self, frame):
        frame.columnconfigure(1, weight=1)
        self._entry(frame, "作品サーバー", self.server, 0)
        ttk.Button(frame, text="作品一覧を更新", command=self.refresh).grid(row=0, column=2, padx=8)
        self._entry(frame, "音声生成用Python", self.python, 1)
        ttk.Button(frame, text="参照", command=lambda: self.browse_file(self.python)).grid(row=1, column=2)
        for row, key in enumerate(MODEL_SPECS, 2):
            self._entry(frame, MODEL_SPECS[key]["label"] + " のモデル", self.paths[key], row)
            ttk.Button(frame, text="参照", command=lambda k=key: self.browse_dir(self.paths[k]))\
                .grid(row=row, column=2)
        planner_row = 2 + len(MODEL_SPECS)
        self._entry(frame, "ACE 専用planner 1.7B のモデル", self.ace_planner_path, planner_row)
        ttk.Button(frame, text="参照", command=lambda: self.browse_dir(self.ace_planner_path))\
            .grid(row=planner_row, column=2)
        output_row = planner_row + 1
        self._entry(frame, "生成結果の保存先", self.output, output_row)
        ttk.Button(frame, text="参照", command=lambda: self.browse_dir(self.output)).grid(row=output_row, column=2)
        ttk.Label(frame, text="初回は①実行環境を準備 → ②選択モデルを取得・変換。取得済みのDiffusers形式フォルダーも指定できます。",
                  wraplength=920).grid(row=output_row + 1, column=0, columnspan=3, sticky="w", pady=(10, 4))
        prep = ttk.Frame(frame)
        prep.grid(row=output_row + 2, column=0, columnspan=3, sticky="w")
        self.setup_button = ttk.Button(prep, text="① 実行環境を準備", command=self.setup_runtime)
        self.setup_button.pack(side="left", padx=(0, 8))
        self.prepare_button = ttk.Button(prep, text="② 選択モデルを取得・変換", command=self.prepare_model)
        self.prepare_button.pack(side="left", padx=8)
        self.planner_prepare_button = ttk.Button(prep, text="③ ACE plannerを取得", command=self.prepare_planner)
        self.planner_prepare_button.pack(side="left", padx=8)
        ttk.Button(prep, text="ログを開く", command=self.open_log).pack(side="left", padx=8)
        ttk.Label(frame, text="auto精度: Stable Audioはfloat32、ACE-StepはCUDAでbfloat16。標準は各8ステップです。",
                  wraplength=920).grid(row=output_row + 3, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Separator(frame).grid(row=output_row + 4, column=0, columnspan=3, sticky="ew", pady=8)
        auth = ttk.Frame(frame)
        auth.grid(row=output_row + 5, column=0, columnspan=3, sticky="w")
        self.auth_button = ttk.Button(auth, text="Hugging Faceアクセス確認", command=self.check_hub_access)
        self.auth_button.pack(side="left", padx=(0, 8))
        ttk.Button(auth, text="選択モデルの利用条件を開く", command=self.open_model_terms).pack(side="left", padx=8)
        ttk.Label(frame, text="Stable Audioは既存のHFログインと利用条件への同意が必要です。ACE-Step 1.5は公開モデルを取得します。",
                  wraplength=920).grid(row=output_row + 6, column=0, columnspan=3, sticky="w", pady=6)
        save = ttk.LabelFrame(frame, text="音声の保存", padding=10)
        save.grid(row=output_row + 7, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(save, text="形式").pack(side="left")
        ttk.Combobox(save, textvariable=self.output_format, values=["mp3", "wav"],
                     state="readonly", width=7).pack(side="left", padx=6)
        ttk.Label(save, text="MP3 kbps").pack(side="left")
        ttk.Combobox(save, textvariable=self.mp3_bitrate, values=list(MP3_BITRATES),
                     state="readonly", width=7).pack(side="left", padx=6)
        ttk.Checkbutton(save, text="MP3保存時も原音WAVを残す（容量大）", variable=self.keep_wav).pack(side="left", padx=8)
        ttk.Label(frame, text="既定はMP3 192kbps。WAVを選ぶとPCM16とFLOAT32原音を保存します。",
                  wraplength=920).grid(row=output_row + 8, column=0, columnspan=3, sticky="w")

    def _loop_settings(self, frame):
        frame.columnconfigure(1, weight=1)
        self._entry(frame, "入力音声", self.loop_source, 0)
        ttk.Button(frame, text="音声を選択", command=self.browse_loop_source).grid(row=0, column=2)
        ttk.Button(frame, text="結果一覧の曲を入力にする", command=self.choose_loop_source).grid(row=1, column=1, sticky="w")
        self._combo(frame, "方式", 2, lambda: None, variable=self.loop_method, values=list(LOOP_METHODS.values()))
        self._entry(frame, "探索区間の目安（秒）", self.loop_duration, 3)
        self._entry(frame, "接続の重ね幅（秒）", self.loop_fade, 4)
        self._combo(frame, "探索候補数（自動探索）", 5, lambda: None, variable=self.loop_candidates, values=[1, 2, 3])
        self._entry(frame, "AI修復幅（2〜8秒）", self.loop_bridge, 6)
        self._entry(frame, "AI参照幅（片側10〜20秒）", self.loop_context, 7)
        self.loop_button = ttk.Button(frame, text="ループ音源を作成", command=self.create_loop)
        self.loop_button.grid(row=8, column=1, sticky="w", pady=12)
        ttk.Label(frame, textvariable=self.loop_hint, wraplength=930).grid(row=9, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Label(frame, text="初回は元曲の冒頭から再生し、ループ終端で戻り先へ移ります。結果一覧に両方の秒数を表示します。"
                  "自動探索は拍・和音・音量が近い区間を選び、探索＋インペインティングはその接続部分だけをAIで補修します。"
                  "AI補修はStable AudioのSmall/Mediumを選択してください。ACE-Stepの音声もクロスフェード・自動探索が可能です。"
                  "保存形式は「環境・モデル準備」と共通です。",
                  wraplength=930).grid(row=10, column=0, columnspan=3, sticky="w", pady=8)

    def choose_loop_source(self):
        selected = self.selected_result()
        if not selected:
            self.loop_hint.set("結果一覧から元の曲を選択してください。")
            return
        directory, result, metadata = selected
        source = Path(metadata.get("float_audio_path") or "")
        if not source.is_file():
            source = self._result_audio_path(directory, result, metadata)
        self.loop_source.set(str(source))
        self.loop_source_metadata = metadata
        self._loop_source_path = source.resolve()
        self.loop_hint.set("入力: " + directory.name + " / 元の曲を残して別の音源を作成します。")
        self.tabs.select(self.loop_tab)

    def browse_loop_source(self):
        path = filedialog.askopenfilename(filetypes=[("音声", "*.mp3 *.wav"), ("すべて", "*")])
        if path:
            self.loop_source.set(path)
            self.loop_source_metadata = self._metadata_for_audio(Path(path))
            self._loop_source_path = Path(path).resolve()

    def create_loop(self):
        try:
            if self.process is not None or self.pending_llm is not None:
                raise ValueError("現在の処理が完了してから実行してください。")
            source = Path(self.loop_source.get().strip()).expanduser().resolve()
            if not source.is_file():
                raise ValueError("入力音声を選択してください。")
            method = next(key for key, label in LOOP_METHODS.items() if label == self.loop_method.get())
            source_metadata = self.loop_source_metadata if source == self._loop_source_path else self._metadata_for_audio(source)
            request = {"source_audio": str(source), "source_generation": source_metadata,
                       "method": method, "target_duration": float(self.loop_duration.get()),
                       "crossfade_seconds": float(self.loop_fade.get()),
                       "bridge_seconds": float(self.loop_bridge.get()),
                       "bridge_context_seconds": float(self.loop_context.get()),
                       "candidate_count": int(self.loop_candidates.get()), "seed": int(self.seed.get()),
                       "model": self.model.get(), "model_path": self.paths[self.model.get()].get().strip(),
                       "device": self.device.get(), "dtype": self.dtype.get(), "steps": int(self.steps.get()),
                       "cpu_offload": self.offload.get(), **self._save_options()}
            from scripts.audio.loops import validate_loop_request
            validate_loop_request(request)
            release_request = self._llm_release_request()
            if method in {"ai_bridge", "smart_ai_bridge"} and release_request:
                request["llm_release"] = release_request
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), "loop")
            self._start_process([self._runtime_python(), "-u", "-X", "utf8",
                                 str(Path(__file__).with_name("loop_runner.py")),
                                 "--request", str(run_dir / "request.json"), "--output-dir", str(run_dir)],
                                run_dir, "loop", request=request)
            self.status.set("ループ候補を作成しています…")
        except (OSError, ValueError, StopIteration) as exc:
            messagebox.showerror("ループ作成", str(exc))

    def _save_options(self):
        return {"output_format": self.output_format.get(), "mp3_bitrate": int(self.mp3_bitrate.get()),
                "keep_wav": self.keep_wav.get()}

    @staticmethod
    def _metadata_for_audio(source):
        metadata = read_json(source.parent / "generation.json")
        paths = [metadata.get(key) for key in ("audio_path", "float_audio_path", "wav_audio_path")]
        result = read_json(source.parent / "result.json")
        if result.get("ok"):
            paths.extend(result.get(key) for key in ("audio_path", "float_audio_path", "wav_audio_path"))
        names = [Path(value).name for value in paths if isinstance(value, str) and value]
        if not names:
            names = ["output.mp3", "output.wav", "output-float.wav"]
        return metadata if source.name in names else {}

    def _llm_settings(self, frame):
        frame.columnconfigure(1, weight=1)
        self.llm_backend_box = self._combo(
            frame, "プロンプト生成方式", 0, self._llm_backend_changed,
            variable=self.llm_backend, values=["llama_cpp", "ollama", "openai"])
        self.llm_local_box = self._combo(
            frame, "ローカルGGUFモデル", 1, lambda: None, variable=self.llm_local_model)
        self.llm_reload_button = ttk.Button(frame, text="再読込", command=self.reload_llm_models)
        self.llm_reload_button.grid(row=1, column=2, padx=8)
        self._entry(frame, "外部LLM API", self.llm_url, 2)
        self._entry(frame, "外部LLMモデル名", self.llm_model, 3)
        ttk.Label(frame, textvariable=self.llm_hint, wraplength=920).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=10)
        ttk.Label(
            frame,
            text="llama_cppは本体で登録済みのGGUFを専用プロセスで読み込み、プロンプト作成後に解放します。"
                 "Ollamaも使用後にモデルを解放してから音声生成へ進みます。"
                 "「シーンから作成」はLLM不要です。",
            wraplength=920,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=8)
        self.reload_llm_models()
        self._llm_backend_changed()

    def reload_llm_models(self):
        try:
            from scripts.audio.llm_runtime import default_native_model, list_native_models

            models = list_native_models(ROOT)
            self.llm_local_box["values"] = models
            if not self.llm_local_model.get():
                self.llm_local_model.set(default_native_model(ROOT))
            self.llm_hint.set(f"本体の設定からGGUFモデル{len(models)}件を読み込みました。")
        except Exception as exc:  # noqa: BLE001 -- optional native registry must not prevent startup
            self.llm_local_box["values"] = []
            self.llm_hint.set("GGUFモデル一覧を取得できません: " + str(exc))

    def _llm_backend_changed(self):
        native = self.llm_backend.get() == "llama_cpp"
        self.llm_local_box.configure(state="readonly" if native else "disabled")
        self.llm_reload_button.configure(state="normal" if native else "disabled")

    @staticmethod
    def _entry(frame, label, variable, row):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=4)

    @staticmethod
    def _combo(frame, label, row, callback, variable=None, values=()):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        box = ttk.Combobox(frame, state="readonly", textvariable=variable, values=values)
        box.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)
        box.bind("<<ComboboxSelected>>", lambda _event: callback())
        return box

    def _model_changed(self):
        spec = MODEL_SPECS[self.model.get()]
        self.model_choice.set(spec["label"])
        self.model_hint.set(f"{spec['label']} / 標準{spec['default_steps']}ステップ")
        if self.model.get() == "ace15_turbo":
            self.generation_hint.set(
                "インスト固定。Gemma → 解放 → 専用planner → 解放 → Diffusersで音声化。"
                if self.ace_planner_enabled.get() else
                "インスト固定。Gemma → 解放 → Diffusersで音声化（plannerなし）。"
            )
            self.ace_controls.grid()
            self.ace_planner_controls.grid()
        else:
            self.generation_hint.set("蒸留モデルの標準: 8ステップ・CFG 1.0。ネガティブ指定は使用しません。")
            self.ace_controls.grid_remove()
            self.ace_planner_controls.grid_remove()

    def _model_selected(self):
        key = next(key for key, spec in MODEL_SPECS.items() if spec["label"] == self.model_choice.get())
        previous = self.model.get()
        self.model.set(key)
        if (previous == "ace15_turbo") != (key == "ace15_turbo"):
            self.prompt_revision += 1
            if self._prompt_source_kind == "llm":
                base_prompt = self._llm_base_prompt
                caption = re.sub(r"\b(?:TrackType|VocalType)\s*:\s*[^,.;\n]+[,.;]?\s*", "",
                                 base_prompt, flags=re.IGNORECASE).strip()
                if key == "ace15_turbo":
                    if not caption.lower().startswith("instrumental"):
                        caption = "Instrumental background music. " + caption
                else:
                    bpm = self._ace_generation_bpm()
                    if bpm not in {"", "0"}:
                        caption += f" BPM: {bpm}."
                    if self.ace_keyscale.get():
                        caption += f" Key: {self.ace_keyscale.get()}."
                    if self.ace_timesignature.get():
                        meter = self.ace_timesignature.get()
                        caption += f" Time signature: {meter}/{'8' if meter == '6' else '4'}."
                    caption += " TrackType: Music, VocalType: Instrumental."
                self._replace_prompt(caption, source="llm", interpretation=self.scene_interpretation.get(),
                                     base_prompt=base_prompt)
        self._model_changed()

    def browse_dir(self, variable):
        chosen = filedialog.askdirectory(initialdir=variable.get() if Path(variable.get()).is_dir() else ROOT)
        if chosen:
            variable.set(chosen)

    def browse_file(self, variable):
        chosen = filedialog.askopenfilename(filetypes=[("Python", "*.exe"), ("すべて", "*")])
        if chosen:
            variable.set(chosen)

    def _async(self, kind, revision, work):
        def run():
            try:
                self.events.put((kind, revision, work(), None))
            except Exception as exc:  # noqa: BLE001 -- surface failures at the thread boundary
                self.events.put((kind, revision, None, str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def refresh(self):
        self.catalog_revision += 1
        self.prompt_revision += 1
        self._clear_scene_prompt()
        self.projects, self.chapters, self.scenes = [], [], []
        for box in [self.project_box, self.chapter_box, self.scene_box]:
            box.set("")
            box["values"] = []
        self._set_preview("")
        self.story_context_hint.set("")
        self.catalog_status.set("作品一覧を取得中…")
        base = self.server.get().strip()
        self._async("projects", self.catalog_revision, lambda: CoordinatorCatalog(base).projects())

    def project_changed(self):
        self.catalog_revision += 1
        self.prompt_revision += 1
        self._clear_scene_prompt()
        self.chapters, self.scenes = [], []
        for box in [self.chapter_box, self.scene_box]:
            box.set("")
            box["values"] = []
        self._set_preview("")
        self.story_context_hint.set("")
        index = self.project_box.current()
        if index < 0:
            return
        project = self.projects[index]
        base = self.server.get().strip()
        self.catalog_status.set("選択した制作版の章・シーンを取得中…")
        self._async("chapters", self.catalog_revision, lambda: CoordinatorCatalog(base).chapters(project))

    def chapter_changed(self):
        self.prompt_revision += 1
        index = self.chapter_box.current()
        self.scenes = self.chapters[index].scenes if index >= 0 else []
        self.scene_box["values"] = [scene.label for scene in self.scenes]
        self.scene_box.set("")
        if self.scenes:
            self.scene_box.current(0)
        self.scene_changed()

    def selected_scene(self):
        index = self.scene_box.current()
        return self.scenes[index] if 0 <= index < len(self.scenes) else None

    def _clear_scene_prompt(self):
        if self._prompt_source_kind in {"llm", "simple"}:
            self._replace_prompt("", source="empty")
            self._active_scene_key = None

    def scene_changed(self):
        scene = self.selected_scene()
        self._set_preview(scene.preview if scene else "この章には生成済みのシーンがありません。")
        self.story_context_hint.set(self._context_hint(scene))
        scene_key = (scene.context.get("project_id"), scene.context.get("storyline_id"),
                     scene.context.get("narrative_artifact_id"), scene.context.get("chapter_number"),
                     scene.id, json.dumps(scene.context, ensure_ascii=False, sort_keys=True)) if scene else None
        if scene_key == self._active_scene_key:
            return
        self._active_scene_key = scene_key
        self.prompt_revision += 1
        if scene and self.auto_scene_prompt.get():
            self.draft()
        else:
            self._replace_prompt("", source="empty")
            self.status.set("シーンを確認し、「LLMで作成」を実行してください。")

    @staticmethod
    def _context_hint(scene):
        if not scene:
            return ""
        story = scene.context.get("story_context") or {}
        brief = story.get("brief") or {}
        available = []
        if any(brief.get(key) for key in ("genre", "mood", "notes", "prompt", "setting")):
            available.append("作品設定")
        outline = story.get("outline") or {}
        if any(outline.get(key) for key in ("ending", "chapters", "character_arcs", "foreshadowing")):
            available.append("全体プロット")
        chapter = story.get("chapter") or {}
        if chapter.get("role") or chapter.get("summary"):
            available.append("章の役割")
        if not available:
            return "LLM参照: シーンのみ（作品設定・全体プロット未取得）"
        text = "LLM参照: " + "・".join(available) + "・シーン"
        if "作品設定" not in available or "全体プロット" not in available:
            missing = [item for item in ("作品設定", "全体プロット") if item not in available]
            text += "（" + "・".join(missing) + "未取得）"
        genre = brief.get("genre")
        if isinstance(genre, str) and genre.strip():
            text += "\n作品ジャンル: " + genre.strip()[:100]
        return text

    def _set_preview(self, text):
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def prompt_options(self):
        tempo = int(self.tempo.get())
        maximum = 300 if self.model.get() == "ace15_turbo" else 240
        if tempo != 0 and not 30 <= tempo <= maximum:
            raise ValueError(f"BPMは0（自動）、または30〜{maximum}を指定してください。")
        options = {"style": list(STYLE_PRESETS)[max(0, self.style_box.current())],
                    "mood": list(MOOD_PRESETS)[max(0, self.mood_box.current())], "tempo": tempo}
        if self.model.get() == "ace15_turbo":
            options["music_backend"] = "ace_step15"
        return options

    def _ace_generation_bpm(self):
        value = self.ace_bpm.get().strip()
        # A requested tempo can also be used with a manually written caption.
        # Generated metadata never changes the next scene's automatic request.
        if value not in {"", "0"}:
            return value
        return "0" if self._prompt_source_kind == "quality_control" else self.tempo.get().strip()

    def _replace_prompt(self, prompt, *, source="manual", interpretation="", base_prompt=None):
        self._llm_base_prompt = (prompt if base_prompt is None else base_prompt) if source == "llm" else ""
        if source in {"empty", "simple", "quality_control"}:
            self.ace_bpm.set("0")
            self.ace_keyscale.set("")
            self.ace_timesignature.set("")
        self._prompt_snapshot = prompt.strip()
        self._prompt_source_kind = source
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", prompt)
        self.prompt.edit_modified(False)
        labels = {"llm": "LLM作成", "simple": "簡易作成", "manual": "手動入力・編集",
                  "quality_control": "品質確認用", "empty": "未作成"}
        self.prompt_source.set(labels[source])
        self.scene_interpretation.set(interpretation)

    def _prompt_edited(self, _event=None):
        if not self.prompt.edit_modified():
            return
        self.prompt.edit_modified(False)
        prompt = self.prompt.get("1.0", "end").strip()
        if prompt != self._prompt_snapshot:
            self._prompt_snapshot = prompt
            self._prompt_source_kind = "manual"
            self.prompt_source.set("手動入力・編集")
            self.scene_interpretation.set("")
            self.prompt_revision += 1

    def draft(self):
        scene = self.selected_scene()
        if not scene:
            messagebox.showinfo("シーン", "生成済みのシーンを選択してください。直接入力でも音声を生成できます。")
            return
        try:
            self._replace_prompt(generate_prompt(scene, **self.prompt_options()), source="simple")
            self.prompt_revision += 1
        except ValueError as exc:
            messagebox.showerror("プロンプト", str(exc))

    def draft_quality(self):
        """Use a simple musical control to separate prompt bias from model quality."""
        self._replace_prompt(QUALITY_PRESETS[self.quality_preset.get()], source="quality_control")
        self.prompt_revision += 1
        if self.model.get() != "ace15_turbo":
            self.model.set("medium")
        elif metadata := QUALITY_METADATA.get(self.quality_preset.get()):
            self.ace_bpm.set(str(metadata["bpm"]))
            self.ace_keyscale.set(metadata["keyscale"])
            self.ace_timesignature.set(metadata["timesignature"])
        self._model_changed()
        self.steps.set("8")
        self.seed.set("-1")
        label = "ACE-Step 1.5 Turbo" if self.model.get() == "ace15_turbo" else "Medium"
        self.status.set(f"品質確認用の音楽指定をセットしました。{label}・8ステップ・Seedランダムで比較できます。")

    def draft_ai(self):
        scene = self.selected_scene()
        if not scene:
            messagebox.showinfo("シーン", "生成済みのシーンを選択してください。")
            return
        try:
            if self.process is not None:
                raise ValueError("現在の処理が完了してから実行してください。")
            options = self.prompt_options()
            python = self._console_python()
            request = {
                "backend": self.llm_backend.get(), "local_model": self.llm_local_model.get().strip(),
                "base_url": self.llm_url.get().strip(), "model": self.llm_model.get().strip(),
                "scene": {"id": scene.id, "label": scene.label,
                          "context": scene.context, "preview": scene.preview},
                "options": options,
            }
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), "prompt")
            self._start_process(
                [python, "-u", "-X", "utf8", str(Path(__file__).with_name("llm_runner.py")),
                 "--request", str(run_dir / "request.json"), "--output-dir", str(run_dir)],
                run_dir, "prompt", request=request)
            self.prompt_revision += 1
            self.pending_llm = (self.prompt_revision, self.prompt.get("1.0", "end").strip(), options)
            self.llm_run_revision = self.prompt_revision
            self.ai_button.configure(state="disabled")
            self.status.set("LLMから英語の音楽プロンプトを取得中… 完了時にモデルを解放します。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("LLMプロンプト", str(exc))

    def _save_settings(self):
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        value = {name: getattr(self, name).get() for name in
                 ["server", "python", "output", "llm_url", "llm_model"]}
        value.update({name: getattr(self, name).get() for name in ["llm_backend", "llm_local_model"]})
        value["auto_scene_prompt"] = self.auto_scene_prompt.get()
        value["ace_planner_enabled"] = self.ace_planner_enabled.get()
        value["ace_planner_path"] = self.ace_planner_path.get()
        value["ace_continuous"] = self.ace_continuous.get()
        value.update(self._save_options())
        value.update({f"path_{key}": variable.get() for key, variable in self.paths.items()})
        SETTINGS.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def _runtime_python(self):
        path = Path(self.python.get().strip()).expanduser().resolve()
        if not path.is_file():
            raise ValueError("音声生成用Pythonがありません。「環境・モデル準備」で①実行環境を準備してください。")
        return str(path)

    @staticmethod
    def _console_python():
        python = Path(sys.executable)
        if python.name.casefold() == "pythonw.exe":
            python = python.with_name("python.exe")
        if not python.is_file():
            raise ValueError("LLM・環境準備用のPythonが見つかりません。")
        return str(python)

    def _llm_release_request(self):
        backend = self.llm_backend.get()
        base_url = self.llm_url.get().strip()
        try:
            parsed = urlsplit(base_url)
            known_ollama = (parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                            and parsed.port == 11434)
        except ValueError:
            known_ollama = False
        if backend == "ollama" or (backend == "llama_cpp" and known_ollama):
            return {"base_url": base_url, "model": self.llm_model.get().strip()}
        return None

    def _start_process(self, command, run_dir, kind, request=None):
        if self.process is not None:
            raise ValueError("現在の処理が完了してから実行してください。")
        if kind != "prompt" and self.pending_llm is not None:
            raise ValueError("LLMの結果が画面へ反映されるまでお待ちください。")
        run_dir.mkdir(parents=True, exist_ok=False)
        if request is not None:
            (run_dir / "request.json").write_text(
                json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        self._save_settings()
        self._log_stream = (run_dir / "process.log").open("wb")
        env = hub_environment(RUNTIME / "cache/huggingface", os.environ)
        env["PYTHONUTF8"] = "1"
        try:
            self._process_job = WindowsChildJob()
            self.process = subprocess.Popen(command, cwd=ROOT, env=env,
                                            stdin=subprocess.DEVNULL,
                                            stdout=self._log_stream, stderr=subprocess.STDOUT,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self._process_job.assign(self.process)
        except Exception:
            if self._process_job is not None:
                self._process_job.close()
                self._process_job = None
            if self.process is not None:
                self.process.kill()
                self.process.wait(timeout=10)
                self.process = None
            self._log_stream.close()
            self._log_stream = None
            raise
        self.run_dir, self.run_kind = run_dir, kind
        if kind in {"generate", "loop"}:
            self.stop_playback()
            self.history.selection_remove(*self.history.selection())
            self._timeline_key = None
            self.timeline.reset()
            self.playback_label.set("新しい音源を作成しています。完了後に候補を選択してください。")
            self.loop_points.set("新しい結果ができるまで試聴する音源は選択されていません。")
        self.started_at, self.stop_at = time.monotonic(), None
        for button in [self.generate_button, self.prepare_button, self.planner_prepare_button, self.setup_button, self.auth_button,
                       self.ai_button, self.loop_button]:
            button.configure(state="disabled")
        self.stop_button.configure(state="normal" if kind in {"generate", "prepare", "prompt", "loop"} else "disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)

    def generate(self):
        try:
            python = self._runtime_python()
            scene = self.selected_scene()
            prompt = self.prompt.get("1.0", "end").strip()
            context = dict(scene.context) if scene else {}
            source = self._prompt_source_kind if prompt == self._prompt_snapshot else "manual"
            context["prompt_generation"] = {"method": source,
                                            "scene_interpretation": self.scene_interpretation.get()
                                            if source == "llm" else ""}
            quality_title = next((title for title, text in QUALITY_PRESETS.items() if prompt == text), None)
            if quality_title:
                context["evaluation"] = {"kind": "quality_control", "preset": quality_title,
                                         "scene_independent": True}
                context["scene_label"] = "品質確認: " + quality_title
            data = {"model": self.model.get(), "model_path": self.paths[self.model.get()].get().strip(),
                        "prompt": prompt, "duration": float(self.duration.get()),
                        "steps": int(self.steps.get()), "seed": int(self.seed.get()), "device": self.device.get(),
                        "dtype": self.dtype.get(), "cpu_offload": self.offload.get(),
                        "context": context, **self._save_options()}
            if self.model.get() == "ace15_turbo":
                data.update(normalize_ace_metadata({"bpm": self._ace_generation_bpm(),
                    "keyscale": self.ace_keyscale.get(), "timesignature": self.ace_timesignature.get()},
                    allow_unspecified=True))
                context["prompt_generation"]["music_backend"] = "ace_step15"
                data.update(ace_planner=self.ace_planner_enabled.get(),
                            planner_model_path=self.ace_planner_path.get().strip(),
                            ace_continuous=self.ace_continuous.get())
            release_request = self._llm_release_request()
            if release_request:
                data["llm_release"] = release_request
            validate_request(data)
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve())
            request_file = run_dir / "request.json"
            self._start_process([python, "-u", "-X", "utf8", str(Path(__file__).with_name("runner.py")),
                                 "--request", str(request_file), "--output-dir", str(run_dir)],
                                run_dir, "generate", request=data)
            self.status.set("モデルを読み込み中… 初回は時間がかかります。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("生成開始", str(exc))

    def prepare_model(self):
        try:
            python = self._runtime_python()
            key = self.model.get()
            target = Path(self.paths[key].get().strip()).expanduser().resolve()
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), "prepare")
            command = [python, "-u", "-X", "utf8", str(Path(__file__).with_name("prepare.py")),
                       "--model", key, "--output-dir", str(target),
                       "--cache-dir", str(RUNTIME / "cache"), "--status-dir", str(run_dir)]
            if key == "ace15_turbo":
                command.extend(["--dtype", "bfloat16" if self.dtype.get() == "auto" else self.dtype.get()])
            self._start_process(command, run_dir, "prepare")
            self.status.set(f"{MODEL_SPECS[key]['label']}の取得・Diffusers形式への変換中…")
        except (OSError, ValueError) as exc:
            messagebox.showerror("モデル準備", str(exc))

    def setup_runtime(self):
        try:
            python = self._console_python()
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), "setup")
            self._start_process([python, "-u", "-X", "utf8",
                                 str(Path(__file__).with_name("setup_runtime.py")),
                                 "--status-dir", str(run_dir)],
                                run_dir, "setup")
            self.status.set("音声生成専用のPython・Diffusers環境を準備中… ログで進捗を確認できます。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("環境準備", str(exc))

    def prepare_planner(self):
        try:
            python = self._runtime_python()
            target = Path(self.ace_planner_path.get().strip()).expanduser().resolve()
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), "prepare-planner")
            command = [python, "-u", "-X", "utf8", str(Path(__file__).with_name("ace_planner_prepare.py")),
                       "--output-dir", str(target), "--cache-dir", str(RUNTIME / "cache"),
                       "--status-dir", str(run_dir)]
            self._start_process(command, run_dir, "prepare")
            self.status.set("ACE専用planner 1.7Bを取得しています（約3.8GB）。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("planner準備", str(exc))

    def open_model_terms(self):
        webbrowser.open("https://huggingface.co/" + MODEL_SPECS[self.model.get()]["repo_id"])

    def check_hub_access(self):
        try:
            python = self._runtime_python()
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), "access")
            self._start_process([python, "-u", "-X", "utf8", str(Path(__file__).with_name("hub_auth.py")),
                                 "--model", self.model.get(), "--status-dir", str(run_dir)], run_dir, "access")
            self.status.set("Hugging Faceのログインとモデルへのアクセスを確認しています。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("アクセス確認", str(exc))

    def stop(self):
        if self.process is None or self.run_kind not in {"generate", "prepare", "prompt", "loop"}:
            return
        (self.run_dir / "stop.request").touch()
        self.stop_at = time.monotonic()
        self.stop_button.configure(state="disabled")
        self.status.set("停止処理中… GPUメモリを解放します。")

    def poll(self):
        while not self.events.empty():
            kind, revision, data, error = self.events.get_nowait()
            if kind == "prompt":
                pending = self.pending_llm
                if pending is None or pending[0] != revision:
                    continue
                self.pending_llm = None
                self.ai_button.configure(state="normal")
                if revision != self.prompt_revision:
                    continue
                try:
                    edited = (self.prompt.get("1.0", "end").strip() != pending[1]
                              or self.prompt_options() != pending[2])
                except ValueError:
                    edited = True
                if edited:
                    self.status.set("入力が変更されたため、LLMの結果は反映しませんでした。")
                    continue
                if error:
                    self.status.set("LLMプロンプト作成に失敗しました。")
                    messagebox.showerror("LLMプロンプト", error)
                else:
                    prompt = data.get("prompt", "") if isinstance(data, dict) else data
                    interpretation = data.get("scene_interpretation", "") if isinstance(data, dict) else ""
                    if self.model.get() == "ace15_turbo":
                        try:
                            metadata = normalize_ace_metadata(data.get("ace_metadata"))
                        except (TypeError, ValueError) as exc:
                            self.status.set("LLMの音楽設定を読み込めませんでした。")
                            messagebox.showerror("LLMプロンプト", str(exc))
                            continue
                        self.ace_bpm.set(str(metadata["bpm"]))
                        self.ace_keyscale.set(metadata["keyscale"])
                        self.ace_timesignature.set(metadata["timesignature"])
                    self._replace_prompt(prompt, source="llm", interpretation=interpretation)
                    self.status.set("LLMのプロンプトを作成しました。内容を確認して生成してください。")
            elif revision != self.catalog_revision:
                continue
            elif error:
                self.catalog_status.set("シーン取得に失敗: " + error)
            elif kind == "projects":
                self.projects = data
                self.project_box["values"] = [f"{p.title} [{p.id[:8]}]" for p in data]
                self.catalog_status.set(f"{len(data)}作品。作品を選択してください。")
                if data:
                    self.project_box.current(0)
                    self.project_changed()
            elif kind == "chapters":
                self.chapters = data
                self.chapter_box["values"] = [f"第{c.number}章・{c.title}" for c in data]
                self.catalog_status.set(f"{len(data)}章を取得しました。")
                if data:
                    self.chapter_box.current(0)
                    self.chapter_changed()
        if self.process is not None:
            state = read_json(self.run_dir / "status.json")
            elapsed = time.monotonic() - self.started_at
            if self.stop_at is None:
                self.status.set(f"{state.get('message', '処理中…')} / {elapsed:.0f}秒経過")
                if state.get("phase") == "generating" and state.get("steps"):
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=state["steps"], value=state.get("step", 0))
                elif str(self.progress["mode"]) != "indeterminate":
                    self.progress.configure(mode="indeterminate")
                    self.progress.start(12)
            elif (self.run_kind in {"generate", "prompt", "loop"} and time.monotonic() - self.stop_at > 5
                  and self.process.poll() is None):
                if self._process_job is not None:
                    self._process_job.close()
                    self._process_job = None
                self.process.terminate()
            code = self.process.poll()
            if code is not None:
                self._finish_process(code)
        if self.close_pending and self.process is None:
            self.close()
            return
        self._poll_playback_status()
        if self.playback_process is not None and self.playback_process.poll() is not None:
            playback = self.playback_process
            self.playback_process = None
            if self._playback_job is not None:
                self._playback_job.close()
                self._playback_job = None
            error = playback.stderr.read().decode("utf-8", errors="replace") if playback.stderr else ""
            if playback.stderr:
                playback.stderr.close()
            self._cleanup_playback_session()
            self.timeline.set_seeking_enabled(True)
            self._playback_key = None
            self._playback_mode = None
            self._update_loop_display()
            if playback.returncode and not self.close_pending:
                messagebox.showerror("試聴", error.strip() or f"終了コード: {playback.returncode}")
        self.root.after(150, self.poll)

    def _finish_process(self, code):
        result = read_json(self.run_dir / "result.json")
        self._log_stream.close()
        self._log_stream = None
        self.process = None
        if self._process_job is not None:
            self._process_job.close()
            self._process_job = None
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        for button in [self.generate_button, self.prepare_button, self.planner_prepare_button, self.setup_button, self.auth_button,
                       self.ai_button, self.loop_button]:
            button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self.run_kind == "prompt":
            revision = self.llm_run_revision
            self.llm_run_revision = None
            if self.stop_at is not None or result.get("cancelled"):
                if self.pending_llm and self.pending_llm[0] == revision:
                    self.pending_llm = None
                self.status.set("LLMプロンプト作成を停止しました。")
            else:
                prompt = result.get("prompt")
                error = None
                if code != 0 or not result.get("ok") or not isinstance(prompt, str) or not prompt.strip():
                    error = (result.get("error") or read_json(self.run_dir / "status.json").get("message")
                             or f"終了コード: {code}")
                    error += f"\n\nログ: {self.run_dir / 'process.log'}"
                self.events.put(("prompt", revision, {"prompt": prompt,
                                  "scene_interpretation": result.get("scene_interpretation", ""),
                                  "ace_metadata": result.get("ace_metadata")}, error))
            return
        if self.stop_at is not None or result.get("cancelled"):
            self.status.set("生成を停止しました。")
        elif code == 0 and (self.run_kind == "setup" or result.get("ok")):
            self.status.set("生成完了。結果を選んで試聴できます。" if self.run_kind in {"generate", "loop"} else "準備完了。")
            if self.run_kind == "generate":
                self._add_result(self.run_dir, result)
            elif self.run_kind == "loop":
                for candidate in result.get("candidates") or []:
                    directory = Path(candidate.get("directory") or Path(candidate["audio_path"]).parent)
                    self._add_result(directory, read_json(directory / "result.json"))
                self.loop_hint.set(result.get("message") or "候補を結果一覧へ追加しました。継ぎ目を試聴して比較してください。")
            elif self.run_kind == "access":
                self.status.set(result.get("message", "モデルへのアクセスを確認しました。"))
        else:
            self.status.set("処理に失敗しました。ログを確認してください。")
            error = result.get("error") or read_json(self.run_dir / "status.json").get("message")
            messagebox.showerror("音声生成", (error or f"終了コード: {code}")
                                 + f"\n\nログ: {self.run_dir / 'process.log'}")

    def _add_result(self, directory, result):
        key = str(directory)
        if key in self.results:
            return
        metadata = result.get("metadata") or read_json(directory / "generation.json")
        settings = metadata.get("settings", {})
        context = metadata.get("context", {})
        self.results[key] = (directory, result, metadata)
        scene_name = context.get("scene_label") or context.get("scene_id") or directory.name
        model_label = settings.get("model", "?")
        if model_label == "ace15_turbo":
            model_label = "ACE + planner" if metadata.get("ace_step", {}).get("planner_enabled") else "ACE (plannerなし)"
        self.history.insert("", "end", iid=key, values=(model_label,
                            f"{metadata.get('duration_seconds', 0):.1f}", settings.get("seed", "?"),
                            f"{metadata.get('elapsed_seconds', 0):.1f}秒", scene_name))
        self.history.selection_set(key)
        self.history.see(key)
        self.result_selected()

    def load_history(self):
        parent = Path(self.output.get().strip()).expanduser()
        count = 0
        paths = list(parent.glob("*/result.json")) + list(parent.glob("*/candidate-*/result.json"))
        for path in sorted(paths):
            if path.parent.name.startswith("candidate-"):
                experiment = read_json(path.parent.parent / "result.json")
                if not experiment.get("ok") or not experiment.get("candidates"):
                    continue
            result = read_json(path)
            if result.get("ok") and not result.get("candidates") and self._result_audio_path(
                    path.parent, result, result.get("metadata") or {}).is_file():
                self._add_result(path.parent.resolve(), result)
                count += 1
        self.status.set(f"保存先の生成結果{count}件を読み込みました。")

    def selected_result(self):
        selection = self.history.selection()
        return self.results.get(selection[0]) if selection else None

    def result_selected(self, _event=None):
        selected = self.selected_result()
        if not selected:
            self.stop_playback()
            self._timeline_key = None
            self.timeline.reset()
            return
        directory, _result, m = selected
        key = str(directory)
        changed = self._timeline_key != key
        if self._playback_key is not None and self._playback_key != key:
            self.stop_playback()
        review = read_json(directory / "review.json")
        self.rating.set(str(review.get("rating") or "未評価"))
        self.note.set(review.get("note", ""))
        warning = " / 無音の可能性" if m.get("silent") else ""
        preview_gain = m.get("preview_gain", 1.0)
        if preview_gain < 1.0:
            warning += f" / 試聴版は音量を{100 * preview_gain:.1f}%へ減衰"
        if m.get("loop"):
            info = m["loop"]
            warning += " / ループ: " + LOOP_METHODS.get(info.get("method"), str(info.get("method", "?")))
            if info.get("warnings"):
                warning += " / 要確認: " + "・".join(map(str, info["warnings"]))
        audio_path = self._result_audio_path(directory, _result, m)
        if audio_path.is_file():
            warning += f" / {audio_path.suffix[1:].upper()} {audio_path.stat().st_size / 1024**2:.2f}MiB"
        silence = m.get("silence_analysis", {})
        if silence.get("count"):
            warning += (f" / ほぼ無音 {silence['count']}区間・計{silence['total_seconds']:.1f}秒"
                        f"（最長{silence['longest_seconds']:.1f}秒、−50dBFS以下）")
        self.metrics.set(f"{m.get('sample_rate', 0)}Hz / {m.get('channels', 0)}ch / "
                         f"ピーク {m.get('peak', 0):.3f} / RMS {m.get('rms', 0):.3f} / "
                         f"範囲超過 {100 * m.get('clipping_fraction', 0):.2f}%{warning}")
        if changed:
            self.loop.set(bool(m.get("loop")))
            self.timeline.set_position(0)
        self._timeline_key = key
        title = f"{directory.parent.name}/{directory.name}" if directory.name.startswith("candidate-") else directory.name
        if m.get("source_audio"):
            title += " / 元曲: " + Path(m["source_audio"]).parent.name
        self.playback_label.set("選択音源: " + title)
        try:
            plan = loop_playback_plan(m)
            duration = plan.end_sample / 44100 if plan is not None else float(m.get("duration_seconds", 0))
            info = m.get("loop") or {}
            edit_start = info.get("edited_file_start_sample")
            edit_end = info.get("edited_file_end_sample")
            if changed:
                self.timeline.set_audio(duration,
                    loop_start_seconds=plan.start_sample / 44100 if plan else None,
                    loop_end_seconds=plan.end_sample / 44100 if plan else None,
                    edit_start_seconds=edit_start / 44100 if isinstance(edit_start, (int, float)) else None,
                    edit_end_seconds=edit_end / 44100 if isinstance(edit_end, (int, float)) else None)
            self.timeline.set_enabled(duration > 0)
            self.timeline.set_seeking_enabled(self._playback_mode != "seam_preview")
        except (OSError, TypeError, ValueError):
            self.timeline.reset()
        self._update_loop_display()

    def _update_loop_display(self):
        selected = self.selected_result()
        if not selected:
            self.loop_points.set("曲を選ぶと再生範囲とループの戻り先を表示します。")
            return
        try:
            plan = loop_playback_plan(selected[2])
            if plan is None:
                duration = float(selected[2].get("duration_seconds", 0))
                message = f"再生: 0.00 → {duration:.2f}秒"
                message += " / 戻り先: 0.00秒（曲全体を繰り返す）" if self.loop.get() else " / 終端で停止"
            else:
                message = plan.description
                if not self.loop.get():
                    message += " / 今回は初回だけ再生して停止"
                if plan.warning:
                    message += "\n" + plan.warning
                if plan.restored_intro:
                    message += "\n旧音源の冒頭を試聴時に補完しています。保存ファイルは変更していません。"
                info = selected[2].get("loop") or {}
                edit_start = info.get("edited_file_start_sample")
                edit_end = info.get("edited_file_end_sample")
                if isinstance(edit_start, (int, float)) and isinstance(edit_end, (int, float)):
                    message += f" / AI修復 {edit_start / 44100:.2f} → {edit_end / 44100:.2f}秒"
            if self._playback_mode == "seam_preview":
                message = "継ぎ目だけ試聴中: Bの手前 → Aの直後（各最大3秒・3回） / シーク無効\n" + message
            self.loop_points.set(message)
        except (OSError, TypeError, ValueError) as exc:
            self.loop_points.set("ループ範囲を読み込めません: " + str(exc))

    def _loop_mode_changed(self):
        if (self._playback_mode == "normal" and self.playback_process is not None
                and self.playback_process.poll() is None):
            self.play(start_seconds=self.timeline._position)
        else:
            self._update_loop_display()

    @staticmethod
    def _result_audio_path(directory, result, metadata):
        for value in (result.get("audio_path"), metadata.get("audio_path"),
                      directory / "output.mp3", directory / "output.wav"):
            if value and Path(value).is_file():
                return Path(value)
        return directory / "output.mp3"

    def play(self, *, seam_preview=False, start_seconds=0.0):
        selected = self.selected_result()
        if not selected:
            return
        try:
            self.stop_playback()
            path = self._result_audio_path(*selected)
            command = [self._runtime_python(), "-u", "-X", "utf8", str(Path(__file__).with_name("player.py")),
                       "--input", str(path)]
            plan = loop_playback_plan(selected[2])
            if plan is not None:
                command.extend(plan.cli_options)
            if seam_preview:
                command.append("--seam-preview")
            elif self.loop.get():
                command.append("--loop")
            self._playback_session = PLAYBACK_ROOT / uuid.uuid4().hex
            self._playback_session.mkdir(parents=True, exist_ok=False)
            self._playback_status_path = self._playback_session / "status.json"
            self._playback_control_path = self._playback_session / "control.json"
            self._playback_sequence = 0
            self._playback_key = str(selected[0])
            self._playback_mode = "seam_preview" if seam_preview else "normal"
            start_sample = max(0, round(float(start_seconds) * 44100))
            command.extend(["--status-file", str(self._playback_status_path),
                            "--control-file", str(self._playback_control_path),
                            "--start-sample", str(start_sample)])
            self.timeline.set_position(start_seconds)
            self.timeline.set_seeking_enabled(not seam_preview)
            self._update_loop_display()
            self._playback_job = WindowsChildJob()
            self.playback_process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self._playback_job.assign(self.playback_process)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self.stop_playback()
            messagebox.showerror("試聴", str(exc))

    def stop_playback(self):
        if self._playback_job is not None:
            self._playback_job.close()
            self._playback_job = None
        if self.playback_process is None:
            self._cleanup_playback_session()
            self._playback_key = None
            self._playback_mode = None
            self.timeline.set_seeking_enabled(True)
            self._update_loop_display()
            return
        playback, self.playback_process = self.playback_process, None
        if playback.poll() is None:
            playback.terminate()
            try:
                playback.wait(timeout=3)
            except subprocess.TimeoutExpired:
                playback.kill()
                playback.wait(timeout=3)
        if playback.stderr:
            playback.stderr.close()
        self._cleanup_playback_session()
        self._playback_key = None
        self._playback_mode = None
        self.timeline.set_seeking_enabled(True)
        self._update_loop_display()

    def _cleanup_playback_session(self):
        session, self._playback_session = self._playback_session, None
        self._playback_status_path = None
        self._playback_control_path = None
        if session is not None:
            for filename in ("status.json", "control.json"):
                try:
                    (session / filename).unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                session.rmdir()
            except OSError:
                pass

    def _poll_playback_status(self):
        if self._playback_status_path is None:
            return
        state = read_json(self._playback_status_path)
        if not state:
            return
        if state.get("state") == "loading":
            return
        if self._playback_key != self._timeline_key:
            return
        # Keep the dragged cursor until the player has consumed the latest seek.
        if self._playback_mode == "normal" and state.get("command_sequence", 0) < self._playback_sequence:
            return
        position = state.get("position_sample")
        rate = state.get("sample_rate", 44100)
        if isinstance(position, (int, float)) and isinstance(rate, (int, float)) and rate > 0:
            self.timeline.set_position(position / rate)

    def _seek_playback(self, seconds):
        selected = self.selected_result()
        if not selected or self._playback_mode == "seam_preview":
            return
        try:
            plan = loop_playback_plan(selected[2])
            end = plan.end_sample if plan else round(float(selected[2].get("duration_seconds", 0)) * 44100)
            if end <= 0:
                return
            sample = min(max(0, round(float(seconds) * 44100)), end - 1)
            self.timeline.set_position(sample / 44100)
            if (self.playback_process is not None and self.playback_process.poll() is None
                    and self._playback_key == str(selected[0])):
                self._playback_sequence += 1
                write_json(self._playback_control_path, {"sequence": self._playback_sequence, "seek_sample": sample})
            else:
                self.play(start_seconds=sample / 44100)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("シーク", str(exc))

    def open_output(self):
        selected = self.selected_result()
        directory = selected[0] if selected else Path(self.output.get().strip())
        if directory.is_dir() and os.name == "nt":
            os.startfile(str(directory))

    def open_log(self):
        if self.run_dir and (self.run_dir / "process.log").is_file() and os.name == "nt":
            subprocess.Popen(["notepad.exe", str(self.run_dir / "process.log")])

    def save_review(self):
        selected = self.selected_result()
        if not selected:
            return
        try:
            value = {"rating": None if self.rating.get() == "未評価" else int(self.rating.get()),
                         "note": self.note.get().strip(), "updated_at": datetime.now().astimezone().isoformat()}
            (selected[0] / "review.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            self.status.set("試聴評価を保存しました。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("評価保存", str(exc))

    def close(self):
        if self.process is not None:
            if self.run_kind == "setup":
                self.close_pending = True
                self.status.set("実行環境の準備が完了すると、このウィンドウを閉じます。")
                return
            if self.run_kind in {"generate", "prepare", "prompt", "loop"}:
                self.close_pending = True
                self.stop()
                return
            self.stop()
            if self._process_job is not None:
                self._process_job.close()
                self._process_job = None
            self.process.terminate()
            self.process.wait(timeout=10)
            self._log_stream.close()
        self.stop_playback()
        self._save_settings()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Stable Audio 3 BGMテストGUI")
    parser.add_argument("--smoke-test", action="store_true", help="GUIを非表示で構築・検証して終了")
    args = parser.parse_args()
    root = Tk()
    if args.smoke_test:
        root.withdraw()
    app = AudioTestApp(root, auto_refresh=not args.smoke_test)
    if args.smoke_test:
        root.update_idletasks()
        assert app.model.get() == "medium"
        assert app.prompt.get("1.0", "end").strip()
        root.destroy()
        print("BGM GUI smoke test passed")
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
