"""Standalone Windows workbench for Qwen Image 2.1 reference-image experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, StringVar, Text, Tk, Toplevel, filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.image_edit.catalog import CoordinatorImageCatalog
from scripts.image_edit.cg_tab import CG_TRANSITIONS, CgProposalTab
from scripts.image_edit.common import (
    MODEL_DIR,
    MODEL_ID,
    RUNTIME,
    SETTINGS,
    hub_environment,
    new_run,
    read_json,
    write_json,
)
from scripts.image_edit.engine import validate_request
from scripts.image_edit.prompts import PORTRAIT_PRESETS, build_portrait_prompt, build_scene_prompt

EDIT_PRESETS = PORTRAIT_PRESETS
RATING_FIELDS = {
    "face": "顔・目の一致",
    "hair": "髪の一致",
    "body": "体格の一致",
    "clothing": "衣装の保持",
    "instruction": "指示への適合",
    "separation": "人物の混同なし",
    "art_style": "画風の一致",
    "anatomy": "手足・人数の自然さ",
    "composition": "構図の適合",
    "impact": "印象の強さ",
}
MODES = ("portrait", "scene")
# (reference resolution, KV cache, text encoder layer by layer, 8-bit transformer, tiled
# decode); measured with three references at 1536x1024. See docs/setup/qwen-image-edit.md.
VRAM_PRESETS = {
    "32GB: 参照1024・再利用あり（標準）": ("1024", True, True, False, False),
    "24GB: 参照768・再利用あり": ("768", True, True, False, False),
    "24GB: 参照1024・再利用なし（約2倍の時間）": ("1024", False, True, False, False),
    "24GB: 8ビット・参照1024・再利用あり": ("1024", True, True, True, True),
    "16GB: 8ビット・参照768・再利用あり": ("768", True, True, True, True),
    "16GB: 8ビット・参照1024・再利用なし（約2倍の時間）": ("1024", False, True, True, True),
    "12GB: 8ビット・参照768・再利用なし": ("768", False, True, True, True),
}
TRANSPARENCY_INSTRUCTION = (
    "This is an RGBA image with transparency. The image has an alpha channel "
    "and the background is transparent."
)


def _canonical_path(value: str) -> str:
    return os.path.normcase(str(Path(value).expanduser().resolve()))


def reference_key(reference: dict) -> tuple:
    """Recognize a selected source even if a catalog download has another filename."""
    source = reference.get("source") or {}
    if source.get("artifact_id"):
        return ("artifact", source.get("artifact_id"), reference.get("character_id"))
    if source.get("image_url"):
        return ("url", source["image_url"])
    if reference.get("sha256"):
        return ("sha256", reference["sha256"])
    return ("path", _canonical_path(reference["path"]))


def checker_preview(path: Path, size: tuple[int, int]):
    """Render RGBA on a checkerboard without changing the saved image."""
    from PIL import Image, ImageDraw

    with Image.open(path) as original:
        picture = original.convert("RGBA")
    picture.thumbnail(size, Image.Resampling.LANCZOS)
    background = Image.new("RGBA", picture.size, "#eeeeee")
    draw = ImageDraw.Draw(background)
    for y in range(0, picture.height, 16):
        for x in range(0, picture.width, 16):
            if (x // 16 + y // 16) % 2:
                draw.rectangle((x, y, x + 15, y + 15), fill="#cccccc")
    return Image.alpha_composite(background, picture).convert("RGB")


class ImageEditTestApp(CgProposalTab):
    def __init__(self, root: Tk, *, auto_refresh: bool = True):
        self.root = root
        root.title("Qwen Image 2.1 • キャラクター画像テスト")
        root.geometry("1240x940")
        root.minsize(1060, 790)
        self.events = queue.SimpleQueue()
        self.catalog_revision = 0
        self.catalog = None
        self.reference_revision = {mode: 0 for mode in MODES}
        self.managed_prompt_segments = {mode: {} for mode in MODES}
        self.references = {mode: [] for mode in MODES}
        self.projects, self.portraits, self.chapters, self.scenes = [], [], [], []
        self.process = None
        self.run_dir = None
        self.run_kind = None
        self.started_at = 0.0
        self.stop_at = None
        self.terminate_at = None
        self.close_pending = False
        self._log_stream = None
        self.results = {}
        self.photos = {}
        self.pending_llm = None
        self.scene_prompt_details = None
        saved = read_json(SETTINGS)
        self.server = StringVar(value=saved.get("server", "http://127.0.0.1:8000"))
        self.python = StringVar(
            value=saved.get("python", str(RUNTIME / ".venv/Scripts/python.exe"))
        )
        self.model_path = StringVar(value=saved.get("model_path", str(MODEL_DIR)))
        self.output = StringVar(value=saved.get("output", str(ROOT / "outputs/image-edit-tests")))
        self.steps = StringVar(value="40")
        self.seed = StringVar(value="-1")
        self.three_seeds = BooleanVar(value=False)
        self.cpu_offload = BooleanVar(value=True)
        self.use_kv_cache = BooleanVar(value=True)
        self.dtype = StringVar(value="bfloat16")
        self.reference_resolution = StringVar(value="1024")
        self.text_encoder_layers = BooleanVar(value=True)
        self.transformer_fp8 = BooleanVar(value=False)
        self.vae_tiling = BooleanVar(value=False)
        self.vram_preset = StringVar(value=next(iter(VRAM_PRESETS)))
        self.widths = {"portrait": StringVar(value="768"), "scene": StringVar(value="1024")}
        self.heights = {"portrait": StringVar(value="1152"), "scene": StringVar(value="576")}
        self.transparent = {"portrait": BooleanVar(value=True), "scene": BooleanVar(value=False)}
        self.reference_names = {mode: StringVar() for mode in MODES}
        self.reference_lists, self.prompts = {}, {}
        self.status = StringVar(value="参照画像を追加し、変更内容を入力して生成してください。")
        self.catalog_status = StringVar(
            value="ローカル画像だけでも試せます。作品一覧は「更新」で取得します。"
        )
        self.metrics = StringVar(value="生成結果を選ぶと、条件と時間を表示します。")
        self.edit_preset = StringVar(value=next(iter(EDIT_PRESETS)))
        self.scene_instruction = StringVar()
        self.scene_interpretation = StringVar(value="LLMが選んだ一瞬と構図を、ここに表示します。")
        self.llm_backend = StringVar(value=saved.get("llm_backend", "llama_cpp"))
        if self.llm_backend.get() not in {"llama_cpp", "ollama", "openai"}:
            self.llm_backend.set("llama_cpp")
        self.llm_local_model = StringVar(value=saved.get("llm_local_model", ""))
        self.llm_url = StringVar(value=saved.get("llm_url", "http://127.0.0.1:8080/v1"))
        self.llm_model = StringVar(value=saved.get("llm_model", "local-model"))
        self.llm_hint = StringVar()
        self.review_target = StringVar(value="全体")
        self.review_note = StringVar()
        self.ratings = {key: StringVar(value="未評価") for key in RATING_FIELDS}
        self._cg_init()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(150, self.poll)
        if auto_refresh:
            self.root.after(200, self.refresh_catalog)

    def _build(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="Qwen Image 2.1 / キャラクターの一貫性を比較",
            font=("Yu Gothic UI", 16, "bold"),
        ).pack(anchor="w")
        catalog = ttk.LabelFrame(outer, text="作品から立ち絵・場面を選ぶ", padding=8)
        catalog.pack(fill="x", pady=8)
        ttk.Label(catalog, text="作品").grid(row=0, column=0)
        self.project_box = ttk.Combobox(catalog, state="readonly", width=30)
        self.project_box.grid(row=0, column=1, padx=6)
        self.project_box.bind("<<ComboboxSelected>>", lambda _: self.project_changed())
        ttk.Button(catalog, text="更新", command=self.refresh_catalog).grid(row=0, column=2)
        ttk.Label(catalog, text="公開版の立ち絵").grid(row=0, column=3, padx=(15, 0))
        self.portrait_box = ttk.Combobox(catalog, state="readonly", width=32)
        self.portrait_box.grid(row=0, column=4, padx=6)
        ttk.Button(catalog, text="現在のタブへ追加", command=self.add_catalog_reference).grid(
            row=0, column=5
        )
        ttk.Label(catalog, textvariable=self.catalog_status, wraplength=1100).grid(
            row=1, column=0, columnspan=6, sticky="w", pady=(5, 0)
        )
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.mode_frames = {}
        for mode, title in (("portrait", "立ち絵の差分"), ("scene", "場面の一枚絵")):
            frame = ttk.Frame(self.tabs, padding=10)
            self.tabs.add(frame, text=title)
            self.mode_frames[mode] = frame
            self._generation_tab(frame, mode)
        self.cg_frame = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(self.cg_frame, text="印象的な一枚（LLM任せ）")
        self._cg_tab(self.cg_frame)
        self.comparison_tab = ttk.Frame(self.tabs, padding=10)
        self.environment_tab = ttk.Frame(self.tabs, padding=12)
        self.llm_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.comparison_tab, text="比較・評価・履歴")
        self.tabs.add(self.llm_tab, text="LLM設定")
        self.tabs.add(self.environment_tab, text="環境・モデル準備")
        self._comparison_tab(self.comparison_tab)
        self._llm_settings_tab(self.llm_tab)
        self._environment_tab(self.environment_tab)
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(8, 4))
        ttk.Label(controls, text="Steps").pack(side="left")
        ttk.Entry(controls, textvariable=self.steps, width=5).pack(side="left", padx=5)
        ttk.Label(controls, text="Seed（-1: ランダム）").pack(side="left")
        ttk.Entry(controls, textvariable=self.seed, width=12).pack(side="left", padx=5)
        ttk.Checkbutton(controls, text="Seed 0・1・2で比較", variable=self.three_seeds).pack(
            side="left", padx=8
        )
        self.generate_button = ttk.Button(controls, text="現在のタブを生成", command=self.generate)
        self.generate_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(controls, text="停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        self.progress = ttk.Progressbar(controls, mode="indeterminate", length=150)
        self.progress.pack(side="right")
        ttk.Label(outer, textvariable=self.status, wraplength=1160).pack(anchor="w")

    def _generation_tab(self, frame, mode):
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(0, weight=1)
        left = ttk.LabelFrame(frame, text="参照画像（上から画像1・画像2…）", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        listing = ttk.Treeview(
            left, columns=("name",), show="headings", height=5, selectmode="browse"
        )
        listing.heading("name", text="キャラクター名 / 参照順")
        listing.column("name", width=240)
        listing.grid(row=0, column=0, sticky="nsew")
        listing.bind(
            "<<TreeviewSelect>>",
            lambda _, selected_mode=mode: self.reference_selected(selected_mode),
        )
        self.reference_lists[mode] = listing
        actions = ttk.Frame(left)
        actions.grid(row=1, column=0, sticky="ew", pady=6)
        for title, callback in (
            ("画像を追加", lambda: self.add_local_references(mode)),
            ("↑", lambda: self.move_reference(mode, -1)),
            ("↓", lambda: self.move_reference(mode, 1)),
            ("削除", lambda: self.remove_reference(mode)),
        ):
            ttk.Button(actions, text=title, command=callback).pack(side="left", padx=2)
        names = ttk.Frame(left)
        names.grid(row=2, column=0, sticky="ew")
        names.columnconfigure(0, weight=1)
        ttk.Entry(names, textvariable=self.reference_names[mode]).grid(row=0, column=0, sticky="ew")
        ttk.Button(names, text="名前を反映", command=lambda: self.rename_reference(mode)).grid(
            row=0, column=1, padx=4
        )
        ttk.Label(left, text="差分は参照1枚。一枚絵は1〜10枚。").grid(
            row=3, column=0, sticky="w", pady=4
        )
        preview = ttk.Label(left, text="参照プレビュー", anchor="center")
        preview.grid(row=4, column=0, sticky="nsew")
        self.photos[f"reference_widget_{mode}"] = preview
        ttk.Button(left, text="参照を拡大", command=lambda: self.enlarge_reference(mode)).grid(
            row=5, column=0, sticky="w"
        )
        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        if mode == "portrait":
            presets = ttk.Frame(right)
            presets.grid(row=0, column=0, sticky="ew")
            ttk.Combobox(
                presets,
                textvariable=self.edit_preset,
                values=list(EDIT_PRESETS),
                state="readonly",
                width=24,
            ).pack(side="left")
            ttk.Button(presets, text="テンプレートを入れる", command=self.apply_edit_preset).pack(
                side="left", padx=6
            )
        else:
            scenes = ttk.Frame(right)
            scenes.grid(row=0, column=0, sticky="ew")
            self.chapter_box = ttk.Combobox(scenes, state="readonly", width=22)
            self.chapter_box.pack(side="left")
            self.chapter_box.bind("<<ComboboxSelected>>", lambda _: self.chapter_changed())
            self.scene_box = ttk.Combobox(scenes, state="readonly", width=26)
            self.scene_box.pack(side="left", padx=5)
            self.scene_box.bind("<<ComboboxSelected>>", lambda _: self.scene_changed())
            self.scene_prompt_button = ttk.Button(
                scenes, text="LLMで指示を作成", command=self.create_scene_prompt
            )
            self.scene_prompt_button.pack(side="left")
            ttk.Button(
                right,
                text="参照番号を追加（手入力用・場面の選択不要）",
                command=self.create_reference_prompt,
            ).grid(row=8, column=0, sticky="w", pady=4)
        ttk.Label(
            right,
            text="生成指示（自由に編集できます。参照や場面の変更で自動上書きしません）",
            wraplength=650,
        ).grid(row=1, column=0, sticky="w", pady=6)
        prompt = Text(right, height=10, wrap="word", font=("Yu Gothic UI", 10))
        prompt.grid(row=2, column=0, sticky="nsew")
        self.prompts[mode] = prompt
        settings = ttk.Frame(right)
        settings.grid(row=3, column=0, sticky="ew", pady=8)
        for label, variable in (("幅", self.widths[mode]), ("高さ", self.heights[mode])):
            ttk.Label(settings, text=label).pack(side="left")
            ttk.Entry(settings, textvariable=variable, width=7).pack(side="left", padx=5)
        ttk.Checkbutton(settings, text="透明背景を指示", variable=self.transparent[mode]).pack(
            side="left", padx=10
        )
        ttk.Label(
            right,
            text="透明背景の指示は出力を保証しません。比較画面でアルファと輪郭を確認できます。",
            wraplength=650,
        ).grid(row=4, column=0, sticky="w")
        if mode == "scene":
            source = ttk.LabelFrame(
                right, text="台本の確認（画像モデルには送信しません）", padding=5
            )
            source.grid(row=5, column=0, sticky="ew", pady=(8, 0))
            self.scene_preview = Text(
                source, height=4, wrap="word", state="disabled", font=("Yu Gothic UI", 9)
            )
            self.scene_preview.pack(fill="x")
            hint = ttk.LabelFrame(
                right, text="LLMへの追加指示（任意: 描く瞬間・構図など）", padding=5
            )
            hint.grid(row=6, column=0, sticky="ew", pady=5)
            ttk.Entry(hint, textvariable=self.scene_instruction).pack(fill="x")
            ttk.Label(right, textvariable=self.scene_interpretation, wraplength=650).grid(
                row=7, column=0, sticky="w", pady=3
            )

    def _llm_settings_tab(self, frame):
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="場面を画像生成の指示へ変換するLLM").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=10
        )
        ttk.Label(frame, text="実行方法").grid(row=1, column=0, sticky="w")
        backend = ttk.Combobox(
            frame,
            textvariable=self.llm_backend,
            values=["llama_cpp", "ollama", "openai"],
            state="readonly",
        )
        backend.grid(row=1, column=1, sticky="ew", padx=8, pady=5)
        backend.bind("<<ComboboxSelected>>", lambda _: self._llm_backend_changed())
        ttk.Label(frame, text="ローカルGGUFモデル").grid(row=2, column=0, sticky="w")
        self.llm_local_box = ttk.Combobox(
            frame, textvariable=self.llm_local_model, state="readonly"
        )
        self.llm_local_box.grid(row=2, column=1, sticky="ew", padx=8, pady=5)
        self.llm_reload_button = ttk.Button(frame, text="再読込", command=self.reload_llm_models)
        self.llm_reload_button.grid(row=2, column=2)
        for row, label, variable in (
            (3, "外部LLM API（/v1）", self.llm_url),
            (4, "外部LLMモデル名", self.llm_model),
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(frame, textvariable=variable).grid(
                row=row, column=1, sticky="ew", padx=8, pady=5
            )
        ttk.Label(frame, textvariable=self.llm_hint, wraplength=1000).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=10
        )
        ttk.Label(
            frame,
            text="既定は本体に登録されたGGUFをllama.cppで実行します。台本を読み、"
            "一枚に描く瞬間・人物の動作と表情・背景・構図を英語の指示にまとめます。\n"
            "プロンプト作成後にローカルLLMを終了し、GPUメモリーを解放してから画像を生成できます。"
            "Ollamaも指定モデルを解放します。openaiはOpenAI互換APIの指定です。",
            wraplength=1000,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=10)
        self.reload_llm_models()
        self._llm_backend_changed()

    def reload_llm_models(self):
        try:
            from scripts.audio.llm_runtime import default_native_model, list_native_models

            models = list_native_models(ROOT)
            self.llm_local_box["values"] = models
            if not self.llm_local_model.get():
                self.llm_local_model.set(default_native_model(ROOT))
            self.llm_hint.set(f"本体に登録されたGGUFモデル{len(models)}件を読み込みました。")
        except Exception as exc:  # noqa: BLE001 -- optional registry cannot prevent GUI startup
            self.llm_local_box["values"] = []
            self.llm_hint.set("ローカルLLMを確認できません: " + str(exc))

    def _llm_backend_changed(self):
        native = self.llm_backend.get() == "llama_cpp"
        self.llm_local_box.configure(state="readonly" if native else "disabled")
        self.llm_reload_button.configure(state="normal" if native else "disabled")

    def _comparison_tab(self, frame):
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(0, weight=1)
        history = ttk.Frame(frame)
        history.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        history.rowconfigure(0, weight=1)
        history.columnconfigure(0, weight=1)
        self.history = ttk.Treeview(
            history, columns=("label", "seed"), show="headings", selectmode="browse"
        )
        self.history.heading("label", text="結果 / モード")
        self.history.heading("seed", text="Seed")
        self.history.column("label", width=235)
        self.history.column("seed", width=65)
        self.history.grid(row=0, column=0, sticky="nsew")
        self.history.bind("<<TreeviewSelect>>", lambda _: self.result_selected())
        ttk.Button(history, text="保存先から履歴を読込", command=self.load_history).grid(
            row=1, column=0, sticky="w", pady=6
        )
        previews = ttk.Frame(frame)
        previews.grid(row=0, column=1, sticky="nsew")
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        ttk.Label(previews, text="元の参照").grid(row=0, column=0)
        ttk.Label(previews, text="生成結果（市松は透明）").grid(row=0, column=1)
        self.original_preview = ttk.Label(previews, text="元画像", anchor="center")
        self.result_preview = ttk.Label(previews, text="生成画像", anchor="center")
        self.original_preview.grid(row=1, column=0, sticky="nsew", padx=4)
        self.result_preview.grid(row=1, column=1, sticky="nsew", padx=4)
        self.original_box = ttk.Combobox(previews, state="readonly", width=25)
        self.original_box.grid(row=2, column=0, pady=6)
        self.original_box.bind("<<ComboboxSelected>>", lambda _: self.show_original())
        ttk.Button(previews, text="生成画像を拡大", command=self.enlarge_result).grid(
            row=2, column=1
        )
        actions = ttk.Frame(previews)
        actions.grid(row=3, column=0, columnspan=2, pady=5)
        ttk.Button(
            actions,
            text="差分の基準をこの結果に置換",
            command=lambda: self.use_result_reference("portrait"),
        ).pack(side="left", padx=4)
        ttk.Button(
            actions, text="結果を一枚絵の参照へ", command=lambda: self.use_result_reference("scene")
        ).pack(side="left", padx=4)
        ttk.Button(actions, text="画像を別名保存", command=self.export_result).pack(
            side="left", padx=4
        )
        ttk.Button(actions, text="元の参照を拡大", command=self.enlarge_original).pack(
            side="left", padx=4
        )
        playback = ttk.Frame(previews)
        playback.grid(row=4, column=0, columnspan=2, pady=(0, 5))
        ttk.Button(playback, text="ティラノで場面を再生", command=self.preview_cg).pack(
            side="left", padx=4
        )
        ttk.Label(playback, text="CGの切り替え").pack(side="left", padx=(10, 2))
        ttk.Combobox(playback, textvariable=self.cg_transition, values=list(CG_TRANSITIONS),
                     state="readonly", width=24).pack(side="left")
        ttk.Combobox(playback, textvariable=self.cg_transition_ms,
                     values=["300", "500", "800", "1200"], width=6).pack(side="left", padx=4)
        ttk.Label(playback, text="ミリ秒").pack(side="left")
        ttk.Label(previews, textvariable=self.metrics, wraplength=720).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=6
        )
        review = ttk.LabelFrame(frame, text="一致の評価（1:低い ～ 5:高い）", padding=8)
        review.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(review, text="評価対象").grid(row=0, column=0)
        self.review_target_box = ttk.Combobox(
            review, textvariable=self.review_target, values=["全体"], state="readonly", width=25
        )
        self.review_target_box.grid(row=0, column=1, columnspan=2, sticky="w", padx=6)
        self.review_target_box.bind("<<ComboboxSelected>>", lambda _: self.load_review())
        for index, (key, label) in enumerate(RATING_FIELDS.items()):
            row, col = 1 + index // 3, (index % 3) * 2
            ttk.Label(review, text=label).grid(row=row, column=col, sticky="e", pady=4)
            ttk.Combobox(
                review,
                textvariable=self.ratings[key],
                values=["未評価", "N/A", "1", "2", "3", "4", "5"],
                state="readonly",
                width=8,
            ).grid(row=row, column=col + 1, padx=6)
        ttk.Label(review, text="メモ").grid(row=5, column=0)
        ttk.Entry(review, textvariable=self.review_note).grid(
            row=5, column=1, columnspan=4, sticky="ew", padx=6
        )
        review.columnconfigure(3, weight=1)
        ttk.Button(review, text="この対象の評価を保存", command=self.save_review).grid(
            row=5, column=5
        )

    def _environment_tab(self, frame):
        frame.columnconfigure(1, weight=1)
        for row, (label, variable, file_select) in enumerate(
            (
                ("作品サーバー", self.server, False),
                ("専用Python", self.python, True),
                ("モデル保存先", self.model_path, False),
                ("実験の保存先", self.output, False),
            )
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8)
            if row:
                ttk.Button(
                    frame, text="参照", command=lambda v=variable, f=file_select: self.browse(v, f)
                ).grid(row=row, column=2)
        ttk.Label(frame, text=f"モデル: {MODEL_ID}", wraplength=1000).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=10
        )
        actions = ttk.Frame(frame)
        actions.grid(row=5, column=0, columnspan=3, sticky="w")
        self.setup_button = ttk.Button(actions, text="① 専用環境を準備", command=self.setup_runtime)
        self.setup_button.pack(side="left", padx=4)
        self.prepare_button = ttk.Button(actions, text="② モデルを取得", command=self.prepare_model)
        self.prepare_button.pack(side="left", padx=4)
        ttk.Button(actions, text="現在のログを開く", command=self.open_log).pack(
            side="left", padx=4
        )
        ttk.Button(actions, text="保存先を開く", command=self.open_output).pack(side="left", padx=4)
        options = ttk.LabelFrame(frame, text="生成環境", padding=12)
        options.grid(row=6, column=0, columnspan=3, sticky="ew", pady=16)
        ttk.Label(options, text="精度").grid(row=0, column=0)
        ttk.Combobox(
            options,
            textvariable=self.dtype,
            values=["bfloat16", "float16", "float32"],
            state="readonly",
            width=12,
        ).grid(row=0, column=1, padx=8)
        ttk.Checkbutton(options, text="CPUオフロード", variable=self.cpu_offload).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=8
        )
        ttk.Checkbutton(options, text="KV cacheを使用", variable=self.use_kv_cache).grid(
            row=2, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(options, text="参照解像度").grid(row=3, column=0, pady=8)
        ttk.Combobox(
            options,
            textvariable=self.reference_resolution,
            values=["512", "768", "1024"],
            state="readonly",
            width=12,
        ).grid(row=3, column=1)
        ttk.Checkbutton(
            options, text="テキストエンコーダーを層ごとに処理（VRAM節約）",
            variable=self.text_encoder_layers,
        ).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            options, text="描画本体を8ビット（fp8）で保持（VRAM約6.6GB減）",
            variable=self.transformer_fp8,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            options, text="仕上げを分割して処理（8ビット時のピークを下げる）",
            variable=self.vae_tiling,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(options, text="VRAMの目安から設定").grid(row=7, column=0, pady=8)
        preset = ttk.Combobox(options, textvariable=self.vram_preset, values=list(VRAM_PRESETS),
                              state="readonly", width=48)
        preset.grid(row=7, column=1, sticky="w")
        preset.bind("<<ComboboxSelected>>", lambda _: self.apply_vram_preset())
        ttk.Label(
            frame,
            text="生成ごとに専用プロセスを起動し、本体と共通のGPUロックで順番に実行します。\n環境準備中の終了はインストール完了を待ちます。生成・モデル取得は停止できます。",
            wraplength=1000,
        ).grid(row=7, column=0, columnspan=3, sticky="w")

    def apply_vram_preset(self):
        resolution, cache, layers, fp8, tiling = VRAM_PRESETS[self.vram_preset.get()]
        self.reference_resolution.set(resolution)
        self.use_kv_cache.set(cache)
        self.text_encoder_layers.set(layers)
        self.transformer_fp8.set(fp8)
        self.vae_tiling.set(tiling)
        self.cpu_offload.set(True)

    def active_mode(self):
        selected = self.tabs.select()
        return next(
            (mode for mode, frame in self.mode_frames.items() if str(frame) == selected), None
        )

    def browse(self, variable, file_select=False):
        value = (
            filedialog.askopenfilename(filetypes=[("Python", "*.exe")])
            if file_select
            else filedialog.askdirectory()
        )
        if value:
            variable.set(value)

    def _async(self, kind, revision, work):
        def execute():
            try:
                value, error = work(), None
            except Exception as exc:  # noqa: BLE001 - surface background failures on the UI thread
                value, error = None, str(exc)
            self.events.put((kind, revision, value, error))

        threading.Thread(target=execute, daemon=True).start()

    def refresh_catalog(self):
        self.catalog_revision += 1
        self.projects, self.portraits, self.chapters, self.scenes = [], [], [], []
        for box in (self.project_box, self.portrait_box, self.chapter_box, self.scene_box):
            box.set("")
            box["values"] = []
        self.scene_changed()
        self._cg_set_chapters()
        self.catalog_status.set("作品一覧を取得中…")
        self.catalog = CoordinatorImageCatalog(self.server.get().strip())
        self._async("projects", self.catalog_revision, self.catalog.projects)

    def project_changed(self):
        index = self.project_box.current()
        if not 0 <= index < len(self.projects):
            return
        self.catalog_revision += 1
        revision = self.catalog_revision
        project_id = self.projects[index]["id"]
        self.portraits, self.chapters, self.scenes = [], [], []
        for box in (self.portrait_box, self.chapter_box, self.scene_box):
            box.set("")
            box["values"] = []
        self.scene_changed()
        self._cg_set_chapters()
        catalog = CoordinatorImageCatalog(self.server.get().strip())
        self.catalog = catalog
        self._async("portraits", revision, lambda: catalog.portraits(project_id))
        self._async("chapters", revision, lambda: catalog.chapters(project_id))

    def chapter_changed(self):
        index = self.chapter_box.current()
        self.scenes = (
            self.chapters[index].get("scenes", []) if 0 <= index < len(self.chapters) else []
        )
        self.scene_box["values"] = [scene["label"] for scene in self.scenes]
        self.scene_box.set("")
        if self.scenes:
            self.scene_box.current(0)
        self.scene_changed()

    def selected_scene(self):
        index = self.scene_box.current()
        return self.scenes[index] if 0 <= index < len(self.scenes) else None

    def scene_changed(self):
        scene = self.selected_scene()
        self.scene_preview.configure(state="normal")
        self.scene_preview.delete("1.0", "end")
        self.scene_preview.insert("1.0", scene.get("preview", "") if scene else "")
        self.scene_preview.configure(state="disabled")

    def replace_prompt(self, mode, value):
        if mode == "scene":
            self.scene_prompt_details = None
            self.scene_interpretation.set("LLMが選んだ一瞬と構図を、ここに表示します。")
        prompt = self.prompts[mode]
        for segment in self.managed_prompt_segments[mode].values():
            prompt.mark_unset(segment["start"], segment["end"])
        self.managed_prompt_segments[mode].clear()
        prompt.delete("1.0", "end")
        prompt.insert("1.0", value)

    @staticmethod
    def _mapping_header(references):
        return "".join(
            f"Reference image {index}: {row.get('name') or f'Character {index}'}.\n"
            for index, row in enumerate(references, 1)
        )

    def _register_prompt_segment(self, mode, kind, start, text):
        prompt = self.prompts[mode]
        start_mark, end_mark = f"gui_{kind}_start", f"gui_{kind}_end"
        prompt.mark_set(start_mark, start)
        length = prompt.tk.call("string", "length", text)
        prompt.mark_set(end_mark, f"{start}+{length}c")
        # New user text at either boundary belongs to the user, outside our range.
        prompt.mark_gravity(start_mark, "right")
        prompt.mark_gravity(end_mark, "left")
        self.managed_prompt_segments[mode][kind] = {
            "start": start_mark,
            "end": end_mark,
            "text": text,
        }

    def _replace_prompt_segment(self, mode, kind, text):
        segment = self.managed_prompt_segments[mode].get(kind)
        if segment is None:
            return False
        prompt = self.prompts[mode]
        start, end = prompt.index(segment["start"]), prompt.index(segment["end"])
        if prompt.get(start, end) != segment["text"]:
            # An edited header/instruction is now user-owned and must stay intact.
            prompt.mark_unset(segment["start"], segment["end"])
            del self.managed_prompt_segments[mode][kind]
            return False
        prompt.delete(start, end)
        prompt.insert(start, text)
        if text:
            self._register_prompt_segment(mode, kind, start, text)
        else:
            prompt.mark_unset(segment["start"], segment["end"])
            del self.managed_prompt_segments[mode][kind]
        return True

    def _register_mapping(self):
        header = self._mapping_header(self.references["scene"])
        prompt = self.prompts["scene"]
        length = prompt.tk.call("string", "length", header)
        if header and prompt.get("1.0", f"1.0+{length}c") == header:
            self._register_prompt_segment("scene", "mapping", "1.0", header)

    def _register_transparency(self, mode):
        prompt = self.prompts[mode]
        content = prompt.get("1.0", "end-1c")
        offset = content.rfind(TRANSPARENCY_INSTRUCTION)
        if offset >= 0:
            if offset and content[offset - 1] == "\n":
                offset -= 1
            prefix_length = prompt.tk.call("string", "length", content[:offset])
            self._register_prompt_segment(
                mode, "transparency", f"1.0+{prefix_length}c", content[offset:]
            )

    def apply_edit_preset(self):
        self.replace_prompt(
            "portrait",
            build_portrait_prompt(
                EDIT_PRESETS[self.edit_preset.get()], self.transparent["portrait"].get()
            ),
        )
        if self.transparent["portrait"].get():
            self._register_transparency("portrait")

    def create_scene_prompt(self):
        scene = self.selected_scene()
        if not scene:
            self.status.set("場面を選んでください。自由入力でも生成できます。")
            return
        if not self.references["scene"]:
            self.status.set("場面に登場する立ち絵を1枚以上追加してください。")
            return
        try:
            snapshot = self._scene_prompt_snapshot()
            request = {
                "scene": snapshot["scene"],
                "references": snapshot["references"],
                "instruction": snapshot["instruction"],
                "backend": self.llm_backend.get(),
                "local_model": self.llm_local_model.get().strip(),
                "base_url": self.llm_url.get().strip(),
                "model": self.llm_model.get().strip(),
            }
            run_dir = new_run(
                Path(self.output.get().strip()).expanduser().resolve(), "scene-prompt"
            )
            self._start_process(
                [
                    self._console_python(),
                    "-u",
                    "-X",
                    "utf8",
                    str(Path(__file__).with_name("llm_runner.py")),
                    "--request",
                    str(run_dir / "request.json"),
                    "--output-dir",
                    str(run_dir),
                ],
                run_dir,
                "prompt",
                request,
            )
            self.pending_llm = snapshot
            self.status.set("LLMで台本を解釈し、一枚絵の指示を作成しています。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("LLMプロンプト作成", str(exc))

    def _scene_prompt_snapshot(self):
        return {
            "scene": copy.deepcopy(self.selected_scene()),
            "references": copy.deepcopy(self.references["scene"]),
            "instruction": self.scene_instruction.get().strip(),
            "previous_prompt": self.prompts["scene"].get("1.0", "end").strip(),
        }

    def _scene_prompt_body(self):
        """Ignore only unchanged GUI-owned mapping and transparency decoration."""
        prompt = self.prompts["scene"]
        intervals = []
        for segment in self.managed_prompt_segments["scene"].values():
            start, end = prompt.index(segment["start"]), prompt.index(segment["end"])
            if prompt.get(start, end) == segment["text"]:
                intervals.append((start, end))
        intervals.sort(key=lambda value: tuple(int(part) for part in value[0].split(".")))
        chunks, cursor = [], "1.0"
        for start, end in intervals:
            chunks.append(prompt.get(cursor, start))
            cursor = end
        chunks.append(prompt.get(cursor, "end-1c"))
        return "".join(chunks).strip()

    def _check_scene_prompt_source(self):
        details = self.scene_prompt_details
        if details and self._scene_prompt_body() == details["prompt_body"]:
            current = self._scene_prompt_snapshot()
            if any(
                current[key] != details["source"][key]
                for key in ("scene", "references", "instruction")
            ):
                raise ValueError(
                    "LLM作成後に場面・参照・追加指示が変わりました。LLMで指示を作り直してください。"
                )

    def create_reference_prompt(self):
        if not self.references["scene"]:
            self.status.set("一枚絵の参照を追加してから指示を整えてください。")
            return
        instruction = self.prompts["scene"].get("1.0", "end").strip()
        instruction = instruction or "Describe the composition, actions and background here."
        self.replace_prompt("scene", build_scene_prompt(instruction, self.references["scene"]))
        self._register_mapping()
        self.status.set(
            "画像の順番とキャラ名を指示へ追加しました。内容を確認して生成してください。"
        )

    def prepare_final_prompt(self, mode):
        if mode == "scene":
            self._check_scene_prompt_source()
        if mode == "scene":
            self._replace_prompt_segment(
                mode, "mapping", self._mapping_header(self.references[mode])
            )
        if not self.transparent[mode].get():
            self._replace_prompt_segment(mode, "transparency", "")
            return
        prompt = self.prompts[mode].get("1.0", "end").strip()
        if prompt and self.transparent[mode].get() and TRANSPARENCY_INSTRUCTION not in prompt:
            widget = self.prompts[mode]
            start = widget.index("end-1c")
            text = "\n" + TRANSPARENCY_INSTRUCTION
            widget.insert(start, text)
            self._register_prompt_segment(mode, "transparency", start, text)

    def add_local_references(self, mode):
        paths = filedialog.askopenfilenames(
            filetypes=[("画像", "*.png *.jpg *.jpeg *.webp"), ("すべて", "*.*")]
        )
        for value in paths:
            path = Path(value).resolve()
            try:
                from PIL import Image

                with Image.open(path) as picture:
                    picture.verify()
                reference = {
                    "path": str(path),
                    "name": path.stem,
                    "source": {"kind": "local", "original_path": str(path)},
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                self.add_reference(mode, reference)
            except (OSError, ValueError) as exc:
                messagebox.showerror("画像の追加", f"{path.name}: {exc}")

    def add_catalog_reference(self):
        mode, index = self.active_mode(), self.portrait_box.current()
        if mode is None or not 0 <= index < len(self.portraits):
            self.status.set("差分または一枚絵のタブを開き、立ち絵を選んでください。")
            return
        portrait = copy.deepcopy(self.portraits[index])
        directory = (
            Path(self.output.get().strip()).expanduser().resolve() / "references" / uuid.uuid4().hex
        )
        revision = (self.catalog_revision, self.reference_revision[mode])
        catalog = self.catalog
        if catalog is None:
            self.status.set("作品一覧を更新してから立ち絵を選んでください。")
            return
        self._async(
            f"reference:{mode}", revision, lambda: catalog.download_reference(portrait, directory)
        )
        self.status.set("選んだ立ち絵を読み込んでいます…")

    def add_reference(self, mode, reference):
        reference = copy.deepcopy(reference)
        reference["path"] = str(Path(reference["path"]).expanduser().resolve())
        reference["name"] = str(reference.get("name") or Path(reference["path"]).stem)
        reference.setdefault("source", {})
        if any(reference_key(row) == reference_key(reference) for row in self.references[mode]):
            self.status.set(f"「{reference['name']}」はすでに参照にあります。追加しませんでした。")
            return False
        if len(self.references[mode]) >= 10:
            self.status.set("参照画像は最大10枚です。不要な参照を削除してください。")
            return False
        self.references[mode].append(reference)
        self.reference_revision[mode] += 1
        self.refresh_references(mode, len(self.references[mode]) - 1)
        self.status.set(
            f"「{reference['name']}」を画像{len(self.references[mode])}として追加しました。"
        )
        return True

    def refresh_references(self, mode, select=None):
        listing = self.reference_lists[mode]
        for iid in listing.get_children():
            listing.delete(iid)
        for index, row in enumerate(self.references[mode]):
            listing.insert("", "end", iid=str(index), values=(f"画像{index + 1}: {row['name']}",))
        if select is not None and 0 <= select < len(self.references[mode]):
            listing.selection_set(str(select))
            listing.focus(str(select))
        self.reference_selected(mode)

    def selected_reference(self, mode):
        selection = self.reference_lists[mode].selection()
        return int(selection[0]) if selection else None

    def reference_selected(self, mode):
        index = self.selected_reference(mode)
        reference = self.references[mode][index] if index is not None else None
        self.reference_names[mode].set(reference["name"] if reference else "")
        self.show_picture(
            self.photos[f"reference_widget_{mode}"],
            Path(reference["path"]) if reference else None,
            f"reference_{mode}",
            (330, 245),
        )

    def rename_reference(self, mode):
        index = self.selected_reference(mode)
        name = self.reference_names[mode].get().strip()
        if index is not None and name:
            self.references[mode][index]["name"] = name
            self.reference_revision[mode] += 1
            self.refresh_references(mode, index)

    def move_reference(self, mode, delta):
        index = self.selected_reference(mode)
        if index is None or not 0 <= index + delta < len(self.references[mode]):
            return
        rows = self.references[mode]
        rows[index], rows[index + delta] = rows[index + delta], rows[index]
        self.reference_revision[mode] += 1
        self.refresh_references(mode, index + delta)

    def remove_reference(self, mode):
        index = self.selected_reference(mode)
        if index is None:
            return
        self.references[mode].pop(index)
        self.reference_revision[mode] += 1
        self.refresh_references(mode, min(index, len(self.references[mode]) - 1))

    def show_picture(self, widget, path, key, size):
        if not path:
            widget.configure(image="", text="画像を選んでください")
            self.photos.pop(key, None)
            return
        try:
            from PIL import ImageTk

            photo = ImageTk.PhotoImage(checker_preview(path, size), master=self.root)
            self.photos[key] = photo
            widget.configure(image=photo, text="")
        except (OSError, ValueError) as exc:
            widget.configure(image="", text=f"画像を表示できません: {exc}")
            self.photos.pop(key, None)

    def enlarge(self, path, title):
        if not path:
            return
        window = Toplevel(self.root)
        window.title(title)
        from PIL import ImageTk

        try:
            picture = checker_preview(Path(path), (1000, 800))
            photo = ImageTk.PhotoImage(picture, master=window)
            label = ttk.Label(window, image=photo)
            label.image = photo
            label.pack(padx=12, pady=12)
        except (OSError, ValueError) as exc:
            window.destroy()
            messagebox.showerror("画像表示", str(exc))

    def enlarge_reference(self, mode):
        index = self.selected_reference(mode)
        if index is not None:
            row = self.references[mode][index]
            self.enlarge(row["path"], row["name"])

    def build_request(self, mode):
        if mode == "scene":
            self._check_scene_prompt_source()
        rows = copy.deepcopy(self.references[mode])
        if mode == "portrait" and len(rows) != 1:
            raise ValueError(
                "立ち絵の差分は参照を1枚にしてください。複数人には一枚絵タブを使えます。"
            )
        if not rows:
            raise ValueError("参照画像を1枚以上追加してください。")
        scene = self.selected_scene() if mode == "scene" else None
        context = copy.deepcopy(scene.get("context", {})) if scene else {}
        if scene:
            context.update(
                {
                    "scene_id": scene["id"],
                    "scene_label": scene["label"],
                    "scene_preview": scene.get("preview", ""),
                }
            )
        if mode == "scene" and self.scene_prompt_details:
            context["prompt_generation"] = copy.deepcopy(self.scene_prompt_details)
            context["prompt_generation"]["manually_edited"] = (
                self._scene_prompt_body() != self.scene_prompt_details["prompt_body"]
            )
        data = {
            "schema_version": 1,
            "mode": mode,
            "prompt": self.prompts[mode].get("1.0", "end").strip(),
            "references": rows,
            "model_path": self.model_path.get().strip(),
            "width": int(self.widths[mode].get()),
            "height": int(self.heights[mode].get()),
            "steps": int(self.steps.get()),
            "seed": int(self.seed.get()),
            "dtype": self.dtype.get(),
            "cpu_offload": self.cpu_offload.get(),
            "use_kv_cache": self.use_kv_cache.get(),
            "reference_resolution": int(self.reference_resolution.get()),
            "text_encoder_offload": "layers" if self.text_encoder_layers.get() else "model",
            "transformer_storage": "fp8" if self.transformer_fp8.get() else "native",
            "vae_tiling": self.vae_tiling.get(),
            "transparent": self.transparent[mode].get(),
            "context": context,
        }
        if self.three_seeds.get():
            data["seeds"] = [0, 1, 2]
        validate_request(data)
        return data

    def _save_settings(self):
        write_json(
            SETTINGS,
            {
                key: getattr(self, key).get()
                for key in (
                    "server",
                    "python",
                    "model_path",
                    "output",
                    "llm_backend",
                    "llm_local_model",
                    "llm_url",
                    "llm_model",
                )
            },
        )

    def _runtime_python(self):
        python = Path(self.python.get().strip()).expanduser().resolve()
        if not python.is_file():
            raise ValueError("専用Pythonがありません。環境タブで①専用環境を準備してください。")
        return str(python)

    @staticmethod
    def _console_python():
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        if not python.is_file():
            raise ValueError("環境準備用Pythonが見つかりません。")
        return str(python)

    def _busy_buttons(self):
        return (self.generate_button, self.setup_button, self.prepare_button,
                self.scene_prompt_button, self.cg_propose_button, self.cg_all_button)

    def _start_process(self, command, run_dir, kind, request=None):
        if self.process is not None:
            raise ValueError("現在の処理が終わるまでお待ちください。")
        run_dir.mkdir(parents=True, exist_ok=False)
        if request is not None:
            write_json(run_dir / "request.json", request)
        self._save_settings()
        self._log_stream = (run_dir / "process.log").open("wb")
        environment = hub_environment(RUNTIME / "cache/huggingface", os.environ)
        environment["PYTHONUTF8"] = "1"
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self._log_stream.close()
            self._log_stream = None
            raise
        self.run_dir, self.run_kind = run_dir, kind
        self.started_at = time.monotonic()
        self.stop_at = self.terminate_at = None
        for button in self._busy_buttons():
            button.configure(state="disabled")
        self.stop_button.configure(state="normal" if kind != "setup" else "disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)

    def generate(self):
        if self.tabs.select() == str(self.cg_frame):
            self.generate_cg()
            return
        mode = self.active_mode()
        if mode is None:
            self.status.set("立ち絵の差分・場面の一枚絵・印象的な一枚のタブを選んでください。")
            return
        try:
            self.prepare_final_prompt(mode)
            data = self.build_request(mode)
            python = self._runtime_python()
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), mode)
            self._start_process(
                [
                    python,
                    "-u",
                    "-X",
                    "utf8",
                    str(Path(__file__).with_name("runner.py")),
                    "--request",
                    str(run_dir / "request.json"),
                    "--output-dir",
                    str(run_dir),
                ],
                run_dir,
                "generate",
                data,
            )
            self.status.set("モデルを読み込み中… 初回は時間がかかります。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("生成開始", str(exc))

    def setup_runtime(self):
        try:
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), "setup")
            self._start_process(
                [
                    self._console_python(),
                    "-u",
                    "-X",
                    "utf8",
                    str(Path(__file__).with_name("setup_runtime.py")),
                    "--status-dir",
                    str(run_dir),
                ],
                run_dir,
                "setup",
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("環境準備", str(exc))

    def prepare_model(self):
        try:
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), "prepare")
            self._start_process(
                [
                    self._runtime_python(),
                    "-u",
                    "-X",
                    "utf8",
                    str(Path(__file__).with_name("prepare.py")),
                    "--model-dir",
                    self.model_path.get().strip(),
                    "--status-dir",
                    str(run_dir),
                ],
                run_dir,
                "prepare",
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("モデル取得", str(exc))

    def stop(self):
        if self.process is None or self.run_kind == "setup" or self.stop_at is not None:
            return
        (self.run_dir / "stop.request").touch()
        self.stop_at = time.monotonic()
        self.stop_button.configure(state="disabled")
        self.status.set("停止処理中… この実験のプロセス終了とGPU解放を待っています。")

    def poll(self):
        while not self.events.empty():
            kind, revision, value, error = self.events.get()
            if kind == "cg_preview":
                if revision == self.cg_preview_revision:
                    self._cg_preview_ready(value, error)
                continue
            if kind == "cg_references":
                if revision == (self.catalog_revision, self.cg_revision):
                    self._cg_references_ready(value, error)
                continue
            if kind.startswith("reference:"):
                mode = kind.split(":", 1)[1]
                if revision != (self.catalog_revision, self.reference_revision[mode]):
                    continue
                if error:
                    self.status.set(f"立ち絵の読み込みに失敗: {error}")
                else:
                    self.add_reference(mode, value)
                continue
            if revision != self.catalog_revision:
                continue
            if error:
                hint = (
                    " / 画像実験APIを使うには作品サーバーを再起動してください。"
                    if "404" in error
                    else ""
                )
                self.catalog_status.set(f"作品の読み込みに失敗: {error}{hint}")
                continue
            if kind == "projects":
                self.projects = value
                self.project_box["values"] = [row["title"] for row in value]
                self.catalog_status.set(
                    f"{len(value)}作品。作品を選ぶと立ち絵と場面を読み込みます。"
                )
            elif kind == "portraits":
                self.portraits = value
                self.cg_scene_changed()
                self.portrait_box["values"] = [row["name"] for row in value]
                if value:
                    self.portrait_box.current(0)
                self.catalog_status.set(f"{len(value)}枚の立ち絵を取得しました。")
            elif kind == "chapters":
                self.chapters = value
                self.chapter_box["values"] = [f"{row['number']}章: {row['title']}" for row in value]
                if value:
                    self.chapter_box.current(0)
                self.chapter_changed()
                self._cg_set_chapters()
        if self.process is not None:
            state = read_json(self.run_dir / "status.json")
            elapsed = time.monotonic() - self.started_at
            if self.stop_at is None:
                self.status.set(f"{state.get('message', '処理中…')} / {elapsed:.0f}秒")
                if state.get("step") is not None and state.get("steps"):
                    self.progress.stop()
                    self.progress.configure(
                        mode="determinate", maximum=state["steps"], value=state["step"]
                    )
            elif (
                time.monotonic() - self.stop_at > 5
                and self.process.poll() is None
                and self.terminate_at is None
            ):
                self.process.terminate()
                self.terminate_at = time.monotonic()
            elif (
                self.terminate_at is not None
                and time.monotonic() - self.terminate_at > 2
                and self.process.poll() is None
            ):
                self.process.kill()
            code = self.process.poll()
            if code is not None:
                self.process.wait(timeout=0)
                self._finish_process(code)
        if self.close_pending and self.process is None:
            self.close()
            return
        self.root.after(150, self.poll)

    def _finish_process(self, code):
        result = read_json(self.run_dir / "result.json")
        if self._log_stream:
            self._log_stream.close()
        self._log_stream = None
        self.process = None
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        for button in self._busy_buttons():
            button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self.run_kind == "prompt":
            self._finish_scene_prompt(code, result)
            return
        if self.run_kind == "cg_proposals":
            self._finish_cg_proposals(code, result)
            return
        queued, self.cg_queue = self.cg_queue, []
        if self.stop_at is not None or result.get("cancelled"):
            self._add_result(self.run_dir, result)
            self.status.set(
                "実験を停止しました。完了済みの画像があれば「中断/一部」として比較できます。"
            )
        elif code == 0 and result.get("ok"):
            if self.run_kind == "generate":
                self._add_result(self.run_dir, result)
                if queued:
                    self.cg_queue = queued
                    self._cg_next()
                    return
                self.tabs.select(self.comparison_tab)
                self.status.set("生成完了。元の参照と比較し、一致を評価できます。")
            else:
                self.status.set(result.get("message", "準備完了。"))
        else:
            self._add_result(self.run_dir, result)
            error = (
                result.get("error")
                or read_json(self.run_dir / "status.json").get("message")
                or f"終了コード: {code}"
            )
            self.status.set(f"処理に失敗: {error}")
            if not self.close_pending:
                messagebox.showerror(
                    "Qwen Image", f"{error}\n\nログ: {self.run_dir / 'process.log'}"
                )

    def _finish_scene_prompt(self, code, result):
        source, self.pending_llm = self.pending_llm, None
        if self.stop_at is not None or result.get("cancelled"):
            self.status.set("LLMの指示作成を停止しました。現在の生成指示は保持しています。")
            return
        if code != 0 or not result.get("ok") or not result.get("prompt"):
            error = result.get("error") or f"終了コード: {code}"
            self.status.set(f"LLMの指示作成に失敗: {error}")
            if not self.close_pending:
                messagebox.showerror(
                    "LLMプロンプト作成", f"{error}\n\nログ: {self.run_dir / 'process.log'}"
                )
            return
        if source is None or source != self._scene_prompt_snapshot():
            self.status.set(
                "作成中に場面・参照・入力が変わったため指示を反映しませんでした。結果は保存先に残しています。"
            )
            return
        backend = result.get("backend", "llama_cpp")
        if backend in {"llama_cpp", "ollama"} and not result.get("llm_released"):
            self.status.set("LLMのGPU解放を確認できません。保存されたログを確認してください。")
            return
        self.replace_prompt("scene", build_scene_prompt(result["prompt"], self.references["scene"]))
        self._register_mapping()
        interpretation = str(result.get("scene_interpretation") or "")
        self.scene_interpretation.set("LLMの場面解釈: " + interpretation)
        self.scene_prompt_details = {
            "kind": "llm",
            "run_dir": str(self.run_dir.resolve()),
            "source": source,
            "final_prompt": self.prompts["scene"].get("1.0", "end").strip(),
            "prompt_body": self._scene_prompt_body(),
            "scene_interpretation": interpretation,
            "backend": backend,
            "local_model": result.get("local_model"),
            "llm_released": result.get("llm_released"),
        }
        self.tabs.select(self.mode_frames["scene"])
        self.status.set("LLMが一枚絵の指示を作成しました。場面解釈と構図を確認して生成できます。")

    def _add_result(self, directory, result):
        if not result.get("outputs"):
            return
        request = read_json(directory / "request.json")
        generation = read_json(directory / "generation.json")
        request = copy.deepcopy(request)
        if generation.get("references"):
            request["references"] = [
                {**row, "path": row.get("normalized_path", row["path"])}
                for row in generation["references"]
            ]
        for index, output in enumerate(result.get("outputs", [])):
            path = Path(output["path"])
            if not path.is_absolute():
                path = directory / path
            if not path.is_file():
                continue
            iid = f"{directory.resolve()}::{index}"
            self.results[iid] = {
                "directory": directory,
                "output": output,
                "path": path,
                "request": request,
                "generation": generation,
                "index": index,
            }
            if self.history.exists(iid):
                continue
            partial = " / 中断・一部" if not result.get("ok") else ""
            self.history.insert(
                "",
                "end",
                iid=iid,
                values=(
                    f"{directory.name} / {request.get('mode', '')}{partial}",
                    output.get("seed", "?"),
                ),
            )
            self.history.selection_set(iid)
        self.result_selected()

    def load_history(self):
        parent = Path(self.output.get().strip()).expanduser().resolve()
        for path in sorted(parent.glob("*/result.json")):
            self._add_result(path.parent, read_json(path))
        self.status.set(f"{len(self.results)}枚の結果を履歴に読み込みました。")

    def selected_result(self):
        selection = self.history.selection()
        return self.results.get(selection[0]) if selection else None

    def result_selected(self):
        selected = self.selected_result()
        if not selected:
            return
        rows = selected["request"].get("references", [])
        self.original_box["values"] = [f"画像{i + 1}: {row['name']}" for i, row in enumerate(rows)]
        self.original_box.set("")
        if rows:
            self.original_box.current(0)
        self.show_original()
        self.show_picture(self.result_preview, selected["path"], "result", (360, 330))
        data, output = selected["generation"], selected["output"]
        alpha = (
            f"アルファ範囲 {output['alpha_range']}"
            if output.get("alpha_range") is not None
            else "アルファ情報なし"
        )
        warnings = " / ".join(output.get("warnings", []))
        proposal = (selected["request"].get("context") or {}).get("cg_proposal") or {}
        if proposal:
            row = proposal["proposal"]
            edited = " / 指示を手修正" if proposal.get("manually_edited") else ""
            warnings = (f"案{proposal['index'] + 1}: {row['shot']} / {row['angle']}{edited} / "
                        f"{row['moment']}" + (" / " + warnings if warnings else ""))
        self.metrics.set(
            f"Seed {output.get('seed', '?')} / {output.get('width', selected['request'].get('width', '?'))}×{output.get('height', selected['request'].get('height', '?'))} / {selected['request'].get('steps', '?')} steps / {data.get('elapsed_seconds', output.get('elapsed_seconds', '?'))}秒 / {alpha}\n{warnings + chr(10) if warnings else ''}{selected['path']}"
        )
        targets = ["全体"] + [f"画像{i + 1}: {row['name']}" for i, row in enumerate(rows)]
        self.review_target_box["values"] = targets
        self.review_target.set("全体")
        self.load_review()

    def original_reference(self):
        selected = self.selected_result()
        index = self.original_box.current()
        rows = selected["request"].get("references", []) if selected else []
        return rows[index] if 0 <= index < len(rows) else None

    def show_original(self):
        reference = self.original_reference()
        self.show_picture(
            self.original_preview,
            Path(reference["path"]) if reference else None,
            "original",
            (360, 330),
        )

    def enlarge_original(self):
        reference = self.original_reference()
        if reference:
            self.enlarge(reference["path"], reference["name"])

    def enlarge_result(self):
        selected = self.selected_result()
        if selected:
            self.enlarge(selected["path"], "生成結果")

    def use_result_reference(self, mode):
        selected = self.selected_result()
        if not selected:
            return
        rows = selected["request"].get("references", [])
        name = (
            rows[0]["name"]
            if selected["request"].get("mode") == "portrait" and len(rows) == 1
            else selected["path"].stem
        )
        reference = {
            "path": str(selected["path"].resolve()),
            "name": name,
            "source": {
                "kind": "generated",
                "run_dir": str(selected["directory"].resolve()),
                "output_index": selected["index"],
                "seed": selected["output"].get("seed"),
                "mode": selected["request"].get("mode"),
                "parents": copy.deepcopy(rows),
            },
        }
        if len(rows) == 1 and rows[0].get("character_id"):
            reference["character_id"] = rows[0]["character_id"]
        if mode == "portrait":
            self.references[mode] = []
            self.reference_revision[mode] += 1
        if self.add_reference(mode, reference):
            self.tabs.select(self.mode_frames[mode])

    def export_result(self):
        selected = self.selected_result()
        if not selected:
            return
        destination = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=selected["path"].name, filetypes=[("PNG", "*.png")]
        )
        if destination:
            try:
                from PIL import Image

                with Image.open(selected["path"]) as picture:
                    picture.save(destination, format="PNG")
                self.status.set(f"保存しました: {destination}")
            except OSError as exc:
                messagebox.showerror("画像保存", str(exc))

    def _review_key(self):
        selected = self.selected_result()
        if not selected:
            return None
        targets = list(self.review_target_box["values"])
        target = self.review_target.get()
        index = targets.index(target) if target in targets else 0
        return (
            f"output_{selected['index']}:{'overall' if index == 0 else 'reference_' + str(index)}"
        )

    def load_review(self):
        selected, key = self.selected_result(), self._review_key()
        evaluation = (
            read_json(selected["directory"] / "review.json").get("evaluations", {}).get(key, {})
            if selected
            else {}
        )
        for field, variable in self.ratings.items():
            variable.set(str(evaluation.get("ratings", {}).get(field) or "未評価"))
        self.review_note.set(evaluation.get("note", ""))

    def save_review(self):
        selected, key = self.selected_result(), self._review_key()
        if not selected:
            return
        try:
            path = selected["directory"] / "review.json"
            value = read_json(path)
            value["schema_version"] = 1
            value.setdefault("evaluations", {})[key] = {
                "target": self.review_target.get(),
                "ratings": {
                    field: None
                    if variable.get() == "未評価"
                    else "N/A"
                    if variable.get() == "N/A"
                    else int(variable.get())
                    for field, variable in self.ratings.items()
                },
                "note": self.review_note.get().strip(),
                "updated_at": datetime.now().astimezone().isoformat(),
            }
            write_json(path, value)
            self.status.set("評価を保存しました。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("評価保存", str(exc))

    def open_log(self):
        if self.run_dir and (self.run_dir / "process.log").is_file():
            subprocess.Popen(["notepad.exe", str(self.run_dir / "process.log")])

    def open_output(self):
        path = Path(self.output.get().strip()).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)

    def close(self):
        if self.process is not None:
            self.close_pending = True
            if self.run_kind == "setup":
                self.status.set("環境準備が完了してからウィンドウを閉じます。")
            else:
                self.stop()
            return
        self._save_settings()
        self.cg_preview_server.close()
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qwen Image 2.1 キャラクター画像テスト")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    root = Tk()
    if args.smoke_test:
        root.withdraw()
    app = ImageEditTestApp(root, auto_refresh=not args.smoke_test)
    if args.smoke_test:
        root.update_idletasks()
        assert len(app.tabs.tabs()) == 6
        assert app.cpu_offload.get()
        root.destroy()
        print("Qwen Image test GUI smoke test passed")
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
