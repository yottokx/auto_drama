"""The "LLM chooses cast and camera" tab of the image experiment workbench."""

from __future__ import annotations

import copy
import uuid
import webbrowser
from pathlib import Path
from tkinter import BooleanVar, StringVar, Text, messagebox, ttk

from scripts.image_edit.cg_preview import PreviewServer, prepare
from scripts.image_edit.cg_proposals import (
    PROMPT_VERSION,
    PROPOSAL_COUNT,
    compile_prompt,
    scene_candidates,
)
from scripts.image_edit.common import new_run
from scripts.image_edit.engine import validate_request

CG_VISIBILITY = {"face_front": "正面", "face_profile": "横顔", "back": "背中", "partial": "一部"}
CG_TRANSITIONS = {"なし（現在の本編と同じ）": "cut", "クロスフェード": "dissolve", "暗転": "fade"}
CG_HINT = "場面を選び「LLMで案を作成」を押すと、描く瞬間・人物・構図の案がここに並びます。"


class CgProposalTab:
    """Mixin for ImageEditTestApp; owns only the proposal tab's state."""

    def _cg_init(self):
        self.cg_scenes, self.cg_proposals, self.cg_queue = [], [], []
        self.cg_revision = 0
        self.cg_preview_revision = 0
        self.cg_preview_server = PreviewServer()
        self.cg_transition = StringVar(value="クロスフェード")
        self.cg_transition_ms = StringVar(value="500")
        self.cg_current = None
        self.cg_source = self.cg_run_dir = self.cg_pending = None
        self.cg_instruction = StringVar()
        self.cg_allow_extras = BooleanVar(value=True)
        self.cg_message_area = BooleanVar(value=False)
        self.cg_face_items = BooleanVar(value=True)
        self.cg_width, self.cg_height = StringVar(value="1536"), StringVar(value="1024")
        self.cg_cast = StringVar(value="場面を選ぶと、立ち絵のある登場人物を表示します。")
        self.cg_detail = StringVar(value=CG_HINT)

    def _cg_tab(self, frame):
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)
        top = ttk.Frame(frame)
        top.grid(row=0, column=0, sticky="ew")
        self.cg_chapter_box = ttk.Combobox(top, state="readonly", width=22)
        self.cg_chapter_box.pack(side="left")
        self.cg_chapter_box.bind("<<ComboboxSelected>>", lambda _: self.cg_chapter_changed())
        self.cg_scene_box = ttk.Combobox(top, state="readonly", width=44)
        self.cg_scene_box.pack(side="left", padx=5)
        self.cg_scene_box.bind("<<ComboboxSelected>>", lambda _: self.cg_scene_changed())
        self.cg_propose_button = ttk.Button(
            top, text="LLMで案を作成", command=self.create_cg_proposals)
        self.cg_propose_button.pack(side="left")
        self.cg_all_button = ttk.Button(top, text="全ての案を順に生成", command=self.generate_cg_all)
        self.cg_all_button.pack(side="left", padx=5)
        ttk.Label(frame, textvariable=self.cg_cast, wraplength=1150).grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        options = ttk.Frame(frame)
        options.grid(row=2, column=0, sticky="ew", pady=6)
        ttk.Label(options, text="LLMへの追加指示（任意）").pack(side="left")
        ttk.Entry(options, textvariable=self.cg_instruction, width=36).pack(side="left", padx=5)
        ttk.Checkbutton(options, text="立ち絵のない人物を人影として許可",
                        variable=self.cg_allow_extras).pack(side="left", padx=6)
        ttk.Checkbutton(options, text="下部のメッセージ枠を避ける",
                        variable=self.cg_message_area,
                        command=self._cg_recompile).pack(side="left", padx=6)
        ttk.Checkbutton(options, text="顔の小物を追記", variable=self.cg_face_items,
                        command=self._cg_recompile).pack(side="left", padx=6)
        for label, variable in (("幅", self.cg_width), ("高さ", self.cg_height)):
            ttk.Label(options, text=label).pack(side="left")
            ttk.Entry(options, textvariable=variable, width=6).pack(side="left", padx=4)
        self.cg_list = ttk.Treeview(
            frame, columns=("cast", "shot", "angle", "display", "moment"),
            show="headings", height=PROPOSAL_COUNT, selectmode="browse")
        for column, title, width in (("cast", "人物（見え方）", 240), ("shot", "ショット", 110),
                                     ("angle", "アングル", 100), ("display", "表示する発話", 130),
                                     ("moment", "描く内容", 560)):
            self.cg_list.heading(column, text=title)
            self.cg_list.column(column, width=width, stretch=column == "moment")
        self.cg_list.grid(row=3, column=0, sticky="ew")
        self.cg_list.bind("<<TreeviewSelect>>", lambda _: self.cg_proposal_selected())
        ttk.Label(frame, textvariable=self.cg_detail, wraplength=1150).grid(
            row=4, column=0, sticky="w", pady=6)
        body = ttk.Frame(frame)
        body.grid(row=5, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(1, weight=1)
        ttk.Label(body, text="画像モデルへ送る指示（選んだ案。編集できます）").grid(
            row=0, column=0, sticky="w")
        ttk.Label(body, text="台本の確認（画像モデルには送信しません）").grid(
            row=0, column=1, sticky="w", padx=(8, 0))
        self.cg_prompt = Text(body, height=10, wrap="word", font=("Yu Gothic UI", 10))
        self.cg_prompt.grid(row=1, column=0, sticky="nsew")
        self.cg_preview = Text(body, height=10, wrap="word", state="disabled",
                               font=("Yu Gothic UI", 9))
        self.cg_preview.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

    def _cg_set_chapters(self):
        self.cg_chapter_box["values"] = self.chapter_box["values"]
        self.cg_chapter_box.set("")
        if self.chapters:
            self.cg_chapter_box.current(0)
        self.cg_chapter_changed()

    def cg_chapter_changed(self):
        index = self.cg_chapter_box.current()
        self.cg_scenes = (
            self.chapters[index].get("scenes", []) if 0 <= index < len(self.chapters) else [])
        self.cg_scene_box["values"] = [scene["label"] for scene in self.cg_scenes]
        self.cg_scene_box.set("")
        if self.cg_scenes:
            self.cg_scene_box.current(0)
        self.cg_scene_changed()

    def cg_selected_scene(self):
        index = self.cg_scene_box.current()
        return self.cg_scenes[index] if 0 <= index < len(self.cg_scenes) else None

    def cg_scene_changed(self):
        scene = self.cg_selected_scene()
        self.cg_revision += 1
        self.cg_queue = []
        self._cg_show_proposals([], None)
        self.cg_preview.configure(state="normal")
        self.cg_preview.delete("1.0", "end")
        self.cg_preview.insert("1.0", scene.get("preview", "") if scene else "")
        self.cg_preview.configure(state="disabled")
        if not scene:
            self.cg_cast.set("場面を選ぶと、立ち絵のある登場人物を表示します。")
            return
        found = scene_candidates(scene, self.portraits)
        text = "候補: " + (" / ".join(f"{row['tag']}={row['name']}（{row['lines']}発話）"
                                    for row in found["candidates"]) or "なし")
        if found["unreferenced"]:
            text += "　立ち絵なし: " + "、".join(found["unreferenced"])
        self.cg_cast.set(text)

    def _cg_snapshot(self):
        scene = self.cg_selected_scene()
        found = scene_candidates(scene, self.portraits) if scene else {}
        return {"scene": copy.deepcopy(scene), "instruction": self.cg_instruction.get().strip(),
                "allow_extras": self.cg_allow_extras.get(),
                "candidates": copy.deepcopy(found.get("candidates", [])),
                "unreferenced": list(found.get("unreferenced", []))}

    def create_cg_proposals(self):
        snapshot = self._cg_snapshot()
        if not snapshot["scene"]:
            self.status.set("作品・章・場面を選んでください。")
            return
        if not snapshot["candidates"]:
            self.status.set("この場面には、公開版の立ち絵がある登場人物がいません。")
            return
        try:
            request = {"task": "cg_proposals", "prompt_version": PROMPT_VERSION, **snapshot,
                       "count": PROPOSAL_COUNT,
                       "timeout": 240, "backend": self.llm_backend.get(),
                       "local_model": self.llm_local_model.get().strip(),
                       "base_url": self.llm_url.get().strip(),
                       "model": self.llm_model.get().strip()}
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), "cg-proposals")
            self._start_process(
                [self._console_python(), "-u", "-X", "utf8",
                 str(Path(__file__).with_name("llm_runner.py")),
                 "--request", str(run_dir / "request.json"), "--output-dir", str(run_dir)],
                run_dir, "cg_proposals", request)
            self.cg_pending = snapshot
            self.status.set("LLMが場面を読み、描く瞬間・人物・構図の案を作成しています。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("LLMの案の作成", str(exc))

    def _finish_cg_proposals(self, code, result):
        source, self.cg_pending = self.cg_pending, None
        if self.stop_at is not None or result.get("cancelled"):
            self.status.set("案の作成を停止しました。")
            return
        if code != 0 or not result.get("ok") or not result.get("proposals"):
            error = result.get("error") or f"終了コード: {code}"
            self.status.set(f"案の作成に失敗: {error}")
            if not self.close_pending:
                messagebox.showerror(
                    "LLMの案の作成", f"{error}\n\nログ: {self.run_dir / 'process.log'}")
            return
        if source is None or source != self._cg_snapshot():
            self.status.set(
                "作成中に場面や入力が変わったため案を反映しませんでした。結果は保存先に残しています。")
            return
        if result.get("backend") in {"llama_cpp", "ollama"} and not result.get("llm_released"):
            self.status.set("LLMのGPU解放を確認できません。保存されたログを確認してください。")
            return
        self.cg_source, self.cg_run_dir = source, self.run_dir.resolve()
        self._cg_show_proposals(result["proposals"], source)
        self.tabs.select(self.cg_frame)
        self.status.set("LLMが案を作成しました。案を選んで内容を確認し、生成できます。")

    def _cg_show_proposals(self, proposals, source):
        self.cg_current = None
        self.cg_proposals = []
        for iid in self.cg_list.get_children():
            self.cg_list.delete(iid)
        self.cg_prompt.delete("1.0", "end")
        self.cg_detail.set(CG_HINT)
        if not proposals:
            self.cg_source = None
            return
        names = {row["tag"]: row["name"] for row in source["candidates"]}
        for index, proposal in enumerate(proposals):
            row = copy.deepcopy(proposal)
            row["prompt"] = row["compiled"] = self._cg_compile(row, source)
            self.cg_proposals.append(row)
            cast = "、".join(f"{names[member['tag']]}（{CG_VISIBILITY[member['visibility']]}）"
                            for member in row["cast"])
            span = row["display_to"] - row["display_from"] + 1
            self.cg_list.insert("", "end", iid=str(index), values=(
                cast, row["shot"], row["angle"],
                f"{row['display_from']}〜{row['display_to']}（{span}発話）", row["moment"]))
        self.cg_list.selection_set("0")
        self.cg_proposal_selected()

    def _cg_compile(self, proposal, source):
        return compile_prompt(proposal, allow_extras=source["allow_extras"],
                              message_area=self.cg_message_area.get(),
                              face_items=self.cg_face_items.get())

    def _cg_recompile(self):
        """Prompt options replace only prompts the user has not edited."""
        self._cg_store_edit()
        for row in self.cg_proposals:
            compiled = self._cg_compile(row, self.cg_source)
            if row["prompt"] == row["compiled"]:
                row["prompt"] = compiled
            row["compiled"] = compiled
        if self.cg_current is not None:
            self.cg_prompt.delete("1.0", "end")
            self.cg_prompt.insert("1.0", self.cg_proposals[self.cg_current]["prompt"])

    def _cg_store_edit(self):
        if self.cg_current is not None and self.cg_current < len(self.cg_proposals):
            self.cg_proposals[self.cg_current]["prompt"] = self.cg_prompt.get("1.0", "end").strip()

    def cg_proposal_selected(self):
        selection = self.cg_list.selection()
        if not selection:
            return
        self._cg_store_edit()
        self.cg_current = int(selection[0])
        row = self.cg_proposals[self.cg_current]
        self.cg_prompt.delete("1.0", "end")
        self.cg_prompt.insert("1.0", row["prompt"])
        extras = f"\n立ち絵のない人物: {row['extras']}" if row["extras"] else ""
        names = {item["tag"]: item["name"] for item in self.cg_source["candidates"]}
        items = "、".join(f"{names[member['tag']]}={member['face_items']}"
                         for member in row["cast"]
                         if member["face_items"] and member["visibility"].startswith("face"))
        extras += f"\n顔の小物: {items}" if items else ""
        self.cg_detail.set(f"描く内容: {row['moment']}\n見どころ: {row['visual_hook']}{extras}")

    def generate_cg(self):
        if self.cg_current is None:
            self.status.set("生成する案を選んでください。先に「LLMで案を作成」を実行します。")
            return
        self._cg_start([self.cg_current])

    def generate_cg_all(self):
        if not self.cg_proposals:
            self.status.set("先に「LLMで案を作成」を実行してください。")
            return
        self._cg_start(list(range(len(self.cg_proposals))))

    def _cg_start(self, indices):
        if self.process is not None:
            self.status.set("現在の処理が終わるまでお待ちください。")
            return
        self._cg_store_edit()
        self.cg_queue = list(indices)
        self._cg_next()

    def _cg_next(self):
        """Download only the chosen cast's portraits, then start one generation."""
        index = self.cg_queue.pop(0)
        proposal, catalog = self.cg_proposals[index], self.catalog
        if catalog is None:
            self.cg_queue = []
            self.status.set("作品一覧を更新してから生成してください。")
            return
        portraits = {row["tag"]: row["portrait"] for row in self.cg_source["candidates"]}
        chosen = [copy.deepcopy(portraits[member["tag"]]) for member in proposal["cast"]]
        directory = (Path(self.output.get().strip()).expanduser().resolve()
                     / "references" / uuid.uuid4().hex)
        self._async("cg_references", (self.catalog_revision, self.cg_revision), lambda: {
            "index": index,
            "references": [catalog.download_reference(row, directory) for row in chosen]})
        self.status.set(f"案{index + 1}の人物の立ち絵を読み込んでいます…")

    def _cg_references_ready(self, value, error):
        if error:
            self.cg_queue = []
            self.status.set(f"立ち絵の読み込みに失敗: {error}")
            return
        try:
            index = value["index"]
            data = self.build_cg_request(index, value["references"])
            run_dir = new_run(Path(self.output.get().strip()).expanduser().resolve(), "cg")
            self._start_process(
                [self._runtime_python(), "-u", "-X", "utf8",
                 str(Path(__file__).with_name("runner.py")),
                 "--request", str(run_dir / "request.json"), "--output-dir", str(run_dir)],
                run_dir, "generate", data)
            self.status.set(f"案{index + 1}を生成します。モデルを読み込み中…")
        except (OSError, ValueError) as exc:
            self.cg_queue = []
            messagebox.showerror("生成開始", str(exc))

    def preview_cg(self):
        """Open the whole chapter in the Tyrano player with this image in its interval."""
        selected = self.selected_result()
        if not selected or not (selected["request"].get("context") or {}).get("cg_proposal"):
            self.status.set("「印象的な一枚（LLM任せ）」タブで生成した結果を選んでください。")
            return
        record, path = copy.deepcopy(selected["request"]), selected["path"]
        server = self.server.get().strip()
        try:
            transition = (CG_TRANSITIONS[self.cg_transition.get()], int(self.cg_transition_ms.get()))
        except (KeyError, ValueError):
            self.status.set("CGの切り替えの時間はミリ秒の整数で指定してください。")
            return
        self.cg_preview_revision += 1
        self._async("cg_preview", self.cg_preview_revision,
                    lambda: prepare(record, path, server, transition))
        self.status.set("公開版の台本と素材を読み込み、CGの区間を差し替えています…")

    def _cg_preview_ready(self, value, error):
        if error:
            self.status.set(f"再生の準備に失敗: {error}")
            return
        url = self.cg_preview_server.add(*value)
        webbrowser.open(url)
        self.status.set("ブラウザーで開きました。開始画面の「つづきから」で、この場面の冒頭から"
                        f"再生します。 {url}")

    def build_cg_request(self, index, references):
        proposal, source = self.cg_proposals[index], self.cg_source
        scene = source["scene"]
        context = copy.deepcopy(scene.get("context", {}))
        context.update({
            "scene_id": scene["id"], "scene_label": scene["label"],
            "scene_preview": scene.get("preview", ""),
            "cg_proposal": {
                "run_dir": str(self.cg_run_dir), "index": index,
                "prompt_version": PROMPT_VERSION,
                "proposal": {key: copy.deepcopy(proposal[key]) for key in (
                    "moment", "visual_hook", "cast", "shot", "angle", "display_from",
                    "display_to", "layout", "light", "extras", "english_prompt")},
                "candidates": [{key: row[key] for key in ("tag", "character_id", "name")}
                               for row in source["candidates"]],
                "instruction": source["instruction"], "allow_extras": source["allow_extras"],
                "message_area": self.cg_message_area.get(),
                "face_items": self.cg_face_items.get(),
                "manually_edited": proposal["prompt"] != proposal["compiled"]}})
        data = {"schema_version": 1, "mode": "scene", "prompt": proposal["prompt"],
                "references": copy.deepcopy(references),
                "model_path": self.model_path.get().strip(),
                "width": int(self.cg_width.get()), "height": int(self.cg_height.get()),
                "steps": int(self.steps.get()), "seed": int(self.seed.get()),
                "dtype": self.dtype.get(), "cpu_offload": self.cpu_offload.get(),
                "use_kv_cache": self.use_kv_cache.get(),
                "reference_resolution": int(self.reference_resolution.get()),
                "text_encoder_offload": "layers" if self.text_encoder_layers.get() else "model",
                "transformer_storage": "fp8" if self.transformer_fp8.get() else "native",
                "vae_tiling": self.vae_tiling.get(),
                "transparent": False, "context": context}
        if self.three_seeds.get():
            data["seeds"] = [0, 1, 2]
        validate_request(data)
        return data
