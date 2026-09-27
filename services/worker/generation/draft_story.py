"""Text-only draft experiment: save prose first, leave literary evaluation to readers."""

from __future__ import annotations

import copy
import html
import json
import time
from pathlib import Path

from . import pipeline
from .cancellation import GenerationCancelled, check_cancelled
from .causal_runtime import digest, resource_limits
from .draft_context import fit_context
from .llm import ContextBudgetError, write_json
from .model_routing import RoutedLLM, model_configuration_identity
from .narrative import _story_cast
from .processes import gpu_lock
from .text_debug import _input_payload, _output_lock, extract_input
from .workflow_version import generator_protocol

POLICY = "story_draft_v1"
COMMON = (
    "日本語の物語を作ります。承認された人物・世界設定を尊重してください。"
    "実本文を一次資料にし、構成案や予定を既に起きた出来事と扱わないでください。"
    "人物の決断とその結果を次の行動につなぎ、前章の導入や発見を再演しないでください。"
    "必要な脇役や場所は自然に登場させて構いません。"
    "思考過程や作業説明を出力せず、依頼した文章だけを出力してください。"
)
OUTLINE = COMMON + (
    "短い全体構成と第1章の執筆依頼を合わせて1500字以内を目安に書いてください。"
    "中心の問題、各章で何が変わるか、次章が必要になる理由、結末への方向を示してください。"
    "第1章で起こす具体的な行動と到達点も含めます。自然文でよく、JSONやIDは不要です。"
)
HANDOFF = COMMON + (
    "保存済みの実本文から、次章への引き継ぎメモと執筆依頼を1000字以内を目安に書いてください。"
    "実際の出来事と選択、現在地と目的、残る問題・約束・迷い、初めてのように繰り返しては"
    "いけないこと、その結果から次章で必要になる行動と新しい変化を短くまとめます。"
    "実績と今後の予定を分け、未実施の予定は未実施とします。合否判定や修復依頼はしません。"
)
WRITER = COMMON + (
    "今回の章の本文を2000〜4000字程度を目安に、会話・行動・場面転換を含む小説として"
    "書いてください。字数は目安です。保存済みの前章の到達点から具体的な行動を起こし、"
    "人物の選択によって状況を変えてください。再紹介、既知情報の初耳扱い、同じ対立や和解"
    "のやり直しを避けてください。JSON・話者ID・あらすじ・検査結果は不要です。本文だけを返します。"
)


class DraftExecutionError(RuntimeError):
    """A required writing operation could not produce usable text."""


class DraftBudgetError(DraftExecutionError):
    """The persisted experiment resource allowance has been exhausted."""


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _reply_text(reply, *, writer):
    if not isinstance(reply, dict) or reply.get("tool_calls"):
        raise DraftExecutionError("Expected a plain text response, not a tool call.")
    content = reply.get("content")
    finish = reply.get("_finish_reason")
    if not isinstance(content, str) or not content.strip():
        raise DraftExecutionError("The model returned empty text.")
    if finish not in ({"stop", "length"} if writer else {"stop"}):
        raise DraftExecutionError(f"The model did not finish its response: {finish}.")
    return content


class DraftRun:
    def __init__(self, output, manifest, config, payload, target):
        self.output, self.manifest, self.config = output, manifest, config
        self.payload, self.target = payload, target
        self.limits = resource_limits(payload)
        self.max_calls = payload.get("workflow_limits", {}).get("max_calls", 160)
        self.timeout = config["llm"]["total_timeout_seconds"]
        self.path = output / "draft-state.json"
        if not self.path.exists() and any(
                path.is_file() and path.name not in {"experiment.json", ".text-debug.lock"}
                for path in output.rglob("*")):
            raise ValueError("Draft journal is missing; existing prose and usage must not be reset.")
        self.state = _read(self.path) if self.path.exists() else {
            "input_sha256": manifest["input_sha256"], "steps": {}, "chapters": [],
            "notes": [], "fallbacks": [], "sessions": [], "request_ordinal": 0,
            "partial": None,
        }
        if self.state.get("input_sha256") != manifest["input_sha256"]:
            raise ValueError("Draft journal belongs to another experiment.")
        for name in ("chapters", "notes", "requests", "llm"):
            (output / name).mkdir(exist_ok=True)
        self.session, self.started, self.llm = None, None, None
        self._verify_chapters()
        self._recover_requests()
        self.persist()

    def persist(self):
        write_json(self.path, self.state)

    def _verify_chapters(self):
        for number, chapter in enumerate(self.state["chapters"], 1):
            if chapter["number"] != number or digest(chapter["text"]) != chapter["sha256"]:
                raise ValueError("Saved chapter sequence or text was changed.")
            path = self.output / "chapters" / f"chapter-{number:03d}.md"
            if path.exists() and path.read_text(encoding="utf-8") != chapter["text"]:
                raise ValueError("Saved chapter file was changed.")
            if not path.exists():
                _text(path, chapter["text"])
        expected = {f"chapter-{number:03d}.md" for number in range(1, len(self.state["chapters"]) + 1)}
        if any(path.name not in expected and not path.name.endswith(".partial.md")
               for path in (self.output / "chapters").glob("chapter-*.md")):
            raise ValueError("A saved chapter has no matching draft journal entry.")
        if len(self.state["chapters"]) > self.target:
            raise ValueError("chapter_limit cannot precede a saved chapter.")

    def rows(self):
        return [row for step in self.state["steps"].values() for row in step["attempts"]]

    def _recover_requests(self):
        # A killed process may have written the model response before the journal.
        # Recover that response, otherwise retain its reserved maximum token charge.
        for step in self.state["steps"].values():
            for row in step["attempts"]:
                if row["status"] not in {"pending", "interrupted"} or "result" in step:
                    continue
                if row["status"] == "pending":
                    row.update(status="interrupted", elapsed_seconds=self.timeout)
                cache = self.output / row["cache_file"] if row.get("cache_file") else None
                if cache is None or not cache.is_file():
                    continue
                value = _read(cache)
                if value.get("request_sha256") != row["request_sha256"]:
                    raise ValueError("Cached response does not match the saved draft request.")
                choice = value["response"]["choices"][0]
                reply = {**choice["message"], "_finish_reason": choice.get("finish_reason")}
                row["reply"] = reply
                self._record_usage(row, value["response"].get("usage"))
                try:
                    _reply_text(reply, writer=row["purpose"] == "draft-chapter")
                except DraftExecutionError:
                    row["status"] = "failed"
                else:
                    row["status"], step["result"] = "completed", reply
        for session in self.state["sessions"]:
            if session["status"] == "running":
                # Unknown wall time is charged conservatively instead of reset.
                session.update(status="interrupted",
                               elapsed_seconds=max(session.get("elapsed_seconds", 0), self.timeout))

    @staticmethod
    def _record_usage(row, usage):
        if isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0
                                          for key in ("prompt_tokens", "completion_tokens")):
            row["usage"] = {key: usage[key] for key in ("prompt_tokens", "completion_tokens")}
            row["charged_tokens"] = sum(row["usage"].values())

    def elapsed(self):
        return sum(time.monotonic() - self.started if item is self.session else item["elapsed_seconds"]
                   for item in self.state["sessions"])

    def check_budget(self, number, extra_tokens=0):
        rows = self.rows()
        chapter = [row for row in rows if row["chapter"] == number]
        def charged(values):
            return sum(row.get("charged_tokens", row.get("reserved_tokens", 0)) for row in values)
        for used, limit, label in (
            (charged(rows) + extra_tokens, self.limits["max_story_tokens"], "story tokens"),
            (charged(chapter) + extra_tokens, self.limits["max_tokens"], "chapter tokens"),
            (self.elapsed(), self.limits["max_story_elapsed_seconds"], "story elapsed time"),
            (sum(row.get("elapsed_seconds", 0) for row in chapter),
             self.limits["max_elapsed_seconds"], "chapter elapsed time"),
        ):
            if used > limit:
                raise DraftBudgetError(f"Draft resource budget exceeded: {label} ({used} > {limit}).")
        if extra_tokens and sum(bool(row.get("dispatched")) for row in chapter) >= self.max_calls:
            raise DraftBudgetError("Draft chapter request count limit reached.")

    def save(self, status, error=None):
        if self.session is not None:
            self.session["elapsed_seconds"] = time.monotonic() - self.started
        self.persist()
        rows = self.rows()
        measured = [row for row in rows if "usage" in row]
        dispatched = [row for row in rows if row.get("dispatched")]
        report = {
            "schema_version": 1, "workflow_policy": self.manifest["workflow_policy"],
            "execution_mode": "text_only",
            "input_sha256": self.manifest["input_sha256"], "status": status,
            "approved_chapter_count": self.manifest["approved_chapter_count"],
            "target_chapter_count": self.target, "saved_chapter_count": len(self.state["chapters"]),
            "partial_chapter": self.state["partial"]["number"] if self.state["partial"] else None,
            "quality_acceptance": "not_evaluated", "human_review": "not_performed",
            "public_build_created": False, "fallbacks": self.state["fallbacks"],
            "resource_limits": {**self.limits, "max_calls_per_chapter": self.max_calls},
            "metrics": {
                "requests": len(dispatched), "elapsed_seconds": round(self.elapsed(), 3),
                "prompt_tokens_measured": sum(row["usage"]["prompt_tokens"] for row in measured),
                "completion_tokens_measured": sum(row["usage"]["completion_tokens"] for row in measured),
                "charged_tokens": sum(row.get("charged_tokens", row.get("reserved_tokens", 0)) for row in rows),
                "unmeasured_requests": sum("usage" not in row for row in dispatched),
                "model_loads": sum(item.get("model_loads", 0) for item in self.state["sessions"]),
                "model_switches": sum(item.get("model_switches", 0) for item in self.state["sessions"]),
            },
        }
        if error:
            report["error"] = error
        write_json(self.output / "report.json", report)
        self.render(status)
        return report

    def render(self, status):
        title = self.manifest["approval_snapshot"]["world"]["result"]["title"]
        sections = [f"# {title}\n\n下書き・内容未評価。状態: {status}\n"]
        page_sections = []
        chapters = [*self.state["chapters"]]
        if self.state["partial"]:
            chapters.append({**self.state["partial"], "partial": True})
        for chapter in chapters:
            label = f"第{chapter['number']}章" + ("（途中原稿）" if chapter.get("partial") else "")
            sections.append(f"## {label}\n\n{chapter['text']}\n")
            page_sections.append(f"<h2>{label}</h2><div class='prose'>{html.escape(chapter['text'])}</div>")
        _text(self.output / "story.md", "\n".join(sections))
        page = ("<!doctype html><html lang='ja'><meta charset='utf-8'>"
                f"<title>{html.escape(title)}</title><style>"
                "body{max-width:48rem;margin:3rem auto;padding:0 1.5rem;line-height:2;"
                "font-family:serif;background:#faf8f2;color:#282521}"
                ".prose{white-space:pre-wrap;overflow-wrap:anywhere}h2{margin-top:3rem}"
                f"</style><h1>{html.escape(title)}</h1><p>下書き・内容未評価。状態: {status}</p>"
                + "\n".join(page_sections) + "</html>")
        _text(self.output / "story.html", page)

    def request(self, key, purpose, number, system, context):
        step = self.state["steps"].setdefault(key, {"attempts": []})
        if "result" in step:
            return step["result"]
        while len(step["attempts"]) < 2:
            check_cancelled()
            self.check_budget(number)
            self.llm.select_purpose(purpose)
            row = {"attempt": len(step["attempts"]) + 1, "chapter": number, "purpose": purpose,
                   "status": "pending", "profile": copy.deepcopy(self.llm.profile),
                   "reserved_tokens": 0, "dispatched": False}
            step["attempts"].append(row)
            self.persist()
            started, trace_start = time.monotonic(), len(self.llm.trace)
            try:
                self.llm._ensure_runtime()
                messages, selection = fit_context(self.llm, purpose, system, **context)
                row["selection"] = selection
                budget = selection["budget"]
                reservation = budget["prompt_tokens"] + budget["output_tokens"]
                self.check_budget(number, reservation)
                ordinal = self.state["request_ordinal"] + 1
                request, fingerprint = self.llm._chat_request(ordinal, messages, {})
                row.update(request=ordinal, request_sha256=fingerprint,
                           cache_file=f"llm/{ordinal:02d}-{purpose}-{fingerprint[:16]}.json",
                           reserved_tokens=reservation, dispatched=True)
                self.state["request_ordinal"] = ordinal
                write_json(self.output / "requests" / f"{key}-{row['attempt']}.json",
                           {"request": request, "selection": selection, "request_sha256": fingerprint})
                self.persist()
                reply = self.llm.chat(purpose, messages, allow_truncated=True)
                row["reply"] = reply
                _reply_text(reply, writer=purpose == "draft-chapter")
                step["result"], row["status"] = reply, "completed"
            except (KeyboardInterrupt, GenerationCancelled):
                row["status"] = "interrupted"
                raise
            except (ContextBudgetError, DraftBudgetError) as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                if hasattr(exc, "selection"):
                    row["selection"] = exc.selection
                raise
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                row["elapsed_seconds"] = time.monotonic() - started
                for event in reversed(self.llm.trace[trace_start:]):
                    if event.get("type") == "llm_generation":
                        self._record_usage(row, event.get("usage"))
                        break
                self.llm.requests = self.state["request_ordinal"]
                self.save("running")
            if "result" in step:
                return step["result"]
            self.check_budget(number)
        raise DraftExecutionError(f"{key}: technical retry exhausted. "
                                  + str(step["attempts"][-1].get("error", "interrupted")))

    def write_chapters(self):
        snapshot = self.payload["approval_snapshot"]
        setting = {"world": snapshot["world"]["result"],
                   "characters": _story_cast([row["result"] for row in snapshot["characters"]]),
                   "relationships": snapshot["relationships"]["result"]}
        context = {"setting": setting, "outline": "", "brief": "", "chapters": [], "notes": []}
        outline = self.request("outline", "draft-outline", 1, OUTLINE, context)["content"]
        _text(self.output / "outline.md", outline)
        count = self.manifest["approved_chapter_count"]
        for number in range(len(self.state["chapters"]) + 1, self.target + 1):
            context = {"setting": setting, "outline": outline, "brief": "",
                       "chapters": self.state["chapters"], "notes": self.state["notes"]}
            position = f"全{count}章の第{number}章です。"
            if number == count:
                position += "最終章なので、ここまでの結果を受けて中心の問題に決着をつけてください。"
            if number > 1:
                try:
                    brief = self.request(f"handoff-{number:03d}", "draft-handoff", number,
                                         HANDOFF + position, context)["content"]
                except DraftBudgetError:
                    raise
                except (DraftExecutionError, ContextBudgetError) as exc:
                    brief = ("引き継ぎメモなし。保存済みの前章本文の結果から続きを判断してください。"
                             "構成案の予定を実績として扱わないでください。")
                    if not any(row["chapter"] == number for row in self.state["fallbacks"]):
                        self.state["fallbacks"].append({"chapter": number, "reason": str(exc)})
                    self.persist()
                else:
                    if not any(row["number"] == number - 1 for row in self.state["notes"]):
                        self.state["notes"].append({"number": number - 1, "text": brief})
                        self.persist()
                        _text(self.output / "notes" / f"chapter-{number - 1:03d}.md", brief)
                context["brief"] = brief
            reply = self.request(f"chapter-{number:03d}", "draft-chapter", number,
                                 WRITER + position, context)
            text = reply["content"]
            self.partial(number, text)
            if reply["_finish_reason"] == "length":
                reply = self.request(f"continue-{number:03d}", "draft-chapter", number,
                                     WRITER + position + "今回の章の執筆済み部分の末尾から続筆し、"
                                     "章を完結させてください。執筆済み部分は再掲しません。",
                                     {**context, "continuation": text})
                text += reply["content"]
                self.partial(number, text)
                if reply["_finish_reason"] == "length":
                    raise DraftExecutionError("One continuation also reached the output limit; partial prose saved.")
            self.state["chapters"].append({"number": number, "text": text, "sha256": digest(text)})
            self.state["partial"] = None
            self.persist()
            _text(self.output / "chapters" / f"chapter-{number:03d}.md", text)
            partial_path = self.output / "chapters" / f"chapter-{number:03d}.partial.md"
            partial_path.unlink(missing_ok=True)
            self.save("running")
            self.check_budget(number)

    def partial(self, number, text):
        self.state["partial"] = {"number": number, "text": text}
        self.persist()
        _text(self.output / "chapters" / f"chapter-{number:03d}.partial.md", text)
        self.save("running")

    def run(self):
        self.started = time.monotonic()
        self.session = {"status": "running", "elapsed_seconds": 0}
        self.state["sessions"].append(self.session)
        self.save("running")
        status, error = "running", None
        try:
            self.check_budget(max(1, len(self.state["chapters"])))
            if len(self.state["chapters"]) < self.target:
                with (
                    gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock",
                             self.config["gpu_lock_timeout_seconds"]),
                    RoutedLLM(pipeline.ROOT, self.config, self.payload, self.output / "llm") as llm,
                ):
                    self.llm = llm
                    llm.requests = self.state["request_ordinal"]
                    # Runtime logs from earlier invocations remain inspectable.
                    llm._load_number = max((int(path.name.split("-", 1)[0])
                                            for path in (llm.output / "runtimes").glob("*")
                                            if path.name.split("-", 1)[0].isdigit()), default=0)
                    self.write_chapters()
            status = ("draft_complete" if len(self.state["chapters"]) ==
                      self.manifest["approved_chapter_count"] else "chapter_limit_reached")
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, GenerationCancelled)) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.session["status"] = status
            if self.llm is not None:
                loads = sum(row.get("type") == "model_switch" and row.get("status") == "ready"
                            for row in self.llm.trace)
                self.session.update(model_loads=loads, model_switches=max(0, loads - 1))
                write_json(self.output / "llm" / f"session-{len(self.state['sessions']):03d}-metrics.json",
                           self.llm.trace)
            report = self.save(status, error)
        return report


def run_draft_debug(input_value: dict, output_dir: Path, *,
                    context_size: int | None = None, chapter_limit: int | None = None,
                    resume: bool = False, seed: int | None = None) -> dict:
    return _run_text_experiment(input_value, output_dir, context_size=context_size,
                                chapter_limit=chapter_limit, resume=resume, seed=seed,
                                policy=POLICY, runner_type=DraftRun)


def _run_text_experiment(input_value: dict, output_dir: Path, *,
                         context_size=None, chapter_limit=None, resume=False, seed=None,
                         policy, runner_type, plot_only=False) -> dict:
    """Shared experiment identity/ownership only; the runner owns its output contract."""
    snapshot, profile, source_seed = extract_input(input_value)
    source = _input_payload(input_value)
    seed = source_seed if seed is None else seed
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1.")
    if context_size is not None:
        if type(context_size) is not int or context_size < 1024:
            raise ValueError("context_size must be an integer of at least 1024 tokens.")
        profile["context_size"] = context_size
    count = snapshot["world"]["result"]["chapterCount"]
    target = count if chapter_limit is None else chapter_limit
    if type(target) is not int or not 1 <= target <= count:
        raise ValueError("chapter_limit must be within the approved chapter count.")
    profiles = copy.deepcopy(source.get("profiles", {}))
    if not isinstance(profiles, dict) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, dict)
            for key, value in profiles.items()):
        raise TypeError("Purpose profiles must map nonempty names to profile objects.")
    payload = {"approval_snapshot": snapshot, "seed": seed, "profile": profile,
               "profiles": profiles, "workflow_policy": policy, "story_workflow_version": 2,
               "workflow_limits": copy.deepcopy(source.get("workflow_limits", {}))}
    if plot_only:
        if policy != "script_continuation_v1":
            raise ValueError("plot_only is available only for script_continuation_v1.")
        payload["plot_only"] = True
    resource_limits(payload)
    config = pipeline.load_config()
    identity = {"schema_version": 1, "execution_mode": "text_only", "workflow_policy": policy,
                "generator_protocol": generator_protocol("causal", policy),
                "approval_snapshot": snapshot, "approved_chapter_count": count,
                "seed": seed, "profile": profile, "profiles": profiles,
                "workflow_limits": payload["workflow_limits"], "generation_config": config,
                "model_configuration": model_configuration_identity(pipeline.ROOT, config)}
    if policy == "script_continuation_v1":
        from .script_cast import ScriptOptions
        from .script_examples import example_catalog_identity

        options = ScriptOptions.model_validate(source.get("script_options", {})).model_dump(mode="json")
        payload["script_options"] = options
        identity["script_options"] = options
        identity["script_examples"] = example_catalog_identity()
    fingerprint = digest(identity)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        path = output / "experiment.json"
        if path.exists():
            if not resume:
                raise ValueError("Use a new output directory or --resume.")
            manifest = _read(path)
            if (manifest.get("input_sha256") != fingerprint
                    or digest({key: manifest.get(key) for key in identity}) != fingerprint):
                raise ValueError("Experiment input, model settings, protocol or resource limits changed.")
        else:
            if any(path.name != ".text-debug.lock" for path in output.iterdir()):
                raise ValueError("A new experiment requires an empty output directory.")
            manifest = {**identity, "input_sha256": fingerprint}
            write_json(path, manifest)
        return runner_type(output, manifest, config, payload, target).run()
