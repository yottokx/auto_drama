"""Read the coordinator's selected chapters without creating audio jobs or assets.

The client sends GET requests only. Selection and frozen-history filtering belong
to the coordinator, rather than being reimplemented from a mutable local database.
Published builds pin their narrative through their validation record.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_CONTEXT_CHAPTERS = 24
MAX_CONTEXT_CAST = 32
MAX_CONTEXT_TEXT = 2000


class CatalogError(ValueError):
    """A coordinator response could not be read as a selected scene catalog."""


@dataclass(frozen=True)
class Project:
    id: str
    title: str


@dataclass(frozen=True)
class Scene:
    id: str
    label: str
    context: dict
    preview: str


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    scenes: list[Scene]


@dataclass(frozen=True)
class _NarrativeSource:
    document: dict | None
    artifact_id: str | None
    export_artifact_id: str | None = None
    approval: dict | None = None
    warnings: tuple[str, ...] = ()


def _story_context(narrative: dict, approval: dict | None, sources: dict,
                   warnings: list[str]) -> dict:
    """Bound immutable story material; a plan describes intent, not scene facts."""
    truncated = []

    def text(value, field, limit=MAX_CONTEXT_TEXT):
        value = value if isinstance(value, str) else ""
        if len(value) > limit:
            truncated.append(field)
            marker = "\n[… omitted …]\n"
            head = (limit - len(marker)) * 2 // 3
            tail = limit - len(marker) - head
            return value[:head] + marker + value[-tail:]
        return value

    def rows(value, field, limit):
        value = [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []
        if len(value) > limit:
            truncated.append(field)
        return value[:limit]

    approval = approval or {}
    world = approval.get("world") or {}
    world = world.get("result", world) if isinstance(world, dict) else {}
    world = world if isinstance(world, dict) else {}
    brief = {key: text(world.get(key), "brief." + key) for key in
             ("title", "genre", "mood", "notes", "prompt", "setting")}
    brief["chapter_count"] = world.get("chapterCount") if isinstance(
        world.get("chapterCount"), int) else None

    def chapter(value):
        return {"number": value.get("number"),
                **{key: text(value.get(key), "outline.chapters." + key, 1000)
                   for key in ("title", "role", "summary")}}

    original_outline = narrative.get("outline") or {}
    original_outline = original_outline if isinstance(original_outline, dict) else {}
    all_chapters = original_outline.get("chapters") or []
    current = next((row for row in all_chapters if isinstance(row, dict)
                    and row.get("number") == narrative.get("chapter_number")), {}) if isinstance(
                        all_chapters, list) else {}
    # Preserve this chapter even when a long series exceeds the overview limit.
    current_chapter = chapter(current) if current else {}
    outline = {
        "ending": text(original_outline.get("ending"), "outline.ending"),
        "chapters": [chapter(row) for row in rows(
            all_chapters, "outline.chapters", MAX_CONTEXT_CHAPTERS)],
        "character_arcs": [{"character_id": row.get("character_id"),
                            "change": text(row.get("change"), "outline.character_arcs.change", 1000)}
                           for row in rows(original_outline.get("character_arcs"),
                                           "outline.character_arcs", 10)],
        "foreshadowing": [{"setup_chapter": row.get("setup_chapter"),
                           "payoff_chapter": row.get("payoff_chapter"),
                           "detail": text(row.get("detail"), "outline.foreshadowing.detail", 500)}
                          for row in rows(original_outline.get("foreshadowing"),
                                          "outline.foreshadowing", 30)],
    }
    cast = []
    seen_cast = set()
    character_rows = rows(approval.get("characters"), "cast.approved", MAX_CONTEXT_CAST)
    character_rows += rows(narrative.get("supporting_characters"), "cast.supporting", 100)
    character_rows = [character for row in character_rows
                      if isinstance(character := row.get("result", row), dict)]
    present_ids = {identifier for scene in narrative.get("scenes", [])
                   for identifier in (scene.get("plan") or {}).get("character_ids", [])}
    # Keep actors in this chapter before background cast when a long story hits
    # the overview bound. Stable sorting preserves the source order within each.
    character_rows.sort(key=lambda row: row.get("id") not in present_ids)
    for character in character_rows:
        if not character.get("id"):
            continue
        identifier = str(character["id"])
        if identifier in seen_cast:
            continue
        if len(cast) == MAX_CONTEXT_CAST:
            truncated.append("cast")
            break
        seen_cast.add(identifier)
        cast.append({"id": identifier,
                     "name": text(character.get("name"), "cast.name", 200),
                     "role": text(character.get("role"), "cast.role", 300),
                     "settings": text(character.get("settings") or character.get("freeform"),
                                      "cast.settings", 1000)})
    relationships = approval.get("relationships") or {}
    if isinstance(relationships, dict):
        relationships = relationships.get("result", relationships)
    pairs = relationships.get("pairs") if isinstance(relationships, dict) else None
    pairs = [{"character_ids": row.get("characterIds", []),
              "summary": text(row.get("summary"), "relationships.summary", 800)}
             for row in rows(pairs, "relationships", 45)]
    missing = []
    if not world:
        missing.append("brief")
    if not outline["ending"] and not outline["chapters"]:
        missing.append("outline")
    if not current_chapter:
        missing.append("chapter")
    if not cast:
        missing.append("cast")
    available = bool(world or original_outline or cast)
    return {"schema_version": 1,
            "status": "partial" if available and missing else "complete" if available else "unavailable",
            "brief": brief, "outline": outline, "chapter": current_chapter,
            "cast": cast, "relationships": pairs, "sources": sources,
            "missing": missing, "warnings": [warning[:500] for warning in warnings[:8]],
            "truncated": list(dict.fromkeys(truncated))}


def _cast_appearance(narrative: dict, approval: dict | None) -> dict:
    """Approved looks by character ID, kept beside the story context for image experiments."""
    rows = list((approval or {}).get("characters") or [])
    rows += list(narrative.get("supporting_characters") or [])
    result = {}
    for row in rows:
        character = row.get("result", row) if isinstance(row, dict) else None
        if (isinstance(character, dict) and character.get("id")
                and isinstance(character.get("appearance"), str)):
            result.setdefault(str(character["id"]), character["appearance"][:600])
    return result


def fetch_bytes(url: str, timeout: float = 20) -> bytes:
    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
            content = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise CatalogError(f"Coordinator の読み込みに失敗しました（HTTP {exc.code}）。") from exc
    except (URLError, OSError) as exc:
        raise CatalogError(f"Coordinator に接続できません: {exc}") from exc
    if len(content) > MAX_RESPONSE_BYTES:
        raise CatalogError("Coordinator の応答が32MiBを超えています。")
    return content


def fetch_json(url: str, timeout: float = 20) -> dict:
    content = fetch_bytes(url, timeout)
    try:
        result = json.loads(content)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CatalogError("Coordinator の応答をJSONとして読めません。") from exc
    if not isinstance(result, dict):
        raise CatalogError("Coordinator のJSON応答がオブジェクトではありません。")
    return result


class CoordinatorCatalog:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 20):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CatalogError("Coordinator のURLには http:// または https:// を指定してください。")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _json(self, path: str) -> dict:
        return fetch_json(self.base_url + path, self.timeout)

    def _artifact(self, artifact_id: str) -> dict:
        return self._json(f"/api/artifacts/{quote(artifact_id, safe='')}/content")

    def projects(self) -> list[Project]:
        rows = self._json("/api/projects").get("projects", [])
        if not isinstance(rows, list):
            raise CatalogError("作品一覧の形式が正しくありません。")
        return [Project(str(row["id"]), str(row.get("title") or "無題の作品"))
                for row in rows if isinstance(row, dict) and row.get("id")]

    def _narrative(self, chapter: dict) -> _NarrativeSource:
        build = chapter.get("build") or {}
        validation = build.get("validation") or {}
        if isinstance(validation, str):
            try:
                validation = json.loads(validation)
            except ValueError as exc:
                raise CatalogError("公開版の本文参照が正しくありません。") from exc
        if not isinstance(validation, dict):
            raise CatalogError("公開版の本文参照が正しくありません。")
        # Prefer the published text even if a later adopted narrative exists.
        pinned_id = validation.get("narrative_artifact_id")
        if pinned_id:
            return _NarrativeSource(self._artifact(str(pinned_id)), str(pinned_id))
        if build.get("export_artifact_id"):
            # Older builds do not have a lineage field, but contain immutable prose.
            artifact_id = str(build["export_artifact_id"])
            content = fetch_bytes(
                self.base_url + f"/api/artifacts/{quote(artifact_id, safe='')}/content",
                self.timeout,
            )
            try:
                with ZipFile(io.BytesIO(content)) as archive:
                    info = archive.getinfo("narrative.json")
                    if info.file_size > MAX_RESPONSE_BYTES:
                        raise CatalogError("公開版の本文が32MiBを超えています。")
                    narrative = json.loads(archive.read(info))
                    approval = None
                    warnings = []
                    # The same immutable export binds the legacy prose and setup.
                    # Missing optional approval must not discard readable prose.
                    try:
                        approval_info = archive.getinfo("approval.json")
                        if approval_info.file_size > MAX_RESPONSE_BYTES:
                            raise ValueError("approval.json exceeds the response limit")
                        approval = json.loads(archive.read(approval_info))
                        if not isinstance(approval, dict):
                            raise TypeError("approval.json is not an object")
                    except (BadZipFile, KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
                        approval = None
                        warnings.append("公開エクスポートの作品設定を読み込めません: " + str(exc))
            except (BadZipFile, KeyError, ValueError, UnicodeDecodeError) as exc:
                raise CatalogError("公開版の本文をエクスポートから読み込めません。") from exc
            if not isinstance(narrative, dict):
                raise CatalogError("公開版の本文の形式が正しくありません。")
            return _NarrativeSource(narrative, None, artifact_id, approval, tuple(warnings))
        artifact_id = chapter.get("narrative_artifact_id")
        return (_NarrativeSource(self._artifact(str(artifact_id)), str(artifact_id))
                if artifact_id else _NarrativeSource(None, None))

    def _approval(self, production: dict, project_id: str) -> tuple[dict | None, list[str]]:
        """Read the production's immutable M2 approval, never the current M2 draft."""
        artifact_id = production.get("approval_artifact_id")
        if not artifact_id:
            return None, ["制作版に固定された作品設定の参照がありません。"]
        try:
            approval = self._artifact(str(artifact_id))
            if (approval.get("projectId") != project_id
                    or (production.get("approval_id")
                        and approval.get("id") != production["approval_id"])):
                raise CatalogError("作品設定の承認版が選択中の制作版と一致しません。")
            return approval, []
        except CatalogError as exc:
            return None, [str(exc)]

    def chapters(self, project: Project | str) -> list[Chapter]:
        project_id = project.id if isinstance(project, Project) else str(project)
        project_title = project.title if isinstance(project, Project) else ""
        selected = self._json(f"/api/m3/projects/{quote(project_id, safe='')}")
        production = selected.get("production")
        if not production:
            return []
        if not isinstance(production, dict):
            raise CatalogError("選択中の制作版の形式が正しくありません。")
        rows = production.get("chapters")
        if rows is None:
            rows = [{**production, "production_id": production.get("id")}]
        if not isinstance(rows, list):
            raise CatalogError("章一覧の形式が正しくありません。")
        chapters = []
        seen_numbers = set()
        approval_loaded = False
        approval = None
        approval_warnings = []
        for row in rows:
            if not isinstance(row, dict):
                raise CatalogError("章一覧の形式が正しくありません。")
            try:
                number = int(row["chapter_number"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CatalogError("章番号が正しくありません。") from exc
            if number < 1 or number in seen_numbers:
                raise CatalogError("章番号が重複しているか範囲外です。")
            seen_numbers.add(number)
            source = self._narrative(row)
            narrative, artifact_id = source.document, source.artifact_id
            if narrative is None:
                chapters.append(Chapter(number, f"第{number}章（本文未生成）", []))
                continue
            if narrative.get("chapter_number", number) != number:
                raise CatalogError("本文の章番号が選択中の章と一致しません。")
            storyline_id = production.get("storyline_id") or production.get("id")
            if narrative.get("storyline_id") and narrative["storyline_id"] != storyline_id:
                raise CatalogError("本文の制作版が選択中の作品と一致しません。")
            if source.export_artifact_id:
                chapter_approval = source.approval
                warnings = list(source.warnings)
                approval_source = "published_export" if chapter_approval else None
                approval_artifact_id = None
            else:
                if not approval_loaded:
                    approval, approval_warnings = self._approval(production, project_id)
                    approval_loaded = True
                chapter_approval = approval
                warnings = approval_warnings
                approval_source = "production_pinned_approval" if approval else None
                approval_artifact_id = production.get("approval_artifact_id")
            story_context = _story_context(narrative, chapter_approval, {
                "project_id": project_id,
                "storyline_id": storyline_id,
                "production_id": row.get("production_id"),
                "narrative_artifact_id": artifact_id,
                "published_build_id": (row.get("build") or {}).get("id"),
                "history_frozen": bool(production.get("history_frozen")),
                "approval_id": (chapter_approval or {}).get("id") or (
                    chapter_approval or {}).get("approval_id"),
                "approval_artifact_id": approval_artifact_id,
                "approval_export_artifact_id": source.export_artifact_id,
                "outline": "selected_narrative" if narrative.get("outline") else None,
                "approval": approval_source,
            }, warnings)
            locations = {value["id"]: value for value in narrative.get("locations", [])}
            scenes = []
            for index, value in enumerate(narrative.get("scenes", []), 1):
                plan = value.get("plan") or {}
                location = locations.get(plan.get("location_id"), {})
                objectives = str(plan.get("objectives") or "")
                place = str(location.get("name") or "場所未指定")
                label = f"{index}. {place}"
                if objectives:
                    label += " — " + objectives[:44].replace("\n", " ")
                context = {
                    **value,
                    "scene_id": str(value["id"]),
                    "scene_label": label,
                    "location": location,
                    "project_id": project_id,
                    "project_title": project_title,
                    "chapter_number": number,
                    "chapter_title": narrative.get("title", f"第{number}章"),
                    "production_id": row.get("production_id"),
                    "storyline_id": storyline_id,
                    "narrative_artifact_id": artifact_id,
                    "published_build_id": (row.get("build") or {}).get("id"),
                    "history_frozen": bool(production.get("history_frozen")),
                    "story_context": story_context,
                    "cast_appearance": _cast_appearance(narrative, chapter_approval),
                }
                preview = "\n".join([
                    f"場所: {place} / {location.get('time_of_day', '')}",
                    f"目的: {objectives}",
                    f"開始: {plan.get('start_state', '')}",
                    f"終了: {plan.get('end_state', '')}",
                    f"雰囲気: {plan.get('atmosphere', '')}",
                    "", "本文:", str(value.get("raw_text") or ""),
                ])
                scenes.append(Scene(str(value["id"]), label, context, preview))
            chapters.append(Chapter(number, str(narrative.get("title") or f"第{number}章"), scenes))
        return sorted(chapters, key=lambda chapter: chapter.number)
