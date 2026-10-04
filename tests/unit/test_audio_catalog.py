"""Selected-history and immutable-publication boundaries for the BGM browser."""

import io
import json
from zipfile import ZipFile

import pytest

from scripts.audio import catalog
from scripts.audio.catalog import CatalogError, CoordinatorCatalog, Project


def narrative(number=1, storyline="selected-story", scene_id="scene-1", title="再会"):
    return {
        "chapter_number": number, "title": title, "storyline_id": storyline,
        "locations": [{"id": "cafe", "name": "喫茶店", "description": "窓辺の席",
                       "time_of_day": "夕方", "atmosphere": "静か"}],
        "scenes": [{
            "id": scene_id, "raw_text": "窓辺に二人は座った。懐かしい笑顔に安堵する。",
            "plan": {"location_id": "cafe", "character_ids": ["alice", "bob"],
                     "objectives": "旧友との再会", "start_state": "互いに緊張している",
                     "end_state": "信頼を取り戻す", "atmosphere": "温かな安堵",
                     "required_events": [{"id": "event", "description": "昔の話をする"}]},
            "utterances": [{"id": "line", "display_text": "久しぶりだね。",
                            "inner_emotion": "再会の喜び", "voice_emotion": "happy"}],
            "directions": [],
        }],
    }


def stub_json(monkeypatch, responses):
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        response = responses[url.removeprefix("http://coordinator")]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(catalog, "fetch_json", fetch)
    return calls


def test_catalog_uses_selected_frozen_chapters_and_full_scene_context(monkeypatch):
    calls = stub_json(monkeypatch, {
        "/api/projects": {"projects": [{"id": "project", "title": "テスト作品"}]},
        "/api/m3/projects/project": {"production": {
            "id": "selected-story", "storyline_id": "selected-story", "history_frozen": True,
            "chapters": [
                {"chapter_number": 2, "production_id": "selected-chapter-2",
                 "narrative_artifact_id": "frozen-2", "build": None},
                {"chapter_number": 1, "production_id": "selected-story",
                 "narrative_artifact_id": "frozen-1", "build": None},
                {"chapter_number": 3, "production_id": None, "narrative_artifact_id": None},
            ],
        }},
        "/api/artifacts/frozen-1/content": narrative(),
        "/api/artifacts/frozen-2/content": narrative(2, scene_id="scene-2", title="帰路"),
    })
    client = CoordinatorCatalog("http://coordinator")
    projects = client.projects()
    assert projects == [Project("project", "テスト作品")]
    chapters = client.chapters(projects[0])
    assert [chapter.number for chapter in chapters] == [1, 2, 3]
    assert [chapter.title for chapter in chapters][:2] == ["再会", "帰路"]
    assert chapters[2].scenes == []
    scene = chapters[0].scenes[0]
    assert scene.id == "scene-1"
    assert "喫茶店" in scene.label and "旧友との再会" in scene.label
    assert scene.context["raw_text"] == narrative()["scenes"][0]["raw_text"]
    assert scene.context["scene_id"] == scene.id
    assert scene.context["scene_label"] == scene.label
    assert scene.context["utterances"][0]["display_text"] == "久しぶりだね。"
    assert scene.context["plan"]["end_state"] == "信頼を取り戻す"
    assert scene.context["location"]["time_of_day"] == "夕方"
    assert scene.context["history_frozen"] is True
    assert scene.context["storyline_id"] == "selected-story"
    assert scene.context["narrative_artifact_id"] == "frozen-1"
    assert "窓辺に二人" in scene.preview and "温かな安堵" in scene.preview
    assert calls == ["http://coordinator/api/projects",
                     "http://coordinator/api/m3/projects/project",
                     "http://coordinator/api/artifacts/frozen-2/content",
                     "http://coordinator/api/artifacts/frozen-1/content"]


def test_published_build_pins_prose_instead_of_later_adopted_artifact(monkeypatch):
    calls = stub_json(monkeypatch, {
        "/api/m3/projects/project": {"production": {
            "id": "selected-story", "chapters": [{
                "chapter_number": 1, "production_id": "selected-story",
                "narrative_artifact_id": "new-unpublished-text",
                "build": {"id": "published", "validation": {"narrative_artifact_id": "published-text"}},
            }],
        }},
        "/api/artifacts/published-text/content": narrative(title="公開本文"),
    })
    chapter = CoordinatorCatalog("http://coordinator").chapters("project")[0]
    assert chapter.title == "公開本文"
    assert chapter.scenes[0].context["published_build_id"] == "published"
    assert calls[-1].endswith("/published-text/content")
    assert not any("new-unpublished-text" in url for url in calls)


def test_legacy_publication_reads_immutable_export_narrative(monkeypatch):
    stub_json(monkeypatch, {
        "/api/m3/projects/project": {"production": {
            "id": "selected-story", "chapters": [{
                "chapter_number": 1, "narrative_artifact_id": "mutable",
                "build": {"id": "legacy", "export_artifact_id": "export", "validation": {}},
            }],
        }},
    })
    content = io.BytesIO()
    with ZipFile(content, "w") as archive:
        archive.writestr("narrative.json", json.dumps(narrative(title="旧公開版")))
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return content.getvalue()

    monkeypatch.setattr(catalog, "fetch_bytes", fetch)
    chapter = CoordinatorCatalog("http://coordinator").chapters("project")[0]
    assert chapter.title == "旧公開版"
    assert calls == ["http://coordinator/api/artifacts/export/content"]


@pytest.mark.parametrize("document", [narrative(number=2), narrative(storyline="other-story")])
def test_mismatched_selected_chapter_is_rejected(monkeypatch, document):
    stub_json(monkeypatch, {
        "/api/m3/projects/project": {"production": {
            "id": "selected-story", "chapters": [{
                "chapter_number": 1, "narrative_artifact_id": "artifact", "build": None,
            }],
        }},
        "/api/artifacts/artifact/content": document,
    })
    with pytest.raises(CatalogError, match="一致しません"):
        CoordinatorCatalog("http://coordinator").chapters("project")


def test_no_selected_production_is_empty(monkeypatch):
    calls = stub_json(monkeypatch, {
        "/api/m3/projects/project": {"project_id": "project", "production": None},
    })
    assert CoordinatorCatalog("http://coordinator").chapters("project") == []
    assert len(calls) == 1


def test_http_client_only_reads_with_get(monkeypatch):
    calls = []

    def open_request(request, timeout):
        calls.append((request.get_method(), request.full_url, request.data, timeout))
        return io.BytesIO(b'{"projects": []}')

    monkeypatch.setattr(catalog, "urlopen", open_request)
    assert CoordinatorCatalog("http://coordinator", timeout=8).projects() == []
    assert calls == [("GET", "http://coordinator/api/projects", None, 8)]


def story_narrative(number=1, title="出会い", ending="くだらない夜の友好"):
    document = narrative(number=number, title=title)
    document["outline"] = {
        "ending": ending,
        "character_arcs": [{"character_id": "alice", "change": "威厳より日常を楽しむ"}],
        "chapters": [
            {"number": 1, "title": "出会い", "role": "大げさな要求と淡々とした応対の対比",
             "summary": "予算内の食べ物を探し、身近な便利さに驚く。"},
            {"number": 2, "title": "小さな争い", "role": "敵同士をくだらない争いへ導く",
             "summary": "食べ物をきっかけに、張り合いながら共通点を見つける。"},
        ],
        "foreshadowing": [{"setup_chapter": 1, "payoff_chapter": 2,
                           "detail": "同じ食べ物の好みが休戦につながる。"}],
    }
    return document


def story_approval():
    return {
        "id": "pinned-approval", "projectId": "project",
        "world": {"result": {
            "title": "深夜の小さな騒動", "genre": "現代コメディ／ゆるファンタジー",
            "mood": "テンポ速め、くだらない、ちょっとシュール。重い展開なし。",
            "notes": "権威が日常の利便性に負ける可笑しさ。",
            "prompt": "壮大な設定で身近な買い物をする。", "chapterCount": 2,
        }},
        "characters": [{"result": {
            "id": "alice", "name": "アリス", "role": "尊大な異世界の客",
            "settings": "威厳はあるが、便利さには素直に感動する。"}},
            {"result": {"id": "bob", "name": "ボブ", "role": "店員",
                        "freeform": "驚くより先に淡々と店の規則を説明する。"}}],
        "relationships": {"result": {"pairs": [{
            "characterIds": ["alice", "bob"], "summary": "真剣な威厳を日常の接客が受け流す。"}]}},
    }


def test_story_context_uses_each_fixed_outline_and_shared_approved_setup(monkeypatch):
    first = story_narrative()
    first["scenes"].append({**first["scenes"][0], "id": "scene-2"})
    calls = stub_json(monkeypatch, {
        "/api/m3/projects/project": {"production": {
            "id": "selected-story", "history_frozen": True,
            "approval_id": "pinned-approval", "approval_artifact_id": "approval",
            "chapters": [
                {"chapter_number": 1, "production_id": "selected-story",
                 "narrative_artifact_id": "text-1"},
                {"chapter_number": 2, "production_id": "chapter-2",
                 "narrative_artifact_id": "text-2"}],
        }},
        "/api/artifacts/text-1/content": first,
        "/api/artifacts/text-2/content": story_narrative(
            number=2, title="小さな争い", ending="第2章で固定された着地"),
        "/api/artifacts/approval/content": story_approval(),
    })
    chapters = CoordinatorCatalog("http://coordinator").chapters("project")
    story = chapters[0].scenes[0].context["story_context"]
    assert story["status"] == "complete" and story["missing"] == []
    assert story["brief"]["genre"] == "現代コメディ／ゆるファンタジー"
    assert "重い展開なし" in story["brief"]["mood"]
    assert story["brief"]["chapter_count"] == 2
    assert story["chapter"]["role"] == "大げさな要求と淡々とした応対の対比"
    assert len(story["outline"]["chapters"]) == 2
    assert story["cast"][1]["settings"].startswith("驚くより先に")
    assert story["relationships"][0]["character_ids"] == ["alice", "bob"]
    assert story["sources"]["history_frozen"] is True
    assert story["sources"]["approval_id"] == "pinned-approval"
    assert story["sources"]["approval_artifact_id"] == "approval"
    assert story["sources"]["outline"] == "selected_narrative"
    assert story["sources"]["approval"] == "production_pinned_approval"
    assert chapters[0].scenes[1].context["story_context"] == story
    second = chapters[1].scenes[0].context["story_context"]
    assert second["chapter"]["number"] == 2
    assert second["outline"]["ending"] == "第2章で固定された着地"
    assert second["sources"]["narrative_artifact_id"] == "text-2"
    assert calls.count("http://coordinator/api/artifacts/approval/content") == 1
    assert len(calls) == 4  # Project, two chapter texts and one shared fixed approval.


def test_story_context_pins_published_outline_instead_of_latest_plot(monkeypatch):
    calls = stub_json(monkeypatch, {
        "/api/m3/projects/project": {"production": {
            "id": "selected-story", "approval_id": "pinned-approval",
            "approval_artifact_id": "approval", "chapters": [{
                "chapter_number": 1, "production_id": "selected-story",
                "narrative_artifact_id": "unpublished-new-text",
                "build": {"id": "publication", "validation": {
                    "narrative_artifact_id": "published-text"}},
            }],
        }},
        "/api/artifacts/published-text/content": story_narrative(ending="公開本文の着地"),
        "/api/artifacts/approval/content": story_approval(),
    })
    story = CoordinatorCatalog("http://coordinator").chapters("project")[0].scenes[0].context[
        "story_context"]
    assert story["outline"]["ending"] == "公開本文の着地"
    assert story["sources"]["published_build_id"] == "publication"
    assert story["sources"]["narrative_artifact_id"] == "published-text"
    assert all("/m2/" not in url and "/plot" not in url and "unpublished-new" not in url
               for url in calls)


@pytest.mark.parametrize("field,value", [("projectId", "other-project"), ("id", "new-approval")])
def test_unrelated_approval_is_optional_and_never_mixed_into_scene(monkeypatch, field, value):
    approval = story_approval()
    approval[field] = value
    stub_json(monkeypatch, {
        "/api/m3/projects/project": {"production": {
            "id": "selected-story", "approval_id": "pinned-approval",
            "approval_artifact_id": "approval", "chapters": [{
                "chapter_number": 1, "narrative_artifact_id": "text"}],
        }},
        "/api/artifacts/text/content": story_narrative(),
        "/api/artifacts/approval/content": approval,
    })
    scene = CoordinatorCatalog("http://coordinator").chapters("project")[0].scenes[0]
    story = scene.context["story_context"]
    assert "窓辺に二人" in scene.context["raw_text"]
    assert story["status"] == "partial"
    assert story["brief"]["genre"] == "" and story["cast"] == []
    assert "brief" in story["missing"] and "cast" in story["missing"]
    assert "一致しません" in story["warnings"][0]
    assert story["sources"]["approval"] is None


def test_missing_optional_setup_preserves_plot_and_scene(monkeypatch):
    stub_json(monkeypatch, {
        "/api/m3/projects/project": {"production": {
            "id": "selected-story", "approval_artifact_id": "deleted-approval",
            "chapters": [{"chapter_number": 1, "narrative_artifact_id": "text"}],
        }},
        "/api/artifacts/text/content": story_narrative(),
        "/api/artifacts/deleted-approval/content": CatalogError("HTTP 404"),
    })
    scene = CoordinatorCatalog("http://coordinator").chapters("project")[0].scenes[0]
    story = scene.context["story_context"]
    assert story["status"] == "partial"
    assert story["outline"]["ending"] == "くだらない夜の友好"
    assert story["warnings"] == ["HTTP 404"]
    assert scene.context["raw_text"] == narrative()["scenes"][0]["raw_text"]


def test_legacy_context_uses_approval_from_same_export(monkeypatch):
    calls = stub_json(monkeypatch, {
        "/api/m3/projects/project": {"production": {
            "id": "selected-story", "approval_artifact_id": "new-approval",
            "chapters": [{"chapter_number": 1, "narrative_artifact_id": "mutable",
                          "build": {"id": "legacy", "export_artifact_id": "export"}}],
        }},
    })
    content = io.BytesIO()
    with ZipFile(content, "w") as archive:
        archive.writestr("narrative.json", json.dumps(story_narrative(ending="旧公開版の着地")))
        archive.writestr("approval.json", json.dumps({
            "approval_id": "old-approval", "world": {"genre": "旧版の明るいコメディ"},
            "characters": [{"id": "alice", "name": "アリス", "settings": "旧版の人物像"}],
        }))
    monkeypatch.setattr(catalog, "fetch_bytes", lambda url, timeout: content.getvalue())
    story = CoordinatorCatalog("http://coordinator").chapters("project")[0].scenes[0].context[
        "story_context"]
    assert story["brief"]["genre"] == "旧版の明るいコメディ"
    assert story["outline"]["ending"] == "旧公開版の着地"
    assert story["cast"][0]["settings"] == "旧版の人物像"
    assert story["sources"]["approval"] == "published_export"
    assert story["sources"]["approval_export_artifact_id"] == "export"
    assert story["sources"]["approval_id"] == "old-approval"
    assert story["sources"]["approval_artifact_id"] is None
    assert len(calls) == 1


def test_story_overview_is_bounded_and_retains_selected_late_chapter(monkeypatch):
    document = story_narrative(number=100)
    document["outline"]["ending"] = "終" * (catalog.MAX_CONTEXT_TEXT + 10)
    document["outline"]["chapters"] = [{
        "number": number, "title": str(number), "role": "位置づけ", "summary": "内容",
    } for number in range(1, 101)]
    document["supporting_characters"] = [{
        "id": "cast-" + str(number), "name": str(number), "settings": "設定",
    } for number in range(100)]
    document["scenes"][0]["plan"]["character_ids"] = ["cast-99"]
    stub_json(monkeypatch, {
        "/api/m3/projects/project": {"production": {
            "id": "selected-story", "chapters": [{
                "chapter_number": 100, "narrative_artifact_id": "text"}],
        }},
        "/api/artifacts/text/content": document,
    })
    story = CoordinatorCatalog("http://coordinator").chapters("project")[0].scenes[0].context[
        "story_context"]
    assert len(story["outline"]["ending"]) == catalog.MAX_CONTEXT_TEXT
    assert len(story["outline"]["chapters"]) == catalog.MAX_CONTEXT_CHAPTERS
    assert len(story["cast"]) == catalog.MAX_CONTEXT_CAST
    assert story["cast"][0]["id"] == "cast-99"
    assert story["chapter"]["number"] == 100
    assert set(story["truncated"]) >= {"outline.ending", "outline.chapters", "cast"}
