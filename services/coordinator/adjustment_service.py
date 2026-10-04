"""Complete-story presentation drafts and atomically selected publication editions."""
from __future__ import annotations

import copy
import io
import json
import wave
import zipfile
import zlib
from uuid import uuid4

from packages.contracts.adjustments import AdjustmentCharacter
from packages.contracts.m3 import EMOTION_TAGS, PortraitSetting
from packages.tyrano_export.portrait import PortraitSource

from .m3_bundle import validate_bundle
from .m3_service import M3Service
from .service import ServiceError, public, required

LIMITS = {"upload_bytes": 32 * 1024 * 1024, "image_max_side": 4096,
          "audio_min_seconds": 0.25, "audio_max_seconds": 30}


class AdjustmentService:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.db, self.store, self.clock = coordinator.db, coordinator.store, coordinator.clock
        self.m3 = M3Service(coordinator)
        self.history = self.m3.history

    def _context(self, connection, project_id):
        required(connection, "project", project_id)
        root = self.m3._selected_production(connection, project_id)
        if root is None:
            return None, []
        count = self.m3._snapshot(connection, root)["world"]["result"]["chapterCount"]
        chapters = [dict(row) for row in connection.execute(
            "SELECT * FROM m3_production WHERE storyline_id=? ORDER BY chapter_number", (root["id"],))]
        for chapter in chapters:
            chapter["build"] = self.m3._selected_build(connection, chapter)
        return root, chapters if len(chapters) == count and all(row["build"] for row in chapters) else []

    def _draft(self, connection, project_id):
        selection = self.history.selection(connection, project_id)
        return required(connection, "adjustment_draft", selection["adjustment_draft_id"]) if (
            selection and selection["adjustment_draft_id"]) else None

    def _editable(self, connection, project_id, revision):
        root, chapters = self._context(connection, project_id)
        draft = self._draft(connection, project_id)
        if not chapters or not draft or draft["production_id"] != root["id"]:
            raise ServiceError(409, "全章の完成後に調整を開始してください。")
        if draft["revision"] != revision:
            raise ServiceError(409, "調整版が更新されました。最新の内容を確認してください。")
        selection = self.history.selection(connection, project_id)
        if selection["edition_id"] != draft["base_edition_id"]:
            raise ServiceError(409, "公開版が変更されています。新しい調整を開始してください。")
        if draft["status"] == "applying":
            raise ServiceError(409, "全章へ反映中です。完了または失敗を確認してください。")
        return draft, root, chapters

    def _candidate(self, connection, draft, identifier, cid, kind):
        row = required(connection, "adjustment_candidate", identifier)
        if (row["production_id"] != draft["production_id"] or row["project_id"] != draft["project_id"]
                or row["character_id"] != cid or row["kind"] != kind or not row["artifact_id"]):
            raise ServiceError(422, "この人物の完成済み素材候補を指定してください。")
        return row

    def _geometry(self, connection, draft, state):
        from packages.tyrano_export.presentation import make_portrait_baseline, stage_geometry

        sources = {}
        for value in state["characters"]:
            if not value["image_candidate_id"]:
                continue
            candidate = self._candidate(connection, draft, value["image_candidate_id"], value["character_id"], "image")
            image = required(connection, "artifact", candidate["artifact_id"])
            if image["kind"] != "character":
                continue
            sources[value["character_id"]] = PortraitSource(self.store.read(image), value["framing"],
                                                            value["height_cm"], value["body_bounds"])
        baseline = make_portrait_baseline(sources, state.get("baseline"))
        geometry = stage_geometry(sources, baseline=baseline, adjustments={
            row["character_id"]: {"offset_y": row["offset_y"], "scale": row["scale"]}
            for row in state["characters"]})
        return baseline, geometry

    def _inspect(self, connection, root, chapters):
        snapshot = self.m3._snapshot(connection, root)
        main = {row["id"]: row["result"] for row in snapshot["characters"]}
        cast = {cid: {"character_id": cid, "name": result["name"], "role": "main",
                     "result": result, "chapter_numbers": []} for cid, result in main.items()}
        scenes = []
        for chapter in chapters:
            narrative = self.m3._load_narrative(connection, chapter)
            people = {**main, **{row.id: row.model_dump(mode="json") for row in narrative.supporting_characters}}
            requirements = [dict(row) for row in connection.execute(
                "SELECT * FROM m3_requirement WHERE production_id=?", (chapter["id"],))]
            locations = {row.id: row.name for row in narrative.locations}
            for number, scene in enumerate(narrative.scenes, 1):
                background = next(row for row in requirements if row["kind"] == "m3_background"
                                  and row["target_id"] == scene.plan.location_id)
                scenes.append({"chapter_number": chapter["chapter_number"], "scene_id": scene.id,
                    "production_id": chapter["id"],
                    "music_prompt": next((json.loads(row["descriptor"]).get("prompt", "") for row in requirements
                                          if row["kind"] == "m3_music" and row["target_id"] == scene.id), ""),
                    "title": f"シーン{number}・{locations[scene.plan.location_id]}", "character_ids": scene.plan.character_ids,
                    "background_url": f"/api/artifacts/{background['artifact_id']}/content"})
                for cid in scene.plan.character_ids:
                    if cid == "NARRATOR":
                        continue
                    cast.setdefault(cid, {"character_id": cid, "name": people[cid]["name"],
                        "role": "main" if cid in main else "supporting", "result": people[cid], "chapter_numbers": []})
                    if chapter["chapter_number"] not in cast[cid]["chapter_numbers"]:
                        cast[cid]["chapter_numbers"].append(chapter["chapter_number"])
        for person in cast.values():
            person["source_prompts"] = {kind: person["result"].get(field, "")
                                        for kind, field in (("image", "appearance"), ("voice", "voice"))}
        return list(cast.values()), scenes

    @staticmethod
    def _provenance_prompt(provenance, kind):
        details = provenance.get("prompt_details", {})
        value = details.get("effective") if isinstance(details, dict) else None
        if not isinstance(value, str) or not value.strip():
            media = provenance.get(kind, {})
            value = media.get("prompt" if kind == "image" else "caption") if isinstance(media, dict) else None
        return value if isinstance(value, str) and value.strip() else None

    def _artifact_prompt(self, connection, artifact_id, kind):
        # Approved main-character images have an immutable M3 alias pointing
        # back to the artifact containing their actual generation prompt.
        visited = set()
        while artifact_id and artifact_id not in visited:
            visited.add(artifact_id)
            artifact = connection.execute("SELECT provenance FROM artifact WHERE id=?", (artifact_id,)).fetchone()
            if not artifact:
                return None
            provenance = json.loads(artifact["provenance"])
            prompt = self._provenance_prompt(provenance, kind)
            if prompt is not None:
                return prompt
            artifact_id = provenance.get("approved_artifact_id")
        return None

    def _baseline_prompt(self, connection, production_id, cid, kind):
        original = connection.execute("SELECT artifact_id FROM adjustment_candidate WHERE production_id=? "
            "AND character_id=? AND kind=? AND source='original' ORDER BY created_at,id LIMIT 1",
            (production_id, cid, kind)).fetchone()
        return self._artifact_prompt(connection, original["artifact_id"], kind) if original else None

    def _edition(self, connection, identifier):
        if not identifier:
            return None
        row = required(connection, "publication_edition", identifier)
        return {"id": row["id"], "source_edition_id": row["source_edition_id"], "chapters": [
            {"chapter_number": b["chapter_number"], "build_id": b["id"], "player_url": f"/player/{b['id']}/",
             "export_url": f"/api/artifacts/{b['export_artifact_id']}/content"}
            for b in connection.execute("SELECT b.* FROM edition_build e JOIN chapter_build b ON b.id=e.build_id "
                "WHERE e.edition_id=? ORDER BY b.chapter_number", (identifier,))]}

    def _view(self, connection, project_id, preview=None):
        root, chapters = self._context(connection, project_id)
        draft = self._draft(connection, project_id)
        if draft and (not root or draft["production_id"] != root["id"]):
            draft = None
        candidates = [dict(row) for row in connection.execute(
            "SELECT * FROM adjustment_candidate WHERE production_id=? ORDER BY created_at,id",
            (root["id"],))] if root else []
        jobs = [public(dict(row)) for row in connection.execute(
            "SELECT j.* FROM job j JOIN adjustment_job a ON a.job_id=j.id WHERE a.draft_id=? ORDER BY j.created_at,j.id",
            (draft["id"],))] if draft else []
        by_id = {row["id"]: row for row in jobs}
        for row in candidates:
            image_available = (row["kind"] != "image" or not row["artifact_id"]
                or required(connection, "artifact", row["artifact_id"])["kind"] == "character")
            row["url"] = f"/api/artifacts/{row['artifact_id']}/content" if row["artifact_id"] and image_available else None
            row["sample_url"] = f"/api/artifacts/{row['sample_artifact_id']}/content" if row["sample_artifact_id"] else None
            row["metadata"] = json.loads(row["metadata"])
            row["status"] = "completed" if row["artifact_id"] else by_id.get(row["job_id"], {}).get("status", "pending")
        cast, scenes = self._inspect(connection, root, chapters) if chapters else ([], [])
        people = {row["character_id"]: row for row in cast}
        for row in candidates:
            person = people.get(row["character_id"])
            row["prompt_details"] = None
            if not person or row["source"] == "upload":
                continue
            source = person["source_prompts"][row["kind"]]
            stored = row["metadata"].get("prompt_details", {})
            if not isinstance(stored, dict):
                stored = {}
            row["prompt_details"] = {
                "source": stored.get("source", source), "input": stored.get("input", source),
                "instruction": stored.get("instruction", row["prompt"] or ""),
                "effective": stored.get("effective") or self._artifact_prompt(connection, row["artifact_id"], row["kind"]),
                "baseline": stored.get("baseline") or self._baseline_prompt(connection, row["production_id"],
                    row["character_id"], row["kind"]),
            }
        state = preview or json.loads(draft["state"]) if draft else None
        from .music_adjustments import inspect_music
        from .music_service import effective_settings, plan_metadata

        automatic = {chapter["id"]: plan_metadata(connection, chapter, self.m3, state) for chapter in chapters}
        for scene in scenes:
            scene["music_plan"] = automatic[scene["production_id"]][scene["scene_id"]]

        music_candidates, music_jobs = inspect_music(self, connection, root, draft)
        jobs.extend(music_jobs)
        if draft:
            baseline, geometry = self._geometry(connection, draft, state)
            draft = {**draft, "characters": state["characters"], "baseline": baseline, "geometry": geometry,
                     "scene_music": effective_settings(connection, chapters, self.m3, state)}
            draft.pop("state", None)
            selected = {row["character_id"]: row for row in state["characters"]}
            for row in cast:
                row.update(selected[row["character_id"]])
                for kind in ("image", "voice"):
                    candidate = next((c for c in candidates if c["id"] == row[kind + "_candidate_id"]), None)
                    row[kind + "_url"] = candidate["url"] if candidate else None
                    if kind == "voice":
                        row["reference_text"] = candidate["reference_text"] if candidate else ""
        selection = self.history.selection(connection, project_id)
        return {"project_id": project_id, "complete": bool(chapters), "readonly": not bool(chapters),
                "busy": bool(draft and draft["status"] == "applying") or any(j["status"] in {"pending", "running"} for j in jobs),
                "draft": draft, "cast": cast, "scenes": scenes, "candidates": candidates, "music_candidates": music_candidates, "jobs": jobs,
                "edition": self._edition(connection, selection["edition_id"]) if selection else None, "limits": LIMITS}

    def project(self, project_id):
        self.recover(project_id)
        with self.db.transaction() as connection:
            self.history.finish(connection, project_id)
            return self._view(connection, project_id)

    def _store_candidate(self, connection, root, cid, kind, artifact_id, reference_text, source,
                         prompt="", original_id=None, metadata=None, job_id=None):
        identifier = uuid4().hex
        connection.execute("INSERT INTO adjustment_candidate (id,project_id,production_id,character_id,kind,"
            "artifact_id,original_artifact_id,reference_text,source,prompt,job_id,metadata,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (identifier, root["project_id"], root["id"], cid, kind,
            artifact_id, original_id, reference_text, source, prompt, job_id, json.dumps(metadata or {}), self.clock()))
        return identifier

    def start(self, project_id, request):
        with self.db.transaction() as connection:
            root, chapters = self._context(connection, project_id)
            if not chapters:
                raise ServiceError(409, "全章が公開されてから調整できます。")
            self.history.ensure(connection, project_id)
            selection = self.history.selection(connection, project_id)
            if request.expected_edition_id != selection["edition_id"]:
                raise ServiceError(409, "公開版が変更されました。最新の作品を確認してください。")
            current = self._draft(connection, project_id)
            if current and current["status"] == "applying":
                raise ServiceError(409, "調整版を反映中です。")
            if current and current["base_edition_id"] == selection["edition_id"] and current["status"] != "applied":
                state = json.loads(current["state"])
                if "scene_music" not in state:
                    from .music_service import original_settings

                    self.history.begin(connection, project_id, "既存の調整に場面のBGM設定を追加", "m3-adjustment-start", concurrent=True)
                    state["scene_music"] = original_settings(connection, chapters, self.m3)
                    connection.execute("UPDATE adjustment_draft SET state=?,revision=revision+1 WHERE id=?",
                                       (json.dumps(state), current["id"]))
                    self.history.finish(connection, project_id)
                return self._view(connection, project_id)
            self.history.begin(connection, project_id, "完成後の調整を開始", "m3-adjustment-start", concurrent=True)
            edition_id = selection["edition_id"]
            if edition_id:
                state = json.loads(required(connection, "publication_edition", edition_id)["state"])
                if "scene_music" not in state:
                    from .music_service import original_settings

                    state["scene_music"] = original_settings(connection, chapters, self.m3)
            else:
                cast, _scenes = self._inspect(connection, root, chapters)
                settings = {}
                requirements = {}
                for chapter in chapters:
                    requirements[chapter["id"]] = [dict(row) for row in connection.execute(
                        "SELECT * FROM m3_requirement WHERE production_id=?", (chapter["id"],))]
                    settings.update({row["character_id"]: row for row in self.m3._published_portraits(
                        connection, chapter, chapter["build"])})
                characters = []
                for person in cast:
                    cid = person["character_id"]
                    values = {}
                    for kind, job_kind in (("image", "m3_image"), ("voice", "m3_voice")):
                        material = next((row for rows in requirements.values() for row in rows
                                         if row["kind"] == job_kind and row["target_id"] == cid), None)
                        if material:
                            artifact_id = material["artifact_id"]
                            transcript = json.loads(material["descriptor"]).get("reference_text") if kind == "voice" else None
                        else:
                            # Approved main characters can be previewed even if
                            # this story never calls them onto the stage.
                            approved = next(row for row in self.m3._snapshot(connection, root)["characters"] if row["id"] == cid)
                            artifact, data = self.m3._reference(connection, root, approved[kind], job_kind.replace("m3_", "m2_"))
                            if kind == "image":
                                artifact = self.m3._artifact(connection, root, "adjustment-approved-" + artifact["id"],
                                    "character", "image.png", data, provenance={"approved_artifact_id": artifact["id"]})
                            artifact_id = artifact["id"]
                            voice = json.loads(artifact["provenance"]).get("voice", {}) if kind == "voice" else {}
                            transcript = voice.get("reference_text") or voice.get("text") if kind == "voice" else None
                        values[kind + "_candidate_id"] = self._store_candidate(connection, root, cid, kind,
                            artifact_id, transcript, "original")
                    setting = settings.get(cid, {})
                    characters.append(AdjustmentCharacter(character_id=cid, **values,
                        framing=setting.get("framing", "auto"), height_cm=setting.get("height_cm", person["result"].get("height_cm")),
                        body_bounds=setting.get("body_bounds")).model_dump(mode="json"))
                from .music_service import original_settings

                state = {"characters": characters, "baseline": None, "requirements": requirements,
                         "scene_music": original_settings(connection, chapters, self.m3)}
                edition_id = uuid4().hex
                temporary = {"project_id": project_id, "production_id": root["id"]}
                state["baseline"], _ = self._geometry(connection, temporary, state)
                self._insert_edition(connection, edition_id, root, None, state,
                                     {row["id"]: row["build"]["id"] for row in chapters})
            identifier = uuid4().hex
            connection.execute("INSERT INTO adjustment_draft VALUES (?,?,?,?,1,?,'draft',NULL,NULL,?)",
                (identifier, project_id, root["id"], edition_id, json.dumps(state), self.clock()))
            connection.execute("UPDATE project_history_state SET edition_id=?,adjustment_draft_id=?,version=version+1 WHERE project_id=?",
                               (edition_id, identifier, project_id))
            self.history.finish(connection, project_id)
            return self._view(connection, project_id)

    def _insert_edition(self, connection, identifier, root, source, state, builds):
        connection.execute("INSERT INTO publication_edition VALUES (?,?,?,?,?,?)",
            (identifier, root["project_id"], root["id"], source, json.dumps(state), self.clock()))
        for production_id, build_id in builds.items():
            connection.execute("INSERT INTO edition_build VALUES (?,?,?)", (identifier, production_id, build_id))

    def _save_state(self, connection, draft, request):
        state = json.loads(draft["state"])
        prior = {row["character_id"]: row for row in state["characters"]}
        values = [row.model_dump(mode="json") for row in request.characters]
        if {row["character_id"] for row in values} != set(prior):
            raise ServiceError(422, "作品の全人物の表示設定を指定してください。")
        for value in values:
            cid = value["character_id"]
            if value["image_candidate_id"]:
                self._candidate(connection, draft, value["image_candidate_id"], cid, "image")
            elif prior[cid]["image_candidate_id"]:
                raise ServiceError(422, "立ち絵の素材候補を選択してください。")
            self._candidate(connection, draft, value["voice_candidate_id"], cid, "voice")
            if (value["image_candidate_id"] != prior[cid]["image_candidate_id"]
                    and value["body_bounds"] == prior[cid]["body_bounds"]):
                value["body_bounds"] = None
        state["characters"] = values
        if request.scene_music is not None:
            from .music_adjustments import validate_settings

            validate_settings(self, connection, draft, state, request.scene_music)
        state["baseline"], _ = self._geometry(connection, draft, state)
        return state

    def save(self, project_id, request, *, preview=False):
        with self.db.transaction() as connection:
            draft, _root, _chapters = self._editable(connection, project_id, request.expected_revision)
            state = self._save_state(connection, draft, request)
            if preview:
                return self._view(connection, project_id, preview=state)
            self.history.begin(connection, project_id, "調整中の表示・素材を保存", "m3-adjustment-save", concurrent=True)
            connection.execute("UPDATE adjustment_draft SET revision=revision+1,state=?,status='draft',error=NULL WHERE id=?",
                               (json.dumps(state), draft["id"]))
            self.history.finish(connection, project_id)
            return self._view(connection, project_id)

    def _enqueue(self, connection, draft, root, kind, purpose, cid, *, candidate_id=None, apply_id=None,
                 production=None, target_id=None, material=None):
        identifier, run_id = uuid4().hex, uuid4().hex
        frozen = connection.execute("SELECT j.* FROM job j JOIN m3_production_job p ON p.job_id=j.id "
            "WHERE p.production_id=? AND j.kind='m3_narrative' ORDER BY j.created_at LIMIT 1", (root["id"],)).fetchone()
        settings = json.loads(frozen["settings_snapshot"])
        payload = {"schema_version": 1, "approval_snapshot": self.m3._snapshot(connection, root),
            "production_id": (production or root)["id"], "seed": int(identifier[:8], 16) & 0x7fffffff,
            "profile": settings.get("profile", {}), "tts_profile": settings.get("tts_profile"),
            "adjustment": {"draft_id": draft["id"], "revision": draft["revision"], "purpose": purpose,
                           "character_id": cid, "apply_id": apply_id}, **(material or {})}
        project = required(connection, "project", root["project_id"])
        connection.execute("INSERT INTO generation_run(id,project_id,settings_version,story_revision_id,policy,status,created_at) "
            "VALUES (?,?,?,?,?,'pending',?)", (run_id, root["project_id"], project["settings_version"],
                draft["id"], json.dumps({"mode": "adjustment"}), self.clock()))
        connection.execute("INSERT INTO job(id,project_id,run_id,kind,payload,settings_snapshot,status,max_attempts,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,'pending',3,?,?)", (identifier, root["project_id"], run_id, kind,
                json.dumps(payload), json.dumps(settings), self.clock(), self.clock()))
        connection.execute("INSERT INTO adjustment_job (job_id,draft_id,purpose,apply_id,candidate_id,character_id,"
            "target_id,production_id,artifact_id,adoption_revision) VALUES (?,?,?,?,?,?,?,?,NULL,?)",
            (identifier, draft["id"], purpose, apply_id, candidate_id, cid, target_id,
             (production or root)["id"], draft["revision"]))
        return identifier

    def generate(self, project_id, request):
        with self.db.transaction() as connection:
            draft, root, chapters = self._editable(connection, project_id, request.expected_revision)
            cast, _ = self._inspect(connection, root, chapters)
            character = next((row["result"] for row in cast if row["character_id"] == request.character_id), None)
            if not character:
                raise ServiceError(422, "作品に登場する人物を指定してください。")
            self.history.begin(connection, project_id, "素材候補を生成", "m3-adjustment-generate", concurrent=True)
            draft = {**draft, "revision": draft["revision"] + 1}
            transcript = (request.reference_text or character["selfIntroduction"]) if request.kind == "voice" else None
            source = character["appearance" if request.kind == "image" else "voice"]
            prompt_details = {"source": source, "input": request.source_prompt if request.source_prompt is not None else source,
                "instruction": request.instruction, "effective": None,
                "baseline": self._baseline_prompt(connection, root["id"], request.character_id, request.kind)}
            candidate = self._store_candidate(connection, root, request.character_id, request.kind, None, transcript,
                                              "generated", request.instruction, metadata={"prompt_details": prompt_details})
            job = self._enqueue(connection, draft, root, "m3_image" if request.kind == "image" else "m3_voice",
                "candidate", request.character_id, candidate_id=candidate,
                material={"character_id": request.character_id, "character_result": character,
                          "instruction": request.instruction,
                          **({"source_prompt": request.source_prompt} if request.source_prompt is not None else {}),
                          **({"reference_text": transcript} if transcript else {})})
            connection.execute("UPDATE adjustment_candidate SET job_id=? WHERE id=?", (job, candidate))
            connection.execute("UPDATE adjustment_draft SET revision=revision+1 WHERE id=?", (draft["id"],))
            return self._view(connection, project_id)

    def upload(self, project_id, revision, cid, kind, original, normalized, metadata, reference_text):
        with self.db.transaction() as connection:
            draft, root, _ = self._editable(connection, project_id, revision)
            if cid not in {row["character_id"] for row in json.loads(draft["state"])["characters"]}:
                raise ServiceError(422, "登録された人物を指定してください。")
            if kind == "voice" and (not reference_text or not reference_text.strip()):
                raise ServiceError(422, "基準音声に対応する読み上げ文を入力してください。")
            self.history.begin(connection, project_id, "素材候補をアップロード", "m3-adjustment-upload", concurrent=True)
            draft = {**draft, "revision": draft["revision"] + 1}
            raw = self.m3._artifact(connection, root, "adjustment-original-" + uuid4().hex,
                "adjustment_original", "original.bin", original, provenance=metadata)
            connection.execute("UPDATE artifact SET media_type='application/octet-stream' WHERE id=?", (raw["id"],))
            media = self.m3._artifact(connection, root, "adjustment-upload-" + uuid4().hex,
                "character" if kind == "image" else "audio", "image.png" if kind == "image" else "voice.wav",
                normalized, provenance=metadata)
            candidate = self._store_candidate(connection, root, cid, kind, media["id"], reference_text,
                "upload", original_id=raw["id"], metadata=metadata)
            if kind == "voice":
                self._sample(connection, draft, root, candidate)
            connection.execute("UPDATE adjustment_draft SET revision=revision+1 WHERE id=?", (draft["id"],))
            self.history.finish(connection, project_id)
            return self._view(connection, project_id)

    def _sample(self, connection, draft, root, candidate_id):
        candidate = required(connection, "adjustment_candidate", candidate_id)
        for chapter in connection.execute("SELECT * FROM m3_production WHERE storyline_id=? ORDER BY chapter_number", (root["id"],)):
            for scene in self.m3._load_narrative(connection, chapter).scenes:
                utterance = next((u for u in scene.utterances if u.speaker_id == candidate["character_id"]), None)
                if utterance:
                    record = required(connection, "artifact", candidate["artifact_id"])
                    descriptor = json.loads(connection.execute("SELECT descriptor FROM m3_requirement WHERE production_id=? "
                        "AND kind='m3_voice_clone' AND target_id=?", (chapter["id"], utterance.id)).fetchone()[0])
                    return self._enqueue(connection, draft, root, "m3_voice_clone", "sample", candidate["character_id"],
                        candidate_id=candidate_id, material={**descriptor, "dialogue_text": utterance.spoken_text[:120],
                            "reference_voice": {"artifact_id": record["id"], "sha256": record["sha256"], "text": candidate["reference_text"]}})
        return None

    def sample(self, project_id, request):
        with self.db.transaction() as connection:
            draft, root, _ = self._editable(connection, project_id, request.expected_revision)
            candidate = required(connection, "adjustment_candidate", request.candidate_id)
            self._candidate(connection, draft, candidate["id"], candidate["character_id"], "voice")
            self.history.begin(connection, project_id, "選択した声で台詞を試聴", "m3-adjustment-sample", concurrent=True)
            draft = {**draft, "revision": draft["revision"] + 1}
            self._sample(connection, draft, root, candidate["id"])
            connection.execute("UPDATE adjustment_draft SET revision=revision+1 WHERE id=?", (draft["id"],))
            self.history.finish(connection, project_id)
            return self._view(connection, project_id)

    def apply(self, project_id, request):
        with self.db.transaction() as connection:
            draft, root, chapters = self._editable(connection, project_id, request.expected_revision)
            state = json.loads(draft["state"])
            source = json.loads(required(connection, "publication_edition", draft["base_edition_id"])["state"])
            self.history.begin(connection, project_id, "調整版を全章へ反映", "m3-adjustment-apply", concurrent=True)
            identifier = uuid4().hex
            connection.execute("INSERT INTO adjustment_apply VALUES (?,?,?,?,'pending',NULL,NULL)",
                (identifier, draft["id"], draft["revision"], json.dumps(state)))
            prior = {row["character_id"]: row for row in source["characters"]}
            for value in state["characters"]:
                cid = value["character_id"]
                if value["voice_candidate_id"] == prior[cid]["voice_candidate_id"]:
                    continue
                candidate = self._candidate(connection, draft, value["voice_candidate_id"], cid, "voice")
                record = required(connection, "artifact", candidate["artifact_id"])
                for chapter in chapters:
                    for requirement in source["requirements"][chapter["id"]]:
                        descriptor = json.loads(requirement["descriptor"])
                        if requirement["kind"] == "m3_voice_clone" and descriptor["character_id"] == cid:
                            self._enqueue(connection, draft, root, "m3_voice_clone", "dialogue", cid,
                                apply_id=identifier, production=chapter, target_id=requirement["target_id"],
                                material={**descriptor, "reference_voice": {"artifact_id": record["id"],
                                    "sha256": record["sha256"], "text": candidate["reference_text"]}})
            connection.execute("UPDATE adjustment_draft SET status='applying',active_apply_id=?,error=NULL WHERE id=?",
                               (identifier, draft["id"]))
        self.advance(identifier)
        return self.project(project_id)

    def advance(self, identifier):
        try:
            with self.db.transaction() as connection:
                operation = required(connection, "adjustment_apply", identifier)
                if operation["status"] == "completed":
                    return
                draft = required(connection, "adjustment_draft", operation["draft_id"])
                selection = self.history.selection(connection, draft["project_id"])
                if (draft["active_apply_id"] != identifier or draft["revision"] != operation["revision"]
                        or selection["edition_id"] != draft["base_edition_id"]):
                    raise ServiceError(409, "以前の調整版は現在の公開版へ反映できません。")
                jobs = [dict(row) for row in connection.execute("SELECT j.*,a.artifact_id,a.production_id,a.target_id FROM job j "
                    "JOIN adjustment_job a ON a.job_id=j.id WHERE a.apply_id=?", (identifier,))]
                if any(row["status"] == "failed" for row in jobs):
                    raise ValueError("Dialogue regeneration failed.")
                if any(row["status"] != "completed" for row in jobs):
                    return
                root, chapters = self._context(connection, draft["project_id"])
                if not chapters:
                    raise ValueError("Complete source publication is missing.")
                state = json.loads(operation["state"])
                requirements = copy.deepcopy(state["requirements"])
                for value in state["characters"]:
                    for rows in requirements.values():
                        for row in rows:
                            if row["target_id"] == value["character_id"] and row["kind"] in {"m3_image", "m3_voice"}:
                                kind = "image" if row["kind"] == "m3_image" else "voice"
                                candidate = self._candidate(connection, draft, value[kind + "_candidate_id"], value["character_id"], kind)
                                row["artifact_id"] = candidate["artifact_id"]
                for job in jobs:
                    for row in requirements[job["production_id"]]:
                        if row["kind"] == "m3_voice_clone" and row["target_id"] == job["target_id"]:
                            row["artifact_id"] = job["artifact_id"]
                state["requirements"] = requirements
                state["baseline"], _ = self._geometry(connection, draft, state)
                source = json.loads(required(connection, "publication_edition", draft["base_edition_id"])["state"])
                music_fields = {"scene_music", "music_plans"}
                unchanged = state == source
                music_only = ({key: value for key, value in state.items() if key not in music_fields}
                              == {key: value for key, value in source.items() if key not in music_fields})
                builds = {row["id"]: row["build"]["id"] for row in chapters}
                publisher = _AdjustmentPublisher(self.coordinator, builds, state)
                for chapter in chapters:
                    before = publisher._latest_build(connection, chapter["id"])
                    def bindings(value, production_id):
                        return {row["scene_id"]: row for row in value.get("scene_music", [])
                                if row["production_id"] == production_id}
                    same_plan = state.get("music_plans", {}).get(chapter["id"]) == source.get("music_plans", {}).get(chapter["id"])
                    if unchanged:
                        # Refresh the runtime without recalculating adopted
                        # portrait placement, music bindings, or scene settings.
                        # History can select an older build than the latest one.
                        latest = self._refresh_player(connection, chapter, chapter["build"])
                    elif music_only and same_plan and bindings(state, chapter["id"]) == bindings(source, chapter["id"]):
                        latest = before
                    else:
                        publisher._publish(connection, chapter, requirements[chapter["id"]], presentation=state)
                        latest = publisher._latest_build(connection, chapter["id"])
                    # Even unchanged exports receive a distinct build identity:
                    # an old URL can never navigate into a newer edition.
                    if latest["id"] == before["id"]:
                        latest = self._clone_build(connection, required(connection, "chapter_build", builds[chapter["id"]]))
                    builds[chapter["id"]] = latest["id"]
                edition = uuid4().hex
                self._insert_edition(connection, edition, root, draft["base_edition_id"], state, builds)
                self.history.set_selection(connection, draft["project_id"], root["id"], builds[root["id"]], None)
                connection.execute("UPDATE project_history_state SET edition_id=?,adjustment_draft_id=? WHERE project_id=?",
                                   (edition, draft["id"], draft["project_id"]))
                connection.execute("UPDATE adjustment_apply SET status='completed',edition_id=?,error=NULL WHERE id=?", (edition, identifier))
                connection.execute("UPDATE adjustment_draft SET status='applied',error=NULL WHERE id=?", (draft["id"],))
                self.history.finish(connection, draft["project_id"])
        except (OSError, ValueError, ServiceError, zipfile.BadZipFile):
            with self.db.transaction() as connection:
                operation = required(connection, "adjustment_apply", identifier)
                if operation["status"] != "completed":
                    error = "調整版を反映できませんでした。元の公開版は引き続き鑑賞できます。"
                    connection.execute("UPDATE adjustment_apply SET status='failed',error=? WHERE id=?", (error, identifier))
                    connection.execute("UPDATE adjustment_draft SET status='failed',error=? WHERE id=? AND active_apply_id=?",
                                       (error, operation["draft_id"], identifier))

    def _refresh_player(self, connection, chapter, build):
        from packages.contracts import Script
        from packages.tyrano_export import compile_bundle

        script = Script.model_validate_json(self.store.read(required(connection, "artifact", build["script_artifact_id"])))
        assets = {asset.id: self.store.read(required(connection, "artifact", asset.artifact_id)) for asset in script.assets}
        original = self.store.read(required(connection, "artifact", build["export_artifact_id"]))
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            documents = {name: archive.read(name) for name in archive.namelist()
                         if name in {"approval.json", "narrative.json", "chapter-manifest.json"}
                         or name.startswith("sources/")}
        bundle = compile_bundle(script, assets, documents=documents)
        if bundle == original:
            return self._clone_build(connection, build)
        record = self.m3._artifact(connection, chapter, "export-" + chapter["id"],
                                  "tyrano_export", "tyrano-source.zip", bundle)
        return self._clone_build(connection, {**build, "export_artifact_id": record["id"]})

    def _clone_build(self, connection, build):
        value = dict(build)
        value["id"], value["created_at"] = uuid4().hex, self.clock()
        value["revision"] = connection.execute("SELECT MAX(revision)+1 FROM chapter_build WHERE production_id=?",
                                               (build["production_id"],)).fetchone()[0]
        columns = list(value)
        connection.execute("INSERT INTO chapter_build (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")",
                           tuple(value[key] for key in columns))
        return value

    def recover(self, project_id=None):
        with self.db.transaction() as connection:
            operations = [row[0] for row in connection.execute("SELECT a.id FROM adjustment_apply a JOIN adjustment_draft d ON d.id=a.draft_id "
                "WHERE a.status='pending'" + (" AND d.project_id=?" if project_id else ""), (project_id,) if project_id else ())]
        for identifier in operations:
            self.advance(identifier)

    def validate_retry(self, connection, job):
        link = required(connection, "adjustment_draft", json.loads(job["payload"])["adjustment"]["draft_id"])
        selected = self._draft(connection, job["project_id"])
        selection = self.history.selection(connection, job["project_id"])
        if (not selected or selected["id"] != link["id"]
                or selection["edition_id"] != link["base_edition_id"]):
            raise ServiceError(409, "以前の調整ジョブは再試行できません。")
        if not self.history.has_operation(connection, job["project_id"], "m3"):
            self.history.begin(connection, job["project_id"], "素材生成・試聴を再試行", "m3-adjustment-retry", concurrent=True)
        association = connection.execute("SELECT * FROM adjustment_job WHERE job_id=?", (job["id"],)).fetchone()
        connection.execute("UPDATE adjustment_job SET adoption_revision=? WHERE job_id=?", (link["revision"], job["id"]))
        if association["apply_id"]:
            operation = required(connection, "adjustment_apply", association["apply_id"])
            if operation["revision"] != link["revision"] or link["active_apply_id"] != operation["id"]:
                raise ServiceError(409, "新しい調整版を確認してください。")
            connection.execute("UPDATE adjustment_apply SET status='pending',error=NULL WHERE id=?", (operation["id"],))
            connection.execute("UPDATE adjustment_draft SET status='applying',error=NULL WHERE id=?", (link["id"],))

    def retry(self, project_id, request):
        with self.db.transaction() as connection:
            selected = self._draft(connection, project_id)
            music_failed = selected and connection.execute("SELECT 1 FROM job j WHERE j.status='failed' AND j.id IN "
                "(SELECT job_id FROM music_adjustment_job WHERE draft_id=? UNION SELECT job_id FROM music_replan_job WHERE draft_id=?)",
                (selected["id"], selected["id"])).fetchone()
        if music_failed:
            from .music_adjustments import MusicAdjustmentService

            return MusicAdjustmentService(self.coordinator).retry(project_id, request)
        with self.db.transaction() as connection:
            draft, _root, _ = self._editable(connection, project_id, request.expected_revision)
            identifier = draft["active_apply_id"]
            if draft["status"] == "failed" and identifier:
                jobs = [row[0] for row in connection.execute("SELECT j.id FROM job j JOIN adjustment_job a ON a.job_id=j.id "
                    "WHERE a.apply_id=? AND j.status='failed'", (identifier,))]
                connection.execute("UPDATE adjustment_apply SET status='pending',error=NULL WHERE id=?", (identifier,))
                connection.execute("UPDATE adjustment_draft SET status='applying',error=NULL WHERE id=?", (draft["id"],))
            else:
                identifier = None
                rows = connection.execute("SELECT j.id FROM job j JOIN adjustment_job a ON a.job_id=j.id "
                    "WHERE a.draft_id=? AND a.apply_id IS NULL AND j.status='failed'", (draft["id"],)).fetchall()
                if not rows:
                    raise ServiceError(409, "失敗した処理がありません。")
                self.history.begin(connection, project_id, "失敗した素材生成・試聴を再試行", "m3-adjustment-retry", concurrent=True)
                jobs = [row["id"] for row in rows]
                connection.execute("UPDATE adjustment_draft SET revision=revision+1 WHERE id=?", (draft["id"],))
        for job_id in jobs:
            self.coordinator.retry(job_id)
        if identifier:
            self.advance(identifier)
        return self.project(project_id)

    def complete(self, job_id, worker_id, lease_id, data):
        with self.db.transaction() as connection:
            job, _attempt = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
        try:
            envelope, files = validate_bundle(data, job["kind"])
            payload = json.loads(job["payload"])
            voice = envelope["provenance"].get("voice", {})
            if job["kind"] == "m3_voice" and voice.get("reference_text") != payload["reference_text"]:
                raise ValueError("reference speech changed")
            if job["kind"] == "m3_voice_clone":
                reference = payload["reference_voice"]
                if (voice.get("spoken_text") != payload["dialogue_text"]
                        or voice.get("text") != EMOTION_TAGS[payload["voice_emotion"]] + payload["dialogue_text"]
                        or voice.get("voice_emotion") != payload["voice_emotion"]
                        or any(voice.get(key) != reference[source] for key, source in (
                            ("reference_artifact_id", "artifact_id"), ("reference_sha256", "sha256"),
                            ("reference_text", "text")))):
                    raise ValueError("speech or immutable voice reference changed")
        except (ValueError, TypeError, KeyError, AttributeError, zipfile.BadZipFile,
                EOFError, wave.Error, zlib.error) as exc:
            raise ServiceError(422, "生成結果の素材・音声情報が入力と一致しません。") from exc
        filename = "image.png" if job["kind"] == "m3_image" else "voice.wav"
        if filename not in files:
            raise ServiceError(422, "素材候補には実際の画像・音声が必要です。")
        with self.db.transaction() as connection:
            job, attempt = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            link = dict(connection.execute("SELECT * FROM adjustment_job WHERE job_id=?", (job_id,)).fetchone())
            draft = required(connection, "adjustment_draft", link["draft_id"])
            root = required(connection, "m3_production", draft["production_id"])
            selected = self._draft(connection, draft["project_id"])
            selection = self.history.selection(connection, draft["project_id"])
            current = (selected is not None and selected["id"] == draft["id"]
                       and draft["revision"] == link["adoption_revision"]
                       and selection["edition_id"] == draft["base_edition_id"])
            record = self.m3._artifact(connection, root, "adjustment-media-" + job_id,
                "character" if filename == "image.png" else "audio", filename, files[filename],
                provenance=envelope["provenance"], job=job, attempt=attempt)
            bundle = self.m3._artifact(connection, root, "adjustment-result-" + job_id, "m3_bundle", "result.zip", data,
                                       job=job, attempt=attempt)
            connection.execute("UPDATE adjustment_job SET artifact_id=? WHERE job_id=?", (record["id"], job_id))
            if link["purpose"] == "candidate":
                candidate = required(connection, "adjustment_candidate", link["candidate_id"])
                prior = json.loads(candidate["metadata"])
                metadata = {**prior, **envelope["provenance"]}
                details = prior.get("prompt_details")
                if isinstance(details, dict):
                    metadata["prompt_details"] = {**details,
                        "effective": self._provenance_prompt(envelope["provenance"], candidate["kind"])}
                connection.execute("UPDATE adjustment_candidate SET artifact_id=?,metadata=? WHERE id=?",
                    (record["id"], json.dumps(metadata), link["candidate_id"]))
                if job["kind"] == "m3_voice" and current:
                    self._sample(connection, draft, root, link["candidate_id"])
            elif link["purpose"] == "sample" and current:
                connection.execute("UPDATE adjustment_candidate SET sample_artifact_id=? WHERE id=?", (record["id"], link["candidate_id"]))
            now = self.clock()
            connection.execute("UPDATE job_attempt SET status='completed',ended_at=? WHERE id=?", (now, attempt["id"]))
            connection.execute("UPDATE job SET status='completed',result_artifact_id=?,updated_at=?,error=NULL WHERE id=?", (bundle["id"], now, job_id))
            connection.execute("UPDATE generation_run SET status='completed' WHERE id=?", (job["run_id"],))
        if link["apply_id"]:
            self.advance(link["apply_id"])
        return {"job": self.coordinator.job(job_id)["job"], "artifact": public(bundle)}


class _AdjustmentPublisher(M3Service):
    def __init__(self, coordinator, builds, state):
        super().__init__(coordinator)
        self.builds, self.presentation = builds, state

    def _selected_build(self, connection, production):
        return required(connection, "chapter_build", self.builds[production["id"]])

    def _portrait_settings(self, connection, production):
        values = {}
        for row in self.presentation["characters"]:
            values[row["character_id"]] = PortraitSetting.model_validate({key: row[key] for key in
                ("character_id", "framing", "height_cm", "body_bounds") if key in row} | {
                "image_artifact_id": next((candidate["artifact_id"] for candidate in connection.execute(
                    "SELECT artifact_id FROM adjustment_candidate WHERE id=?", (row["image_candidate_id"],))), None)})
        return None, values
