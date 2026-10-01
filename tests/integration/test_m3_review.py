"""Cross-boundary regression checks from the independent M3 implementation review."""

import io
from zipfile import ZipFile

import httpx
from fastapi.testclient import TestClient

from packages.contracts import Script
from packages.contracts.m3 import M3_KINDS
from packages.narrative.validation import script_character_id
from services.coordinator.app import create_app
from services.worker.client import WorkerClient
from tests.integration.test_m2 import action, report_voice_inventory
from tests.integration.test_m3 import approved, narrative, output, production


def test_real_worker_transport_replay_and_later_draft_edits_keep_approved_chapter(tmp_path):
    with TestClient(create_app(tmp_path / "coordinator")) as client:
        project, _ = approved(client)
        initial = production(client, project)
        frozen = initial["jobs"][0]["payload"]["approval_snapshot"]
        main = frozen["characters"][0]
        action(client, project, "edit-character", character_id=main["id"],
               patch={"name": "承認後に編集した未承認の名前"})

        class LoseOneResponse:
            base_url = client.base_url

            def __init__(self):
                self.lost = False
                self.completions = []

            def get(self, *args, **kwargs):
                return client.get(*args, **kwargs)

            def post(self, url, **kwargs):
                response = client.post(url, **kwargs)
                if url.startswith("/api/m3/jobs/") and url.endswith("/complete"):
                    self.completions.append((url, kwargs["content"]))
                    if not self.lost:
                        self.lost = True
                        assert response.status_code == 200
                        raise httpx.ReadError("Response lost after result adoption")
                return response

        transport = LoseOneResponse()
        executions = []

        def generate(job, directory):
            executions.append(job["id"])
            assert job["payload"]["approval_snapshot"] == frozen
            if job["kind"] == "m3_voice_clone":
                assert (directory / "reference-voice.wav").is_file()
            return output(job)

        worker = WorkerClient(transport, generation_runner=generate,
                              generation_kinds=M3_KINDS, work_dir=tmp_path / "worker",
                              extra_capabilities=["tts_download"])
        report_voice_inventory(client, worker.register())
        for _ in range(40):
            result = worker.run_once()
            if result == "idle":
                break
            assert result == "completed"
        else:
            raise AssertionError("Chapter did not finish within its fixture job budget")
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        assert len(executions) == len(set(executions)) == state["total_jobs"]
        assert transport.completions[0] == transport.completions[1]
        assert len(transport.completions) == state["total_jobs"] + 1
        with ZipFile(io.BytesIO(client.get(state["export_url"]).content)) as archive:
            script = Script.model_validate_json(archive.read("script.json"))
            adopted = next(c for c in script.characters if c.id == script_character_id(main["id"]))
            assert adopted.name == main["result"]["name"]
            assert all(c.name != "承認後に編集した未承認の名前" for c in script.characters)
            for scene in narrative(frozen)["scenes"]:
                assert archive.read(f"sources/{scene['id']}.txt") == scene["raw_text"].encode()
        assert client.get(state["player_url"] + "data/scenario/first.ks").status_code == 200
