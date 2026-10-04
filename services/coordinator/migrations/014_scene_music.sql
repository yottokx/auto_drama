ALTER TABLE m3_production ADD COLUMN music_enabled INTEGER NOT NULL DEFAULT 0 CHECK(music_enabled IN (0,1));
CREATE TABLE music_candidate (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    storyline_id TEXT NOT NULL REFERENCES m3_production(id),
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    scene_id TEXT NOT NULL,
    artifact_id TEXT REFERENCES artifact(id),
    source_artifact_id TEXT REFERENCES artifact(id),
    source TEXT NOT NULL,
    prompt TEXT NOT NULL,
    metadata TEXT NOT NULL,
    job_id TEXT REFERENCES job(id),
    created_at REAL NOT NULL
);
CREATE TABLE music_adjustment_job (
    job_id TEXT PRIMARY KEY REFERENCES job(id),
    draft_id TEXT NOT NULL REFERENCES adjustment_draft(id),
    candidate_id TEXT NOT NULL REFERENCES music_candidate(id),
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    scene_id TEXT NOT NULL,
    adoption_revision INTEGER NOT NULL CHECK(adoption_revision>0)
);
CREATE INDEX music_candidate_story ON music_candidate(storyline_id,created_at);
