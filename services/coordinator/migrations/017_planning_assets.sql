-- Portraits and reference voices of planned supporting characters, made while the plan is reviewed.
CREATE TABLE planning_asset (
    id TEXT PRIMARY KEY,
    planning_id TEXT NOT NULL REFERENCES planning_draft(id),
    character_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('image','voice')),
    source_sha256 TEXT NOT NULL,
    job_id TEXT REFERENCES job(id),
    artifact_id TEXT REFERENCES artifact(id),
    created_at REAL NOT NULL
);
CREATE INDEX planning_asset_current ON planning_asset(planning_id,character_id,kind,created_at);
