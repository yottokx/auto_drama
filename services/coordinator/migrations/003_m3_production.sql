CREATE TABLE m3_production (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    approval_id TEXT NOT NULL UNIQUE REFERENCES m2_approval(id),
    approval_artifact_id TEXT NOT NULL REFERENCES artifact(id),
    narrative_artifact_id TEXT REFERENCES artifact(id),
    error TEXT,
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE TABLE m3_requirement (
    id TEXT PRIMARY KEY,
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    descriptor TEXT NOT NULL,
    job_id TEXT UNIQUE REFERENCES job(id),
    artifact_id TEXT REFERENCES artifact(id),
    UNIQUE(production_id, kind, target_id)
);
CREATE TABLE m3_production_job (
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    job_id TEXT NOT NULL UNIQUE REFERENCES job(id),
    PRIMARY KEY(production_id, job_id)
);
CREATE TABLE chapter_build (
    id TEXT PRIMARY KEY,
    production_id TEXT NOT NULL UNIQUE REFERENCES m3_production(id),
    project_id TEXT NOT NULL REFERENCES project(id),
    chapter_number INTEGER NOT NULL CHECK(chapter_number = 1),
    script_artifact_id TEXT NOT NULL REFERENCES artifact(id),
    export_artifact_id TEXT NOT NULL REFERENCES artifact(id),
    manifest TEXT NOT NULL,
    validation TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status = 'published'),
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE INDEX m3_production_project ON m3_production(project_id, created_at);
CREATE INDEX m3_requirement_production ON m3_requirement(production_id);
CREATE TRIGGER chapter_build_immutable BEFORE UPDATE ON chapter_build BEGIN
    SELECT RAISE(ABORT, 'published chapter builds are immutable');
END;
