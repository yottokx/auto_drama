-- Preserve every M3 production/build/artifact ID while extending chapters.
CREATE TABLE m3_production_next (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    approval_id TEXT NOT NULL REFERENCES m2_approval(id),
    approval_artifact_id TEXT NOT NULL REFERENCES artifact(id),
    narrative_artifact_id TEXT REFERENCES artifact(id),
    error TEXT,
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    storyline_id TEXT NOT NULL REFERENCES m3_production(id),
    chapter_number INTEGER NOT NULL DEFAULT 1 CHECK(chapter_number BETWEEN 1 AND 100),
    previous_narrative_artifact_id TEXT REFERENCES artifact(id),
    previous_state_hash TEXT,
    control_state TEXT NOT NULL DEFAULT 'running'
        CHECK(control_state IN ('running','stopping','paused','interrupted')),
    m4_enabled INTEGER NOT NULL DEFAULT 0 CHECK(m4_enabled IN (0,1)),
    UNIQUE(approval_id, chapter_number)
);
INSERT INTO m3_production_next
    (id,project_id,approval_id,approval_artifact_id,narrative_artifact_id,error,schema_version,
     created_at,storyline_id)
    SELECT id,project_id,approval_id,approval_artifact_id,narrative_artifact_id,error,
           schema_version,created_at,id FROM m3_production;
DROP TABLE m3_production;
ALTER TABLE m3_production_next RENAME TO m3_production;
CREATE INDEX m3_production_project ON m3_production(project_id, created_at);
CREATE INDEX m3_production_storyline ON m3_production(storyline_id, chapter_number);

CREATE TABLE chapter_build_next (
    id TEXT PRIMARY KEY,
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    project_id TEXT NOT NULL REFERENCES project(id),
    chapter_number INTEGER NOT NULL CHECK(chapter_number BETWEEN 1 AND 100),
    script_artifact_id TEXT NOT NULL REFERENCES artifact(id),
    export_artifact_id TEXT NOT NULL REFERENCES artifact(id),
    manifest TEXT NOT NULL,
    validation TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status = 'published'),
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    revision INTEGER NOT NULL CHECK(revision >= 1),
    UNIQUE(production_id, revision)
);
INSERT INTO chapter_build_next SELECT * FROM chapter_build;
DROP TABLE chapter_build;
ALTER TABLE chapter_build_next RENAME TO chapter_build;
CREATE TRIGGER chapter_build_immutable BEFORE UPDATE ON chapter_build BEGIN
    SELECT RAISE(ABORT, 'published chapter builds are immutable');
END;
