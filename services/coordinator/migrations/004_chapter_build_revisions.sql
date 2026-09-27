-- Rendering changes create a new immutable build from the same adopted inputs.
CREATE TABLE chapter_build_next (
    id TEXT PRIMARY KEY,
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    project_id TEXT NOT NULL REFERENCES project(id),
    chapter_number INTEGER NOT NULL CHECK(chapter_number = 1),
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
INSERT INTO chapter_build_next
    (id, production_id, project_id, chapter_number, script_artifact_id,
     export_artifact_id, manifest, validation, status, schema_version, created_at, revision)
    SELECT id, production_id, project_id, chapter_number, script_artifact_id,
           export_artifact_id, manifest, validation, status, schema_version, created_at, 1
    FROM chapter_build;
DROP TABLE chapter_build;
ALTER TABLE chapter_build_next RENAME TO chapter_build;
CREATE TRIGGER chapter_build_immutable BEFORE UPDATE ON chapter_build BEGIN
    SELECT RAISE(ABORT, 'published chapter builds are immutable');
END;
