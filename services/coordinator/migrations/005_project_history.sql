-- History stores small state documents and immutable artifact/build references.
-- Media and export archives are never copied by a checkpoint or a restore.
CREATE TABLE project_revision (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    number INTEGER NOT NULL CHECK (number > 0),
    label TEXT NOT NULL,
    action TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    restored_from_id TEXT REFERENCES project_revision(id),
    created_at REAL NOT NULL,
    UNIQUE(project_id, number)
);
CREATE INDEX project_revision_project ON project_revision(project_id, number);
CREATE TRIGGER project_revision_immutable BEFORE UPDATE ON project_revision BEGIN
    SELECT RAISE(ABORT, 'project revisions are immutable');
END;
CREATE TABLE project_history_state (
    project_id TEXT PRIMARY KEY REFERENCES project(id),
    version INTEGER NOT NULL DEFAULT 1,
    current_revision_id TEXT REFERENCES project_revision(id),
    operation TEXT,
    production_id TEXT REFERENCES m3_production(id),
    build_id TEXT REFERENCES chapter_build(id),
    portrait_settings_id TEXT REFERENCES artifact(id),
    production_snapshot TEXT,
    production_frozen INTEGER NOT NULL DEFAULT 0 CHECK (production_frozen IN (0,1))
);
