CREATE TABLE m2_draft (
    project_id TEXT PRIMARY KEY REFERENCES project(id),
    state TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE m2_approval (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    revision INTEGER NOT NULL,
    artifact_id TEXT NOT NULL UNIQUE REFERENCES artifact(id),
    created_at REAL NOT NULL
);
CREATE INDEX m2_approval_by_project ON m2_approval(project_id, created_at);
