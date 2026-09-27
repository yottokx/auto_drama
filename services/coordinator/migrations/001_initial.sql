CREATE TABLE project (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    instructions TEXT NOT NULL,
    chapter_count INTEGER NOT NULL CHECK (chapter_count > 0),
    settings_version INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE TABLE generation_run (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    settings_version INTEGER NOT NULL,
    story_revision_id TEXT NOT NULL,
    policy TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE TABLE worker (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    capabilities TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    last_seen_at REAL NOT NULL
);
CREATE TABLE job (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    run_id TEXT NOT NULL REFERENCES generation_run(id),
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    settings_snapshot TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    result_artifact_id TEXT REFERENCES artifact(id),
    error TEXT,
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE job_dependency (
    job_id TEXT NOT NULL REFERENCES job(id),
    depends_on_id TEXT NOT NULL REFERENCES job(id),
    PRIMARY KEY (job_id, depends_on_id),
    CHECK (job_id != depends_on_id)
);
CREATE TABLE job_attempt (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES job(id),
    worker_id TEXT NOT NULL REFERENCES worker(id),
    attempt INTEGER NOT NULL,
    lease_id TEXT NOT NULL UNIQUE,
    lease_expires_at REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'expired')),
    error_kind TEXT,
    error TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    UNIQUE (job_id, attempt)
);
CREATE TABLE artifact (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    logical_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    kind TEXT NOT NULL,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    storage_key TEXT NOT NULL,
    source_job_id TEXT REFERENCES job(id),
    source_attempt_id TEXT REFERENCES job_attempt(id),
    provenance TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    UNIQUE (project_id, logical_id, version)
);
CREATE INDEX job_ready ON job(status, priority DESC, created_at);
CREATE INDEX job_by_project ON job(project_id);
CREATE INDEX artifact_by_project ON artifact(project_id);
CREATE INDEX attempt_active ON job_attempt(status, lease_expires_at);
