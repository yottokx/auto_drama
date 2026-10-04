CREATE TABLE publication_edition (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    source_edition_id TEXT REFERENCES publication_edition(id),
    state TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE edition_build (
    edition_id TEXT NOT NULL REFERENCES publication_edition(id),
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    build_id TEXT NOT NULL UNIQUE REFERENCES chapter_build(id),
    PRIMARY KEY(edition_id,production_id)
);
CREATE TRIGGER publication_edition_immutable BEFORE UPDATE ON publication_edition BEGIN
    SELECT RAISE(ABORT,'published editions are immutable');
END;
CREATE TRIGGER edition_build_immutable BEFORE UPDATE ON edition_build BEGIN
    SELECT RAISE(ABORT,'edition chapter selections are immutable');
END;
CREATE TABLE adjustment_draft (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    base_edition_id TEXT NOT NULL REFERENCES publication_edition(id),
    revision INTEGER NOT NULL CHECK(revision>0),
    state TEXT NOT NULL,
    status TEXT NOT NULL,
    active_apply_id TEXT,
    error TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE adjustment_candidate (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    character_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('image','voice')),
    artifact_id TEXT REFERENCES artifact(id),
    original_artifact_id TEXT REFERENCES artifact(id),
    reference_text TEXT,
    source TEXT NOT NULL,
    prompt TEXT NOT NULL,
    job_id TEXT REFERENCES job(id),
    sample_artifact_id TEXT REFERENCES artifact(id),
    metadata TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE adjustment_apply (
    id TEXT PRIMARY KEY,
    draft_id TEXT NOT NULL REFERENCES adjustment_draft(id),
    revision INTEGER NOT NULL,
    state TEXT NOT NULL,
    status TEXT NOT NULL,
    edition_id TEXT REFERENCES publication_edition(id),
    error TEXT
);
CREATE TABLE adjustment_job (
    job_id TEXT PRIMARY KEY REFERENCES job(id),
    draft_id TEXT NOT NULL REFERENCES adjustment_draft(id),
    purpose TEXT NOT NULL CHECK(purpose IN ('candidate','dialogue','sample')),
    apply_id TEXT REFERENCES adjustment_apply(id),
    candidate_id TEXT REFERENCES adjustment_candidate(id),
    character_id TEXT NOT NULL,
    target_id TEXT,
    production_id TEXT REFERENCES m3_production(id),
    artifact_id TEXT REFERENCES artifact(id),
    adoption_revision INTEGER NOT NULL CHECK(adoption_revision>0)
);
ALTER TABLE project_history_state ADD COLUMN edition_id TEXT REFERENCES publication_edition(id);
ALTER TABLE project_history_state ADD COLUMN adjustment_draft_id TEXT REFERENCES adjustment_draft(id);
