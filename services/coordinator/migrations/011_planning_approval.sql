CREATE TABLE planning_draft (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    main_approval_id TEXT NOT NULL UNIQUE REFERENCES m2_approval(id),
    revision INTEGER NOT NULL DEFAULT 1 CHECK(revision>0),
    content TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    active_job_id TEXT REFERENCES job(id),
    approved_plan_id TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE planning_approval (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    planning_id TEXT NOT NULL REFERENCES planning_draft(id),
    revision INTEGER NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES artifact(id),
    sha256 TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(planning_id,revision)
);
CREATE TRIGGER planning_approval_immutable BEFORE UPDATE ON planning_approval BEGIN
    SELECT RAISE(ABORT, 'approved plans are immutable');
END;
CREATE TABLE planning_job (
    planning_id TEXT NOT NULL REFERENCES planning_draft(id),
    job_id TEXT NOT NULL UNIQUE REFERENCES job(id),
    revision INTEGER NOT NULL,
    PRIMARY KEY(planning_id,job_id)
);
ALTER TABLE project_history_state ADD COLUMN planning_id TEXT REFERENCES planning_draft(id);

-- Preserve existing IDs and legacy execution rules; new series pin their plan.
CREATE TABLE m3_production_planned (
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
    plan_approval_id TEXT REFERENCES planning_approval(id),
    sequential_publication INTEGER NOT NULL DEFAULT 0 CHECK(sequential_publication IN (0,1)),
    UNIQUE(storyline_id,chapter_number)
);
INSERT INTO m3_production_planned
    (id,project_id,approval_id,approval_artifact_id,narrative_artifact_id,error,schema_version,
     created_at,storyline_id,chapter_number,previous_narrative_artifact_id,previous_state_hash,
     control_state,m4_enabled)
    SELECT id,project_id,approval_id,approval_artifact_id,narrative_artifact_id,error,schema_version,
           created_at,storyline_id,chapter_number,previous_narrative_artifact_id,previous_state_hash,
           control_state,m4_enabled FROM m3_production;
DROP TABLE m3_production;
ALTER TABLE m3_production_planned RENAME TO m3_production;
CREATE INDEX m3_production_project ON m3_production(project_id,created_at);
CREATE INDEX m3_production_storyline ON m3_production(storyline_id,chapter_number);
CREATE UNIQUE INDEX m3_production_plan_root ON m3_production(plan_approval_id)
    WHERE chapter_number=1 AND plan_approval_id IS NOT NULL;
