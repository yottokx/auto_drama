-- Expression pictures of a portrait that is not a main character's STEP3 look: a planned
-- supporting character's, or one chosen while adjusting a finished work. Each is edited from
-- one portrait and belongs to it; the newest finished row of an expression is the one shown.
CREATE TABLE portrait_expression (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    character_id TEXT NOT NULL,
    base_artifact_id TEXT NOT NULL REFERENCES artifact(id),
    expression TEXT NOT NULL,
    job_id TEXT REFERENCES job(id),
    artifact_id TEXT REFERENCES artifact(id),
    removed INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX portrait_expression_current ON portrait_expression(project_id,base_artifact_id,expression,created_at);
