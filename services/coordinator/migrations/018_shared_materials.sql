-- Outfits and items several characters share; each holds one or more design sheets.
CREATE TABLE shared_material (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES project(id),
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    removed INTEGER NOT NULL DEFAULT 0 CHECK(removed IN (0,1)),
    created_at REAL NOT NULL
);
CREATE INDEX shared_material_project ON shared_material(project_id,created_at);
CREATE TABLE shared_material_image (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES shared_material(id),
    label TEXT NOT NULL DEFAULT '',
    wish TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL CHECK(mode IN ('text','derive','extract')),
    source TEXT NOT NULL DEFAULT '{}',
    job_id TEXT REFERENCES job(id),
    artifact_id TEXT REFERENCES artifact(id),
    detail TEXT,
    removed INTEGER NOT NULL DEFAULT 0 CHECK(removed IN (0,1)),
    created_at REAL NOT NULL
);
CREATE INDEX shared_material_image_material ON shared_material_image(material_id,created_at);
