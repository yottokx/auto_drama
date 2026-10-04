CREATE TABLE event_cg_settings (
    id INTEGER PRIMARY KEY CHECK(id=1),
    profile TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK(revision>0)
);
CREATE TABLE event_cg_policy (
    project_id TEXT PRIMARY KEY REFERENCES project(id),
    max_cgs INTEGER NOT NULL CHECK(max_cgs BETWEEN 0 AND 100),
    max_variants_per_cg INTEGER NOT NULL CHECK(max_variants_per_cg BETWEEN 0 AND 10),
    revision INTEGER NOT NULL CHECK(revision>0)
);
CREATE TABLE event_cg_production (
    production_id TEXT PRIMARY KEY REFERENCES m3_production(id),
    policy TEXT NOT NULL
);
CREATE TRIGGER event_cg_production_immutable BEFORE UPDATE ON event_cg_production BEGIN
    SELECT RAISE(ABORT, 'production CG policies are immutable');
END;
