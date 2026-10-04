CREATE TABLE music_replan_job (
    job_id TEXT PRIMARY KEY REFERENCES job(id),
    draft_id TEXT NOT NULL REFERENCES adjustment_draft(id),
    production_id TEXT NOT NULL REFERENCES m3_production(id),
    adoption_revision INTEGER NOT NULL CHECK(adoption_revision>0)
);
CREATE INDEX music_replan_draft ON music_replan_job(draft_id);
