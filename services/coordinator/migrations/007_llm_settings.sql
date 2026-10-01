CREATE TABLE llm_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    value TEXT NOT NULL
);
ALTER TABLE worker ADD COLUMN llm_models TEXT NOT NULL DEFAULT '[]';
