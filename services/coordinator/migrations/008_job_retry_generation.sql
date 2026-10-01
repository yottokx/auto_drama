ALTER TABLE job ADD COLUMN retry_generation INTEGER NOT NULL DEFAULT 0 CHECK (retry_generation >= 0);
