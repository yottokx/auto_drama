ALTER TABLE tts_worker_inventory ADD COLUMN generation_ready INTEGER NOT NULL DEFAULT 0
    CHECK(generation_ready IN (0,1));
ALTER TABLE tts_worker_inventory ADD COLUMN generation_purposes TEXT NOT NULL DEFAULT '[]';
