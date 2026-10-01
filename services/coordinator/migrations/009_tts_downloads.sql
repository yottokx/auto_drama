CREATE TABLE tts_settings (
    id INTEGER PRIMARY KEY CHECK(id=1),
    value TEXT NOT NULL
);

CREATE TABLE tts_worker_inventory (
    worker_id TEXT NOT NULL REFERENCES worker(id),
    model_id TEXT NOT NULL,
    precision TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    file_download_ready INTEGER NOT NULL CHECK(file_download_ready IN (0,1)),
    updated_at REAL NOT NULL,
    PRIMARY KEY(worker_id,model_id,precision,manifest_id)
);

CREATE TABLE tts_download (
    id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL REFERENCES worker(id),
    model_id TEXT NOT NULL,
    precision TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    manifest TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'queued','downloading','verifying','cancelling','cancelled','interrupted','failed','completed'
    )),
    phase TEXT,
    done_bytes INTEGER NOT NULL DEFAULT 0 CHECK(done_bytes>=0),
    total_bytes INTEGER NOT NULL DEFAULT 0 CHECK(total_bytes>=0),
    current_file TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
    lease_id TEXT,
    lease_expires_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX tts_download_worker_status ON tts_download(worker_id,status,created_at);
CREATE INDEX tts_download_manifest ON tts_download(worker_id,manifest_id);
