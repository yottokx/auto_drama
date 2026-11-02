-- The BGM settings new works start with; each work keeps the precision of its first track.
CREATE TABLE music_settings (
    id INTEGER PRIMARY KEY CHECK(id=1),
    profile TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK(revision>0)
);
