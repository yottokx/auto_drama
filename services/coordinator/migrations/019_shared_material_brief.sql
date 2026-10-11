-- A material declared in the STEP1 brief keeps the identifier the brief gave it.
ALTER TABLE shared_material ADD COLUMN brief_id TEXT;
