CREATE EXTENSION IF NOT EXISTS vector;

-- One row per label (SPL set). raw keeps the full openFDA record so extraction can be
-- scored against the structured sections without a second download.
CREATE TABLE IF NOT EXISTS labels (
    set_id          TEXT PRIMARY KEY,
    brand_name      TEXT,
    generic_name    TEXT,
    manufacturer    TEXT,
    effective_time  DATE,
    raw             JSONB NOT NULL
);

-- One row per chunk of a label section. position is the chunk's order inside the section.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    set_id      TEXT NOT NULL REFERENCES labels(set_id) ON DELETE CASCADE,
    section     TEXT NOT NULL,
    position    INT  NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(384) NOT NULL,
    UNIQUE (set_id, section, position)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_set_id_idx ON chunks (set_id);

-- Full-text index for hybrid retrieval. Dense vectors miss exact terms ("boxed warning", a drug
-- name); Postgres FTS catches them, so no separate BM25 library is needed.
-- The full-text column carries the drug's names and the section name as well as the text, so
-- "boxed warning for rivaroxaban" matches a boxed-warning chunk that only ever says "XARELTO", and
-- "boxed warning" matches chunks that start with "WARNING: ..." without the word "boxed".
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS names TEXT NOT NULL DEFAULT '';  -- generic + brand, copied at ingest
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', names || '. ' || replace(section, '_', ' ') || '. ' || content)) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
