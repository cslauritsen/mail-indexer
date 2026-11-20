-- Mail search index: SQLite + FTS5 + sqlite-vec.
-- {DIM} is replaced at startup by EMBED_DIM (1024 for bge-m3).

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- One row per email. id = sha256(Message-ID), or sha256(raw file) as fallback.
CREATE TABLE IF NOT EXISTS messages (
    id            TEXT PRIMARY KEY,
    message_id    TEXT,
    in_reply_to   TEXT,
    refs          TEXT,                -- References header, space-separated, no angle brackets
    folder        TEXT NOT NULL,       -- 'INBOX' or a Maildir++ folder such as '.Church'
    maildir_name  TEXT NOT NULL,       -- filename WITHOUT the ':2,FLAGS' suffix
    date_utc      INTEGER NOT NULL,
    from_addr     TEXT,
    from_name     TEXT,
    to_addrs      TEXT,
    subject       TEXT,
    snippet       TEXT,
    has_attach    INTEGER NOT NULL DEFAULT 0,
    indexed_at    INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id);
CREATE INDEX IF NOT EXISTS idx_messages_location   ON messages(folder, maildir_name);
CREATE INDEX IF NOT EXISTS idx_messages_date       ON messages(date_utc);
CREATE INDEX IF NOT EXISTS idx_messages_from       ON messages(from_addr);
CREATE INDEX IF NOT EXISTS idx_messages_reply      ON messages(in_reply_to);

-- One row per text chunk. chunks.id is the shared rowid for FTS and vectors.
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    msg_id    TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    chunk_no  INTEGER NOT NULL,
    text      TEXT NOT NULL,
    UNIQUE (msg_id, chunk_no)
);

-- Exact-word search. rowid = chunks.id.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, subject, sender,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Semantic search. rowid = chunks.id.
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
    embedding float[{DIM}]
);

-- sqlite-vec has no foreign keys, so clean FTS and vectors here.
CREATE TRIGGER IF NOT EXISTS trg_chunks_delete AFTER DELETE ON chunks
BEGIN
    DELETE FROM chunks_fts    WHERE rowid = old.id;
    DELETE FROM chunk_vectors WHERE rowid = old.id;
END;
