-- Persist an inspectable, ordered execution conversation for every routine run.
CREATE TABLE IF NOT EXISTS proactive_run_messages (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES proactive_runs(id) ON DELETE CASCADE,
    entry_type TEXT NOT NULL CHECK (
        entry_type IN ('prompt', 'trigger', 'status', 'tool', 'assistant', 'error')
    ),
    status TEXT,
    title TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_messages_run_id_id
    ON proactive_run_messages(run_id, id);
