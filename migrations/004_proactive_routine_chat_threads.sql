-- Keep one normal chat conversation per routine and record which runs
-- have already been imported into that conversation.
ALTER TABLE proactive_routines
    ADD COLUMN IF NOT EXISTS chat_thread_id TEXT;

ALTER TABLE proactive_runs
    ADD COLUMN IF NOT EXISTS chat_thread_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_routines_chat_thread
    ON proactive_routines(chat_thread_id)
    WHERE chat_thread_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_runs_chat_thread
    ON proactive_runs(chat_thread_id)
    WHERE chat_thread_id IS NOT NULL;
