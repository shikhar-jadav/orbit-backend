-- Indexes for recent-thread navigation and bounded active-message history.
-- Partial active-message index also accelerates preview, count, and history page lookups.
CREATE INDEX IF NOT EXISTS threads_user_updated_id_idx
    ON threads (user_id, updated_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS messages_active_thread_id_desc_idx
    ON messages (thread_id, id DESC)
    WHERE is_active = 1;

CREATE INDEX IF NOT EXISTS artifacts_thread_turn_idx
    ON artifacts (thread_id, turn_id);
