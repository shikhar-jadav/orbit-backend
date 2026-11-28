-- Named bots with persistent instructions and bot-owned conversations.
CREATE TABLE IF NOT EXISTS bots (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    instructions TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS bots_user_name_unique
    ON bots (user_id, LOWER(name));
CREATE INDEX IF NOT EXISTS bots_user_created_idx
    ON bots (user_id, created_at ASC);
CREATE TABLE IF NOT EXISTS bot_threads (
    thread_id TEXT PRIMARY KEY REFERENCES threads(thread_id) ON DELETE CASCADE,
    bot_id TEXT NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS bot_threads_user_bot_idx
    ON bot_threads (user_id, bot_id, created_at DESC);
