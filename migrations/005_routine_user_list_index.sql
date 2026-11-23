-- Accelerate the user's routine sidebar, which filters by user_id and sorts newest first.
CREATE INDEX IF NOT EXISTS idx_routines_user_created
    ON proactive_routines (user_id, created_at DESC);
