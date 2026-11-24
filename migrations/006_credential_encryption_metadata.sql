-- Backward-compatible credential encryption metadata and reconnection state.
-- Preserves all existing records without dropping or truncating any tables.

ALTER TABLE proactive_provider_accounts
    ADD COLUMN IF NOT EXISTS encryption_version TEXT;

ALTER TABLE proactive_provider_accounts
    ADD COLUMN IF NOT EXISTS encryption_key_version TEXT;

ALTER TABLE proactive_provider_accounts
    ADD COLUMN IF NOT EXISTS credential_status TEXT NOT NULL DEFAULT 'active';

ALTER TABLE proactive_provider_accounts
    ADD COLUMN IF NOT EXISTS credential_error TEXT;

ALTER TABLE proactive_github_hooks
    ADD COLUMN IF NOT EXISTS encryption_version TEXT;

ALTER TABLE proactive_github_hooks
    ADD COLUMN IF NOT EXISTS encryption_key_version TEXT;

ALTER TABLE proactive_github_hooks
    ADD COLUMN IF NOT EXISTS credential_status TEXT NOT NULL DEFAULT 'active';

ALTER TABLE proactive_routines
    ADD COLUMN IF NOT EXISTS encryption_version TEXT;

ALTER TABLE proactive_routines
    ADD COLUMN IF NOT EXISTS encryption_key_version TEXT;

ALTER TABLE proactive_routines
    ADD COLUMN IF NOT EXISTS credential_status TEXT NOT NULL DEFAULT 'active';

ALTER TABLE user_personal_connectors
    ADD COLUMN IF NOT EXISTS encryption_version TEXT;

ALTER TABLE user_personal_connectors
    ADD COLUMN IF NOT EXISTS encryption_key_version TEXT;

ALTER TABLE user_personal_connectors
    ADD COLUMN IF NOT EXISTS credential_status TEXT NOT NULL DEFAULT 'active';
