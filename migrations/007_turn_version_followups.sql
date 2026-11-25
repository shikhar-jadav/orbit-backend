-- Store active follow-up message IDs per turn version so switching versions
-- hides other branches' follow-ups and restores this version's follow-ups.
ALTER TABLE turn_versions
    ADD COLUMN IF NOT EXISTS followup_message_ids_json TEXT;
