CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    avatar_color TEXT NOT NULL DEFAULT '#673ab7',
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member')),
    is_active SMALLINT NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_login_at TEXT
);
CREATE INDEX IF NOT EXISTS users_email_idx ON users(email);

CREATE TABLE IF NOT EXISTS threads (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    parent_thread_id TEXT,
    branch_from_message_id BIGINT,
    branch_label TEXT,
    user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS threads_user_id_idx ON threads(user_id);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_active SMALLINT NOT NULL DEFAULT 1,
    selected_version INTEGER NOT NULL DEFAULT 1,
    failed SMALLINT NOT NULL DEFAULT 0,
    tools_json TEXT,
    parts_json TEXT
);
CREATE INDEX IF NOT EXISTS messages_thread_id_id ON messages(thread_id, id);

CREATE TABLE IF NOT EXISTS turn_versions (
    user_message_id BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    assistant_message_id BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    version_index INTEGER NOT NULL,
    user_content TEXT NOT NULL,
    assistant_content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    failed SMALLINT NOT NULL DEFAULT 0,
    tools_json TEXT,
    parts_json TEXT,
    PRIMARY KEY (user_message_id, version_index)
);

CREATE TABLE IF NOT EXISTS attachments (
    attachment_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    message_id BIGINT REFERENCES messages(id) ON DELETE SET NULL,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    storage_name TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    extracted_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS attachments_thread_message ON attachments(thread_id, message_id);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS canvases (
    id BIGSERIAL PRIMARY KEY,
    canvas_id TEXT NOT NULL UNIQUE,
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS canvases_thread_id ON canvases(thread_id);

CREATE TABLE IF NOT EXISTS artifacts (
    id BIGSERIAL PRIMARY KEY,
    artifact_id TEXT NOT NULL UNIQUE,
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    turn_id BIGINT,
    title TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    language TEXT,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_thread_id ON artifacts(thread_id);

CREATE TABLE IF NOT EXISTS user_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    remember_me SMALLINT NOT NULL DEFAULT 0,
    user_agent TEXT NOT NULL DEFAULT '',
    ip_address TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked SMALLINT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS user_sessions_token_hash_idx ON user_sessions(token_hash);
CREATE INDEX IF NOT EXISTS user_sessions_user_id_idx ON user_sessions(user_id);

CREATE TABLE IF NOT EXISTS user_recovery_codes (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    code_hash TEXT NOT NULL,
    used SMALLINT NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS user_recovery_codes_user_idx ON user_recovery_codes(user_id, used);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS user_otp_codes (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    purpose TEXT NOT NULL CHECK (purpose IN ('login', 'recovery')),
    code_hash TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    used SMALLINT NOT NULL DEFAULT 0,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS user_otp_codes_lookup_idx ON user_otp_codes(user_id, purpose, used);

CREATE TABLE IF NOT EXISTS user_personal_skills (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL DEFAULT '',
    is_enabled SMALLINT NOT NULL DEFAULT 1,
    is_draft SMALLINT NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'custom',
    files_json TEXT NOT NULL DEFAULT '[]',
    security_scan_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS user_personal_skills_user_idx ON user_personal_skills(user_id);

CREATE TABLE IF NOT EXISTS user_personal_connectors (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    mcp_url TEXT NOT NULL,
    connector_type TEXT NOT NULL DEFAULT 'Web',
    auth_mode TEXT NOT NULL DEFAULT 'none' CHECK (auth_mode IN ('now', 'later', 'none')),
    headers_json TEXT NOT NULL DEFAULT '[]',
    advanced_json TEXT NOT NULL DEFAULT '{}',
    tools_json TEXT NOT NULL DEFAULT '[]',
    tool_permissions_json TEXT NOT NULL DEFAULT '{}',
    is_connected SMALLINT NOT NULL DEFAULT 0,
    last_status_code INTEGER,
    last_checked_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS user_personal_connectors_user_idx ON user_personal_connectors(user_id);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    thread_id TEXT,
    tool_name TEXT NOT NULL,
    connector_id TEXT,
    action_summary TEXT NOT NULL,
    risk_level TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'pending',
    details_json TEXT NOT NULL DEFAULT '{}',
    decision_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS approvals_user_idx ON approvals(user_id);
CREATE INDEX IF NOT EXISTS approvals_status_idx ON approvals(status);

CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    thread_id TEXT,
    tool_name TEXT NOT NULL,
    action_name TEXT NOT NULL,
    policy_evaluated TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    status TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT 'system',
    input_payload_json TEXT NOT NULL DEFAULT '{}',
    output_payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_logs_user_idx ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS audit_logs_created_idx ON audit_logs(created_at);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    thread_id TEXT,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    input TEXT NOT NULL,
    result TEXT,
    error TEXT,
    retry_count INTEGER DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_thread ON tasks(thread_id);

CREATE TABLE IF NOT EXISTS task_events (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    payload TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_events_task_id ON task_events(task_id);

CREATE TABLE IF NOT EXISTS proactive_routines (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    action_prompt TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    routine_type TEXT NOT NULL DEFAULT 'reminder',
    schedule_expression TEXT,
    schedule_timezone TEXT NOT NULL DEFAULT 'UTC',
    schedule_interval_seconds INTEGER,
    schedule_display TEXT,
    one_shot SMALLINT NOT NULL DEFAULT 0,
    next_run_at TEXT,
    last_run_at TEXT,
    enabled SMALLINT NOT NULL DEFAULT 1,
    target_thread_id TEXT,
    webhook_token TEXT,
    webhook_secret TEXT,
    require_approval SMALLINT NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    trigger_definition TEXT,
    execution_definition TEXT,
    event_source TEXT,
    event_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_routines_sched ON proactive_routines(enabled, next_run_at);
CREATE INDEX IF NOT EXISTS idx_routines_whk ON proactive_routines(webhook_token);

CREATE TABLE IF NOT EXISTS proactive_subscriptions (
    id TEXT PRIMARY KEY,
    routine_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_subscription_id TEXT,
    event_type TEXT,
    native_filters TEXT,
    residual_filters TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    level INTEGER,
    expires_at TEXT,
    last_verified_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_subscriptions_routine ON proactive_subscriptions(routine_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_provider ON proactive_subscriptions(provider_subscription_id);

CREATE TABLE IF NOT EXISTS proactive_events (
    id TEXT PRIMARY KEY,
    provider_event_id TEXT,
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    subscription_id TEXT,
    received_at TEXT NOT NULL,
    processing_status TEXT NOT NULL DEFAULT 'received',
    payload_reference TEXT,
    payload TEXT,
    match_reason TEXT,
    UNIQUE(provider_event_id, tenant_id)
);
CREATE INDEX IF NOT EXISTS idx_events_provider ON proactive_events(provider_event_id);
CREATE INDEX IF NOT EXISTS idx_events_status ON proactive_events(processing_status);

CREATE TABLE IF NOT EXISTS proactive_inbox (
    id TEXT PRIMARY KEY,
    routine_id TEXT NOT NULL,
    event_record_id TEXT,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inbox_pending ON proactive_inbox(status, created_at);

CREATE TABLE IF NOT EXISTS proactive_runs (
    id TEXT PRIMARY KEY,
    routine_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    trigger_source TEXT NOT NULL,
    duration_ms INTEGER DEFAULT 0,
    result_summary TEXT,
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_routine ON proactive_runs(routine_id, created_at DESC);

CREATE TABLE IF NOT EXISTS proactive_notifications (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    routine_id TEXT,
    run_id TEXT,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    notification_type TEXT NOT NULL DEFAULT 'reminder',
    is_read SMALLINT NOT NULL DEFAULT 0,
    data TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notifs_user ON proactive_notifications(user_id, is_read, created_at DESC);

CREATE TABLE IF NOT EXISTS proactive_provider_accounts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_account_id TEXT NOT NULL,
    account_name TEXT NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    scopes TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(user_id, provider, provider_account_id)
);
CREATE INDEX IF NOT EXISTS idx_provider_accounts_lookup ON proactive_provider_accounts(provider, provider_account_id);

CREATE TABLE IF NOT EXISTS proactive_github_hooks (
    id TEXT PRIMARY KEY,
    token TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    repository TEXT NOT NULL,
    routine_id TEXT NOT NULL,
    hook_id BIGINT,
    secret TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_github_hooks_routine ON proactive_github_hooks(routine_id);

CREATE TABLE IF NOT EXISTS proactive_gmail_watch_state (
    user_id TEXT NOT NULL,
    email_address TEXT NOT NULL,
    history_id TEXT NOT NULL,
    expiration TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(user_id, email_address)
);

CREATE TABLE IF NOT EXISTS memory_store (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_memory_ns ON memory_store(namespace);
