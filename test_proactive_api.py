import pytest
import server
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from chat_store import PostgresChatStore
from proactive_engine import ProactiveEngine, ProactiveStore
from proactive_integrations import ProviderIntegrationStore
from server import app
import auth
from conftest import EmptyCheckpointer, MemoryObjectStorage


def configure_provider_store(monkeypatch, tmp_path):
    monkeypatch.setenv("ORBIT_PROVIDER_WEBHOOK_MAX_BYTES", "1048576")
    store = ProviderIntegrationStore(
        server.database,
        encryption_key=Fernet.generate_key().decode("ascii"),
    )
    monkeypatch.setattr(server, "provider_integration_store", store)
    return store


@pytest.fixture(autouse=True)
def isolated_stores(postgres_database, monkeypatch):
    names = (
        "database",
        "chat_store",
        "proactive_store",
        "proactive_engine",
        "provider_integration_store",
        "object_storage",
        "checkpointer",
    )
    previous = {name: getattr(server, name) for name in names}
    store = PostgresChatStore(postgres_database)
    proactive_store = ProactiveStore(postgres_database)
    monkeypatch.setattr(server, "database", postgres_database)
    monkeypatch.setattr(server, "chat_store", store)
    monkeypatch.setattr(server, "proactive_store", proactive_store)
    monkeypatch.setattr(
        server,
        "proactive_engine",
        ProactiveEngine(store=proactive_store, chat_store=store),
    )
    monkeypatch.setattr(
        server,
        "provider_integration_store",
        ProviderIntegrationStore(postgres_database),
    )
    monkeypatch.setattr(server, "object_storage", MemoryObjectStorage())
    monkeypatch.setattr(server, "checkpointer", EmptyCheckpointer())
    yield
    for name, value in previous.items():
        setattr(server, name, value)


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    from datetime import datetime, timezone, timedelta
    from uuid import uuid4
    unique_email = f"proactive_{uuid4().hex[:8]}@example.com"
    user, _ = server.chat_store.create_user(
        email=unique_email,
        display_name="Proactive Tester",
        password_hash=auth.hash_password("Password123!"),
    )
    raw_token = auth.generate_session_token()
    exp = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    server.chat_store.create_session(
        user_id=user["user_id"],
        token_hash=auth.hash_token(raw_token),
        expires_at=exp,
    )
    return {"Authorization": f"Bearer {raw_token}"}, user["user_id"]


def test_routines_api_crud(test_client, auth_headers):
    headers, user_id = auth_headers

    # 1. Create a reminder
    res_create = test_client.post(
        "/api/routines",
        json={
            "name": "Hydration Reminder",
            "action_prompt": "Drink water",
            "schedule": "in 10 minutes",
        },
        headers=headers,
    )
    assert res_create.status_code == 201
    data = res_create.json()
    assert data["name"] == "Hydration Reminder"
    assert data["schedule_timezone"] == "UTC"
    routine_id = data["id"]

    # 2. List routines
    res_list = test_client.get("/api/routines", headers=headers)
    assert res_list.status_code == 200
    routines = res_list.json()["routines"]
    assert any(r["id"] == routine_id for r in routines)

    # 3. Update routine (pause)
    res_patch = test_client.patch(
        f"/api/routines/{routine_id}",
        json={"enabled": False},
        headers=headers,
    )
    assert res_patch.status_code == 200
    assert res_patch.json()["enabled"] is False

    # 4. Trigger test run
    res_run = test_client.post(f"/api/routines/{routine_id}/run", headers=headers)
    assert res_run.status_code == 200

    # 5. List runs
    res_runs = test_client.get(f"/api/routines/{routine_id}/runs", headers=headers)
    assert res_runs.status_code == 200
    assert len(res_runs.json()["runs"]) >= 1
    run_id = res_runs.json()["runs"][0]["id"]

    # 5b. Inspect the isolated run conversation.
    res_detail = test_client.get(
        f"/api/routines/{routine_id}/runs/{run_id}",
        headers=headers,
    )
    assert res_detail.status_code == 200
    assert res_detail.json()["run"]["id"] == run_id
    assert any(
        item["entry_type"] == "assistant"
        for item in res_detail.json()["messages"]
    )

    # Continue the run in a normal chat without reusing the automation transcript.
    res_continue = test_client.post(
        f"/api/routines/{routine_id}/runs/{run_id}/continue",
        headers=headers,
    )
    assert res_continue.status_code == 200
    continued_thread = res_continue.json()["thread"]
    assert continued_thread["title"] == "Hydration Reminder"
    persisted_thread = server.chat_store.get_thread(
        continued_thread["thread_id"],
        user_id=user_id,
        include_extras=False,
    )
    assert persisted_thread is not None
    assert len(persisted_thread["messages"]) == 2
    assert persisted_thread["messages"][0]["content"].startswith("Drink water")
    saved_assistant = persisted_thread["messages"][1]
    assert saved_assistant["content"]
    assert any(part.get("type") == "context" for part in saved_assistant["parts"])
    assert any(part.get("type") == "text" for part in saved_assistant["parts"])
    continued_history = server.history_for_document_follow_up(
        persisted_thread["messages"], None
    )
    assert "historical Orbit autonomous routine execution" in continued_history[1]["content"]

    # Reopening the same run must reuse the existing routine thread without
    # importing the same execution transcript a second time.
    res_continue_again = test_client.post(
        f"/api/routines/{routine_id}/runs/{run_id}/continue",
        headers=headers,
    )
    assert res_continue_again.status_code == 200
    assert res_continue_again.json()["thread"]["thread_id"] == continued_thread["thread_id"]
    assert len(
        server.chat_store.get_thread(
            continued_thread["thread_id"],
            user_id=user_id,
            include_extras=False,
        )["messages"]
    ) == 2

    # A later run from this routine is imported into the same routine chat.
    res_run_again = test_client.post(f"/api/routines/{routine_id}/run", headers=headers)
    assert res_run_again.status_code == 200
    next_runs = test_client.get(f"/api/routines/{routine_id}/runs", headers=headers).json()["runs"]
    next_run_id = next(item["id"] for item in next_runs if item["id"] != run_id)
    res_continue_next = test_client.post(
        f"/api/routines/{routine_id}/runs/{next_run_id}/continue",
        headers=headers,
    )
    assert res_continue_next.status_code == 200
    assert res_continue_next.json()["thread"]["thread_id"] == continued_thread["thread_id"]
    persisted_thread = server.chat_store.get_thread(
        continued_thread["thread_id"],
        user_id=user_id,
        include_extras=False,
    )
    assert len(persisted_thread["messages"]) == 4

    # 6. Notifications API
    res_notifs = test_client.get("/api/notifications", headers=headers)
    assert res_notifs.status_code == 200
    notifs = res_notifs.json()["notifications"]
    assert len(notifs) >= 1
    notif_id = notifs[0]["id"]

    # Mark as read
    res_read = test_client.post(f"/api/notifications/{notif_id}/read", headers=headers)
    assert res_read.status_code == 200

    # 7. Delete routine
    res_del = test_client.delete(f"/api/routines/{routine_id}", headers=headers)
    assert res_del.status_code == 200


def test_recurring_routine_keeps_selected_timezone(test_client, auth_headers):
    headers, _ = auth_headers
    response = test_client.post(
        "/api/routines",
        json={
            "name": "Weekday check-in",
            "action_prompt": "Send the daily check-in",
            "schedule": "30 9 * * 1-5",
            "schedule_timezone": "Asia/Kolkata",
        },
        headers=headers,
    )

    assert response.status_code == 201
    routine = response.json()
    assert routine["schedule_timezone"] == "Asia/Kolkata"
    assert routine["schedule_expression"] == "30 9 * * 1-5"


def test_interval_routine_keeps_elapsed_interval(test_client, auth_headers):
    headers, _ = auth_headers
    response = test_client.post(
        "/api/routines",
        json={
            "name": "Every two days",
            "action_prompt": "Check in",
            "schedule": "every 2 days",
        },
        headers=headers,
    )

    assert response.status_code == 201
    routine = response.json()
    assert routine["schedule_interval_seconds"] == 2 * 24 * 60 * 60


def test_routines_api_rejects_unknown_schedule_timezone(test_client, auth_headers):
    headers, _ = auth_headers
    response = test_client.post(
        "/api/routines",
        json={
            "name": "Invalid time zone",
            "action_prompt": "Check the schedule",
            "schedule": "30 9 * * *",
            "schedule_timezone": "Not/A-Timezone",
        },
        headers=headers,
    )

    assert response.status_code == 400
    assert "Unknown schedule time zone" in response.json()["detail"]


def test_routine_prompt_uses_model_interpretation(test_client, auth_headers, monkeypatch):
    headers, _ = auth_headers

    class FakeModel:
        async def ainvoke(self, _messages):
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"needs_clarification": false, "name": "Water break", '
                        '"action_prompt": "Remind me to drink water", '
                        '"schedule": "every 2 hours"}'
                    )
                },
            )()

    class FakeProviderManager:
        def get_provider_by_index(self, *_args, **_kwargs):
            return FakeModel(), None, 1

    monkeypatch.setattr(server, "provider_manager", FakeProviderManager())
    monkeypatch.setattr(
        server.chat_store,
        "get_settings",
        lambda **_kwargs: {"provider_index": 0, "model": "test-model"},
    )
    response = test_client.post(
        "/api/routines/from-prompt",
        json={
            "prompt": "Remind me to drink water every two hours",
            "schedule_timezone": "Asia/Kolkata",
        },
        headers=headers,
    )

    assert response.status_code == 201
    routine = response.json()
    assert routine["name"] == "Water break"
    assert routine["schedule_interval_seconds"] == 2 * 60 * 60
    assert routine["schedule_timezone"] == "Asia/Kolkata"


def test_routine_prompt_requests_clarification_when_ambiguous(
    test_client, auth_headers, monkeypatch
):
    headers, _ = auth_headers

    class FakeModel:
        async def ainvoke(self, _messages):
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"needs_clarification": true, '
                        '"clarification": "What time should I use?"}'
                    )
                },
            )()

    class FakeProviderManager:
        def get_provider_by_index(self, *_args, **_kwargs):
            return FakeModel(), None, 1

    monkeypatch.setattr(server, "provider_manager", FakeProviderManager())
    monkeypatch.setattr(
        server.chat_store,
        "get_settings",
        lambda **_kwargs: {"provider_index": 0, "model": "test-model"},
    )
    response = test_client.post(
        "/api/routines/from-prompt",
        json={"prompt": "Remind me to drink water"},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "What time should I use?"


def test_webhook_endpoint(test_client):
    # Register routine with webhook
    token = "whk_api_test_999"
    routine = server.proactive_store.create_routine(
        user_id="default_user",
        name="Webhook Test",
        action_prompt="Process webhook event",
        trigger_type="webhook",
        webhook_token=token,
    )

    res = test_client.post(
        f"/api/webhooks/{token}",
        json={"alert": "high_cpu", "value": 98},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "accepted"


def test_event_routine_setup_and_webhook_signature(test_client, auth_headers):
    headers, user_id = auth_headers
    response = test_client.post(
        "/api/routines/events",
        headers=headers,
        json={
            "name": "Bug triage",
            "instruction": "Investigate the new issue and summarize it.",
            "event_type": "issues.opened",
            "when_to_run": "When a new issue is opened with the bug label.",
            "filters": {
                "field": "payload.labels",
                "operator": "contains",
                "value": "bug",
            },
        },
    )

    assert response.status_code == 201
    data = response.json()
    routine = data["routine"]
    assert routine["trigger_type"] == "event"
    assert routine["event_source"] == "webhook"
    assert data["webhook_url"].endswith(routine["webhook_token"])
    assert data["webhook_secret"]
    assert data["subscription"]["residual_filters"]["field"] == "payload.labels"

    unsigned = test_client.post(
        data["webhook_url"],
        json={
            "event_type": "issues.opened",
            "provider_event_id": "issue-1",
            "labels": ["bug"],
        },
    )
    assert unsigned.status_code == 400
    assert unsigned.json()["detail"] == "Missing or invalid Bearer webhook key"

    bearer = test_client.post(
        data["webhook_url"],
        json={
            "event_type": "issues.opened",
            "provider_event_id": "issue-bearer-1",
            "labels": ["bug"],
        },
        headers={"Authorization": f"Bearer {data['webhook_secret']}"},
    )
    assert bearer.status_code == 200
    assert bearer.json()["status"] == "accepted"

    invalid_bearer = test_client.post(
        data["webhook_url"],
        json={"event_type": "issues.opened", "provider_event_id": "issue-bearer-2"},
        headers={"Authorization": "Bearer incorrect-key"},
    )
    assert invalid_bearer.status_code == 400

    import hashlib
    import hmac
    import json
    from datetime import datetime, timezone

    payload = {
        "event_type": "issues.opened",
        "provider_event_id": "issue-1",
        "labels": ["bug"],
    }
    raw_body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    signature = hmac.new(
        data["webhook_secret"].encode(),
        f"{timestamp}.{raw_body}".encode(),
        hashlib.sha256,
    ).hexdigest()
    signed = test_client.post(
        data["webhook_url"],
        json=payload,
        headers={"x-signature": signature, "x-timestamp": timestamp},
    )
    assert signed.status_code == 200
    assert signed.json()["status"] == "accepted"

    listed = test_client.get("/api/routines", headers=headers).json()["routines"]
    listed_routine = next(item for item in listed if item["id"] == routine["id"])
    assert "webhook_secret" not in listed_routine


def test_event_prompt_compiles_and_creates_custom_webhook(
    test_client, auth_headers, monkeypatch
):
    headers, _ = auth_headers

    class FakeModel:
        async def ainvoke(self, _messages):
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"needs_clarification":false,"name":"Webhook triage",'
                        '"instruction":"Summarize the incoming alert.",'
                        '"source":"webhook","event_type":"alert.created",'
                        '"when_to_run":"When a new alert arrives"}'
                    )
                },
            )()

    class FakeProviderManager:
        def get_provider_by_index(self, *_args, **_kwargs):
            return FakeModel(), None, 1

    monkeypatch.setattr(server, "provider_manager", FakeProviderManager())
    monkeypatch.setattr(
        server.chat_store,
        "get_settings",
        lambda **_kwargs: {"provider_index": 0, "model": "test-model"},
    )
    response = test_client.post(
        "/api/routines/events/from-prompt",
        headers=headers,
        json={"prompt": "When an alert is created, summarize it."},
    )

    assert response.status_code == 201
    result = response.json()
    assert result["routine"]["event_source"] == "webhook"
    assert result["routine"]["trigger_definition"]["event_type"] == "alert.created"
    assert result["webhook_url"].endswith(result["routine"]["webhook_token"])


def test_github_oauth_start_requires_config_and_uses_account_callback(
    test_client, auth_headers, monkeypatch, tmp_path
):
    configure_provider_store(monkeypatch, tmp_path)
    headers, _ = auth_headers
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("ORBIT_PUBLIC_URL", "https://orbit.example.test")

    response = test_client.post(
        "/api/integrations/github/oauth/start",
        headers=headers,
    )

    assert response.status_code == 200
    auth_url = response.json()["auth_url"]
    assert "scope=write%3Arepo_hook" in auth_url
    assert "redirect_uri=https%3A%2F%2Forbit.example.test%2Fapi%2Fintegrations%2Fgithub%2Foauth%2Fcallback" in auth_url


def test_provider_oauth_rejects_an_unsafe_public_origin(
    test_client, auth_headers, monkeypatch, tmp_path
):
    configure_provider_store(monkeypatch, tmp_path)
    headers, _ = auth_headers
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("ORBIT_PUBLIC_URL", "https://user:pass@orbit.example.test/callback")
    pending_states = set(server.PENDING_EVENT_OAUTH_STATES)

    response = test_client.post("/api/integrations/github/oauth/start", headers=headers)

    assert response.status_code == 503
    assert set(server.PENDING_EVENT_OAUTH_STATES) == pending_states


def test_provider_webhook_body_limit_fails_closed_and_rejects_oversize(
    test_client, monkeypatch
):
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing-secret")
    monkeypatch.delenv("ORBIT_PROVIDER_WEBHOOK_MAX_BYTES", raising=False)

    unconfigured = test_client.post(
        "/api/integrations/slack/events",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert unconfigured.status_code == 503

    monkeypatch.setenv("ORBIT_PROVIDER_WEBHOOK_MAX_BYTES", "1")
    oversized = test_client.post(
        "/api/integrations/slack/events",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert oversized.status_code == 413


def test_github_routine_rejects_invalid_repository_before_provider_lookup(
    test_client, auth_headers, monkeypatch, tmp_path
):
    configure_provider_store(monkeypatch, tmp_path)
    headers, _ = auth_headers
    monkeypatch.setenv("ORBIT_PUBLIC_URL", "https://orbit.example.test")

    response = test_client.post(
        "/api/routines/events",
        headers=headers,
        json={
            "name": "Invalid repo",
            "instruction": "Do nothing.",
            "source": "github",
            "event_type": "issues.opened",
            "when_to_run": "When an issue opens",
            "repository": "owner/../target",
        },
    )

    assert response.status_code == 400
    assert "valid GitHub owner/repository" in response.json()["detail"]


def test_event_routine_rejects_unbounded_filter_tree(
    test_client, auth_headers, monkeypatch, tmp_path
):
    configure_provider_store(monkeypatch, tmp_path)
    headers, _ = auth_headers

    response = test_client.post(
        "/api/routines/events",
        headers=headers,
        json={
            "name": "Unsafe filter",
            "instruction": "Do nothing.",
            "source": "webhook",
            "event_type": "alert.created",
            "when_to_run": "When an alert arrives",
            "filters": {
                "operator": "AND",
                "conditions": [
                    {
                        "field": "payload.text",
                        "operator": "matches",
                        "value": "x" * 501,
                    }
                ],
            },
        },
    )

    assert response.status_code == 400
    assert "filter value" in response.json()["detail"].lower()


def test_github_webhook_is_registered_verified_and_filtered(
    test_client, auth_headers, monkeypatch, tmp_path
):
    import hashlib
    import hmac
    import json

    headers, user_id = auth_headers
    store = configure_provider_store(monkeypatch, tmp_path)
    monkeypatch.setenv("ORBIT_PUBLIC_URL", "https://orbit.example.test")
    account = store.save_account(
        user_id=user_id,
        provider="github",
        provider_account_id="github-user",
        account_name="octocat",
        access_token="private-github-token",
    )
    assert account

    async def fake_create_webhook(**kwargs):
        assert kwargs["repository"] == "octo/repo"
        assert kwargs["provider_event"] == "issues"
        return 837

    async def fake_delete_webhook(**_kwargs):
        return True

    monkeypatch.setattr(server, "create_github_webhook", fake_create_webhook)
    monkeypatch.setattr(server, "delete_github_webhook", fake_delete_webhook)
    created = test_client.post(
        "/api/routines/events",
        headers=headers,
        json={
            "name": "Bug triage",
            "instruction": "Summarize the issue.",
            "source": "github",
            "event_type": "issues.opened",
            "when_to_run": "When a bug issue opens in octo/repo",
            "repository": "octo/repo",
            "filters": {
                "field": "payload.labels",
                "operator": "contains",
                "value": "bug",
            },
        },
    )
    assert created.status_code == 201, created.text
    body = {
        "action": "opened",
        "repository": {"full_name": "octo/repo"},
        "issue": {"number": 7, "labels": [{"name": "bug"}]},
    }
    raw_body = json.dumps(body).encode()
    secret = created.json()["routine"]["webhook_secret"]
    signature = "sha256=" + hmac.new(
        secret.encode(), raw_body, hashlib.sha256
    ).hexdigest()
    callback = created.json()["provider_webhook_url"]
    response = test_client.post(
        callback.replace("https://orbit.example.test", ""),
        content=raw_body,
        headers={
            "content-type": "application/json",
            "x-hub-signature-256": signature,
            "x-github-event": "issues",
            "x-github-delivery": "delivery-837",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "accepted"
    deleted = test_client.delete(
        f"/api/routines/{created.json()['routine']['id']}",
        headers=headers,
    )
    assert deleted.status_code == 200
    store.delete_account(user_id, "github")


def test_slack_event_callback_verifies_signature_and_queues_matching_event(
    test_client, auth_headers, monkeypatch, tmp_path
):
    import hashlib
    import hmac
    import json
    import time

    headers, user_id = auth_headers
    store = configure_provider_store(monkeypatch, tmp_path)
    monkeypatch.setenv("ORBIT_PUBLIC_URL", "https://orbit.example.test")
    signing_secret = "slack-signing-secret"
    monkeypatch.setenv("SLACK_SIGNING_SECRET", signing_secret)
    store.save_account(
        user_id=user_id,
        provider="slack",
        provider_account_id="T4321",
        account_name="Orbit workspace",
        access_token="private-slack-token",
    )

    async def fake_resolve_channel(_access_token, _channel):
        return "C123"

    monkeypatch.setattr(server, "resolve_slack_channel_id", fake_resolve_channel)
    created = test_client.post(
        "/api/routines/events",
        headers=headers,
        json={
            "name": "Support triage",
            "instruction": "Summarize the request here.",
            "source": "slack",
            "event_type": "message.channels",
            "when_to_run": "When a message in #support contains needs repro",
            "channel_name": "support",
            "filters": {
                "field": "payload.text",
                "operator": "contains",
                "value": "needs repro",
            },
        },
    )
    assert created.status_code == 201, created.text

    envelope = {
        "type": "event_callback",
        "team_id": "T4321",
        "event_id": "Ev123",
        "event": {
            "type": "message",
            "channel": "C123",
            "text": "This needs repro",
            "user": "U456",
            "ts": "123.456",
        },
    }
    body = json.dumps(envelope).encode()
    timestamp = str(int(time.time()))
    base = f"v0:{timestamp}:".encode() + body
    signature = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    response = test_client.post(
        "/api/integrations/slack/events",
        content=body,
        headers={
            "content-type": "application/json",
            "x-slack-request-timestamp": timestamp,
            "x-slack-signature": signature,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "accepted"
    server.proactive_store.delete_routine(created.json()["routine"]["id"])
    store.delete_account(user_id, "slack")


def test_gmail_oauth_start_requests_read_only_offline_access(
    test_client, auth_headers, monkeypatch, tmp_path
):
    configure_provider_store(monkeypatch, tmp_path)
    headers, _ = auth_headers
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "google-client")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "google-secret")
    monkeypatch.setenv("ORBIT_PUBLIC_URL", "https://orbit.example.test")

    response = test_client.post("/api/integrations/gmail/oauth/start", headers=headers)

    assert response.status_code == 200
    auth_url = response.json()["auth_url"]
    assert "gmail.readonly" in auth_url
    assert "access_type=offline" in auth_url
    assert "redirect_uri=https%3A%2F%2Forbit.example.test%2Fapi%2Fintegrations%2Fgmail%2Foauth%2Fcallback" in auth_url


def test_gmail_pubsub_requires_authenticated_delivery(test_client, monkeypatch):
    monkeypatch.delenv("GMAIL_PUBSUB_AUDIENCE", raising=False)
    monkeypatch.delenv("GMAIL_PUBSUB_PUSH_EMAIL", raising=False)
    response = test_client.post(
        "/api/integrations/gmail/events",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 401


def test_gmail_watch_renewal_keeps_history_cursor(monkeypatch, tmp_path):
    import asyncio

    store = ProviderIntegrationStore(
        server.database,
        encryption_key=Fernet.generate_key().decode("ascii"),
    )
    monkeypatch.setattr(server, "provider_integration_store", store)
    account = store.save_account(
        user_id="gmail-renewal-user",
        provider="gmail",
        provider_account_id="renew@example.com",
        account_name="renew@example.com",
        access_token="gmail-access-token",
    )
    store.save_gmail_watch_state(
        user_id="gmail-renewal-user",
        email_address="renew@example.com",
        history_id="12345678901234567890",
        expiration="2025-01-01T00:00:00+00:00",
    )

    async def fake_watch_request(_account):
        return {
            "history_id": "22345678901234567890",
            "expiration": "2025-01-08T00:00:00+00:00",
        }

    monkeypatch.setattr(server, "_gmail_watch_request", fake_watch_request)
    asyncio.run(server._gmail_renew_watch(account))

    state = store.get_gmail_watch_state("gmail-renewal-user", "renew@example.com")
    assert state["history_id"] == "12345678901234567890"
    assert state["expiration"] == "2025-01-08T00:00:00+00:00"


def test_gmail_pubsub_processes_new_messages_and_advances_cursor(
    test_client, monkeypatch, tmp_path
):
    import base64
    import json

    store = configure_provider_store(monkeypatch, tmp_path)
    monkeypatch.setenv("ORBIT_GMAIL_HISTORY_PAGE_LIMIT", "3")
    monkeypatch.setenv("ORBIT_GMAIL_MESSAGE_LIMIT", "5")
    async def verify_request(_request):
        return None

    monkeypatch.setattr(server, "_verify_gmail_pubsub_request", verify_request)
    account = store.save_account(
        user_id="gmail-test-user",
        provider="gmail",
        provider_account_id="user@example.com",
        account_name="user@example.com",
        access_token="gmail-access-token",
        refresh_token="gmail-refresh-token",
        metadata={"expires_at": 4_000_000_000},
    )
    store.save_gmail_watch_state(
        user_id="gmail-test-user",
        email_address="user@example.com",
        history_id="100",
        expiration="2099-01-01T00:00:00+00:00",
    )

    async def fake_access_token(value):
        return value

    async def fake_handle_event(event):
        received_events.append(event)

    class FakeResponse:
        status_code = 200

        def __init__(self, value):
            self.value = value

        def json(self):
            return self.value

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, **_kwargs):
            if url.endswith("/history"):
                return FakeResponse(
                    {
                        "history": [
                            {"messagesAdded": [{"message": {"id": "message-1"}}]}
                        ]
                    }
                )
            return FakeResponse(
                {
                    "id": "message-1",
                    "threadId": "thread-1",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "sender@example.com"},
                            {"name": "Subject", "value": "Hello"},
                        ]
                    },
                }
            )

    received_events = []
    monkeypatch.setattr(server, "_gmail_access_token", fake_access_token)
    monkeypatch.setattr(server.proactive_engine, "handle_incoming_event", fake_handle_event)
    monkeypatch.setattr(server.httpx, "AsyncClient", FakeAsyncClient)
    notification = {
        "emailAddress": "user@example.com",
        "historyId": "102",
    }
    encoded = base64.urlsafe_b64encode(json.dumps(notification).encode()).decode().rstrip("=")

    response = test_client.post(
        "/api/integrations/gmail/events",
        json={"message": {"data": encoded}},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"status": "accepted", "messages_processed": 1}
    assert received_events[0]["event_id"] == "gmail_user@example.com_message-1"
    assert (
        store.get_gmail_watch_state("gmail-test-user", "user@example.com")["history_id"]
        == "102"
    )


def test_gmail_message_limit_rejects_batch_without_advancing_cursor(
    test_client, monkeypatch, tmp_path
):
    import base64
    import json

    store = configure_provider_store(monkeypatch, tmp_path)
    monkeypatch.setenv("ORBIT_GMAIL_HISTORY_PAGE_LIMIT", "3")
    monkeypatch.setenv("ORBIT_GMAIL_MESSAGE_LIMIT", "1")
    async def verify_request(_request):
        return None

    monkeypatch.setattr(server, "_verify_gmail_pubsub_request", verify_request)
    account = store.save_account(
        user_id="gmail-limit-user",
        provider="gmail",
        provider_account_id="limit@example.com",
        account_name="limit@example.com",
        access_token="gmail-access-token",
        refresh_token="gmail-refresh-token",
        metadata={"expires_at": 4_000_000_000},
    )
    assert account
    store.save_gmail_watch_state(
        user_id="gmail-limit-user",
        email_address="limit@example.com",
        history_id="100",
        expiration="2099-01-01T00:00:00+00:00",
    )

    async def fake_access_token(value):
        return value

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "message-1"}},
                            {"message": {"id": "message-2"}},
                        ]
                    }
                ]
            }

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(server, "_gmail_access_token", fake_access_token)
    monkeypatch.setattr(server.httpx, "AsyncClient", FakeAsyncClient)
    notification = {"emailAddress": "limit@example.com", "historyId": "102"}
    encoded = base64.urlsafe_b64encode(json.dumps(notification).encode()).decode().rstrip("=")

    response = test_client.post(
        "/api/integrations/gmail/events",
        json={"message": {"data": encoded}},
    )

    assert response.status_code == 503
    assert "cursor was not advanced" in response.json()["detail"]
    assert (
        store.get_gmail_watch_state("gmail-limit-user", "limit@example.com")["history_id"]
        == "100"
    )

def test_custom_webhook_credentials_are_protected_at_rest(test_client, auth_headers):
    headers, user_id = auth_headers
    webhook_token = "orbit-test-webhook-bearer-token"
    webhook_secret = "orbit-test-webhook-signing-secret"
    routine = server.proactive_store.create_routine(
        user_id=user_id,
        name="Encrypted webhook",
        action_prompt="Summarize the incoming event.",
        trigger_type="webhook",
        webhook_token=webhook_token,
        webhook_secret=webhook_secret,
    )

    with server.database.connection() as connection:
        row = connection.execute(
            "SELECT webhook_token, webhook_secret FROM proactive_routines WHERE id = %s",
            (routine["id"],),
        ).fetchone()
    assert row["webhook_token"] != webhook_token
    assert webhook_token not in row["webhook_token"]
    assert webhook_secret not in row["webhook_secret"]
    assert row["webhook_secret"].startswith("enc:v1:")

    found = server.proactive_store.get_routine_by_webhook(webhook_token)
    assert found is not None
    assert found["webhook_secret"] == webhook_secret

    listed = server.proactive_store.list_routines(user_id=user_id)
    listed_routine = next(item for item in listed if item["id"] == routine["id"])
    assert "webhook_token" not in listed_routine
    assert "webhook_secret" not in listed_routine

    accepted = test_client.post(
        f"/api/webhooks/{webhook_token}",
        json={"event": "test"},
        headers={"Authorization": f"Bearer {webhook_secret}"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "accepted"


def test_legacy_webhook_credentials_are_migrated(postgres_database):
    from credential_vault import CredentialVault

    store = ProactiveStore(postgres_database)
    routine = store.create_routine(
        user_id="legacy-webhook-user",
        name="Legacy webhook",
        action_prompt="Process alert",
        trigger_type="webhook",
        webhook_token="legacy-webhook-token",
        webhook_secret="legacy-webhook-secret",
    )

    with postgres_database.connection() as connection:
        connection.execute(
            """
            UPDATE proactive_routines
            SET webhook_token = %s, webhook_secret = %s
            WHERE id = %s
            """,
            ("legacy-raw-token", "legacy-raw-secret", routine["id"]),
        )

    migrated = store.migrate_webhook_credentials()
    assert migrated["webhook_tokens_hashed"] >= 1
    assert migrated["webhook_secrets_encrypted"] >= 1
    with postgres_database.connection() as connection:
        row = connection.execute(
            "SELECT webhook_token, webhook_secret FROM proactive_routines WHERE id = %s",
            (routine["id"],),
        ).fetchone()
    assert row["webhook_token"] != "legacy-raw-token"
    assert row["webhook_secret"] != "legacy-raw-secret"
    assert CredentialVault().decrypt_text(row["webhook_secret"]) == "legacy-raw-secret"
    assert store.get_routine_by_webhook("legacy-raw-token")["webhook_secret"] == "legacy-raw-secret"

