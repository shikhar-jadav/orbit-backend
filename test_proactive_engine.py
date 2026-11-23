import asyncio
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import pytest

from test_database import IsolatedPostgresDatabase

from proactive_engine import (
    ProactiveStore,
    ProactiveEngine,
    parse_schedule_time,
    format_display_time,
    format_short_local_time,
)


def test_parse_schedule_relative_minutes():
    res = parse_schedule_time("in 15 minutes")
    assert res["one_shot"] is True
    assert "In 15 minute(s)" in res["human_readable"]
    assert res["target_time"] is not None


def test_parse_schedule_relative_hours():
    res = parse_schedule_time("in 2 hours")
    assert res["one_shot"] is True
    assert "In 2 hour(s)" in res["human_readable"]


def test_parse_schedule_cron():
    res = parse_schedule_time("0 9 * * 1-5")
    assert res["one_shot"] is False
    assert res["expression"] == "0 9 * * 1-5"


def test_parse_schedule_every_minutes():
    res = parse_schedule_time("every 5 minutes")
    assert res["one_shot"] is False
    assert res["expression"] == "*/5 * * * *"
    assert res["interval_seconds"] == 5 * 60


def test_parse_schedule_every_days_uses_elapsed_interval():
    res = parse_schedule_time("every 2 days")

    assert res["one_shot"] is False
    assert res["human_readable"] == "Every 2 days"
    assert res["interval_seconds"] == 2 * 24 * 60 * 60


def test_parse_schedule_cron_uses_requested_timezone():
    res = parse_schedule_time("30 9 * * *", "Asia/Kolkata")
    next_run = datetime.fromisoformat(res["next_run_iso"])

    assert next_run.astimezone(ZoneInfo("Asia/Kolkata")).hour == 9
    assert next_run.astimezone(ZoneInfo("Asia/Kolkata")).minute == 30


def test_store_crud():
    with IsolatedPostgresDatabase() as database:
        store = ProactiveStore(database)

        # 1. Create Routine
        r = store.create_routine(
            user_id="user_123",
            name="Water Reminder",
            action_prompt="Drink a glass of water",
            trigger_type="timer",
            schedule_expression="2026-11-20T10:00:00Z",
            one_shot=True,
            next_run_at="2026-11-20T10:00:00Z",
        )
        assert r["id"].startswith("rtn_")
        assert r["name"] == "Water Reminder"
        assert r["one_shot"] is True
        assert r["enabled"] is True

        # 2. Get & List
        fetched = store.get_routine(r["id"])
        assert fetched["name"] == "Water Reminder"

        routines = store.list_routines("user_123")
        assert len(routines) == 1

        # 3. Update
        store.update_routine(r["id"], enabled=False)
        updated = store.get_routine(r["id"])
        assert updated["enabled"] is False

        # 4. Due routines check
        past_time = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        store.create_routine(
            user_id="user_123",
            name="Due Routine",
            action_prompt="Do something",
            next_run_at=past_time,
            one_shot=True,
        )
        due = store.find_due_routines()
        assert len(due) == 1
        assert due[0]["name"] == "Due Routine"

        # 5. Notifications
        notif = store.create_notification(
            user_id="user_123",
            title="⏰ Water Reminder",
            message="Drink water now",
            routine_id=r["id"],
        )
        assert notif["id"].startswith("notif_")
        assert store.get_unread_count("user_123") == 1

        store.mark_notification_read(notif["id"])
        assert store.get_unread_count("user_123") == 0


@pytest.mark.asyncio
async def test_proactive_engine_execution():
    with IsolatedPostgresDatabase() as database:
        store = ProactiveStore(database)
        engine = ProactiveEngine(store=store)

        past_time = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        routine = store.create_routine(
            user_id="user_abc",
            name="Standup Alert",
            action_prompt="Time for standup",
            next_run_at=past_time,
            one_shot=True,
        )

        # Trigger isolated execution
        res = await engine.execute_routine_isolated(routine["id"], trigger_source="test")
        assert res["status"] == "completed"

        # Verify run logged
        runs = store.list_runs(routine["id"])
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"
        transcript = store.list_run_messages(runs[0]["id"])
        assert any(item["entry_type"] == "prompt" for item in transcript)
        assert any(item["entry_type"] == "assistant" for item in transcript)
        assert transcript[-1]["status"] == "completed"

        # Verify notification created in outbox
        notifs = store.list_notifications("user_abc")
        assert len(notifs) == 1
        assert "Standup Alert" in notifs[0]["title"]


@pytest.mark.asyncio
async def test_webhook_ingestion():
    import hmac
    import hashlib
    import json

    with IsolatedPostgresDatabase() as database:
        store = ProactiveStore(database)
        engine = ProactiveEngine(store=store)

        token = "whk_test_12345"
        secret = "secret_key_abc"
        routine = store.create_routine(
            user_id="user_whk",
            name="GitHub CI Watcher",
            action_prompt="Alert if build fails",
            trigger_type="webhook",
            webhook_token=token,
            webhook_secret=secret,
        )

        payload = {"event": "ci_failed", "repo": "orbit"}
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        raw_body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        sig = hmac.new(
            secret.encode("utf-8"),
            f"{timestamp}.{raw_body}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # 1. Valid signature
        res = await engine.handle_incoming_webhook(
            token=token, payload=payload, signature=sig, timestamp=timestamp
        )
        assert res["status"] == "accepted"

        # 2. Invalid signature
        res_bad = await engine.handle_incoming_webhook(
            token=token, payload=payload, signature="invalid_sig", timestamp=timestamp
        )
        assert res_bad["status"] == "error"
        assert "Invalid HMAC signature" in res_bad["message"]


@pytest.mark.asyncio
async def test_routine_type_execution_separation():
    with IsolatedPostgresDatabase() as database:
        store = ProactiveStore(database)

        class MockAgent:
            async def astream_run(self, user_query, **kwargs):
                yield {"type": "provider_selected", "data": {"provider": "Mock", "model": "mock-model"}}
                yield {"type": "tool_start", "data": {"tool": "search_repo", "args": {"query": "issue #42"}}}
                yield {"type": "tool_end", "data": {"tool": "search_repo", "output": "Found issue #42."}}
                yield {"type": "done", "data": {"content": "Autonomous task executed successfully."}}

        engine = ProactiveEngine(store=store, agent=MockAgent())

        # 1. Reminder: directly emits prompt without calling LLM
        reminder = store.create_routine(
            user_id="user_test",
            name="Water Reminder",
            action_prompt="Drink water and hydrate!",
            routine_type="reminder",
            one_shot=True,
        )
        res_reminder = await engine.execute_routine_isolated(reminder["id"], trigger_source="test")
        assert res_reminder["summary"] == "Drink water and hydrate!"

        # 2. Task: invokes agent
        task = store.create_routine(
            user_id="user_test",
            name="Daily Repo Digest",
            action_prompt="Check git commits",
            routine_type="task",
            one_shot=False,
        )
        res_task = await engine.execute_routine_isolated(task["id"], trigger_source="test")
        assert res_task["summary"] == "Autonomous task executed successfully."
        task_run = store.list_runs(task["id"])[0]
        task_transcript = store.list_run_messages(task_run["id"])
        assert any(item["entry_type"] == "tool" and item["status"] == "running" for item in task_transcript)
        assert any(item["entry_type"] == "tool" and item["status"] == "completed" for item in task_transcript)
        assert any(item["entry_type"] == "assistant" for item in task_transcript)

        # Secrets in tool details and common bearer/API-key strings must not be stored.
        secret_entry = store.append_run_message(
            run_id=task_run["id"],
            entry_type="tool",
            status="completed",
            title="Credential check",
            content="Authorization: Bearer very-secret-token sk-thisisademokeystring",
            details={"access_token": "sensitive-value", "safe_query": "issue #42"},
        )
        stored_run_messages = store.list_run_messages(task_run["id"])
        persisted_secret = next(item for item in stored_run_messages if item["id"] == secret_entry["id"])
        assert "very-secret-token" not in persisted_secret["content"]
        assert "sk-thisisademokeystring" not in persisted_secret["content"]
        assert persisted_secret["details"]["access_token"] == "[REDACTED]"
        assert persisted_secret["details"]["safe_query"] == "issue #42"


@pytest.mark.asyncio
async def test_event_filtering_prevents_agent_invocation():
    with IsolatedPostgresDatabase() as database:
        store = ProactiveStore(database)

        class CountingAgent:
            def __init__(self):
                self.calls = 0

            async def astream_run(self, user_query, **kwargs):
                self.calls += 1
                yield {"type": "done", "data": {"content": "should not run"}}

        agent = CountingAgent()
        engine = ProactiveEngine(store=store, agent=agent)
        routine = store.create_routine(
            user_id="event-user",
            name="John Mail",
            action_prompt="Process this email",
            trigger_type="event",
            trigger_definition={
                "source": "gmail",
                "event_type": "email.received",
                "filters": {
                    "field": "payload.sender.email",
                    "operator": "equals",
                    "value": "john@example.com",
                },
            },
            routine_type="task",
        )
        store.create_subscription(
            routine_id=routine["id"],
            user_id="event-user",
            provider="gmail",
            event_type="email.received",
        )

        rejected = await engine.handle_incoming_event({
            "event_id": "mail-1",
            "source": "gmail",
            "event_type": "email.received",
            "tenant_id": "event-user",
            "payload": {"sender": {"email": "other@example.com"}},
        })
        await asyncio.sleep(0)

        assert rejected["status"] == "filtered"
        assert agent.calls == 0
        assert store.list_runs(routine["id"]) == []

        cross_user = await engine.handle_incoming_event({
            "event_id": "mail-cross-user",
            "source": "gmail",
            "event_type": "email.received",
            "tenant_id": "another-user",
            "payload": {"sender": {"email": "john@example.com"}},
        })
        assert cross_user["status"] == "filtered"
        assert agent.calls == 0

        store.update_subscription(
            store.list_subscriptions(routine_id=routine["id"])[0]["id"],
            status="disabled",
        )
        disabled = await engine.handle_incoming_event({
            "event_id": "mail-disabled",
            "source": "gmail",
            "event_type": "email.received",
            "tenant_id": "event-user",
            "payload": {"sender": {"email": "john@example.com"}},
        })
        assert disabled["status"] == "filtered"
        assert agent.calls == 0


def test_processing_event_is_recovered_after_restart():
    with IsolatedPostgresDatabase() as database:
        store = ProactiveStore(database)
        event_id = store.enqueue_inbox_event(
            routine_id="routine-recovery",
            event_type="issues.opened",
            payload={"event_id": "event-recovery"},
        )
        claimed = store.claim_next_inbox_event()
        assert claimed["id"] == event_id
        assert claimed["status"] == "processing"

        restarted_store = ProactiveStore(database)
        assert restarted_store.recover_inbox_events() == 1
        recovered = restarted_store.claim_next_inbox_event()
        assert recovered["id"] == event_id
        assert recovered["attempt_count"] == 2


@pytest.mark.asyncio
async def test_matching_event_is_durable_deduplicated_and_reaches_agent():
    with IsolatedPostgresDatabase() as database:
        store = ProactiveStore(database)

        class CountingAgent:
            def __init__(self):
                self.queries = []

            async def astream_run(self, user_query, **kwargs):
                self.queries.append(user_query)
                yield {"type": "done", "data": {"content": "Issue inspected"}}

        agent = CountingAgent()
        engine = ProactiveEngine(store=store, agent=agent)
        routine = store.create_routine(
            user_id="event-user",
            name="Bug issue",
            action_prompt="Investigate the issue",
            trigger_type="event",
            trigger_definition={
                "source": "github",
                "event_type": "issues.opened",
                "filters": {
                    "field": "payload.labels",
                    "operator": "contains",
                    "value": "bug",
                },
            },
            routine_type="task",
        )
        store.create_subscription(
            routine_id=routine["id"],
            user_id="event-user",
            provider="github",
            event_type="issues.opened",
        )
        incoming = {
            "event_id": "github-event-42",
            "source": "github",
            "event_type": "issues.opened",
            "tenant_id": "event-user",
            "payload": {"labels": ["bug"], "title": "Crash on startup"},
            "metadata": {"provider_event_id": "github-event-42"},
        }

        accepted = await engine.handle_incoming_event(incoming)
        duplicate = await engine.handle_incoming_event(incoming)
        assert accepted["matched_routines"] == [routine["id"]]
        assert duplicate["status"] == "duplicate"
        assert len(store.list_inbox_events(routine["id"])) == 1
        assert agent.queries == []

        engine.start()
        try:
            for _ in range(100):
                if store.list_inbox_events(routine["id"])[0]["status"] == "completed":
                    break
                await asyncio.sleep(0.01)
        finally:
            await engine.stop()

        assert store.list_inbox_events(routine["id"])[0]["status"] == "completed"
        assert len(agent.queries) == 1
        assert "Crash on startup" in agent.queries[0]


@pytest.mark.asyncio
async def test_proactive_approval_requests_are_declined_without_interactive_user():
    with IsolatedPostgresDatabase() as database:
        store = ProactiveStore(database)

        class ApprovalAgent:
            async def astream_run(self, user_query, **kwargs):
                assert kwargs["force_approval"] is True
                future = asyncio.get_running_loop().create_future()
                kwargs["active_interrupts"]["approval-1"] = future
                yield {
                    "type": "interrupt",
                    "data": {"request_id": "approval-1", "is_approval": True},
                }
                decision = await future
                assert decision["action"] == "decline"
                yield {"type": "done", "data": {"content": "Approval-required action declined."}}

        engine = ProactiveEngine(store=store, agent=ApprovalAgent())
        routine = store.create_routine(
            user_id="approval-user",
            name="Approved work only",
            action_prompt="Perform a consequential action.",
            routine_type="task",
            require_approval=True,
        )

        result = await engine.execute_routine_isolated(
            routine["id"], trigger_source="test"
        )
        assert result["status"] == "completed"
        assert "declined" in result["summary"].lower()


def test_deterministic_filter_and_or_and_operators():
    from proactive_engine import DeterministicFilterEngine

    engine = DeterministicFilterEngine()
    event = {
        "payload": {
            "priority": "critical",
            "status": "open",
            "subject": "Production incident",
        }
    }
    condition = {
        "operator": "AND",
        "conditions": [
            {"field": "payload.priority", "operator": "equals", "value": "critical"},
            {"field": "payload.subject", "operator": "contains", "value": "incident"},
        ],
    }
    assert engine.evaluate(event, condition)
    assert not engine.evaluate(event, {
        "operator": "OR",
        "conditions": [
            {"field": "payload.priority", "operator": "equals", "value": "low"},
            {"field": "payload.status", "operator": "equals", "value": "closed"},
        ],
    })


def test_format_short_local_time():
    now_iso = datetime.now(timezone.utc).isoformat()
    formatted = format_short_local_time(now_iso)
    assert len(formatted) > 0
    assert ("AM" in formatted) or ("PM" in formatted)
