from datetime import datetime, timedelta, timezone

from tools import manage_routine, WORKSPACE_TOOLS
from proactive_engine import active_proactive_store_var, ProactiveStore
from memory_store import active_user_id_var
from test_database import IsolatedPostgresDatabase


def test_manage_routine_structured_schedule_and_lifecycle():
    with IsolatedPostgresDatabase() as database:
        store = ProactiveStore(database)
        active_proactive_store_var.set(store)
        active_user_id_var.set("user_manage_tool_test")

        target = (datetime.now(timezone.utc) + timedelta(minutes=10)).replace(microsecond=0)
        res_remind = manage_routine.invoke({
            "action": "schedule",
            "name": "Water Break",
            "action_prompt": "Drink water and stretch",
            "run_at": target.isoformat(),
        })
        assert "Reminder scheduled: 'Water Break'" in res_remind
        reminder = store.list_routines("user_manage_tool_test")[0]
        assert reminder["one_shot"] is True
        assert datetime.fromisoformat(reminder["next_run_at"]) == target

        res_routine = manage_routine.invoke({
            "action": "create",
            "name": "Daily Sync",
            "action_prompt": "Summarize today's goals",
            "cron_expression": "0 9 * * 1-5",
        })
        assert "Recurring routine created: 'Daily Sync'" in res_routine

        res_list = manage_routine.invoke({"action": "list"})
        assert "Found 2 scheduled" in res_list
        assert "Water Break" in res_list
        assert "Daily Sync" in res_list

        assert "Successfully paused" in manage_routine.invoke({
            "action": "pause", "routine_id": "Daily Sync"
        })
        assert "Successfully resumed" in manage_routine.invoke({
            "action": "resume", "routine_id": "Daily Sync"
        })
        assert "Test run enqueued" in manage_routine.invoke({
            "action": "test", "routine_id": "Daily Sync"
        })
        assert "Successfully deleted" in manage_routine.invoke({
            "action": "delete", "routine_id": "Water Break"
        })
        assert "Successfully deleted all 1" in manage_routine.invoke({
            "action": "delete", "routine_id": "all"
        })


def test_manage_routine_rejects_unstructured_or_invalid_schedule():
    with IsolatedPostgresDatabase() as database:
        active_proactive_store_var.set(ProactiveStore(database))
        active_user_id_var.set("user_manage_tool_validation")

        assert "include a timezone" in manage_routine.invoke({
            "action": "schedule",
            "name": "Reminder",
            "action_prompt": "Do something",
            "run_at": "2026-11-18T09:00:00",
        })
        assert "valid five-field cron" in manage_routine.invoke({
            "action": "create",
            "name": "Routine",
            "action_prompt": "Do something",
            "cron_expression": "tomorrow at 9am",
        })
        assert "'all'" in manage_routine.invoke({"action": "delete"})
        assert [tool.name for tool in WORKSPACE_TOOLS].count("manage_routine") == 1
        assert not {
            "schedule_reminder", "create_routine", "list_routines", "pause_routine",
            "resume_routine", "delete_routine", "test_routine",
        }.intersection(tool.name for tool in WORKSPACE_TOOLS)


def test_manage_routine_past_run_at_includes_current_time():
    with IsolatedPostgresDatabase() as database:
        active_proactive_store_var.set(ProactiveStore(database))
        active_user_id_var.set("user_manage_tool_past")

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
        past_err = manage_routine.invoke({
            "action": "schedule",
            "name": "Past Reminder",
            "action_prompt": "Stretch",
            "run_at": past.isoformat(),
        })
        assert "must be in the future" in past_err
        assert "current time is " in past_err
        assert "UTC:" in past_err


def test_dynamic_system_prompt_injects_local_iso_and_replaces_placeholder():
    from agent import get_dynamic_system_prompt, SYSTEM_PROMPT

    rendered = get_dynamic_system_prompt(SYSTEM_PROMPT)
    assert "{current_time}" not in rendered
    assert "Local ISO-8601" in rendered
    assert "UTC ISO-8601" in rendered
