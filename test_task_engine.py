import unittest
import asyncio
from uuid import uuid4
from typing import Dict, Any

from test_database import IsolatedPostgresDatabase
from task_engine import (
    TaskStore,
    TaskStatus,
    TaskEventType,
    DurableExecutionEngine,
)
from chat_store import PostgresChatStore
from test_database import IsolatedPostgresDatabase


class DummyAgent:
    """Mock agent for unit testing the task execution engine."""

    def __init__(self):
        self.should_fail = False
        self.should_interrupt = False

    async def astream_run(
        self,
        user_query: str,
        chat_history=None,
        target_provider_index=None,
        target_model=None,
        target_user_id=None,
        chat_store=None,
        active_interrupts=None,
    ):
        yield {
            "type": "provider_selected",
            "data": {"provider": "MockProvider", "model": "mock-model"},
        }

        if self.should_interrupt and active_interrupts is not None:
            req_id = "req-test-1"
            fut = asyncio.get_running_loop().create_future()
            active_interrupts[req_id] = fut
            yield {
                "type": "interrupt",
                "data": {
                    "request_id": req_id,
                    "interrupt_id": req_id,
                    "questions": [{"question": "Proceed?", "options": ["Yes", "No"]}],
                },
            }
            # Wait for resume
            decision = await fut
            yield {
                "type": "token",
                "data": {"token": f"Resumed with {decision.get('action')}. "},
            }

        yield {
            "type": "tool_start",
            "data": {"tool": "calculate", "args": {"expr": "2+2"}, "tool_call_id": "call-1"},
        }
        await asyncio.sleep(0.01)
        yield {
            "type": "tool_end",
            "data": {"tool": "calculate", "output": "4", "tool_call_id": "call-1"},
        }

        if self.should_fail:
            yield {"type": "error", "data": {"error": "Simulated task failure"}}
            return

        yield {"type": "token", "data": {"token": "Result is 4."}}
        full_content = "Result is 4."
        if self.should_interrupt:
            full_content = f"Resumed with {decision.get('action')}. Result is 4."
        yield {
            "type": "done",
            "data": {
                "content": full_content,
                "provider": {"provider": "MockProvider"},
            },
        }


class TestTaskEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_database = IsolatedPostgresDatabase()
        self.store = TaskStore(self.test_database.database)
        self.dummy_agent = DummyAgent()
        self.engine = DurableExecutionEngine(
            task_store=self.store,
            agent=self.dummy_agent,
        )

    def tearDown(self):
        self.test_database.close()

    def test_task_crud_lifecycle(self):
        task = self.store.create_task(
            user_id="user_123",
            input_data={"message": "Research customers"},
            organization_id="org_1",
            task_type="research",
        )
        self.assertEqual(task["status"], TaskStatus.QUEUED)
        self.assertEqual(task["user_id"], "user_123")
        self.assertEqual(task["task_type"], "research")

        checked_out = self.store.checkout_next_queued_task()
        self.assertIsNotNone(checked_out)
        self.assertEqual(checked_out["id"], task["id"])
        self.assertEqual(checked_out["status"], TaskStatus.RUNNING)

        # Nothing left in queue
        self.assertIsNone(self.store.checkout_next_queued_task())

        # Update to completed
        self.store.update_task_status(
            task["id"],
            status=TaskStatus.COMPLETED,
            result={"output": "Done"},
        )
        completed = self.store.get_task(task["id"])
        self.assertEqual(completed["status"], TaskStatus.COMPLETED)
        self.assertEqual(completed["result"]["output"], "Done")

    def test_events_recording(self):
        task = self.store.create_task("user_123", {"message": "Test"})
        evt = self.store.append_event(
            task["id"],
            event_type=TaskEventType.TOOL_STARTED,
            payload={"tool": "calculate"},
        )
        self.assertEqual(evt["event_type"], TaskEventType.TOOL_STARTED)

        events = self.store.get_task_events(task["id"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], TaskEventType.TOOL_STARTED)
        self.assertEqual(events[0]["payload"]["tool"], "calculate")

    def test_crash_recovery(self):
        # Simulate an orphaned task that was RUNNING when process crashed
        task = self.store.create_task("user_123", {"message": "Crashed run"})
        self.store.update_task_status(task["id"], status=TaskStatus.RUNNING)

        # New engine / server startup detects and requeues it
        recovered = self.store.recover_orphaned_tasks()
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["id"], task["id"])

        recovered_task = self.store.get_task(task["id"])
        self.assertEqual(recovered_task["status"], TaskStatus.QUEUED)
        self.assertEqual(recovered_task["retry_count"], 1)

    async def test_worker_end_to_end_execution(self):
        self.engine.start()
        try:
            task = self.engine.enqueue_task(
                user_id="user_exec",
                input_data={"message": "Calculate 2+2"},
            )

            # Wait for task to complete
            for _ in range(50):
                await asyncio.sleep(0.05)
                cur = self.store.get_task(task["id"])
                if cur["status"] in TaskStatus.TERMINAL:
                    break

            final_task = self.store.get_task(task["id"])
            self.assertEqual(final_task["status"], TaskStatus.COMPLETED)
            self.assertIn("Result is 4", final_task["result"]["content"])

            # Check that events were recorded in task_events
            events = self.store.get_task_events(task["id"])
            event_types = [e["event_type"] for e in events]
            self.assertIn(TaskEventType.TASK_STARTED, event_types)
            self.assertIn(TaskEventType.TOOL_STARTED, event_types)
            self.assertIn(TaskEventType.TOOL_COMPLETED, event_types)
            self.assertIn(TaskEventType.TASK_COMPLETED, event_types)

        finally:
            await self.engine.stop()

    async def test_worker_human_in_the_loop_approval(self):
        self.dummy_agent.should_interrupt = True
        self.engine.start()
        try:
            task = self.engine.enqueue_task(
                user_id="user_approval",
                input_data={"message": "Needs approval"},
            )

            # Wait for task to enter waiting_approval
            for _ in range(50):
                await asyncio.sleep(0.05)
                cur = self.store.get_task(task["id"])
                if cur["status"] == TaskStatus.WAITING_APPROVAL:
                    break

            waiting_task = self.store.get_task(task["id"])
            self.assertEqual(waiting_task["status"], TaskStatus.WAITING_APPROVAL)

            # Resume with approval
            resumed = self.engine.resume_task(task["id"], {"action": "approve"})
            self.assertTrue(resumed)

            # Wait for task to finish
            for _ in range(50):
                await asyncio.sleep(0.05)
                cur = self.store.get_task(task["id"])
                if cur["status"] in TaskStatus.TERMINAL:
                    break

            completed_task = self.store.get_task(task["id"])
            self.assertEqual(completed_task["status"], TaskStatus.COMPLETED)
            self.assertIn("Resumed with approve", completed_task["result"]["content"])

        finally:
            await self.engine.stop()

    async def test_cancellation(self):
        task = self.store.create_task("user_cancel", {"message": "To cancel"})
        success = self.engine.cancel_task(task["id"])
        self.assertTrue(success)

        cur = self.store.get_task(task["id"])
        self.assertEqual(cur["status"], TaskStatus.CANCELLED)


if __name__ == "__main__":
    unittest.main()



from httpx import AsyncClient, ASGITransport
import server


class TestTaskServerEndpoints(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_database = IsolatedPostgresDatabase()
        self.original_store = server.chat_store
        self.original_task_store = server.task_store
        self.original_task_engine = server.task_engine
        server.chat_store = PostgresChatStore(self.test_database.database)
        server.task_store = TaskStore(self.test_database.database)
        server.task_engine = DurableExecutionEngine(
            task_store=server.task_store,
            agent=DummyAgent(),
        )
        self.transport = ASGITransport(app=server.app)
        self.client = AsyncClient(transport=self.transport, base_url="http://test")
        # Generate test user & token
        from auth import generate_session_token, hash_token
        user_record, _ = server.chat_store.create_user(
            email=f"taskuser_{uuid4().hex[:8]}@example.com",
            display_name="Task User",
            password_hash="hashed_pw",
        )
        self.user_id = user_record["user_id"]
        self.token = generate_session_token()
        server.chat_store.create_session(
            user_id=self.user_id,
            token_hash=hash_token(self.token),
            expires_at="2099-01-01T00:00:00Z",
        )
        self.headers = {"Authorization": f"Bearer {self.token}"}

    async def asyncTearDown(self):
        await self.client.aclose()
        server.chat_store = self.original_store
        server.task_store = self.original_task_store
        server.task_engine = self.original_task_engine
        self.test_database.close()

    async def test_create_and_get_task_endpoints(self):
        resp = await self.client.post(
            "/api/tasks",
            json={"message": "Perform research on 100 targets"},
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 202)
        task_data = resp.json()
        self.assertEqual(task_data["status"], "queued")
        task_id = task_data["id"]

        # Get task details
        get_resp = await self.client.get(f"/api/tasks/{task_id}", headers=self.headers)
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["id"], task_id)

        # List tasks
        list_resp = await self.client.get("/api/tasks", headers=self.headers)
        self.assertEqual(list_resp.status_code, 200)
        tasks = list_resp.json()
        self.assertTrue(any(t["id"] == task_id for t in tasks))

        # Cancel task
        cancel_resp = await self.client.post(
            f"/api/tasks/{task_id}/cancel", headers=self.headers
        )
        self.assertEqual(cancel_resp.status_code, 200)
        self.assertEqual(cancel_resp.json()["status"], "cancelled")
