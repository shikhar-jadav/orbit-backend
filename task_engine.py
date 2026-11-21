"""
Durable Task Execution Engine for Persistent Agent Runs.

Implements decoupled execution architecture:
- Separate API lifecycle from Agent execution lifecycle
- Persistent task queue & task_events in PostgreSQL
- Background worker execution of LangGraph ReAct Graph with Checkpointers
- Real-time SSE event streaming with replay capability on reconnect
- Human-in-the-loop approvals / pause / resume
- Crash recovery: orphaned running tasks are detected and resumed from checkpoint on startup
"""

import asyncio
import json
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
from uuid import uuid4

from database import PostgresDatabase


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TaskStatus:
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    ALL = {QUEUED, RUNNING, WAITING_APPROVAL, PAUSED, COMPLETED, FAILED, CANCELLED}
    TERMINAL = {COMPLETED, FAILED, CANCELLED}


class TaskEventType:
    TASK_STARTED = "task.started"
    AGENT_THINKING = "agent.thinking"
    TOKEN = "token"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    ARTIFACT_CREATED = "artifact.created"
    A2UI_CREATED = "a2ui.created"
    APPROVAL_REQUIRED = "approval.required"
    TASK_PAUSED = "task.paused"
    TASK_RESUMED = "task.resumed"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"


class TaskStore:
    """Persists durable tasks and task events in PostgreSQL."""

    def __init__(self, database: PostgresDatabase):
        self.database = database

    @contextmanager
    def _connection(self):
        with self.database.connection() as conn:
            yield conn

    def create_task(
        self,
        user_id: str,
        input_data: Dict[str, Any],
        organization_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        task_type: str = "chat",
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        tid = task_id or str(uuid4())
        org_id = organization_id or user_id
        now = _utc_now()
        input_json = json.dumps(input_data, ensure_ascii=False, default=str)

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    id, organization_id, user_id, thread_id, task_type,
                    status, input, retry_count, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 0, %s)
                """,
                (tid, org_id, user_id, thread_id, task_type, TaskStatus.QUEUED, input_json, now),
            )

        return self.get_task(tid)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = %s", (task_id,)).fetchone()
            if not row:
                return None
            return self._row_to_task_dict(row)

    def list_tasks(
        self,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM tasks WHERE 1=1"
        params: List[Any] = []
        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)
        if thread_id:
            query += " AND thread_id = %s"
            params.append(thread_id)
        if status:
            query += " AND status = %s"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_task_dict(r) for r in rows]

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        increment_retry: bool = False,
    ) -> bool:
        updates = ["status = %s"]
        params: List[Any] = [status]

        if result is not None:
            updates.append("result = %s")
            params.append(json.dumps(result, ensure_ascii=False, default=str))
        if error is not None:
            updates.append("error = %s")
            params.append(json.dumps(error, ensure_ascii=False, default=str))
        if started_at:
            updates.append("started_at = %s")
            params.append(started_at)
        if completed_at:
            updates.append("completed_at = %s")
            params.append(completed_at)
        if increment_retry:
            updates.append("retry_count = retry_count + 1")

        params.append(task_id)
        with self._connection() as conn:
            cur = conn.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id = %s", params)
            return cur.rowcount > 0

    def checkout_next_queued_task(self) -> Optional[Dict[str, Any]]:
        """Atomically checkout the oldest queued task and mark it running."""
        now = _utc_now()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT id FROM tasks
                WHERE status = %s
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (TaskStatus.QUEUED,),
            ).fetchone()

            if not row:
                return None

            task_id = row["id"]
            cur = conn.execute(
                """
                UPDATE tasks
                SET status = %s, started_at = coalesce(started_at, %s)
                WHERE id = %s AND status = %s
                """,
                (TaskStatus.RUNNING, now, task_id, TaskStatus.QUEUED),
            )
            if cur.rowcount == 0:
                return None

            updated = conn.execute("SELECT * FROM tasks WHERE id = %s", (task_id,)).fetchone()
            return self._row_to_task_dict(updated)

    def recover_orphaned_tasks(self) -> List[Dict[str, Any]]:
        """Find tasks that were left in RUNNING status on server crash, requeue or mark for resume."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = %s", (TaskStatus.RUNNING,)
            ).fetchall()
            recovered = []
            for r in rows:
                tid = r["id"]
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = %s, retry_count = retry_count + 1
                    WHERE id = %s
                    """,
                    (TaskStatus.QUEUED, tid),
                )
                recovered.append(self._row_to_task_dict(r))
            return recovered

    def append_event(
        self,
        task_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        eid = event_id or str(uuid4())
        now = _utc_now()
        p_json = json.dumps(payload, ensure_ascii=False, default=str) if payload is not None else None

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO task_events (id, task_id, event_type, payload, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (eid, task_id, event_type, p_json, now),
            )

        return {
            "id": eid,
            "task_id": task_id,
            "event_type": event_type,
            "payload": payload,
            "created_at": now,
        }

    def get_task_events(
        self, task_id: str, after_seq: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM task_events
                WHERE task_id = %s
                ORDER BY created_at ASC, rowid ASC
                """,
                (task_id,),
            ).fetchall()

            events = []
            for r in rows:
                payload = json.loads(r["payload"]) if r["payload"] else None
                events.append({
                    "id": r["id"],
                    "task_id": r["task_id"],
                    "event_type": r["event_type"],
                    "payload": payload,
                    "created_at": r["created_at"],
                })
            return events

    def _row_to_task_dict(self, row: dict[str, Any]) -> Dict[str, Any]:
        input_data = json.loads(row["input"]) if row["input"] else {}
        result_data = json.loads(row["result"]) if row["result"] else None
        error_data = json.loads(row["error"]) if row["error"] else None

        return {
            "id": row["id"],
            "organization_id": row["organization_id"],
            "user_id": row["user_id"],
            "thread_id": row["thread_id"],
            "task_type": row["task_type"],
            "status": row["status"],
            "input": input_data,
            "result": result_data,
            "error": error_data,
            "retry_count": row["retry_count"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "created_at": row["created_at"],
        }


class TaskBroadcaster:
    """PubSub broker to push live task events to SSE listeners."""

    def __init__(self):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, task_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(task_id, set()).add(queue)
        return queue

    async def unsubscribe(self, task_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            if task_id in self._subscribers:
                self._subscribers[task_id].discard(queue)
                if not self._subscribers[task_id]:
                    del self._subscribers[task_id]

    async def broadcast(self, task_id: str, event: Dict[str, Any]) -> None:
        async with self._lock:
            queues = list(self._subscribers.get(task_id, []))
        for q in queues:
            await q.put(event)


class DurableExecutionEngine:
    """
    Coordinates persistent task queue, background worker execution,
    LangGraph checkpointer resumption, and live event broadcasting.
    """

    def __init__(
        self,
        task_store: TaskStore,
        agent: Any,
        chat_store: Optional[Any] = None,
        checkpointer: Optional[Any] = None,
    ):
        self.store = task_store
        self.agent = agent
        self.chat_store = chat_store
        self.checkpointer = checkpointer
        self.broadcaster = TaskBroadcaster()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._wake_event = asyncio.Event()

        # Active pause/resume futures for tasks in waiting_approval status
        self._pending_approvals: Dict[str, asyncio.Future] = {}
        self._cancelled_tasks: Set[str] = set()

    def start(self) -> None:
        """Start the background worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

        # Recover any orphaned tasks from a prior server crash
        recovered = self.store.recover_orphaned_tasks()
        if recovered:
            print(f"[TaskEngine] Recovered {len(recovered)} in-flight tasks from previous session.")
            self._wake_event.set()

    async def stop(self) -> None:
        """Gracefully stop the background worker."""
        self._running = False
        self._wake_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def enqueue_task(
        self,
        user_id: str,
        input_data: Dict[str, Any],
        organization_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        task_type: str = "chat",
    ) -> Dict[str, Any]:
        """Submit a task to the durable queue."""
        task = self.store.create_task(
            user_id=user_id,
            input_data=input_data,
            organization_id=organization_id,
            thread_id=thread_id,
            task_type=task_type,
        )
        self._wake_event.set()
        return task

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued or running task."""
        task = self.store.get_task(task_id)
        if not task:
            return False
        if task["status"] in TaskStatus.TERMINAL:
            return False

        self._cancelled_tasks.add(task_id)

        # Cancel any pending approval wait
        fut = self._pending_approvals.pop(task_id, None)
        if fut and not fut.done():
            fut.set_result({"action": "cancelled"})

        now = _utc_now()
        self.store.update_task_status(
            task_id,
            status=TaskStatus.CANCELLED,
            completed_at=now,
        )
        evt = self.store.append_event(
            task_id,
            event_type=TaskEventType.TASK_CANCELLED,
            payload={"reason": "Cancelled by user request."},
        )
        asyncio.create_task(self.broadcaster.broadcast(task_id, evt))
        return True

    def resume_task(self, task_id: str, decision: Dict[str, Any]) -> bool:
        """Resume a task waiting for approval with the user's decision."""
        task = self.store.get_task(task_id)
        if not task or task["status"] != TaskStatus.WAITING_APPROVAL:
            return False

        fut = self._pending_approvals.get(task_id)
        if fut and not fut.done():
            fut.set_result(decision)
            evt = self.store.append_event(
                task_id,
                event_type=TaskEventType.TASK_RESUMED,
                payload={"decision": decision},
            )
            asyncio.create_task(self.broadcaster.broadcast(task_id, evt))
            return True
        return False

    async def _emit_event(
        self, task_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        evt = self.store.append_event(task_id, event_type, payload)
        await self.broadcaster.broadcast(task_id, evt)
        return evt

    async def _worker_loop(self) -> None:
        """Main worker loop: pulls queued jobs and executes them durable across crashes."""
        while self._running:
            try:
                task = self.store.checkout_next_queued_task()
                if not task:
                    self._wake_event.clear()
                    try:
                        await asyncio.wait_for(self._wake_event.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        pass
                    continue

                await self._execute_task(task)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[TaskEngine Worker Error] {e}\n{traceback.format_exc()}")
                await asyncio.sleep(1.0)

    async def _execute_task(self, task: Dict[str, Any]) -> None:
        task_id = task["id"]
        user_id = task["user_id"]
        thread_id = task.get("thread_id") or task_id
        input_data = task["input"]
        user_query = input_data.get("message", "")

        # Emit task.started
        await self._emit_event(
            task_id,
            TaskEventType.TASK_STARTED,
            {"task_id": task_id, "thread_id": thread_id, "input": input_data},
        )

        active_interrupts: Dict[str, asyncio.Future] = {}
        collected_tokens: List[str] = []
        collected_tools: List[Dict[str, Any]] = []

        try:
            # Check for cancellation before executing
            if task_id in self._cancelled_tasks:
                return

            # Execute via agent.astream_run with full checkpoint persistence & interrupt support
            target_provider_idx = input_data.get("provider_index")
            target_model = input_data.get("model")

            async for event in self.agent.astream_run(
                user_query=user_query,
                chat_history=input_data.get("chat_history"),
                target_provider_index=target_provider_idx,
                target_model=target_model,
                target_user_id=user_id,
                chat_store=self.chat_store,
                active_interrupts=active_interrupts,
            ):
                if task_id in self._cancelled_tasks:
                    break

                etype = event.get("type")
                edata = event.get("data", {})

                if etype == "token":
                    tok = str(edata.get("token", ""))
                    if tok:
                        collected_tokens.append(tok)
                        await self._emit_event(
                            task_id,
                            TaskEventType.TOKEN,
                            {"token": tok, "reasoning": edata.get("reasoning", False)},
                        )

                elif etype == "tool_start":
                    t_item = {
                        "tool": edata.get("tool"),
                        "args": edata.get("args"),
                        "tool_call_id": edata.get("tool_call_id"),
                        "status": "running",
                    }
                    collected_tools.append(t_item)
                    await self._emit_event(
                        task_id,
                        TaskEventType.TOOL_STARTED,
                        t_item,
                    )

                elif etype == "tool_end":
                    t_name = edata.get("tool")
                    t_out = edata.get("output")
                    t_id = edata.get("tool_call_id")
                    for t in collected_tools:
                        if t.get("tool_call_id") == t_id or t.get("tool") == t_name:
                            t["output"] = t_out
                            t["status"] = "done"

                    await self._emit_event(
                        task_id,
                        TaskEventType.TOOL_COMPLETED,
                        {
                            "tool": t_name,
                            "output": str(t_out)[:4000],
                            "tool_call_id": t_id,
                        },
                    )

                    # Capture Artifacts / A2UI
                    if t_name == "create_artifact":
                        try:
                            parsed_art = json.loads(str(t_out))
                            await self._emit_event(
                                task_id, TaskEventType.ARTIFACT_CREATED, parsed_art
                            )
                        except Exception:
                            pass
                    elif t_name == "render_a2ui_canvas":
                        try:
                            parsed_a2ui = json.loads(str(t_out))
                            await self._emit_event(
                                task_id, TaskEventType.A2UI_CREATED, parsed_a2ui
                            )
                        except Exception:
                            pass

                elif etype in ("interrupt", "ask_question"):
                    # Transition task status to waiting_approval
                    self.store.update_task_status(
                        task_id, status=TaskStatus.WAITING_APPROVAL
                    )
                    req_id = edata.get("request_id") or edata.get("interrupt_id")
                    fut = asyncio.get_running_loop().create_future()
                    self._pending_approvals[task_id] = fut
                    if req_id:
                        self._pending_approvals[req_id] = fut

                    await self._emit_event(
                        task_id,
                        TaskEventType.APPROVAL_REQUIRED,
                        edata,
                    )

                    # Wire up to active_interrupts so the agent generator resumes
                    int_id = edata.get("interrupt_id") or req_id
                    if int_id and int_id in active_interrupts:
                        agent_fut = active_interrupts[int_id]
                        # Wait for external resume call
                        decision = await fut
                        if not agent_fut.done():
                            agent_fut.set_result(decision)
                    self.store.update_task_status(task_id, status=TaskStatus.RUNNING)

                elif etype == "done":
                    final_text = str(edata.get("content") or "".join(collected_tokens)).strip()
                    res_payload = {
                        "content": final_text,
                        "tools": collected_tools,
                        "provider": edata.get("provider"),
                    }
                    now = _utc_now()
                    self.store.update_task_status(
                        task_id,
                        status=TaskStatus.COMPLETED,
                        result=res_payload,
                        completed_at=now,
                    )
                    await self._emit_event(
                        task_id,
                        TaskEventType.TASK_COMPLETED,
                        res_payload,
                    )
                    await self._deliver_task_completion(task, res_payload, now)
                    return

                elif etype == "error":
                    err_msg = str(edata.get("error") or "Unknown error")
                    now = _utc_now()
                    self.store.update_task_status(
                        task_id,
                        status=TaskStatus.FAILED,
                        error={"error": err_msg},
                        completed_at=now,
                    )
                    await self._emit_event(
                        task_id,
                        TaskEventType.TASK_FAILED,
                        {"error": err_msg},
                    )
                    await self._deliver_task_failure(task, err_msg, now)
                    return

            # Fallback completion if done wasn't explicitly yielded
            now = _utc_now()
            res_payload = {
                "content": "".join(collected_tokens).strip(),
                "tools": collected_tools,
            }
            self.store.update_task_status(
                task_id,
                status=TaskStatus.COMPLETED,
                result=res_payload,
                completed_at=now,
            )
            await self._emit_event(
                task_id,
                TaskEventType.TASK_COMPLETED,
                res_payload,
            )
            await self._deliver_task_completion(task, res_payload, now)

        except Exception as e:
            now = _utc_now()
            err_dict = {"error": str(e), "traceback": traceback.format_exc()}
            self.store.update_task_status(
                task_id,
                status=TaskStatus.FAILED,
                error=err_dict,
                completed_at=now,
            )
            await self._emit_event(
                task_id,
                TaskEventType.TASK_FAILED,
                err_dict,
            )
            await self._deliver_task_failure(task, str(e), now)
        finally:
            self._pending_approvals.pop(task_id, None)
            self._cancelled_tasks.discard(task_id)

    async def _deliver_task_completion(
        self, task: Dict[str, Any], res_payload: Dict[str, Any], completed_at: str
    ) -> None:
        pass

    async def _deliver_task_failure(
        self, task: Dict[str, Any], error_msg: str, completed_at: str
    ) -> None:
        pass

    async def stream_task_events(
        self, task_id: str
    ) -> AsyncGenerator[str, None]:
        """
        SSE generator for task execution:
        1. Emits all historic events already recorded in task_events (safe replay on reconnect).
        2. Streams new live events as they occur.
        3. Exits when task reaches terminal status.
        """
        task = self.store.get_task(task_id)
        if not task:
            yield f"data: {json.dumps({'event': 'error', 'error': 'Task not found'})}\n\n"
            return

        # 1. Replay past events
        seen_event_ids: Set[str] = set()
        past_events = self.store.get_task_events(task_id)
        for pe in past_events:
            seen_event_ids.add(pe["id"])
            yield f"event: {pe['event_type']}\ndata: {json.dumps(pe, ensure_ascii=False, default=str)}\n\n"

        # Check if already terminal
        current_task = self.store.get_task(task_id)
        if current_task and current_task["status"] in TaskStatus.TERMINAL:
            return

        # 2. Subscribe to live events
        queue = await self.broadcaster.subscribe(task_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                    if event["id"] not in seen_event_ids:
                        seen_event_ids.add(event["id"])
                        yield f"event: {event['event_type']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

                    if event["event_type"] in (
                        TaskEventType.TASK_COMPLETED,
                        TaskEventType.TASK_FAILED,
                        TaskEventType.TASK_CANCELLED,
                    ):
                        break
                except asyncio.TimeoutError:
                    # SSE keepalive comment
                    yield ": ping\n\n"
                    # Check DB status in case completed while waiting
                    t_check = self.store.get_task(task_id)
                    if t_check and t_check["status"] in TaskStatus.TERMINAL:
                        break
        finally:
            await self.broadcaster.unsubscribe(task_id, queue)
